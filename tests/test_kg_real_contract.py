from typing import Any

import pytest

from app.core.config import Settings, settings
from app.tools.kg_adapter import LocalKGAdapter, RemoteKGAdapter
from app.tools.response_mappers.kg_mapper import KGResponseMapper


def test_settings_default_kg_api_base_is_project_three_port():
    default = Settings.model_fields["KG_API_BASE"].default
    assert default == "http://127.0.0.1:8002"
    endpoint_default = Settings.model_fields["KG_ENDPOINT"].default
    assert endpoint_default == "/graph_query"


def test_remote_kg_adapter_url_concat(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"records": []}

    def fake_post(url: str, **kwargs):  # noqa: ANN001
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResp()

    import app.tools.kg_adapter as kg_module

    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_API_BASE", "http://127.0.0.1:8002/")
    monkeypatch.setattr(settings, "KG_ENDPOINT", "/graph_query")
    monkeypatch.setattr(kg_module.requests, "post", fake_post)

    adapter = RemoteKGAdapter()
    items = adapter.search(query="q", keywords=["NameError"], top_k=3)
    assert items == []
    assert captured["url"] == "http://127.0.0.1:8002/graph_query"
    assert captured["json"]["query"] == "q"
    assert captured["json"]["entity_terms"] == ["NameError"]
    assert captured["json"]["top_k"] == 3
    assert "request_id" in captured["json"]


def test_project3_records_style_can_map():
    mapper = KGResponseMapper()
    payload = {
        "records": [
            {
                "entity": "NameError",
                "entity_type": "error",
                "relation": "HAS_SOLUTION",
                "target": "变量未定义",
                "evidence": "NameError 通常表示变量未定义或命名不一致",
                "score": 0.91,
                "metadata": {
                    "source": "neo4j_core_seed",
                    "seed_version": "v1",
                    "confidence": 0.91,
                    "cypher_template": "T1",
                    "relation_props": {"w": 1},
                },
            }
        ]
    }
    items = mapper.map_items(payload)
    assert items
    first = items[0]
    assert first["entity"] == "NameError"
    assert first["relation"] == "HAS_SOLUTION"
    assert "neo4j_core_seed" in str((first.get("metadata") or {}).get("source", ""))


def test_empty_records_ok_and_not_crash(monkeypatch):
    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"records": []}

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.kg_adapter as kg_module

    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_API_BASE", "http://127.0.0.1:8002")
    monkeypatch.setattr(settings, "KG_ENDPOINT", "/graph_query")
    monkeypatch.setattr(kg_module.requests, "post", fake_post)

    adapter = RemoteKGAdapter()
    items = adapter.search(query="q", keywords=["k"], top_k=2)
    assert items == []
    assert adapter.last_status["validation_ok"] is True


def test_nonempty_records_validation_ok(monkeypatch):
    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "records": [
                    {
                        "entity": "NameError",
                        "entity_type": "error",
                        "relation": "RELATED_ERROR",
                        "target": "变量未定义",
                        "evidence": "常见于变量未定义或命名不一致",
                        "score": 0.92,
                        "metadata": {"source": "neo4j_core_seed"},
                    }
                ]
            }

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.kg_adapter as kg_module

    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_API_BASE", "http://127.0.0.1:8002")
    monkeypatch.setattr(settings, "KG_ENDPOINT", "/graph_query")
    monkeypatch.setattr(kg_module.requests, "post", fake_post)

    adapter = RemoteKGAdapter()
    items = adapter.search(query="q", keywords=["NameError"], top_k=3)
    assert len(items) >= 1
    assert adapter.last_status["validation_ok"] is True


def test_complete_fields_enter_kgevidenceitem_via_mapper():
    items = KGResponseMapper().map_items(
        {
            "records": [
                {
                    "entity": "变量定义",
                    "entity_type": "kp",
                    "relation": "COMMON_MISUSE",
                    "target": "变量未定义或命名不一致",
                    "evidence": "课堂中常见：先用后定义，或拼写不一致",
                    "score": 0.8,
                }
            ]
        }
    )
    assert items
    first = items[0]
    assert first["entity"]
    assert first["relation"]
    assert first["target"]
    assert first["evidence"]


def test_local_kg_mode_not_affected():
    adapter = LocalKGAdapter()
    items = adapter.search("NameError", ["NameError"], top_k=2)
    assert isinstance(items, list)

