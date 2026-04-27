from typing import Any

from app.core.config import settings
from app.graph.workflow import build_agent_graph
from app.tools.kg_adapter import RemoteKGAdapter
from app.tools.rag_adapter import LocalRAGAdapter, RemoteRAGAdapter
from app.tools.response_mappers.kg_mapper import KGResponseMapper
from app.tools.response_mappers.rag_mapper import RAGResponseMapper


def test_local_provider_returns_unified_rag_schema():
    adapter = LocalRAGAdapter()
    items = adapter.search("函数报错", ["函数"], top_k=2)
    if items:
        item = items[0]
        assert "source_id" in item
        assert "title" in item
        assert "snippet" in item
        assert "score" in item


def test_remote_mapper_handles_field_variants():
    rag_mapper = RAGResponseMapper()
    kg_mapper = KGResponseMapper()
    rag_items = rag_mapper.map_items(
        {"hits": [{"id": "a1", "name": "doc", "text": "body", "similarity": 0.8}]}
    )
    kg_items = kg_mapper.map_items(
        {
            "records": [
                {
                    "subject": "函数",
                    "subject_type": "kp",
                    "predicate": "related_error",
                    "object": "TypeError",
                    "snippet": "类型错误",
                    "confidence": 0.9,
                }
            ]
        }
    )
    assert rag_items and rag_items[0]["source_id"] == "a1"
    assert kg_items and kg_items[0]["entity"] == "函数"


def test_remote_adapter_contract_fail_does_not_crash(monkeypatch):
    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"unexpected": "shape"}

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.rag_adapter as rag_module

    monkeypatch.setattr(rag_module.requests, "post", fake_post)
    adapter = RemoteRAGAdapter()
    items = adapter.search("query", ["k"], 3)
    assert items == []
    assert adapter.last_status["validation_ok"] is False


def test_workflow_debug_trace_contains_provider_mapper_validation(monkeypatch):
    class FakeRagResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"hits": [{"id": "r1", "name": "remote rag", "text": "snippet"}]}

    class FakeKgResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "records": [
                    {"subject": "函数", "predicate": "causes", "object": "TypeError", "snippet": "x"}
                ]
            }

    def fake_rag_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeRagResp()

    def fake_kg_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeKgResp()

    import app.tools.rag_adapter as rag_module
    import app.tools.kg_adapter as kg_module

    monkeypatch.setattr(settings, "RAG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(rag_module.requests, "post", fake_rag_post)
    monkeypatch.setattr(kg_module.requests, "post", fake_kg_post)

    state = build_agent_graph().invoke({"request_text": "请诊断 student_id:STU-0001 的函数报错"})
    trace = state.get("debug_trace", [])
    rag_trace = [t for t in trace if t.get("node_name") == "fetch_rag_evidence"][0]
    kg_trace = [t for t in trace if t.get("node_name") == "fetch_kg_evidence"][0]
    assert rag_trace["rag_provider"] == "remote"
    assert rag_trace["output_summary"]["mapper"] == "RAGResponseMapper"
    assert kg_trace["output_summary"]["validation_ok"] is True


def test_workflow_not_crash_when_remote_contract_invalid(monkeypatch):
    class BadResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"bad_shape": True}

    def fake_bad_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return BadResp()

    import app.tools.rag_adapter as rag_module
    import app.tools.kg_adapter as kg_module

    monkeypatch.setattr(settings, "RAG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(rag_module.requests, "post", fake_bad_post)
    monkeypatch.setattr(kg_module.requests, "post", fake_bad_post)

    state = build_agent_graph().invoke({"request_text": "请诊断 student_id:STU-0001 的报错"})
    assert isinstance(state, dict)
    rag_trace = [t for t in state.get("debug_trace", []) if t.get("node_name") == "fetch_rag_evidence"][0]
    assert rag_trace["output_summary"]["validation_ok"] is False
