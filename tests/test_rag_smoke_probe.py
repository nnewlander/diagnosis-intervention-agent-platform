"""Tests for scripts/rag_smoke_probe.py (joint RAG+KG smoke helpers)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rag_smoke_probe as rsp


def test_is_rag_fallback_item_detects_variants():
    assert rsp.is_rag_fallback_item({"source_id": "FALLBACK-NameError", "metadata": {}})
    assert rsp.is_rag_fallback_item({"source_id": "FALLBACK-x", "metadata": {}})
    assert rsp.is_rag_fallback_item({"source_type": "fallback_error_guide", "source_id": "x"})
    assert rsp.is_rag_fallback_item({"source_id": "ok", "metadata": {"fallback": True}})
    assert not rsp.is_rag_fallback_item({"source_id": "seed-faq-nameerror", "source_type": "faq", "metadata": {}})


def test_joint_exit_code_for_rag_fallback():
    items = [{"source_id": "FALLBACK-NameError", "metadata": {}}]
    assert rsp.joint_exit_code_for_rag_fallback(items, fail_on_rag_fallback=True) == 1
    assert rsp.joint_exit_code_for_rag_fallback(items, fail_on_rag_fallback=False) == 0
    assert rsp.joint_exit_code_for_rag_fallback([], fail_on_rag_fallback=True) == 0


def test_rag_needs_warmup_when_all_flags_not_ready():
    assert rsp.rag_needs_warmup(
        {"faq_ready": False, "bm25_ready": False, "lightweight_search_ready": False}
    )
    assert rsp.rag_needs_warmup({})
    assert not rsp.rag_needs_warmup(
        {"faq_ready": True, "bm25_ready": False, "lightweight_search_ready": False}
    )


def test_run_rag_warmup_and_reprobe_calls_warmup(monkeypatch):
    calls: list[str] = []

    class Resp:
        def __init__(self, payload: dict):
            self.ok = True
            self.status_code = 200
            self.text = ""
            self.content = b"{}"
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    def fake_get(url: str, timeout: float = 5.0):  # noqa: ARG001
        calls.append(url)
        if "/health" in url:
            return Resp({"status": "ok"})
        if "/warmup" in url:
            return Resp({"warmed": True})
        if "/ready" in url:
            if any("/warmup" in u for u in calls):
                return Resp(
                    {
                        "faq_ready": True,
                        "bm25_ready": True,
                        "lightweight_search_ready": True,
                    }
                )
            return Resp(
                {
                    "faq_ready": False,
                    "bm25_ready": False,
                    "lightweight_search_ready": False,
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(rsp.requests, "get", fake_get)

    before = {
        "faq_ready": False,
        "bm25_ready": False,
        "lightweight_search_ready": False,
    }
    out = rsp.run_rag_warmup_and_reprobe(
        "http://127.0.0.1:8001",
        before,
        enabled=True,
        timeout=2.0,
        warmup_timeout=5.0,
    )
    assert out["warmup_called"] is True
    assert any("/warmup" in u for u in calls)
    after = out.get("ready_after") or {}
    assert after.get("faq_ready") is True
    assert out.get("probe_after", {}).get("ready_ok") is True


def test_kg_smoke_probe_import_unchanged():
    import kg_smoke_probe as kp

    assert kp.MOCK_KG_PORT == 9003
    assert "probe_kg_health_ready" in dir(kp)
