from app.core.config import settings
from app.tools.rag_adapter import RemoteRAGAdapter


def test_remote_rag_url_join_correct(monkeypatch):
    captured = {"url": ""}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": []}

    def fake_post(url, json, headers, timeout):  # noqa: ANN001
        captured["url"] = url
        return FakeResp()

    import app.tools.rag_adapter as rag_module

    monkeypatch.setattr(settings, "RAG_API_BASE", "http://127.0.0.1:8001/")
    monkeypatch.setattr(settings, "RAG_ENDPOINT", "/search")
    monkeypatch.setattr(rag_module.requests, "post", fake_post)

    RemoteRAGAdapter().search("q", ["k"], 3)
    assert captured["url"] == "http://127.0.0.1:8001/search"


def test_search_style_response_maps(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": [{"id": "h1", "name": "doc", "text": "snippet", "similarity": 0.9}]}

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.rag_adapter as rag_module

    monkeypatch.setattr(rag_module.requests, "post", fake_post)
    items = RemoteRAGAdapter().search("q", ["k"], 3)
    assert len(items) == 1
    assert items[0]["source_id"] == "h1"


def test_ask_style_response_fallback(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "这是一个NameError，通常是变量未定义。"}

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.rag_adapter as rag_module

    monkeypatch.setattr(rag_module.requests, "post", fake_post)
    adapter = RemoteRAGAdapter()
    items = adapter.search("q", ["k"], 3)
    assert len(items) == 1
    assert "fallback_applied" in adapter.last_status["error"]


def test_empty_hits_no_crash(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": []}

    def fake_post(*args, **kwargs):  # noqa: ANN001, ANN002
        return FakeResp()

    import app.tools.rag_adapter as rag_module

    monkeypatch.setattr(rag_module.requests, "post", fake_post)
    items = RemoteRAGAdapter().search("q", ["k"], 3)
    assert items == []


def test_smoke_real_rag_force_remote(monkeypatch):
    class FakeGraph:
        def invoke(self, payload):  # noqa: ANN001
            return {
                "rag_query": "课堂演示遇到 NameError NameError 变量",
                "error_type": "NameError",
                "parsed_slots": {"knowledge_points": ["变量"]},
                "rag_evidence": [{"source_id": "x", "title": "t", "snippet": "s", "source_type": "faq", "metadata": {"route": "bm25_faq"}}],
                "final_response": "ok",
                "debug_trace": [
                    {"node_name": "fetch_rag_evidence", "output_summary": {"validation_ok": True, "mapper": "m"}}
                ],
            }

    import scripts.smoke_test_real_rag as smoke_module

    monkeypatch.setattr(smoke_module, "build_agent_graph", lambda: FakeGraph())
    monkeypatch.setattr(smoke_module.sys, "argv", ["smoke_test_real_rag.py", "--force-remote"])
    exit_code = smoke_module.main()
    assert exit_code == 0
    assert settings.RAG_PROVIDER == "remote"


def test_smoke_fallback_warn_and_fail_on_fallback(monkeypatch, capsys):
    class FakeGraph:
        def invoke(self, payload):  # noqa: ANN001
            return {
                "rag_query": "q",
                "error_type": "NameError",
                "parsed_slots": {"knowledge_points": ["变量"]},
                "rag_evidence": [
                    {
                        "source_id": "FALLBACK-NameError",
                        "title": "fallback",
                        "snippet": "fallback content",
                        "source_type": "fallback_error_guide",
                        "metadata": {"fallback": True},
                    }
                ],
                "final_response": "ok",
                "debug_trace": [
                    {"node_name": "fetch_rag_evidence", "output_summary": {"validation_ok": True, "mapper": "m"}}
                ],
            }

    import scripts.smoke_test_real_rag as smoke_module

    monkeypatch.setattr(smoke_module, "build_agent_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        smoke_module.sys, "argv", ["smoke_test_real_rag.py", "--force-remote", "--fail-on-fallback"]
    )
    exit_code = smoke_module.main()
    out = capsys.readouterr().out
    assert "fallback evidence" in out
    assert exit_code != 0


def test_smoke_faq_evidence_not_fail(monkeypatch, capsys):
    class FakeGraph:
        def invoke(self, payload):  # noqa: ANN001
            return {
                "rag_query": "q",
                "error_type": "NameError",
                "parsed_slots": {"knowledge_points": ["变量"]},
                "rag_evidence": [
                    {
                        "source_id": "seed-faq-nameerror",
                        "title": "faq",
                        "snippet": "faq content",
                        "source_type": "faq",
                        "metadata": {"route": "bm25_faq", "fallback": False},
                    }
                ],
                "final_response": "ok",
                "debug_trace": [
                    {"node_name": "fetch_rag_evidence", "output_summary": {"validation_ok": True, "mapper": "m"}}
                ],
            }

    import scripts.smoke_test_real_rag as smoke_module

    monkeypatch.setattr(smoke_module, "build_agent_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        smoke_module.sys, "argv", ["smoke_test_real_rag.py", "--force-remote", "--fail-on-fallback"]
    )
    exit_code = smoke_module.main()
    out = capsys.readouterr().out
    assert "真实 FAQ/BM25 evidence" in out
    assert exit_code == 0


def test_smoke_prints_rag_query(monkeypatch, capsys):
    class FakeGraph:
        def invoke(self, payload):  # noqa: ANN001
            return {
                "rag_query": "课堂演示遇到 NameError ...",
                "error_type": "NameError",
                "parsed_slots": {"knowledge_points": ["变量"]},
                "rag_evidence": [],
                "final_response": "ok",
                "debug_trace": [
                    {"node_name": "fetch_rag_evidence", "output_summary": {"validation_ok": True, "mapper": "m"}}
                ],
            }

    import scripts.smoke_test_real_rag as smoke_module

    monkeypatch.setattr(smoke_module, "build_agent_graph", lambda: FakeGraph())
    monkeypatch.setattr(smoke_module.sys, "argv", ["smoke_test_real_rag.py", "--force-remote"])
    smoke_module.main()
    out = capsys.readouterr().out
    assert "rag_query=" in out
