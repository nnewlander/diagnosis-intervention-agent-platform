from app.graph.workflow import build_agent_graph


def test_kg_supported_points_can_cover_multiple_user_points(monkeypatch):
    class FakeKGAdapter:
        provider_name = "remote"

        def __init__(self) -> None:
            self.last_status = {"mapper": "KGResponseMapper", "validation_ok": True, "error": ""}

        def search(self, query: str, keywords: list[str], top_k: int):  # noqa: ARG002
            point = keywords[0] if keywords else ""
            if point == "for循环":
                return [
                    {
                        "entity": "for循环",
                        "relation": "COMMON_MISUSE",
                        "target": "循环边界、缩进和变量更新错误",
                        "evidence": "for循环常见误区",
                        "metadata": {"source": "neo4j_core_seed"},
                    }
                ]
            if point == "条件判断":
                return [
                    {
                        "entity": "条件判断",
                        "relation": "COMMON_MISUSE",
                        "target": "条件表达式和分支结构混淆",
                        "evidence": "条件判断常见误区",
                        "metadata": {"source": "neo4j_core_seed"},
                    }
                ]
            return []

    import app.graph.nodes as nodes_module

    monkeypatch.setattr(nodes_module, "get_kg_adapter", lambda: FakeKGAdapter())
    state = build_agent_graph().invoke({"request_text": "李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个3天干预建议。"})
    kg_supported = state.get("evidence_summary", {}).get("kg_summary", {}).get("kg_supported_points", [])
    assert "for循环" in kg_supported
    assert "条件判断" in kg_supported

