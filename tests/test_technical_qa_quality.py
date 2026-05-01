from app.graph.workflow import build_agent_graph


def _run(request_text: str) -> dict:
    return build_agent_graph().invoke({"request_text": request_text})


def test_nameerror_case_rag_top1_seed_faq():
    state = _run("学生问 NameError 是什么意思，能结合知识图谱和资料给我一个课堂解释吗？")
    rag_items = state.get("rag_evidence", [])
    assert isinstance(rag_items, list)
    assert rag_items
    top1 = rag_items[0]
    assert "nameerror" in str(top1.get("source_id", "")).lower()


def test_nameerror_final_response_contains_classroom_phrase():
    state = _run("学生问 NameError 是什么意思，能结合知识图谱和资料给我一个课堂解释吗？")
    text = str(state.get("final_response", ""))
    assert "程序找不到这个名字" in text or "找不到这个名字" in text


def test_nameerror_final_response_prefers_kg_has_solution():
    state = _run("学生问 NameError 是什么意思，能结合知识图谱和资料给我一个课堂解释吗？")
    text = str(state.get("final_response", ""))
    kg_items = state.get("kg_evidence", [])
    has_solution_in_kg = any(str(x.get("relation", "")) == "HAS_SOLUTION" for x in kg_items)
    if has_solution_in_kg:
        assert "HAS_SOLUTION" in text
    else:
        # fallback: at least keep KG reference section present
        assert "- KG:" in text

