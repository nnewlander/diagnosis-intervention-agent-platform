from app.graph.workflow import build_agent_graph
from app.services.intent_service import parse_request_slots


def _run(request_text: str) -> dict:
    return build_agent_graph().invoke({"request_text": request_text})


def test_user_mentioned_knowledge_points_extract_variable_and_for():
    slots = parse_request_slots("李同学最近几次作业在变量定义和 for循环上反复出错，帮我诊断一下。")
    points = slots.get("user_mentioned_knowledge_points", [])
    assert "变量定义" in points
    assert "for循环" in points


def test_alignment_status_not_forced_to_weak_points_only():
    state = _run("请诊断 student_id:STU-0001 在变量定义和 for循环上的反复出错问题")
    alignment = state.get("mysql_evidence", {}).get("alignment_summary", {})
    assert alignment.get("evidence_alignment_status") in {
        "aligned",
        "partially_aligned",
        "mismatched",
        "insufficient_data",
    }
    # Ensure user focus remains visible in alignment summary.
    mentioned = set(alignment.get("matched_user_mentioned_points", []) + alignment.get("unmatched_user_mentioned_points", []))
    assert "变量定义" in mentioned
    assert "for循环" in mentioned


def test_diagnosis_observed_problem_contains_user_focus_and_data_support():
    state = _run("请诊断 student_id:STU-0001 在变量定义和 for循环上的反复出错问题")
    observed = state.get("diagnosis", {}).get("observed_problem", "")
    assert "用户关注" in observed
    assert "数据支持" in observed
    basis = state.get("diagnosis", {}).get("evidence_basis", {})
    assert "user_mentioned_knowledge_points" in basis
    assert "evidence_alignment_status" in basis


def test_evidence_summary_contains_multi_source_supported_points():
    state = _run("请诊断 student_id:STU-0001 在变量定义和 for循环上的反复出错问题")
    summary = state.get("evidence_summary", {})
    assert "rag_supported_points" in summary.get("rag_summary", {})
    assert "kg_supported_points" in summary.get("kg_summary", {})
    assert "student_data_supported_points" in summary.get("mysql_summary", {})


def test_technical_qa_path_still_short_and_unchanged():
    state = _run("课堂演示遇到 NameError，应该怎么给学生解释？")
    assert state.get("primary_task_type") == "technical_qa"
    assert state.get("routing_mode") == "technical_qa_short_path"
    assert state.get("intervention_plan", {}) in ({}, None)
    assert state.get("recommended_packages", []) == []

