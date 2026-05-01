from app.graph.workflow import build_agent_graph
from app.services.recommendation_service import _extract_matched_knowledge_points
from app.services.diagnosis_service import build_diagnosis


def _run(request_text: str) -> dict:
    return build_agent_graph().invoke({"request_text": request_text})


def test_mixed_forloop_condition_no_variable_definition_residual():
    state = _run("李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个 3 天干预建议。")
    final_response = str(state.get("final_response", ""))
    assert "for循环" in final_response
    assert "条件判断" in final_response
    assert "变量定义方向由学情记录直接支持" not in final_response


def test_mismatched_alignment_confidence_not_high():
    diagnosis = build_diagnosis(
        mysql_evidence={
            "profile_summary": {"student_id": "STU-X"},
            "recent_submission_summary": {"total": 8, "submissions": []},
            "recent_error_summary": {"error_distribution": {"unknown": 8}},
            "weak_point_summary": {"weak_knowledge_points": ["课堂互动", "平台登录"]},
            "alignment_summary": {
                "matched_user_mentioned_points": [],
                "unmatched_user_mentioned_points": ["for循环", "条件判断"],
                "data_weak_points": ["课堂互动", "平台登录"],
                "evidence_alignment_status": "mismatched",
            },
        },
        kg_evidence=[{"entity": "for循环", "relation": "COMMON_MISUSE", "target": "循环边界错误"}],
        rag_evidence=[{"title": "条件判断 FAQ", "source_id": "seed-cond"}],
        user_mentioned_points=["for循环", "条件判断"],
    )
    assert diagnosis.get("confidence_level") != "high"


def test_mixed_final_response_mentions_rag_and_kg_support():
    state = _run("李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个 3 天干预建议。")
    text = str(state.get("final_response", ""))
    assert "RAG 证据当前主要支持" in text
    assert "KG 证据当前主要支持" in text


def test_mixed_final_response_no_repeated_punctuation_and_has_3day_lines():
    state = _run("李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个 3 天干预建议。")
    text = str(state.get("final_response", ""))
    assert "。。" not in text
    assert "；；" not in text
    assert "3天干预建议" in text
    assert "- 第1天：" in text
    assert "- 第2天：" in text
    assert "- 第3天：" in text


def test_extract_matched_knowledge_points_prefers_title_then_reason():
    req = ["for循环", "条件判断"]
    m1 = _extract_matched_knowledge_points(
        title="算法思维_函数返回值、for循环_补练包_065",
        reason="匹配知识点: for循环, 条件判断；年级匹配: 小学高年级",
        request_points=req,
    )
    assert m1 == ["for循环"]
    m2 = _extract_matched_knowledge_points(
        title="Turtle绘图_条件判断_补练包_008",
        reason="匹配知识点: for循环, 条件判断",
        request_points=req,
    )
    assert m2 == ["条件判断"]
    m3 = _extract_matched_knowledge_points(
        title="Turtle绘图_Turtle绘图、条件判断_补练包_057",
        reason="匹配知识点: for循环, 条件判断",
        request_points=req,
    )
    assert m3 == ["条件判断"]

