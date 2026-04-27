from datetime import datetime
from typing import Any

from app.core.config import settings
from app.graph.state import AgentState
from app.services.diagnosis_service import build_diagnosis
from app.services.intent_service import parse_request_slots
from app.services.intervention_service import build_intervention_plan
from app.services.recommendation_service import recommend_and_format_packages
from app.tools.kg_adapter import get_kg_adapter
from app.tools.rag_adapter import get_rag_adapter
from app.tools.student_data_adapter import get_student_data_adapter


def _provider_meta() -> dict[str, str]:
    return {
        "rag_provider": settings.RAG_PROVIDER,
        "kg_provider": settings.KG_PROVIDER,
        "student_data_provider": settings.STUDENT_DATA_PROVIDER,
    }


def _append_trace(
    state: AgentState,
    node_name: str,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    selected_tools: list[str] | None = None,
) -> None:
    trace = state.get("debug_trace", [])
    trace.append(
        {
            "node_name": node_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "selected_task_type": state.get("task_type", "unknown"),
            "selected_tools": selected_tools or [],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **_provider_meta(),
        }
    )
    state["debug_trace"] = trace


def parse_request(state: AgentState) -> AgentState:
    text = state.get("request_text", "")
    slots = parse_request_slots(text)
    state["parsed_slots"] = slots
    state["task_type"] = slots.get("task_type", "unknown")
    state["student_id"] = slots.get("student_id", "")
    state["class_id"] = slots.get("class_id", "")
    state["knowledge_points"] = slots.get("knowledge_points", [])
    state["desired_days"] = slots.get("desired_days", 0)
    state["error_type"] = slots.get("error_type", "")
    state["task_priority"] = slots.get("task_priority", "medium")
    state["student_mention"] = slots.get("student_mention", "")
    _append_trace(
        state,
        node_name="parse_request",
        input_summary={"request_length": len(text)},
        output_summary={"slots": slots},
        selected_tools=["rule_parser_v2"],
    )
    return state


def route_task(state: AgentState) -> AgentState:
    detected = state.get("parsed_slots", {}).get("detected_task_types", [])
    primary = detected[0] if detected else "unknown"
    secondary = detected[1:] if len(detected) > 1 else []
    state["primary_task_type"] = primary
    state["secondary_task_types"] = secondary
    state["task_type"] = "mixed" if len(detected) > 1 else primary
    _append_trace(
        state,
        node_name="route_task",
        input_summary={"detected_tasks": detected},
        output_summary={
            "task_type": state.get("task_type"),
            "primary_task_type": primary,
            "secondary_task_types": secondary,
        },
        selected_tools=["rule_router_v2"],
    )
    return state


def clarify_if_needed(state: AgentState) -> AgentState:
    student_adapter = get_student_data_adapter()
    resolver = student_adapter.resolve_student(
        student_id=state.get("student_id", ""),
        student_mention=state.get("student_mention", ""),
    )
    state["resolver_result"] = resolver
    if resolver.get("student_id"):
        state["student_id"] = resolver.get("student_id", "")

    questions: list[str] = []
    if resolver.get("need_clarify"):
        questions.append(resolver.get("clarify_message", "请补充 student_id。"))
    if state.get("primary_task_type") in {"diagnosis", "intervention", "dispatch"} and not state.get(
        "knowledge_points", []
    ):
        questions.append("请补充关注知识点，如 for循环、函数、字符串、异常处理。")
    state["clarify_questions"] = questions
    state["need_clarify"] = len(questions) > 0
    _append_trace(
        state,
        node_name="clarify_if_needed",
        input_summary={"resolver_result": resolver},
        output_summary={"need_clarify": state.get("need_clarify"), "questions": questions},
        selected_tools=["student_data_adapter.resolve_student"],
    )
    return state


def fetch_rag_evidence(state: AgentState) -> AgentState:
    rag_adapter = get_rag_adapter()
    keywords = state.get("knowledge_points", []) or [state.get("request_text", "")]
    rag = rag_adapter.search(
        query=state.get("request_text", ""),
        keywords=keywords,
        top_k=settings.TOP_K_RAG,
    )
    state["rag_evidence"] = rag
    rag_status = getattr(rag_adapter, "last_status", {})
    _append_trace(
        state,
        node_name="fetch_rag_evidence",
        input_summary={"keywords": keywords[:5]},
        output_summary={
            "rag_hits": len(rag),
            "provider": rag_adapter.provider_name,
            "mapper": rag_status.get("mapper", ""),
            "validation_ok": rag_status.get("validation_ok", True),
            "validation_error": rag_status.get("error", ""),
        },
        selected_tools=["rag_adapter.search"],
    )
    return state


def fetch_kg_evidence(state: AgentState) -> AgentState:
    kg_adapter = get_kg_adapter()
    keywords = state.get("knowledge_points", [])[:]
    if state.get("error_type"):
        keywords.append(state["error_type"])
    if not keywords:
        keywords = [state.get("request_text", "")]
    kg = kg_adapter.search(
        query=state.get("request_text", ""),
        keywords=keywords,
        top_k=settings.TOP_K_KG,
    )
    state["kg_evidence"] = kg
    kg_status = getattr(kg_adapter, "last_status", {})
    _append_trace(
        state,
        node_name="fetch_kg_evidence",
        input_summary={"keywords": keywords[:5]},
        output_summary={
            "kg_hits": len(kg),
            "provider": kg_adapter.provider_name,
            "mapper": kg_status.get("mapper", ""),
            "validation_ok": kg_status.get("validation_ok", True),
            "validation_error": kg_status.get("error", ""),
        },
        selected_tools=["kg_adapter.search"],
    )
    return state


def fetch_mysql_evidence(state: AgentState) -> AgentState:
    student_adapter = get_student_data_adapter()
    student_id = state.get("student_id", "")
    mysql = student_adapter.load_student_evidence(student_id=student_id)
    state["mysql_evidence"] = mysql
    _append_trace(
        state,
        node_name="fetch_mysql_evidence",
        input_summary={"student_id": student_id},
        output_summary={
            "profile_found": bool(mysql.get("profile_summary")),
            "submission_count": mysql.get("recent_submission_summary", {}).get("total", 0),
            "provider": student_adapter.provider_name,
        },
        selected_tools=["student_data_adapter.load_student_evidence"],
    )
    return state


def generate_diagnosis(state: AgentState) -> AgentState:
    diagnosis = build_diagnosis(
        mysql_evidence=state.get("mysql_evidence", {}),
        kg_evidence=state.get("kg_evidence", []),
    )
    state["diagnosis"] = diagnosis
    _append_trace(
        state,
        node_name="generate_diagnosis",
        input_summary={"need_clarify": state.get("need_clarify", False)},
        output_summary={"confidence_level": diagnosis.get("confidence_level", "low")},
        selected_tools=["diagnosis_rule_engine_v2"],
    )
    return state


def generate_intervention(state: AgentState) -> AgentState:
    student_adapter = get_student_data_adapter()
    cases = student_adapter.get_intervention_cases(limit=3)
    state["intervention_case_evidence"] = cases
    plan = build_intervention_plan(
        diagnosis=state.get("diagnosis", {}),
        knowledge_points=state.get("knowledge_points", []),
        intervention_cases=cases,
        desired_days=state.get("desired_days", 0) or 3,
    )
    state["intervention_plan"] = plan
    _append_trace(
        state,
        node_name="generate_intervention",
        input_summary={"desired_days": state.get("desired_days", 0)},
        output_summary={"plan_mode": plan.get("mode", "normal"), "case_count": len(cases)},
        selected_tools=["student_data_adapter.get_intervention_cases", "intervention_rule_engine_v2"],
    )
    return state


def recommend_package(state: AgentState) -> AgentState:
    profile_summary = state.get("mysql_evidence", {}).get("profile_summary", {})
    grade_band = profile_summary.get("grade_band", "")
    difficulty = "基础" if state.get("task_priority") != "high" else "提升"
    packages = recommend_and_format_packages(
        knowledge_points=state.get("knowledge_points", []),
        grade_band=grade_band,
        difficulty_level=difficulty,
    )
    state["recommended_packages"] = packages
    _append_trace(
        state,
        node_name="recommend_package",
        input_summary={"grade_band": grade_band, "difficulty_level": difficulty},
        output_summary={"recommended_count": len(packages)},
        selected_tools=["package_adapter.recommend"],
    )
    return state


def _build_evidence_summary(state: AgentState) -> dict[str, Any]:
    rag_items = state.get("rag_evidence", [])
    kg_items = state.get("kg_evidence", [])
    mysql = state.get("mysql_evidence", {})
    case_items = state.get("intervention_case_evidence", [])
    return {
        "rag_summary": {
            "hit_count": len(rag_items),
            "provider": settings.RAG_PROVIDER,
            "schema": "RAGEvidenceItem",
            "preview": rag_items[:2],
        },
        "kg_summary": {
            "hit_count": len(kg_items),
            "provider": settings.KG_PROVIDER,
            "schema": "KGEvidenceItem",
            "preview": kg_items[:2],
        },
        "mysql_summary": {
            "provider": settings.STUDENT_DATA_PROVIDER,
            "schema": "StudentEvidenceBundle",
            "evidence": mysql,
        },
        "intervention_case_summary": {
            "provider": settings.STUDENT_DATA_PROVIDER,
            "hit_count": len(case_items),
            "preview": case_items[:2],
        },
    }


def build_final_response(state: AgentState) -> AgentState:
    state["evidence_summary"] = _build_evidence_summary(state)
    diagnosis = state.get("diagnosis", {})
    plan = state.get("intervention_plan", {})
    clarify_lines = (
        "\n补充信息：\n- " + "\n- ".join(state.get("clarify_questions", []))
        if state.get("need_clarify")
        else ""
    )
    state["final_response"] = (
        f"我已先按“{state.get('primary_task_type', 'unknown')}”处理你的请求，"
        f"并补充了次任务 {state.get('secondary_task_types', [])} 的结果。\n"
        f"当前观察：{diagnosis.get('observed_problem', '暂无明确观察')}\n"
        f"可能原因：{diagnosis.get('probable_cause', '暂无明确原因')}\n"
        f"建议目标：{plan.get('intervention_goal', '请先补齐关键信息')}\n"
        f"已推荐练习包 {len(state.get('recommended_packages', []))} 个。"
        f"{clarify_lines}"
    )
    _append_trace(
        state,
        node_name="build_final_response",
        input_summary={"task_type": state.get("task_type", "unknown")},
        output_summary={"final_response_non_empty": bool(state.get("final_response", "").strip())},
        selected_tools=["response_builder_v2"],
    )
    return state
