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


PLAN_KEYWORDS = ["干预", "计划", "3天", "练习", "下发", "推荐"]


def _text_has_point(text: str, point: str) -> bool:
    if not text or not point:
        return False
    return point in text


def _extract_supported_points_from_rag(points: list[str], rag_items: list[dict[str, Any]]) -> list[str]:
    supported: list[str] = []
    for p in points:
        for item in rag_items:
            text = " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("snippet", "")),
                    str(item.get("source_id", "")),
                ]
            )
            if _text_has_point(text, p):
                supported.append(p)
                break
    return list(dict.fromkeys(supported))


def _extract_supported_points_from_kg(points: list[str], kg_items: list[dict[str, Any]]) -> list[str]:
    supported: list[str] = []
    for p in points:
        for item in kg_items:
            text = " ".join(
                [
                    str(item.get("entity", "")),
                    str(item.get("relation", "")),
                    str(item.get("target", "")),
                    str(item.get("evidence", "")),
                ]
            )
            if _text_has_point(text, p):
                supported.append(p)
                break
    return list(dict.fromkeys(supported))


def _clean_response_text(text: str) -> str:
    cleaned = text.replace("。。", "。").replace("；；", "；")
    # collapse duplicated spaces but keep line breaks
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


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
    state["user_mentioned_knowledge_points"] = slots.get("user_mentioned_knowledge_points", [])
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
    requires_student_context = primary in {"diagnosis", "intervention", "dispatch"} or any(
        t in {"diagnosis", "intervention", "dispatch"} for t in secondary
    )
    if primary == "technical_qa" and not requires_student_context:
        state["routing_mode"] = "technical_qa_short_path"
    else:
        state["routing_mode"] = "task_based_routing"
    _append_trace(
        state,
        node_name="route_task",
        input_summary={"detected_tasks": detected},
        output_summary={
            "task_type": state.get("task_type"),
            "primary_task_type": primary,
            "secondary_task_types": secondary,
            "routing_mode": state.get("routing_mode"),
        },
        selected_tools=["rule_router_v2"],
    )
    return state


def clarify_if_needed(state: AgentState) -> AgentState:
    primary = state.get("primary_task_type", "unknown")
    secondary = state.get("secondary_task_types", [])
    requires_student_context = primary in {"diagnosis", "intervention", "dispatch"} or any(
        t in {"diagnosis", "intervention", "dispatch"} for t in secondary
    )

    resolver = {}
    questions: list[str] = []

    if requires_student_context:
        student_adapter = get_student_data_adapter()
        resolver = student_adapter.resolve_student(
            student_id=state.get("student_id", ""),
            student_mention=state.get("student_mention", ""),
        )
        state["resolver_result"] = resolver
        if resolver.get("student_id"):
            state["student_id"] = resolver.get("student_id", "")
        if resolver.get("need_clarify"):
            questions.append(resolver.get("clarify_message", "请补充 student_id。"))
    else:
        state["resolver_result"] = {
            "student_id": state.get("student_id", ""),
            "resolved_by": "not_required_for_task",
            "need_clarify": False,
        }

    if requires_student_context and not state.get("knowledge_points", []):
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
    request_text = state.get("request_text", "")
    error_type = state.get("error_type", "")
    knowledge_points = state.get("knowledge_points", [])
    rag_query_parts = [request_text]
    if error_type:
        rag_query_parts.append(error_type)
    if knowledge_points:
        rag_query_parts.extend(knowledge_points)
    rag_query = " ".join([p for p in rag_query_parts if p]).strip()
    state["rag_query"] = rag_query
    keywords = knowledge_points[:] if knowledge_points else [request_text]
    if error_type and error_type not in keywords:
        keywords.append(error_type)
    rag = rag_adapter.search(
        query=rag_query,
        keywords=keywords,
        top_k=settings.TOP_K_RAG,
    )
    state["rag_evidence"] = rag
    rag_status = getattr(rag_adapter, "last_status", {})
    _append_trace(
        state,
        node_name="fetch_rag_evidence",
        input_summary={"keywords": keywords[:5], "rag_query": rag_query},
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
    request_text = state.get("request_text", "")
    error_type = state.get("error_type", "")
    knowledge_points = state.get("knowledge_points", [])[:]
    entity_terms = []
    if error_type:
        entity_terms.append(error_type)
    entity_terms.extend([kp for kp in knowledge_points if kp])
    if not entity_terms:
        entity_terms = [request_text]
    kg_query = " ".join([p for p in [request_text, error_type] + knowledge_points if p]).strip()
    # Multi-point query: try each user-mentioned point for better coverage, then merge deduplicated results.
    user_points = state.get("user_mentioned_knowledge_points", []) or []
    if len(user_points) > 1:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        per_point_results: list[list[dict[str, Any]]] = []
        for point in user_points:
            point_query = f"{request_text} {point}".strip()
            point_keywords = [point]
            if error_type:
                point_keywords.append(error_type)
            part = kg_adapter.search(
                query=point_query,
                keywords=point_keywords,
                top_k=settings.TOP_K_KG,
            )
            per_point_results.append(part)
        # round-robin merge to avoid first point occupying all top_k slots
        max_len = max((len(x) for x in per_point_results), default=0)
        for idx in range(max_len):
            for part in per_point_results:
                if idx >= len(part):
                    continue
                item = part[idx]
                sig = (
                    str(item.get("entity", "")),
                    str(item.get("relation", "")),
                    str(item.get("target", "")),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                merged.append(item)
                if len(merged) >= settings.TOP_K_KG:
                    break
            if len(merged) >= settings.TOP_K_KG:
                break
        kg = merged[: settings.TOP_K_KG]
    else:
        kg = kg_adapter.search(
            query=kg_query or request_text,
            keywords=entity_terms,
            top_k=settings.TOP_K_KG,
        )
    state["kg_evidence"] = kg
    kg_status = getattr(kg_adapter, "last_status", {})
    first = kg[0] if kg else {}
    _append_trace(
        state,
        node_name="fetch_kg_evidence",
        input_summary={"kg_query": kg_query, "entity_terms": entity_terms[:8]},
        output_summary={
            "kg_provider": settings.KG_PROVIDER,
            "kg_api_base": settings.KG_API_BASE,
            "kg_query": kg_query,
            "entity_terms": entity_terms[:8],
            "evidence_count": len(kg),
            "mapper_used": kg_status.get("mapper", ""),
            "validation_ok": kg_status.get("validation_ok", True),
            "validation_error": kg_status.get("error", ""),
            "first_evidence_source": (first.get("metadata") or {}).get("source") if isinstance(first, dict) else "",
            "first_evidence_entity": first.get("entity", "") if isinstance(first, dict) else "",
            "first_evidence_relation": first.get("relation", "") if isinstance(first, dict) else "",
        },
        selected_tools=["kg_adapter.search"],
    )
    return state


def fetch_mysql_evidence(state: AgentState) -> AgentState:
    student_adapter = get_student_data_adapter()
    student_id = state.get("student_id", "")
    user_points = state.get("user_mentioned_knowledge_points", []) or state.get("knowledge_points", [])
    mysql = student_adapter.load_student_evidence(
        student_id=student_id,
        user_mentioned_points=user_points,
    )
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
        rag_evidence=state.get("rag_evidence", []),
        user_mentioned_points=state.get("user_mentioned_knowledge_points", []),
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


def _need_action_plan(state: AgentState) -> bool:
    text = state.get("request_text", "")
    return any(k in text for k in PLAN_KEYWORDS)


def generate_intervention(state: AgentState) -> AgentState:
    if state.get("primary_task_type") == "diagnosis" and not _need_action_plan(state):
        state["intervention_case_evidence"] = []
        state["intervention_plan"] = {}
        _append_trace(
            state,
            node_name="generate_intervention",
            input_summary={"desired_days": state.get("desired_days", 0)},
            output_summary={"plan_mode": "skipped_for_diagnosis_only", "case_count": 0},
            selected_tools=["intervention_rule_engine_v2"],
        )
        return state
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
    if state.get("primary_task_type") == "diagnosis" and not _need_action_plan(state):
        state["recommended_packages"] = []
        _append_trace(
            state,
            node_name="recommend_package",
            input_summary={"grade_band": "", "difficulty_level": ""},
            output_summary={"recommended_count": 0, "mode": "skipped_for_diagnosis_only"},
            selected_tools=["package_adapter.recommend"],
        )
        return state
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
    user_points = state.get("user_mentioned_knowledge_points", []) or state.get("knowledge_points", [])
    alignment = mysql.get("alignment_summary", {}) if isinstance(mysql, dict) else {}
    student_supported_points = alignment.get("matched_user_mentioned_points", []) if isinstance(alignment, dict) else []
    rag_supported_points = _extract_supported_points_from_rag(user_points, rag_items)
    kg_supported_points = _extract_supported_points_from_kg(user_points, kg_items)
    return {
        "rag_summary": {
            "hit_count": len(rag_items),
            "provider": settings.RAG_PROVIDER,
            "schema": "RAGEvidenceItem",
            "preview": rag_items[:2],
            "rag_supported_points": rag_supported_points,
        },
        "kg_summary": {
            "hit_count": len(kg_items),
            "provider": settings.KG_PROVIDER,
            "schema": "KGEvidenceItem",
            "preview": kg_items[:2],
            "kg_supported_points": kg_supported_points,
        },
        "mysql_summary": {
            "provider": settings.STUDENT_DATA_PROVIDER,
            "schema": "StudentEvidenceBundle",
            "evidence": mysql,
            "student_data_supported_points": student_supported_points,
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
    if state.get("primary_task_type") == "technical_qa":
        error_type = state.get("error_type", "") or "常见运行错误"
        rag_items = state.get("rag_evidence", [])
        kg_items = state.get("kg_evidence", [])
        rag_ref = "-"
        if rag_items:
            first = rag_items[0]
            rag_ref = str(first.get("source_id") or first.get("title") or "-")
        kg_ref = "-"
        if kg_items:
            kg_best = None
            kg_related = None
            for item in kg_items:
                relation = str(item.get("relation") or "")
                if relation == "HAS_SOLUTION" and kg_best is None:
                    kg_best = item
                if relation == "RELATED_ERROR" and kg_related is None:
                    kg_related = item
            selected = kg_best or kg_related or kg_items[0]
            kg_ref = (
                f"{selected.get('entity', '-')}"
                f" -> {selected.get('relation', '-')}"
                f" -> {selected.get('target', '-')}"
            )
        state["final_response"] = (
            f"问题判断：这是一个 {error_type} 类问题，可以理解为“程序找不到这个名字”。\n\n"
            "原因解释：NameError 通常表示变量或函数名在使用前未定义，或者命名与实际定义不一致。\n\n"
            "课堂讲法：\n"
            "1. 可以先告诉学生：你在代码里喊了一个名字，但电脑还不知道这个名字是谁；\n"
            "2. 让学生定位报错行，找到报错的变量名或函数名；\n"
            "3. 检查它是否先定义后使用；\n"
            "4. 再检查大小写、拼写和作用域。\n\n"
            "参考证据：\n"
            f"- RAG: {rag_ref}\n"
            f"- KG: {kg_ref}"
        )
    else:
        if state.get("primary_task_type") == "diagnosis":
            basis = diagnosis.get("evidence_basis", {}) if isinstance(diagnosis, dict) else {}
            focus = "、".join(basis.get("user_mentioned_knowledge_points", [])[:3]) or "当前关注知识点"
            matched = "、".join(basis.get("matched_user_mentioned_points", [])[:3]) or "暂无直接命中"
            unmatched = "、".join(basis.get("unmatched_user_mentioned_points", [])[:3]) or "无"
            weak = "、".join(basis.get("data_weak_points", [])[:3]) or "暂无"
            rag_supported = "、".join(
                state.get("evidence_summary", {}).get("rag_summary", {}).get("rag_supported_points", [])[:3]
            ) or "无明显直接命中"
            kg_supported = "、".join(
                state.get("evidence_summary", {}).get("kg_summary", {}).get("kg_supported_points", [])[:3]
            ) or "无明显直接命中"
            is_mixed_intervention = "intervention" in state.get("secondary_task_types", [])
            plan_lines = ""
            if is_mixed_intervention and isinstance(plan, dict) and plan:
                plan_lines = (
                    "\n\n3天干预建议：\n"
                    f"- 第1天：{plan.get('day_1_action', '围绕核心知识点做错因讲解与最小复现。')}\n"
                    f"- 第2天：{plan.get('day_2_action', '安排分层练习并现场反馈。')}\n"
                    f"- 第3天：{plan.get('day_3_action', '组织错题复盘与迁移训练。')}\n"
                )
            boundary_reminder = ""
            if basis.get("evidence_alignment_status") in {"mismatched", "insufficient_data"}:
                boundary_reminder = (
                    "\n证据边界提醒：\n"
                    f"当前学情记录未直接命中 {unmatched}，本建议主要基于教师描述和 RAG/KG 补充证据生成，"
                    "建议后续补充对应作业记录后复核。"
                )
            state["final_response"] = (
                "诊断结论：\n"
                f"该学生本次主要需要关注“{focus}”。\n\n"
                "证据说明：\n"
                f"- 用户描述中明确提到 {focus}；\n"
                f"- 学情记录中直接命中：{matched}；\n"
                f"- RAG 证据当前主要支持：{rag_supported}；\n"
                f"- KG 证据当前主要支持：{kg_supported}；\n"
                f"- 学情记录未直接支持：{unmatched}。\n"
                f"- 系统额外发现历史弱点还包括：{weak}。\n\n"
                "可能原因：\n"
                f"{diagnosis.get('probable_cause', '概念理解与排错路径仍不稳定。')}\n\n"
                "建议：\n"
                f"{diagnosis.get('brief_suggestion', '建议先围绕变量定义做错因复盘，再用逐行跟踪方式讲解 for循环变量变化。')}"
                f"{plan_lines}"
                f"{boundary_reminder}"
                f"{clarify_lines}"
            )
        else:
            state["final_response"] = (
                f"我已先按“{state.get('primary_task_type', 'unknown')}”处理你的请求，"
                f"并补充了次任务 {state.get('secondary_task_types', [])} 的结果。\n"
                f"当前观察：{diagnosis.get('observed_problem', '暂无明确观察')}\n"
                f"可能原因：{diagnosis.get('probable_cause', '暂无明确原因')}\n"
                f"建议目标：{plan.get('intervention_goal', '请先补齐关键信息')}\n"
                f"已推荐练习包 {len(state.get('recommended_packages', []))} 个。"
                f"{clarify_lines}"
            )
    state["final_response"] = _clean_response_text(str(state.get("final_response", "")))
    _append_trace(
        state,
        node_name="build_final_response",
        input_summary={"task_type": state.get("task_type", "unknown")},
        output_summary={"final_response_non_empty": bool(state.get("final_response", "").strip())},
        selected_tools=["response_builder_v2"],
    )
    return state
