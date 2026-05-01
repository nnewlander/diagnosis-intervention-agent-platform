import requests
import streamlit as st

try:
    from frontend.view_formatters import (
        build_demo_final_response,
        build_kg_conclusion,
        select_kg_reference,
    )
except ModuleNotFoundError:
    from view_formatters import (
        build_demo_final_response,
        build_kg_conclusion,
        select_kg_reference,
    )

st.set_page_config(page_title="教学 Agent 演示", page_icon="🧠", layout="wide")
st.title("核桃智能教学诊断与干预 Agent 平台（面试演示版）")
st.caption("展示 Agent 如何编排 RAG + KG + 学情数据，并生成诊断/干预结果。")

EXAMPLES = {
    "技术答疑": "课堂演示遇到 NameError，应该怎么给学生解释？",
    "学情诊断": "李同学最近几次作业在变量定义和 for循环上反复出错，帮我诊断一下。",
    "诊断 + 干预": "李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个 3 天干预建议。",
    "RAG + KG 混合": "学生问 NameError 是什么意思，能结合知识图谱和资料给我一个课堂解释吗？",
}

if "request_text" not in st.session_state:
    st.session_state["request_text"] = EXAMPLES["技术答疑"]
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


def _fmt_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def _build_request_text(raw_text: str, student_id: str, class_id: str) -> str:
    text = (raw_text or "").strip()
    if student_id:
        text = f"{text} student_id:{student_id.strip()}"
    if class_id:
        text = f"{text} class_id:{class_id.strip()}"
    return text.strip()


def _render_badges(items: list[str]) -> None:
    if not items:
        return
    st.caption(" | ".join(items))


def _safe_list(value):
    return value if isinstance(value, list) else []


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _rag_row_tag(item: dict) -> tuple[str, bool]:
    meta = _safe_dict(item.get("metadata"))
    source_type = str(item.get("source_type", ""))
    fallback = bool(meta.get("fallback")) or source_type == "fallback_error_guide"
    if source_type == "faq" and not fallback:
        return "真实 RAG 命中", False
    if fallback:
        return "RAG fallback", True
    return "-", False


def _rag_header_badges(items: list[dict]) -> list[str]:
    if not items:
        return ["RAG"]
    first = items[0]
    tag, is_fallback = _rag_row_tag(first)
    if is_fallback:
        return ["RAG", "fallback"]
    if tag == "真实 RAG 命中":
        return ["RAG", "FAQ", "BM25", "真实命中"]
    return ["RAG"]


def _render_final_response(final_response: str) -> None:
    st.markdown("### final_response")
    if final_response:
        st.markdown(final_response)
    else:
        st.info("暂无 final_response")


def _render_external_status(rag_items: list[dict], kg_items: list[dict], final_response: str) -> None:
    rag_status = "未命中"
    rag_mode = "-"
    if rag_items:
        r0 = rag_items[0]
        r_meta = _safe_dict(r0.get("metadata"))
        r_fallback = bool(r_meta.get("fallback")) or r0.get("source_type") == "fallback_error_guide"
        if r_fallback:
            rag_status = "fallback"
            rag_mode = "fallback"
        elif r0.get("source_type") == "faq":
            rag_status = "真实命中"
            rag_mode = "FAQ-BM25"
    kg_status = "未命中"
    kg_mode = "-"
    if kg_items:
        k0 = kg_items[0]
        k_meta = _safe_dict(k0.get("metadata"))
        if k_meta.get("source") == "neo4j_core_seed":
            kg_status = "真实 KG core_seed 命中"
            kg_mode = "Neo4j core_seed"
        else:
            kg_status = "已命中"
            kg_mode = str(k_meta.get("source", "-"))
    final_status = "已生成" if str(final_response or "").strip() else "未生成"
    with st.container(border=True):
        st.markdown("**外部能力调用状态**")
        c1, c2, c3 = st.columns(3)
        c1.write(f"RAG: remote / {rag_mode} / {rag_status}")
        c2.write(f"KG: remote / {kg_mode} / {kg_status}")
        c3.write(f"Final response: {final_status}")


def _render_rag_section(rag_items: list[dict], *, is_technical_qa: bool, error_type: str) -> None:
    def is_nameerror_relevant(item: dict) -> bool:
        text = " ".join(
            [
                str(item.get("source_id", "")),
                str(item.get("title", "")),
                str(item.get("snippet", "")),
            ]
        )
        keywords = [
            "NameError",
            "变量未定义",
            "名称未定义",
            "函数未定义",
            "变量名错误",
            "拼写",
            "大小写",
            "作用域",
        ]
        return any(k in text for k in keywords)

    with st.container(border=True):
        st.subheader("RAG evidence 区（Top3）")
        _render_badges(_rag_header_badges(rag_items))
        if not rag_items:
            st.info("暂无证据")
            return
        enable_filter = is_technical_qa and error_type == "NameError"
        relevant_items = [x for x in rag_items if is_nameerror_relevant(x)] if enable_filter else rag_items[:]
        low_relevant_items = [x for x in rag_items if x not in relevant_items] if enable_filter else []
        display_items = relevant_items[:3] if relevant_items else rag_items[:3]
        if enable_filter and relevant_items and len(display_items) < 3:
            st.info("当前仅展示高相关 RAG evidence。")
        # top1 默认展开，top2/top3 置于折叠区，降低页面长度。
        top_items = display_items
        for i, item in enumerate(top_items[:1], start=1):
            meta = _safe_dict(item.get("metadata"))
            row_tag, _ = _rag_row_tag(item)
            st.markdown(f"**#{i} {row_tag}**")
            st.write(f"source_id: {item.get('source_id', '-')}")
            st.write(f"title: {item.get('title', '-')}")
            st.write(f"source_type: {item.get('source_type', '-')}")
            st.write(f"score: {_fmt_score(item.get('score'))}")
            st.write(f"metadata.route: {meta.get('route', '-')}")
            st.write(f"metadata.fallback: {bool(meta.get('fallback'))}")
            st.write(f"snippet: {str(item.get('snippet', ''))[:180]}")
            st.divider()
        if len(top_items) > 1:
            with st.expander("查看 top2 / top3", expanded=False):
                for i, item in enumerate(top_items[1:], start=2):
                    meta = _safe_dict(item.get("metadata"))
                    row_tag, _ = _rag_row_tag(item)
                    st.markdown(f"**#{i} {row_tag}**")
                    st.write(f"source_id: {item.get('source_id', '-')}")
                    st.write(f"title: {item.get('title', '-')}")
                    st.write(f"source_type: {item.get('source_type', '-')}")
                    st.write(f"score: {_fmt_score(item.get('score'))}")
                    st.write(f"metadata.route: {meta.get('route', '-')}")
                    st.write(f"metadata.fallback: {bool(meta.get('fallback'))}")
                    st.write(f"snippet: {str(item.get('snippet', ''))[:180]}")
                    st.divider()
        if enable_filter and low_relevant_items:
            with st.expander("低相关 RAG 候选（相关性较低，仅作候选参考）", expanded=False):
                for i, item in enumerate(low_relevant_items[:3], start=1):
                    meta = _safe_dict(item.get("metadata"))
                    st.markdown(f"**候选#{i}**")
                    st.write(f"source_id: {item.get('source_id', '-')}")
                    st.write(f"title: {item.get('title', '-')}")
                    st.write(f"source_type: {item.get('source_type', '-')}")
                    st.write(f"score: {_fmt_score(item.get('score'))}")
                    st.write(f"metadata.route: {meta.get('route', '-')}")
                    st.write(f"snippet: {str(item.get('snippet', ''))[:120]}")
                    st.divider()


def _render_kg_section(kg_items: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("KG evidence 区（Top3）")
        _render_badges(["KG", "Neo4j", "core_seed"])
        st.info(build_kg_conclusion(kg_items))
        solution_hint = select_kg_reference(kg_items)
        if solution_hint != "-" and "HAS_SOLUTION" in solution_hint:
            st.success(f"KG 推荐解决关系：{solution_hint}")
        if not kg_items:
            st.info("暂无证据")
            return
        top_items = kg_items[:3]
        for i, item in enumerate(top_items[:1], start=1):
            meta = _safe_dict(item.get("metadata"))
            source = str(meta.get("source", ""))
            tag = "真实 KG core_seed 命中" if source == "neo4j_core_seed" else "-"
            st.markdown(f"**#{i} {tag}**")
            st.write(f"entity: {item.get('entity', '-')}")
            st.write(f"entity_type: {item.get('entity_type', '-')}")
            st.write(f"relation: {item.get('relation', '-')}")
            st.write(f"target: {item.get('target', '-')}")
            st.write(f"score: {_fmt_score(item.get('score'))}")
            st.write(f"metadata.source: {source or '-'}")
            st.write(f"evidence: {str(item.get('evidence', ''))[:180]}")
            st.divider()
        if len(top_items) > 1:
            with st.expander("查看 top2 / top3", expanded=False):
                for i, item in enumerate(top_items[1:], start=2):
                    meta = _safe_dict(item.get("metadata"))
                    source = str(meta.get("source", ""))
                    tag = "真实 KG core_seed 命中" if source == "neo4j_core_seed" else "-"
                    st.markdown(f"**#{i} {tag}**")
                    st.write(f"entity: {item.get('entity', '-')}")
                    st.write(f"entity_type: {item.get('entity_type', '-')}")
                    st.write(f"relation: {item.get('relation', '-')}")
                    st.write(f"target: {item.get('target', '-')}")
                    st.write(f"score: {_fmt_score(item.get('score'))}")
                    st.write(f"metadata.source: {source or '-'}")
                    st.write(f"evidence: {str(item.get('evidence', ''))[:180]}")
                    st.divider()


def _should_prefer_kg_solution(task_type: str, primary_task_type: str, request_text: str) -> bool:
    if task_type != "technical_qa" and primary_task_type != "technical_qa":
        return False
    text = (request_text or "").lower()
    trigger_words = ["怎么解释", "怎么讲", "怎么办", "如何处理"]
    return any(w in text for w in trigger_words)


def _sort_kg_for_display(kg_items: list[dict], prefer_solution: bool) -> list[dict]:
    if not prefer_solution or not kg_items:
        return kg_items
    # 前端展示排序：优先 HAS_SOLUTION，不改变后端原始返回结构。
    return sorted(
        kg_items,
        key=lambda x: 0 if str(_safe_dict(x).get("relation", "")) == "HAS_SOLUTION" else 1,
    )


def _render_student_data(mysql_evidence: dict, is_technical_qa: bool) -> None:
    with st.container(border=True):
        st.subheader("学情数据区")
        _render_badges(["StudentData", "local_csv_jsonl"])
        if is_technical_qa:
            st.info("当前任务为技术答疑 technical_qa，未触发学情诊断链路。如需展示学情数据，请使用学情诊断或诊断+干预示例。")
            return
        alignment = _safe_dict(mysql_evidence.get("alignment_summary"))
        profile = _safe_dict(mysql_evidence.get("profile_summary"))
        submit = _safe_dict(mysql_evidence.get("recent_submission_summary"))
        recent_error = _safe_dict(mysql_evidence.get("recent_error_summary"))
        error_dist = _safe_dict(recent_error.get("error_distribution"))
        err_text = "，".join([f"{k}={v}" for k, v in error_dist.items()][:5]) or "-"
        weak_text = "、".join(_safe_list(alignment.get("data_weak_points"))) or "-"
        matched_text = "、".join(_safe_list(alignment.get("matched_user_mentioned_points"))) or "-"
        unmatched_text = "、".join(_safe_list(alignment.get("unmatched_user_mentioned_points"))) or "-"
        mentioned_text = "、".join(
            _safe_list(alignment.get("matched_user_mentioned_points")) + _safe_list(alignment.get("unmatched_user_mentioned_points"))
        ) or "-"
        st.markdown("**学情摘要卡片**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(
                f"学生：{profile.get('student_name_masked', '-')} / {profile.get('grade_band', '-')} / "
                f"attention_risk_level={profile.get('attention_risk_level', '-')}"
            )
            st.write(f"最近提交数：{submit.get('total', '-')}")
            st.write(f"用户关注点：{mentioned_text}")
            st.write(f"用户关注点命中：{matched_text}")
        with c2:
            st.write(f"用户关注点未直接支持：{unmatched_text}")
            st.write(f"历史弱点：{weak_text}")
            st.write(f"错误分布：{err_text}")
            st.write(f"证据一致性：{alignment.get('evidence_alignment_status', '-')}")

        with st.expander("查看原始学情数据", expanded=False):
            st.markdown("**student profile summary**")
            st.json(profile)
            st.markdown("**recent submissions**")
            st.json(submit)
            st.markdown("**weak points**")
            st.json(_safe_dict(mysql_evidence.get("weak_point_summary")))
            st.markdown("**recent errors**")
            st.json(recent_error)


def _render_diagnosis(diagnosis: dict, is_technical_qa: bool) -> None:
    if is_technical_qa and not any(diagnosis.values()):
        with st.expander("诊断区（当前任务不适用）", expanded=False):
            st.info("当前任务不适用")
        return
    with st.container(border=True):
        st.subheader("诊断区")
        basis = _safe_dict(diagnosis.get("evidence_basis"))
        st.write("用户关注：" + ("、".join(_safe_list(basis.get("user_mentioned_knowledge_points"))) or "-"))
        st.write("数据支持：" + ("、".join(_safe_list(basis.get("matched_user_mentioned_points"))) or "-"))
        st.write("额外发现：" + ("、".join(_safe_list(basis.get("data_weak_points"))) or "-"))
        st.write("可能原因：" + str(diagnosis.get("probable_cause", "-")))
        st.write(f"置信度：{diagnosis.get('confidence_level', '-')}")
        st.write(f"证据一致性：{basis.get('evidence_alignment_status', '-')}")
        with st.expander("查看 observed_problem 原文", expanded=False):
            st.write(str(diagnosis.get("observed_problem", "-")))


def _render_plan(plan: dict, is_technical_qa: bool, alignment_status: str) -> None:
    if is_technical_qa and not any(plan.values()):
        with st.expander("干预计划区（当前任务不适用）", expanded=False):
            st.info("当前任务不适用")
        return
    with st.container(border=True):
        st.subheader("干预计划区")
        if not any(plan.values()):
            st.info("当前请求为 diagnosis，未触发完整干预计划。若需要生成 3 天干预方案，请使用“诊断 + 干预”示例。")
            with st.expander("查看空干预计划结构（调试用）", expanded=False):
                st.json(plan)
            return
        if alignment_status in {"mismatched", "insufficient_data"}:
            st.info(
                "说明：当前 3 天干预建议主要基于教师描述和 RAG/KG 补充证据生成；"
                "学情提交记录中暂未直接命中 for循环和条件判断，建议后续补充对应作业记录后复核。"
            )
        st.write(f"intervention_goal: {plan.get('intervention_goal', '-')}")
        st.write(f"day_1_action: {plan.get('day_1_action', '-')}")
        st.write(f"day_2_action: {plan.get('day_2_action', '-')}")
        st.write(f"day_3_action: {plan.get('day_3_action', '-')}")
        st.write(f"optional_followup: {plan.get('optional_followup', '-')}")


def _infer_package_points(pkg: dict, fallback_points: list[str]) -> str:
    matched_field = pkg.get("matched_knowledge_points")
    if isinstance(matched_field, list) and matched_field:
        return "、".join([str(x) for x in matched_field if str(x).strip()])
    text = " ".join(
        [
            str(pkg.get("package_name", "")),
            str(pkg.get("title", "")),
            str(pkg.get("reason", "")),
        ]
    )
    points = []
    for p in ["for循环", "条件判断", "变量定义"]:
        if p in text and p not in points:
            points.append(p)
    if points:
        return "、".join(points)
    if fallback_points:
        return "、".join(fallback_points[:3])
    return "-"


def _render_packages(
    pkgs: list[dict],
    is_technical_qa: bool,
    is_diagnosis_only: bool,
    fallback_points: list[str],
) -> None:
    if is_technical_qa and not pkgs:
        with st.expander("推荐练习区（当前任务不适用）", expanded=False):
            st.info("当前任务不适用")
        return
    with st.container(border=True):
        st.subheader("推荐练习区")
        if is_diagnosis_only:
            st.caption("附加建议（非主任务结果）")
        if not pkgs:
            if is_diagnosis_only:
                st.info("当前请求未要求练习推荐，因此未触发推荐练习。若需要练习包，请使用“诊断 + 干预”或明确输入“推荐练习/下发练习”。")
            else:
                st.info("暂无证据")
            return
        for i, pkg in enumerate(pkgs, start=1):
            st.markdown(f"**#{i}**")
            st.write(f"package_id: {pkg.get('package_id', '-')}")
            st.write(f"title: {pkg.get('title') or pkg.get('package_name', '-')}")
            matched_points = _infer_package_points(pkg, fallback_points)
            st.write(f"matched_knowledge_points: {matched_points}")
            req_points = pkg.get("request_knowledge_points")
            if isinstance(req_points, list) and req_points:
                req_text = "、".join([str(x) for x in req_points[:3]])
            else:
                req_text = "、".join(fallback_points[:3]) if fallback_points else "-"
            st.write(f"request_knowledge_points: {req_text}")
            st.write(
                f"knowledge_point: {pkg.get('knowledge_point') or matched_points}"
            )
            st.write(f"difficulty: {pkg.get('difficulty') or pkg.get('difficulty_level', '-')}")
            st.write(f"reason: {pkg.get('reason', '-')}")
            st.divider()


def _render_debug_trace(debug_trace: list[dict]) -> None:
    with st.expander("节点编排与可观测性", expanded=False):
        if not debug_trace:
            st.info("暂无证据")
            return
        for idx, node in enumerate(debug_trace, start=1):
            node_name = str(node.get("node_name", ""))
            out = _safe_dict(node.get("output_summary"))
            st.markdown(f"**{idx}. {node_name}**")
            st.write(f"selected_tools: {node.get('selected_tools', [])}")
            st.write(f"selected_task_type: {node.get('selected_task_type', '-')}")
            # diagnosis-only 场景下可选环节跳过，增加可解释状态，避免误判为失败。
            status = out.get("plan_mode") or out.get("mode")
            if status == "skipped_for_diagnosis_only":
                st.write("status: skipped_by_task_boundary")
            if node_name in {"fetch_rag_evidence", "fetch_kg_evidence"}:
                provider = node.get("rag_provider", "-") if node_name == "fetch_rag_evidence" else node.get("kg_provider", "-")
                mapper = out.get("mapper_used") or out.get("mapper") or "-"
                validation_ok = out.get("validation_ok")
                validation_ok = validation_ok if validation_ok not in (None, "") else "-"
                evidence_count = out.get("evidence_count")
                if evidence_count in (None, ""):
                    evidence_count = out.get("rag_hits", out.get("kg_hits", "-"))
                if provider not in (None, "", "-"):
                    st.write(f"provider: {provider}")
                if mapper != "-":
                    st.write(f"mapper_used: {mapper}")
                if validation_ok != "-":
                    st.write(f"validation_ok: {validation_ok}")
                if evidence_count not in (None, "", "-"):
                    st.write(f"evidence_count: {evidence_count}")
            st.divider()


with st.container(border=True):
    st.subheader("输入区")
    api_url = st.text_input("FastAPI 地址", value="http://127.0.0.1:8000/agent/run")
    c1, c2 = st.columns(2)
    with c1:
        student_id = st.text_input("student_id（可选）", value="")
    with c2:
        class_id = st.text_input("class_id（可选）", value="")
    request_text = st.text_area("request_text", value=st.session_state["request_text"], height=120)

    e1, e2, e3, e4 = st.columns(4)
    for idx, key in enumerate(EXAMPLES):
        if [e1, e2, e3, e4][idx].button(key, use_container_width=True):
            st.session_state["request_text"] = EXAMPLES[key]
            st.rerun()

    if st.button("运行 Agent", type="primary", use_container_width=True):
        payload_text = _build_request_text(request_text, student_id, class_id)
        with st.spinner("Agent 编排执行中..."):
            try:
                resp = requests.post(api_url, json={"request_text": payload_text}, timeout=40)
                resp.raise_for_status()
                st.session_state["last_result"] = resp.json()
            except Exception as exc:
                st.error(f"调用失败: {exc}")
                st.session_state["last_result"] = None

data = st.session_state.get("last_result")
if not data:
    st.info("点击“运行 Agent”后展示总览、证据与编排过程。")
    st.stop()

parsed_slots = _safe_dict(data.get("parsed_slots"))
debug_trace = _safe_list(data.get("debug_trace"))
debug_state = _safe_dict(data.get("debug_state"))
evidence_summary = _safe_dict(data.get("evidence_summary"))
rag_items = _safe_list(debug_state.get("rag_evidence"))
kg_items = _safe_list(debug_state.get("kg_evidence"))
mysql_evidence = _safe_dict(_safe_dict(evidence_summary.get("mysql_summary")).get("evidence"))
rag_summary = _safe_dict(evidence_summary.get("rag_summary"))
kg_summary = _safe_dict(evidence_summary.get("kg_summary"))
mysql_summary = _safe_dict(evidence_summary.get("mysql_summary"))
task_type = str(data.get("task_type", "unknown"))
primary_task_type = str(data.get("primary_task_type", "unknown"))
is_technical_qa = task_type == "technical_qa" or primary_task_type == "technical_qa"
diagnosis = _safe_dict(data.get("diagnosis"))
plan = _safe_dict(data.get("intervention_plan"))
pkgs = _safe_list(data.get("recommended_packages"))
alignment_status = _safe_dict(diagnosis.get("evidence_basis")).get("evidence_alignment_status", "")
user_mentioned_points = _safe_list(_safe_dict(diagnosis.get("evidence_basis")).get("user_mentioned_knowledge_points"))
is_diagnosis_only = primary_task_type == "diagnosis" and not _safe_dict(data.get("intervention_plan"))
request_text_current = str(debug_state.get("request_text", "") or st.session_state.get("request_text", ""))
kg_items_for_display = _sort_kg_for_display(
    kg_items,
    prefer_solution=_should_prefer_kg_solution(task_type, primary_task_type, request_text_current),
)

with st.container(border=True):
    st.subheader("总览区")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("task_type", data.get("task_type", "unknown"))
    c2.metric("primary_task_type", data.get("primary_task_type", "unknown"))
    c3.metric("secondary_task_types", ", ".join(_safe_list(data.get("secondary_task_types"))) or "[]")
    c4.metric("need_clarify", str(bool(debug_state.get("need_clarify", False))))

_render_external_status(rag_items, kg_items_for_display, data.get("final_response", ""))
if primary_task_type == "diagnosis":
    with st.container(border=True):
        st.markdown("**多源证据支持点**")
        st.write("StudentData 支持：" + ("、".join(_safe_list(mysql_summary.get("student_data_supported_points"))) or "-"))
        st.write("RAG 支持：" + ("、".join(_safe_list(rag_summary.get("rag_supported_points"))) or "-"))
        st.write("KG 支持：" + ("、".join(_safe_list(kg_summary.get("kg_supported_points"))) or "-"))
demo_final_response = build_demo_final_response(
    data.get("final_response", ""),
    task_type=task_type,
    primary_task_type=primary_task_type,
    parsed_error_type=str(debug_state.get("error_type", "")),
    rag_items=rag_items,
    kg_items=kg_items_for_display,
)
_render_final_response(demo_final_response)
with st.expander("查看后端原始 final_response（调试用，非演示答案）", expanded=False):
    st.caption("这里展示的是后端原始返回文本，正式演示答案以上方格式化版本为准。")
    st.code(str(data.get("final_response", "")))

if is_technical_qa:
    if student_id and str(student_id).strip():
        st.info("已检测到 student_id，但当前任务为技术答疑 technical_qa，系统不会读取学生学情数据。如需展示学情数据，请使用学情诊断或诊断+干预示例。")
    _render_rag_section(rag_items, is_technical_qa=is_technical_qa, error_type=str(debug_state.get("error_type", "")))
    _render_kg_section(kg_items_for_display)
    _render_debug_trace(debug_trace)
    _render_student_data(mysql_evidence, is_technical_qa=True)
    _render_diagnosis(diagnosis, is_technical_qa=True)
    _render_plan(plan, is_technical_qa=True, alignment_status=alignment_status)
    _render_packages(
        pkgs,
        is_technical_qa=True,
        is_diagnosis_only=is_diagnosis_only,
        fallback_points=user_mentioned_points,
    )
else:
    _render_student_data(mysql_evidence, is_technical_qa=False)
    col_left, col_right = st.columns(2)
    with col_left:
        _render_rag_section(rag_items, is_technical_qa=is_technical_qa, error_type=str(debug_state.get("error_type", "")))
        _render_diagnosis(diagnosis, is_technical_qa=False)
    with col_right:
        _render_kg_section(kg_items_for_display)
        _render_plan(plan, is_technical_qa=False, alignment_status=alignment_status)
    _render_packages(
        pkgs,
        is_technical_qa=False,
        is_diagnosis_only=is_diagnosis_only,
        fallback_points=user_mentioned_points,
    )
    _render_debug_trace(debug_trace)
