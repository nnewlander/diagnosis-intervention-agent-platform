from typing import Any


CAUSE_HINTS = {
    "变量定义": "变量定义方面可能涉及先使用后定义、命名不一致或作用域理解不足",
    "for循环": "for循环方面可能涉及循环边界、缩进和循环变量更新理解不稳",
    "条件判断": "条件判断方面可能涉及条件表达式、比较运算符和 if/elif/else 分支结构理解不稳",
}


def _clean_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("。。", "。").replace("；；", "；").replace("，，", "，")
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _build_probable_cause(user_points: list[str], unmatched: list[str]) -> str:
    causes: list[str] = []
    for p in user_points:
        hint = CAUSE_HINTS.get(p)
        if hint and hint not in causes:
            causes.append(hint)
    if not causes:
        causes.append("可能是概念理解与排错路径仍不稳定。")
    if unmatched:
        causes.append("对未直接命中的知识点，目前主要由教师描述与 RAG/KG 证据补充支持，提交记录中的直接证据不足。")
    return _clean_text("；".join(causes) + ("。" if causes else ""))


def _confidence_by_alignment(alignment_status: str, submission_count: int, matched_count: int) -> str:
    if alignment_status == "aligned" and submission_count >= 5 and matched_count > 0:
        return "high"
    if alignment_status == "partially_aligned":
        return "medium"
    if alignment_status == "mismatched":
        return "cautious_medium"
    return "medium"


def _build_brief_suggestion(user_points: list[str]) -> str:
    if not user_points:
        return _clean_text("建议先围绕当前关注知识点做错因复盘，再通过逐行跟踪与口述排查强化迁移能力。")
    focus = "、".join(user_points[:3])
    return _clean_text(f"建议先围绕{focus}做错因复盘，再用逐行跟踪方式讲解关键变量变化与判断路径。")


def build_diagnosis(
    mysql_evidence: dict[str, Any],
    kg_evidence: list[dict[str, Any]],
    rag_evidence: list[dict[str, Any]] | None = None,
    user_mentioned_points: list[str] | None = None,
) -> dict[str, Any]:
    profile_summary = mysql_evidence.get("profile_summary", {})
    submission_summary = mysql_evidence.get("recent_submission_summary", {})
    error_summary = mysql_evidence.get("recent_error_summary", {})
    weak_summary = mysql_evidence.get("weak_point_summary", {})
    alignment = mysql_evidence.get("alignment_summary", {}) if isinstance(mysql_evidence, dict) else {}
    rag_items = rag_evidence or []
    mentioned = [str(x).strip() for x in (user_mentioned_points or []) if str(x).strip()]

    has_core_evidence = bool(profile_summary) and submission_summary.get("total", 0) > 0
    if not has_core_evidence:
        return {
            "observed_problem": _clean_text("当前学生画像或提交记录不足"),
            "probable_cause": _clean_text("关键证据缺失，暂无法定位稳定问题模式"),
            "evidence_basis": {
                "profile_available": bool(profile_summary),
                "submission_count": submission_summary.get("total", 0),
                "user_mentioned_knowledge_points": mentioned,
                "matched_user_mentioned_points": [],
                "unmatched_user_mentioned_points": mentioned,
                "data_weak_points": [],
                "evidence_alignment_status": "insufficient_data",
                "rag_hit_count": len(rag_items),
                "kg_hit_count": len(kg_evidence),
            },
            "confidence_level": "low",
            "mode": "conservative",
            "brief_suggestion": _clean_text("建议先补充学生最近两周可用提交记录，再进行精细诊断。"),
        }

    weak_points = weak_summary.get("weak_knowledge_points", [])
    error_dist = error_summary.get("error_distribution", {})
    top_error = max(error_dist, key=error_dist.get) if error_dist else "unknown"
    matched = alignment.get("matched_user_mentioned_points", []) if isinstance(alignment, dict) else []
    unmatched = alignment.get("unmatched_user_mentioned_points", []) if isinstance(alignment, dict) else []
    data_weak_points = alignment.get("data_weak_points", weak_points[:5]) if isinstance(alignment, dict) else weak_points[:5]
    align_status = alignment.get("evidence_alignment_status", "insufficient_data") if isinstance(alignment, dict) else "insufficient_data"
    user_focus_text = "、".join(mentioned) if mentioned else "当前提问中的重点知识点"
    matched_text = "、".join(matched) if matched else "暂无直接命中"
    unmatched_text = "、".join(unmatched) if unmatched else "无"
    weak_text = "、".join(data_weak_points[:3]) if data_weak_points else "暂无稳定弱点"
    observed = _clean_text(
        (
        f"用户关注：老师关注{user_focus_text}反复出错；"
        f"数据支持：学情记录直接命中{matched_text}，未直接支持{unmatched_text}；"
        f"系统额外发现：历史弱点集中在{weak_text}，最近提交错误以{top_error}为主。"
        )
    )

    probable_cause = _build_probable_cause(mentioned, unmatched)
    confidence = _confidence_by_alignment(
        align_status,
        submission_summary.get("total", 0),
        len(matched),
    )
    return {
        "observed_problem": observed,
        "probable_cause": probable_cause,
        "evidence_basis": {
            "profile_summary": profile_summary,
            "submission_count": submission_summary.get("total", 0),
            "user_mentioned_knowledge_points": mentioned,
            "matched_user_mentioned_points": matched,
            "unmatched_user_mentioned_points": unmatched,
            "data_weak_points": data_weak_points[:5],
            "evidence_alignment_status": align_status,
            "weak_knowledge_points": weak_points[:5],
            "error_distribution": error_dist,
            "rag_hit_count": len(rag_items),
            "kg_hit_count": len(kg_evidence),
        },
        "confidence_level": confidence,
        "mode": "normal",
        "brief_suggestion": _build_brief_suggestion(mentioned),
    }
