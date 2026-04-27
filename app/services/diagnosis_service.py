from typing import Any


def build_diagnosis(mysql_evidence: dict[str, Any], kg_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    profile_summary = mysql_evidence.get("profile_summary", {})
    submission_summary = mysql_evidence.get("recent_submission_summary", {})
    error_summary = mysql_evidence.get("recent_error_summary", {})
    weak_summary = mysql_evidence.get("weak_point_summary", {})

    has_core_evidence = bool(profile_summary) and submission_summary.get("total", 0) > 0
    if not has_core_evidence:
        return {
            "observed_problem": "当前学生画像或提交记录不足",
            "probable_cause": "关键证据缺失，暂无法定位稳定问题模式",
            "evidence_basis": {
                "profile_available": bool(profile_summary),
                "submission_count": submission_summary.get("total", 0),
                "kg_hit_count": len(kg_evidence),
            },
            "confidence_level": "low",
            "mode": "conservative",
        }

    weak_points = weak_summary.get("weak_knowledge_points", [])
    error_dist = error_summary.get("error_distribution", {})
    top_error = max(error_dist, key=error_dist.get) if error_dist else "unknown"
    observed = (
        f"学生在{','.join(weak_points[:3]) or '基础语法'}上出现重复错误，"
        f"最近提交中以{top_error}最常见。"
    )

    probable_cause = "可能是概念理解与排错顺序不稳定，课堂迁移到独立作业时出现断层。"
    confidence = "high" if submission_summary.get("total", 0) >= 5 else "medium"
    return {
        "observed_problem": observed,
        "probable_cause": probable_cause,
        "evidence_basis": {
            "profile_summary": profile_summary,
            "submission_count": submission_summary.get("total", 0),
            "weak_knowledge_points": weak_points[:5],
            "error_distribution": error_dist,
            "kg_hit_count": len(kg_evidence),
        },
        "confidence_level": confidence,
        "mode": "normal",
    }
