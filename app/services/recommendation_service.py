from typing import Any

from app.core.config import settings
from app.models.evidence import PackageRecommendationItem
from app.tools.package_adapter import get_package_adapter


def _extract_matched_knowledge_points(title: str, reason: str, request_points: list[str]) -> list[str]:
    # Prefer title/package_name match first, then reason match; fallback to request points.
    title_text = str(title or "")
    reason_text = str(reason or "")
    matched_from_title = [p for p in request_points if p and p in title_text]
    if matched_from_title:
        return list(dict.fromkeys(matched_from_title))
    matched_from_reason = [p for p in request_points if p and p in reason_text]
    if matched_from_reason:
        return list(dict.fromkeys(matched_from_reason))
    return request_points[:]


def recommend_and_format_packages(
    knowledge_points: list[str],
    grade_band: str = "",
    difficulty_level: str = "",
) -> list[dict[str, Any]]:
    adapter = get_package_adapter(provider="local")
    packages = adapter.recommend(
        knowledge_points=knowledge_points,
        grade_band=grade_band,
        difficulty_level=difficulty_level,
        top_k=settings.TOP_K_PACKAGES,
    )
    formatted: list[dict[str, Any]] = []
    for item in packages:
        reason_parts = []
        if knowledge_points:
            reason_parts.append(f"匹配知识点: {', '.join(knowledge_points[:3])}")
        if grade_band:
            reason_parts.append(f"年级匹配: {grade_band}")
        if difficulty_level:
            reason_parts.append(f"难度偏好: {difficulty_level}")
        package_name = str(item.get("package_name") or item.get("title") or "未命名练习包")
        normalized = PackageRecommendationItem(
            package_id=str(item.get("package_id", "unknown")),
            package_name=package_name,
            reason="；".join(reason_parts) if reason_parts else "根据学情证据推荐",
            difficulty_level=str(item.get("difficulty_level", "")),
        ).model_dump()
        normalized["target_knowledge_points"] = item.get("target_knowledge_points", [])
        normalized["target_grade_band"] = item.get("target_grade_band", "")
        normalized["request_knowledge_points"] = knowledge_points[:]
        normalized["matched_knowledge_points"] = _extract_matched_knowledge_points(
            title=package_name,
            reason=str(normalized.get("reason", "")),
            request_points=knowledge_points[:],
        )
        formatted.append(normalized)
    return formatted
