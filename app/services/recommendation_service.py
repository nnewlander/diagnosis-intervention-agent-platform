from typing import Any

from app.core.config import settings
from app.tools.package_adapter import get_package_adapter


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
        kp = item.get("target_knowledge_points", [])
        reason_parts = []
        if knowledge_points:
            reason_parts.append(f"匹配知识点: {', '.join(knowledge_points[:3])}")
        if grade_band:
            reason_parts.append(f"年级匹配: {grade_band}")
        if difficulty_level:
            reason_parts.append(f"难度偏好: {difficulty_level}")
        formatted.append(
            {
                "package_id": item.get("package_id", "unknown"),
                "title": item.get("package_name") or item.get("title") or "未命名练习包",
                "target_knowledge_points": kp,
                "difficulty_level": item.get("difficulty_level", ""),
                "target_grade_band": item.get("target_grade_band", ""),
                "reason": "；".join(reason_parts) if reason_parts else "根据学情证据推荐",
            }
        )
    return formatted
