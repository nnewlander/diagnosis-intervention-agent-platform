from typing import Any

from app.data_loader.loader import LocalDataStore


class StudentEntityResolver:
    """Resolve student entity from explicit id or mention text."""

    def __init__(self, store: LocalDataStore) -> None:
        self.store = store
        self.name_to_profiles: dict[str, list[dict[str, Any]]] = {}
        for profile in self.store.student_profiles:
            name = str(profile.get("student_name_masked", "")).strip()
            if not name:
                continue
            self.name_to_profiles.setdefault(name, []).append(profile)

    def resolve(self, student_id: str, student_mention: str) -> dict[str, Any]:
        if student_id:
            profile = self.store.get_student_profile(student_id)
            if profile:
                return {
                    "student_id": student_id,
                    "resolved_by": "student_id",
                    "matched_profile": profile,
                    "need_clarify": False,
                    "clarify_message": "",
                }

        if student_mention and student_mention.endswith("同学"):
            candidates = self.name_to_profiles.get(student_mention, [])
            if len(candidates) == 1:
                target = candidates[0]
                return {
                    "student_id": target.get("student_id", ""),
                    "resolved_by": "name_fuzzy",
                    "matched_profile": target,
                    "need_clarify": False,
                    "clarify_message": "",
                }
            if len(candidates) > 1:
                ids = [c.get("student_id", "") for c in candidates[:3]]
                return {
                    "student_id": "",
                    "resolved_by": "name_ambiguous",
                    "matched_profile": {},
                    "need_clarify": True,
                    "clarify_message": f"检测到{student_mention}对应多个学生，请补充 student_id。候选: {ids}",
                }

        return {
            "student_id": "",
            "resolved_by": "unresolved",
            "matched_profile": {},
            "need_clarify": True,
            "clarify_message": "未能定位学生身份，请提供 student_id 或更明确的学生信息。",
        }
