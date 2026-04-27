from collections import Counter
from typing import Any

from app.core.config import settings
from app.data_loader.loader import LocalDataStore


def _truncate_text(value: Any, limit: int = 120) -> str:
    text = str(value) if value is not None else ""
    return text[:limit]


class StudentEvidenceService:
    """Aggregate local MySQL-style evidence for diagnosis/intervention."""

    def __init__(self, store: LocalDataStore) -> None:
        self.store = store

    def _get_latest_mastery(self, student_id: str) -> dict[str, Any]:
        for item in self.store.mastery_snapshots:
            if str(item.get("student_id", "")) == student_id:
                return item
        return {}

    def _find_recent_feedback(self, student_id: str) -> dict[str, Any]:
        for case in self.store.intervention_cases:
            if str(case.get("student_id", "")) == student_id:
                return {
                    "teacher_acceptance": case.get("teacher_acceptance", ""),
                    "intervention_goal": case.get("intervention_goal", ""),
                    "recommended_actions_raw": _truncate_text(case.get("recommended_actions_raw", ""), 160),
                    "follow_up_days": case.get("follow_up_days", ""),
                }
        return {}

    def build_student_evidence(self, student_id: str) -> dict[str, Any]:
        if not student_id:
            return {
                "profile_summary": {},
                "recent_submission_summary": {"submissions": [], "total": 0},
                "weak_point_summary": {"weak_knowledge_points": [], "mastery_level": ""},
                "recent_error_summary": {"error_distribution": {}},
                "intervention_feedback_summary": {},
            }

        profile = self.store.get_student_profile(student_id)
        submissions = self.store.get_submissions(student_id, limit=settings.MAX_SUBMISSIONS)
        mastery = self._get_latest_mastery(student_id)
        feedback = self._find_recent_feedback(student_id)

        error_counter = Counter(
            [str(item.get("error_type", "unknown") or "unknown") for item in submissions]
        )
        compact_submissions = [
            {
                "submission_id": item.get("submission_id"),
                "knowledge_point": item.get("knowledge_point"),
                "judge_status": item.get("judge_status"),
                "error_type": item.get("error_type"),
                "score": item.get("score"),
                "submitted_at": item.get("submitted_at"),
            }
            for item in submissions[: settings.MAX_SUBMISSIONS]
        ]

        return {
            "profile_summary": {
                "student_id": profile.get("student_id", ""),
                "student_name_masked": profile.get("student_name_masked", ""),
                "grade_band": profile.get("grade_band", ""),
                "current_class_id": profile.get("current_class_id", ""),
                "attention_risk_level": profile.get("attention_risk_level", ""),
            },
            "recent_submission_summary": {
                "total": len(submissions),
                "submissions": compact_submissions,
            },
            "weak_point_summary": {
                "weak_knowledge_points": mastery.get("weak_knowledge_points", [])[:5],
                "mastery_level": mastery.get("mastery_level", ""),
                "class_attention_note": _truncate_text(mastery.get("class_attention_note", ""), 120),
            },
            "recent_error_summary": {
                "error_distribution": dict(error_counter),
                "recent_error_types": mastery.get("recent_error_types", [])[:5],
            },
            "intervention_feedback_summary": feedback,
        }
