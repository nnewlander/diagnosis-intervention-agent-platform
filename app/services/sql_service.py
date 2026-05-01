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

    @staticmethod
    def _build_alignment(
        user_mentioned_points: list[str],
        submissions: list[dict[str, Any]],
        weak_points: list[str],
    ) -> dict[str, Any]:
        normalized_user = [str(x).strip() for x in user_mentioned_points if str(x).strip()]
        if not normalized_user:
            return {
                "matched_user_mentioned_points": [],
                "unmatched_user_mentioned_points": [],
                "data_weak_points": weak_points[:5],
                "evidence_alignment_status": "insufficient_data",
            }
        submission_points = {str(x.get("knowledge_point", "")).strip() for x in submissions if x.get("knowledge_point")}
        weak_set = {str(x).strip() for x in weak_points if str(x).strip()}
        matched: list[str] = []
        unmatched: list[str] = []
        for kp in normalized_user:
            if kp in submission_points or kp in weak_set:
                matched.append(kp)
            else:
                unmatched.append(kp)
        if not submission_points and not weak_set:
            status = "insufficient_data"
        elif len(matched) == len(normalized_user):
            status = "aligned"
        elif matched:
            status = "partially_aligned"
        else:
            status = "mismatched"
        return {
            "matched_user_mentioned_points": matched,
            "unmatched_user_mentioned_points": unmatched,
            "data_weak_points": weak_points[:5],
            "evidence_alignment_status": status,
        }

    def build_student_evidence(
        self,
        student_id: str,
        user_mentioned_points: list[str] | None = None,
    ) -> dict[str, Any]:
        if not student_id:
            return {
                "profile_summary": {},
                "recent_submission_summary": {"submissions": [], "total": 0},
                "weak_point_summary": {"weak_knowledge_points": [], "mastery_level": ""},
                "recent_error_summary": {"error_distribution": {}},
                "intervention_feedback_summary": {},
                "alignment_summary": {
                    "matched_user_mentioned_points": [],
                    "unmatched_user_mentioned_points": [],
                    "data_weak_points": [],
                    "evidence_alignment_status": "insufficient_data",
                },
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
        weak_points = mastery.get("weak_knowledge_points", [])[:5]
        alignment = self._build_alignment(user_mentioned_points or [], compact_submissions, weak_points)

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
                "weak_knowledge_points": weak_points,
                "mastery_level": mastery.get("mastery_level", ""),
                "class_attention_note": _truncate_text(mastery.get("class_attention_note", ""), 120),
            },
            "recent_error_summary": {
                "error_distribution": dict(error_counter),
                "recent_error_types": mastery.get("recent_error_types", [])[:5],
            },
            "intervention_feedback_summary": feedback,
            "alignment_summary": alignment,
        }
