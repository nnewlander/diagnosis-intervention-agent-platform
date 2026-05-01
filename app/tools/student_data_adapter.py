import json
import sqlite3
from typing import Any

from app.core.config import settings
from app.data_loader.loader import LocalDataStore
from app.services.entity_resolver import StudentEntityResolver
from app.services.sql_service import StudentEvidenceService
from app.tools.base import BaseStudentDataAdapter


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {desc[0]: row[idx] for idx, desc in enumerate(cursor.description)}


class LocalCSVJSONLStudentDataAdapter(BaseStudentDataAdapter):
    provider_name = "local_csv_jsonl"

    def __init__(self) -> None:
        self.store = LocalDataStore()
        self.resolver = StudentEntityResolver(store=self.store)
        self.evidence_service = StudentEvidenceService(store=self.store)

    def resolve_student(self, student_id: str, student_mention: str) -> dict[str, Any]:
        return self.resolver.resolve(student_id=student_id, student_mention=student_mention)

    def load_student_evidence(
        self,
        student_id: str,
        user_mentioned_points: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.evidence_service.build_student_evidence(
            student_id=student_id,
            user_mentioned_points=user_mentioned_points or [],
        )

    def get_intervention_cases(self, limit: int) -> list[dict[str, Any]]:
        return self.store.intervention_cases[:limit]


class SQLiteStudentDataAdapter(BaseStudentDataAdapter):
    provider_name = "sqlite"

    def _connect(self) -> sqlite3.Connection:
        db_path = settings.PROJECT_ROOT / settings.SQLITE_DB_PATH
        return sqlite3.connect(str(db_path))

    def resolve_student(self, student_id: str, student_mention: str) -> dict[str, Any]:
        if student_id:
            return {
                "student_id": student_id,
                "resolved_by": "student_id",
                "matched_profile": {},
                "need_clarify": False,
                "clarify_message": "",
            }
        if student_mention.endswith("同学"):
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT student_id, student_name_masked FROM student_profiles WHERE student_name_masked = ? LIMIT 2",
                    (student_mention,),
                )
                rows = cur.fetchall()
                if len(rows) == 1:
                    return {
                        "student_id": rows[0][0],
                        "resolved_by": "name_fuzzy",
                        "matched_profile": {"student_name_masked": rows[0][1]},
                        "need_clarify": False,
                        "clarify_message": "",
                    }
                if len(rows) > 1:
                    return {
                        "student_id": "",
                        "resolved_by": "name_ambiguous",
                        "matched_profile": {},
                        "need_clarify": True,
                        "clarify_message": "同名学生不唯一，请补充 student_id。",
                    }
        return {
            "student_id": "",
            "resolved_by": "unresolved",
            "matched_profile": {},
            "need_clarify": True,
            "clarify_message": "未定位到学生，请提供 student_id。",
        }

    def load_student_evidence(
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
                    "unmatched_user_mentioned_points": user_mentioned_points or [],
                    "data_weak_points": [],
                    "evidence_alignment_status": "insufficient_data",
                },
            }
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM student_profiles WHERE student_id = ? LIMIT 1", (student_id,))
            profile_row = cur.fetchone()
            profile = _row_to_dict(cur, profile_row) if profile_row else {}

            cur.execute(
                "SELECT submission_id, knowledge_point, judge_status, error_type, score, submitted_at "
                "FROM practice_submissions WHERE student_id = ? ORDER BY submitted_at DESC LIMIT ?",
                (student_id, settings.MAX_SUBMISSIONS),
            )
            submissions = [
                {
                    "submission_id": r[0],
                    "knowledge_point": r[1],
                    "judge_status": r[2],
                    "error_type": r[3],
                    "score": r[4],
                    "submitted_at": r[5],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT weak_knowledge_points, mastery_level, recent_error_types, class_attention_note "
                "FROM student_mastery_snapshots WHERE student_id = ? ORDER BY snapshot_date DESC LIMIT 1",
                (student_id,),
            )
            mastery_row = cur.fetchone()
            weak_points = json.loads(mastery_row[0]) if mastery_row and mastery_row[0] else []
            mastery_level = mastery_row[1] if mastery_row else ""
            recent_error_types = json.loads(mastery_row[2]) if mastery_row and mastery_row[2] else []
            class_note = mastery_row[3] if mastery_row else ""

            cur.execute(
                "SELECT teacher_acceptance, intervention_goal, recommended_actions_raw, follow_up_days "
                "FROM intervention_feedback WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
                (student_id,),
            )
            feedback_row = cur.fetchone()
            feedback = (
                {
                    "teacher_acceptance": feedback_row[0],
                    "intervention_goal": feedback_row[1],
                    "recommended_actions_raw": feedback_row[2],
                    "follow_up_days": feedback_row[3],
                }
                if feedback_row
                else {}
            )

        error_dist: dict[str, int] = {}
        for item in submissions:
            key = item.get("error_type") or "unknown"
            error_dist[key] = error_dist.get(key, 0) + 1

        weak_points = weak_points[:5]
        user_points = [str(x).strip() for x in (user_mentioned_points or []) if str(x).strip()]
        submission_points = {str(item.get("knowledge_point", "")).strip() for item in submissions if item.get("knowledge_point")}
        weak_set = {str(x).strip() for x in weak_points if str(x).strip()}
        matched = [kp for kp in user_points if kp in submission_points or kp in weak_set]
        unmatched = [kp for kp in user_points if kp not in matched]
        if not submission_points and not weak_set:
            align_status = "insufficient_data"
        elif user_points and len(matched) == len(user_points):
            align_status = "aligned"
        elif matched:
            align_status = "partially_aligned"
        elif user_points:
            align_status = "mismatched"
        else:
            align_status = "insufficient_data"

        return {
            "profile_summary": {
                "student_id": profile.get("student_id", ""),
                "student_name_masked": profile.get("student_name_masked", ""),
                "grade_band": profile.get("grade_band", ""),
                "current_class_id": profile.get("current_class_id", ""),
                "attention_risk_level": profile.get("attention_risk_level", ""),
            },
            "recent_submission_summary": {"total": len(submissions), "submissions": submissions},
            "weak_point_summary": {
                "weak_knowledge_points": weak_points,
                "mastery_level": mastery_level,
                "class_attention_note": class_note[:120],
            },
            "recent_error_summary": {
                "error_distribution": error_dist,
                "recent_error_types": recent_error_types[:5],
            },
            "intervention_feedback_summary": feedback,
            "alignment_summary": {
                "matched_user_mentioned_points": matched,
                "unmatched_user_mentioned_points": unmatched,
                "data_weak_points": weak_points,
                "evidence_alignment_status": align_status,
            },
        }

    def get_intervention_cases(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT case_id, student_id, class_id, intervention_goal, recommended_actions_raw, follow_up_days "
                "FROM intervention_feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "case_id": row[0],
                "student_id": row[1],
                "class_id": row[2],
                "intervention_goal": row[3],
                "recommended_actions_raw": row[4],
                "follow_up_days": row[5],
            }
            for row in rows
        ]


class MySQLStudentDataAdapter(BaseStudentDataAdapter):
    provider_name = "mysql"

    def resolve_student(self, student_id: str, student_mention: str) -> dict[str, Any]:
        # TODO: connect MySQL and resolve by student_id/name using indexed query.
        return {
            "student_id": student_id,
            "resolved_by": "mysql_placeholder",
            "matched_profile": {},
            "need_clarify": not bool(student_id),
            "clarify_message": "MySQL provider 占位实现，请补充真实连接逻辑。",
        }

    def load_student_evidence(
        self,
        student_id: str,
        user_mentioned_points: list[str] | None = None,
    ) -> dict[str, Any]:
        # TODO: implement MySQL query pipeline for profile/submissions/mastery/feedback.
        return {
            "profile_summary": {},
            "recent_submission_summary": {"submissions": [], "total": 0},
            "weak_point_summary": {"weak_knowledge_points": [], "mastery_level": ""},
            "recent_error_summary": {"error_distribution": {}},
            "intervention_feedback_summary": {
                "note": "mysql provider placeholder",
                "mysql_host": settings.MYSQL_HOST,
                "mysql_db": settings.MYSQL_DB,
            },
            "alignment_summary": {
                "matched_user_mentioned_points": [],
                "unmatched_user_mentioned_points": user_mentioned_points or [],
                "data_weak_points": [],
                "evidence_alignment_status": "insufficient_data",
            },
        }

    def get_intervention_cases(self, limit: int) -> list[dict[str, Any]]:
        # TODO: query intervention history table in MySQL.
        return []


def get_student_data_adapter() -> BaseStudentDataAdapter:
    provider = settings.STUDENT_DATA_PROVIDER.lower()
    if provider == "sqlite":
        return SQLiteStudentDataAdapter()
    if provider == "mysql":
        return MySQLStudentDataAdapter()
    return LocalCSVJSONLStudentDataAdapter()
