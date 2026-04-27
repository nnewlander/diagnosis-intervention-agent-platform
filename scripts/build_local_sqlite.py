from pathlib import Path
import sqlite3
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.data_loader.loader import LocalDataStore


def build_local_sqlite(db_path: Path | None = None) -> Path:
    target = db_path or (settings.PROJECT_ROOT / settings.SQLITE_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    store = LocalDataStore()

    with sqlite3.connect(str(target)) as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS student_profiles;
            DROP TABLE IF EXISTS class_student_map;
            DROP TABLE IF EXISTS practice_submissions;
            DROP TABLE IF EXISTS student_mastery_snapshots;
            DROP TABLE IF EXISTS intervention_feedback;

            CREATE TABLE student_profiles (
                student_id TEXT PRIMARY KEY,
                student_name_masked TEXT,
                grade_band TEXT,
                primary_course_module TEXT,
                current_class_id TEXT,
                attention_risk_level TEXT
            );

            CREATE TABLE class_student_map (
                class_id TEXT,
                student_id TEXT
            );

            CREATE TABLE practice_submissions (
                submission_id TEXT PRIMARY KEY,
                student_id TEXT,
                class_id TEXT,
                knowledge_point TEXT,
                judge_status TEXT,
                error_type TEXT,
                score REAL,
                submitted_at TEXT
            );

            CREATE TABLE student_mastery_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                student_id TEXT,
                snapshot_date TEXT,
                weak_knowledge_points TEXT,
                mastery_level TEXT,
                recent_error_types TEXT,
                class_attention_note TEXT
            );

            CREATE TABLE intervention_feedback (
                case_id TEXT PRIMARY KEY,
                created_at TEXT,
                class_id TEXT,
                student_id TEXT,
                intervention_goal TEXT,
                recommended_actions_raw TEXT,
                follow_up_days INTEGER,
                teacher_acceptance TEXT
            );

            CREATE INDEX idx_profiles_class_id ON student_profiles(current_class_id);
            CREATE INDEX idx_submissions_student_time ON practice_submissions(student_id, submitted_at);
            CREATE INDEX idx_snapshots_student_date ON student_mastery_snapshots(student_id, snapshot_date);
            CREATE INDEX idx_intervention_student_time ON intervention_feedback(student_id, created_at);
            """
        )

        for row in store.student_profiles:
            cur.execute(
                "INSERT OR REPLACE INTO student_profiles VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row.get("student_id", ""),
                    row.get("student_name_masked", ""),
                    row.get("grade_band", ""),
                    row.get("primary_course_module", ""),
                    row.get("current_class_id", ""),
                    row.get("attention_risk_level", ""),
                ),
            )
            cur.execute(
                "INSERT INTO class_student_map VALUES (?, ?)",
                (row.get("current_class_id", ""), row.get("student_id", "")),
            )

        for item in store.practice_submissions:
            cur.execute(
                "INSERT OR REPLACE INTO practice_submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.get("submission_id", ""),
                    item.get("student_id", ""),
                    item.get("class_id", ""),
                    item.get("knowledge_point", ""),
                    item.get("judge_status", ""),
                    item.get("error_type", ""),
                    item.get("score", 0),
                    item.get("submitted_at", ""),
                ),
            )

        for item in store.mastery_snapshots:
            cur.execute(
                "INSERT OR REPLACE INTO student_mastery_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.get("snapshot_id", ""),
                    item.get("student_id", ""),
                    item.get("snapshot_date", ""),
                    json.dumps(item.get("weak_knowledge_points", []), ensure_ascii=False),
                    item.get("mastery_level", ""),
                    json.dumps(item.get("recent_error_types", []), ensure_ascii=False),
                    item.get("class_attention_note", ""),
                ),
            )

        for item in store.intervention_cases:
            cur.execute(
                "INSERT OR REPLACE INTO intervention_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.get("case_id", ""),
                    item.get("created_at", ""),
                    item.get("class_id", ""),
                    item.get("student_id", ""),
                    item.get("intervention_goal", ""),
                    item.get("recommended_actions_raw", ""),
                    item.get("follow_up_days", 0),
                    item.get("teacher_acceptance", ""),
                ),
            )
        conn.commit()
    return target


if __name__ == "__main__":
    db = build_local_sqlite()
    print(f"[sqlite] local database built at: {db}")
