import csv
import json
from pathlib import Path
from typing import Any

from app.core.config import settings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


class LocalDataStore:
    """In-memory local data cache for the mock agent pipeline."""

    def __init__(self) -> None:
        self.teacher_dialogs = read_jsonl(
            settings.RAW_DIR / "raw_teacher_support_dialogs_10pct.jsonl"
        )
        self.intervention_cases = read_jsonl(
            settings.RAW_DIR / "raw_intervention_cases_10pct.jsonl"
        )
        self.assignment_catalog = read_jsonl(
            settings.RAW_DIR / "raw_assignment_package_catalog_10pct.jsonl"
        )
        self.student_profiles = read_csv(
            settings.MYSQL_DIR / "student_profiles_10pct.csv"
        )
        self.practice_submissions = read_jsonl(
            settings.MYSQL_DIR / "practice_submissions_10pct.jsonl"
        )
        self.mastery_snapshots = read_jsonl(
            settings.MYSQL_DIR / "student_mastery_snapshots_10pct.jsonl"
        )
        self.rag_docs = read_jsonl(
            settings.SOURCES_DIR / "reused_project2_rag_subset_10pct.jsonl"
        )
        self.kg_docs = read_jsonl(
            settings.SOURCES_DIR / "reused_project3_kg_subset_10pct.jsonl"
        )

    @staticmethod
    def _first_value(data: dict[str, Any], candidates: list[str]) -> str:
        for key in candidates:
            value = data.get(key)
            if value:
                return str(value)
        return ""

    def get_student_profile(self, student_id: str) -> dict[str, Any]:
        if not student_id:
            return {}
        for row in self.student_profiles:
            row_student_id = self._first_value(
                row, ["student_id", "sid", "studentId", "id"]
            )
            if row_student_id == student_id:
                return row
        return {}

    def get_submissions(self, student_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if not student_id:
            return []
        results: list[dict[str, Any]] = []
        for item in self.practice_submissions:
            item_student_id = self._first_value(
                item, ["student_id", "sid", "studentId", "user_id"]
            )
            if item_student_id == student_id:
                results.append(item)
            if len(results) >= limit:
                break
        return results

    def search_rag(self, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
        if not keywords:
            return []
        results: list[dict[str, Any]] = []
        lowered = [k.lower() for k in keywords if k]
        for doc in self.rag_docs:
            text = json.dumps(doc, ensure_ascii=False).lower()
            if any(k in text for k in lowered):
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def search_kg(self, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
        if not keywords:
            return []
        results: list[dict[str, Any]] = []
        lowered = [k.lower() for k in keywords if k]
        for doc in self.kg_docs:
            text = json.dumps(doc, ensure_ascii=False).lower()
            if any(k in text for k in lowered):
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def recommend_packages(
        self, knowledge_points: list[str], limit: int = 3
    ) -> list[dict[str, Any]]:
        if not knowledge_points:
            return []
        lowered = [k.lower() for k in knowledge_points if k]
        results: list[dict[str, Any]] = []
        for item in self.assignment_catalog:
            text = json.dumps(item, ensure_ascii=False).lower()
            if any(k in text for k in lowered):
                results.append(item)
            if len(results) >= limit:
                break
        return results
