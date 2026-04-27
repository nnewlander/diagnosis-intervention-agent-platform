import json
from typing import Any

from app.data_loader.loader import LocalDataStore
from app.tools.base import BasePackageAdapter


class LocalPackageAdapter(BasePackageAdapter):
    provider_name = "local"

    def __init__(self) -> None:
        self.store = LocalDataStore()

    @staticmethod
    def _score_package(
        package: dict[str, Any],
        knowledge_points: list[str],
        grade_band: str,
        difficulty_level: str,
    ) -> int:
        score = 0
        text = json.dumps(package, ensure_ascii=False).lower()
        for kp in knowledge_points:
            if kp.lower() in text:
                score += 3
        if grade_band and str(package.get("target_grade_band", "")).lower() == grade_band.lower():
            score += 2
        if difficulty_level and str(package.get("difficulty_level", "")).lower() == difficulty_level.lower():
            score += 1
        if package.get("status") == "active":
            score += 1
        return score

    def recommend(
        self,
        knowledge_points: list[str],
        grade_band: str,
        difficulty_level: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.store.assignment_catalog:
            score = self._score_package(item, knowledge_points, grade_band, difficulty_level)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


class RemotePackageAdapter(BasePackageAdapter):
    provider_name = "remote"

    def recommend(
        self,
        knowledge_points: list[str],
        grade_band: str,
        difficulty_level: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        # TODO: integrate real package recommendation service endpoint.
        return [
            {
                "package_id": "PKG-REMOTE-MOCK",
                "package_name": "远程推荐占位包",
                "target_knowledge_points": knowledge_points,
                "target_grade_band": grade_band,
                "difficulty_level": difficulty_level,
                "status": "active",
            }
        ][:top_k]


def get_package_adapter(provider: str = "local") -> BasePackageAdapter:
    if provider.lower() == "remote":
        return RemotePackageAdapter()
    return LocalPackageAdapter()
