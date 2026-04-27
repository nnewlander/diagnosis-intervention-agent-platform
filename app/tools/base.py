from abc import ABC, abstractmethod
from typing import Any


class BaseRAGAdapter(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class BaseKGAdapter(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class BaseStudentDataAdapter(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def resolve_student(self, student_id: str, student_mention: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_student_evidence(self, student_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_intervention_cases(self, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class BasePackageAdapter(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def recommend(
        self,
        knowledge_points: list[str],
        grade_band: str,
        difficulty_level: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError
