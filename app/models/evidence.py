from typing import Any

from pydantic import BaseModel, Field


class RAGEvidenceItem(BaseModel):
    source_id: str = ""
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    source_type: str = "rag"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KGEvidenceItem(BaseModel):
    entity: str = ""
    entity_type: str = ""
    relation: str = ""
    target: str = ""
    evidence: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudentEvidenceBundle(BaseModel):
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    recent_submission_summary: dict[str, Any] = Field(default_factory=dict)
    weak_point_summary: dict[str, Any] = Field(default_factory=dict)
    recent_error_summary: dict[str, Any] = Field(default_factory=dict)
    intervention_feedback_summary: dict[str, Any] = Field(default_factory=dict)


class PackageRecommendationItem(BaseModel):
    package_id: str = ""
    package_name: str = ""
    reason: str = ""
    difficulty_level: str = ""
