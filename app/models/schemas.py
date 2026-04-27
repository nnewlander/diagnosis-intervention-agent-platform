from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    request_text: str = Field(..., description="Teacher natural language request.")


class EvidenceSummary(BaseModel):
    rag_summary: dict[str, Any] = Field(default_factory=dict)
    kg_summary: dict[str, Any] = Field(default_factory=dict)
    mysql_summary: dict[str, Any] = Field(default_factory=dict)
    intervention_case_summary: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    task_type: str
    primary_task_type: str = "unknown"
    secondary_task_types: list[str] = Field(default_factory=list)
    parsed_slots: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: EvidenceSummary
    diagnosis: dict[str, Any]
    intervention_plan: dict[str, Any]
    recommended_packages: list[dict[str, Any]]
    final_response: str
    debug_trace: list[dict[str, Any]] = Field(default_factory=list)
    debug_state: dict[str, Any]
