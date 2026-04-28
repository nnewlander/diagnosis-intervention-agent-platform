from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator


class RemoteQueryContract(BaseModel):
    query: str
    top_k: int = 5
    keywords: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class RemoteRAGOutputContract(BaseModel):
    hits: list[dict[str, Any]] | None = None
    results: list[dict[str, Any]] | None = None
    items: list[dict[str, Any]] | None = None
    data: list[dict[str, Any]] | None = None
    error: str = ""

    @model_validator(mode="after")
    def ensure_any_result_field(self):
        if not any([self.hits, self.results, self.items, self.data]):
            raise ValueError("RAG response must contain one of hits/results/items/data")
        return self


class RemoteKGOutputContract(BaseModel):
    records: list[dict[str, Any]] | None = None
    results: list[dict[str, Any]] | None = None
    paths: list[dict[str, Any]] | None = None
    items: list[dict[str, Any]] | None = None
    data: list[dict[str, Any]] | None = None
    error: str = ""

    @model_validator(mode="after")
    def ensure_any_result_field(self):
        if not any([self.records, self.results, self.paths, self.items, self.data]):
            raise ValueError("KG response must contain one of records/results/paths/items/data")
        return self


def validate_query_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        RemoteQueryContract.model_validate(payload)
        return True, ""
    except ValidationError as exc:
        return False, str(exc)


def validate_rag_output(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    try:
        validated = RemoteRAGOutputContract.model_validate(payload)
        return True, validated.model_dump(), ""
    except ValidationError as exc:
        return False, {}, str(exc)


def validate_kg_output(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    try:
        validated = RemoteKGOutputContract.model_validate(payload)
        return True, validated.model_dump(), ""
    except ValidationError as exc:
        return False, {}, str(exc)
