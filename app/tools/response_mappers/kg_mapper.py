from typing import Any

from app.models.evidence import KGEvidenceItem
from app.tools.response_mappers.base import BaseResponseMapper


class KGResponseMapper(BaseResponseMapper):
    """Compatible mapper for KG response keys: records/results/paths."""

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ["records", "results", "paths", "items", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []

    def map_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for raw in self._extract_items(payload):
            item = KGEvidenceItem(
                entity=str(raw.get("entity") or raw.get("subject") or raw.get("source") or ""),
                entity_type=str(raw.get("entity_type") or raw.get("subject_type") or ""),
                relation=str(raw.get("relation") or raw.get("predicate") or ""),
                target=str(raw.get("target") or raw.get("object") or ""),
                evidence=str(raw.get("evidence") or raw.get("snippet") or raw.get("text") or ""),
                score=float(raw.get("score") or raw.get("confidence") or 0.0),
                metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            )
            mapped.append(item.model_dump())
        return mapped
