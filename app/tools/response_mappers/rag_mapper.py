from typing import Any

from app.models.evidence import RAGEvidenceItem
from app.tools.response_mappers.base import BaseResponseMapper


class RAGResponseMapper(BaseResponseMapper):
    """Compatible mapper for common RAG response keys: hits/results/items."""

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ["hits", "results", "items", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # ask-style fallback: no hits list, only answer-like content.
        for key in ["answer", "response", "output"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [{"id": "ask_fallback", "name": "ask_style_response", "text": value, "similarity": 0.0}]
        return []

    def map_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for raw in self._extract_items(payload):
            item = RAGEvidenceItem(
                source_id=str(raw.get("source_id") or raw.get("id") or raw.get("doc_id") or ""),
                title=str(raw.get("title") or raw.get("name") or ""),
                snippet=str(raw.get("snippet") or raw.get("content") or raw.get("text") or ""),
                score=float(raw.get("score") or raw.get("similarity") or 0.0),
                source_type=str(raw.get("source_type") or raw.get("type") or "rag"),
                metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            )
            mapped.append(item.model_dump())
        return mapped
