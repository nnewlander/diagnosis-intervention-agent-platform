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
            raw_meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            # Preserve project3 meta fields when available (either in metadata or top-level keys).
            meta = {
                "source": raw_meta.get("source") or raw.get("source"),
                "seed_version": raw_meta.get("seed_version") or raw.get("seed_version"),
                "confidence": raw_meta.get("confidence") or raw.get("confidence"),
                "cypher_template": raw_meta.get("cypher_template") or raw.get("cypher_template"),
                "relation_props": raw_meta.get("relation_props") or raw.get("relation_props"),
            }
            # Keep any remaining raw metadata as well.
            for k, v in raw_meta.items():
                if k not in meta or meta[k] is None:
                    meta[k] = v
            item = KGEvidenceItem(
                entity=str(raw.get("entity") or raw.get("subject") or raw.get("source") or ""),
                entity_type=str(raw.get("entity_type") or raw.get("subject_type") or ""),
                relation=str(raw.get("relation") or raw.get("predicate") or ""),
                target=str(raw.get("target") or raw.get("object") or ""),
                evidence=str(raw.get("evidence") or raw.get("snippet") or raw.get("text") or ""),
                score=float(raw.get("score") or raw.get("confidence") or 0.0),
                metadata={k: v for k, v in meta.items() if v is not None},
            )
            mapped.append(item.model_dump())
        return mapped
