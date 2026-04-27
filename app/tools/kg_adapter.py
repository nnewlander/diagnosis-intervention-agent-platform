from typing import Any

import requests

from app.core.config import settings
from app.data_loader.loader import LocalDataStore
from app.models.evidence import KGEvidenceItem
from app.tools.base import BaseKGAdapter
from app.tools.contracts import validate_kg_output, validate_query_contract
from app.tools.response_mappers.kg_mapper import KGResponseMapper


class LocalKGAdapter(BaseKGAdapter):
    provider_name = "local"

    def __init__(self) -> None:
        self.store = LocalDataStore()
        self.last_status = {"mapper": "local_normalize", "validation_ok": True, "error": ""}

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        terms = keywords or [query]
        mapped = []
        for raw in self.store.search_kg(terms, limit=top_k):
            mapped.append(
                KGEvidenceItem(
                    entity=str(raw.get("entity") or raw.get("source") or ""),
                    entity_type=str(raw.get("entity_type") or raw.get("subject_type") or ""),
                    relation=str(raw.get("relation") or raw.get("predicate") or ""),
                    target=str(raw.get("target") or raw.get("object") or ""),
                    evidence=str(raw.get("evidence") or raw.get("content") or str(raw)[:200]),
                    score=float(raw.get("score") or 0.0),
                    metadata={"raw_keys": list(raw.keys())[:8]},
                ).model_dump()
            )
        return mapped


class RemoteKGAdapter(BaseKGAdapter):
    provider_name = "remote"

    def __init__(self) -> None:
        self.mapper = KGResponseMapper()
        self.last_status = {"mapper": "KGResponseMapper", "validation_ok": True, "error": ""}

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        # HTTP API placeholder (future project3 integration endpoint).
        headers = {}
        if settings.KG_API_KEY:
            headers["Authorization"] = f"Bearer {settings.KG_API_KEY}"
        payload = {"query": query, "keywords": keywords, "top_k": top_k}
        ok, error = validate_query_contract(payload)
        if not ok:
            self.last_status = {"mapper": "KGResponseMapper", "validation_ok": False, "error": error}
            return []
        try:
            response = requests.post(
                f"{settings.KG_API_BASE}/graph_query",
                json=payload,
                headers=headers,
                timeout=settings.KG_TIMEOUT,
            )
            response.raise_for_status()
            raw_data = response.json()
            valid, normalized_payload, contract_error = validate_kg_output(raw_data)
            if not valid:
                self.last_status = {
                    "mapper": "KGResponseMapper",
                    "validation_ok": False,
                    "error": contract_error,
                }
                return []
            mapped = self.mapper.map_items(normalized_payload)
            self.last_status = {"mapper": "KGResponseMapper", "validation_ok": True, "error": ""}
            return mapped
        except Exception as exc:
            # TODO: add Neo4j/Cypher execution mode:
            # - initialize Neo4j driver by URI/user/password
            # - map keywords to Cypher templates
            # - normalize graph records to schema-compatible dict
            self.last_status = {"mapper": "KGResponseMapper", "validation_ok": False, "error": str(exc)}
            return []


def get_kg_adapter() -> BaseKGAdapter:
    if settings.KG_PROVIDER.lower() == "remote":
        return RemoteKGAdapter()
    return LocalKGAdapter()
