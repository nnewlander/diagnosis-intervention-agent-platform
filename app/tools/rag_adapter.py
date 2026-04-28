from typing import Any
from uuid import uuid4

import requests

from app.core.config import settings
from app.data_loader.loader import LocalDataStore
from app.models.evidence import RAGEvidenceItem
from app.tools.base import BaseRAGAdapter
from app.tools.contracts import validate_query_contract, validate_rag_output
from app.tools.response_mappers.rag_mapper import RAGResponseMapper


class LocalRAGAdapter(BaseRAGAdapter):
    provider_name = "local"

    def __init__(self) -> None:
        self.store = LocalDataStore()
        self.last_status = {"mapper": "local_normalize", "validation_ok": True, "error": ""}

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        terms = keywords or [query]
        mapped = []
        for raw in self.store.search_rag(terms, limit=top_k):
            mapped.append(
                RAGEvidenceItem(
                    source_id=str(raw.get("source_id") or raw.get("id") or ""),
                    title=str(raw.get("title") or raw.get("name") or "本地文档"),
                    snippet=str(raw.get("snippet") or raw.get("content") or str(raw)[:200]),
                    score=float(raw.get("score") or 0.0),
                    source_type=str(raw.get("source_type") or "rag_local"),
                    metadata={"raw_keys": list(raw.keys())[:8]},
                ).model_dump()
            )
        return mapped


class RemoteRAGAdapter(BaseRAGAdapter):
    provider_name = "remote"

    def __init__(self) -> None:
        self.mapper = RAGResponseMapper()
        self.last_status = {"mapper": "RAGResponseMapper", "validation_ok": True, "error": ""}
        self.last_url = ""

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        headers = {}
        if settings.RAG_API_KEY:
            headers["Authorization"] = f"Bearer {settings.RAG_API_KEY}"
        endpoint = settings.RAG_ENDPOINT if settings.RAG_ENDPOINT.startswith("/") else f"/{settings.RAG_ENDPOINT}"
        request_url = f"{settings.RAG_API_BASE.rstrip('/')}{endpoint}"
        self.last_url = request_url
        payload = {
            "query": query,
            "top_k": top_k,
            "filters": {"keywords": keywords},
            "request_id": f"rag-{uuid4().hex[:12]}",
        }
        ok, error = validate_query_contract(payload)
        if not ok:
            self.last_status = {"mapper": "RAGResponseMapper", "validation_ok": False, "error": error}
            return []
        try:
            response = requests.post(
                request_url,
                json=payload,
                headers=headers,
                timeout=settings.RAG_TIMEOUT,
            )
            response.raise_for_status()
            raw_data = response.json()
            valid, normalized_payload, contract_error = validate_rag_output(raw_data)
            if not valid:
                # Fallback mode for ask-style/other non-search responses.
                fallback_items = self.mapper.map_items(raw_data)
                if fallback_items:
                    self.last_status = {
                        "mapper": "RAGResponseMapper",
                        "validation_ok": False,
                        "error": f"{contract_error}; fallback_applied",
                    }
                    return fallback_items
                self.last_status = {
                    "mapper": "RAGResponseMapper",
                    "validation_ok": False,
                    "error": contract_error,
                }
                return []
            mapped = self.mapper.map_items(normalized_payload)
            self.last_status = {
                "mapper": "RAGResponseMapper",
                "validation_ok": True,
                "error": "",
            }
            return mapped
        except Exception as exc:
            self.last_status = {
                "mapper": "RAGResponseMapper",
                "validation_ok": False,
                "error": str(exc),
            }
            return []


def get_rag_adapter() -> BaseRAGAdapter:
    if settings.RAG_PROVIDER.lower() == "remote":
        return RemoteRAGAdapter()
    return LocalRAGAdapter()
