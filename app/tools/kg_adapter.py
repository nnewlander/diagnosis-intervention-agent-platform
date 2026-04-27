from typing import Any

import requests

from app.core.config import settings
from app.data_loader.loader import LocalDataStore
from app.tools.base import BaseKGAdapter


class LocalKGAdapter(BaseKGAdapter):
    provider_name = "local"

    def __init__(self) -> None:
        self.store = LocalDataStore()

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        terms = keywords or [query]
        return self.store.search_kg(terms, limit=top_k)


class RemoteKGAdapter(BaseKGAdapter):
    provider_name = "remote"

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        # HTTP API placeholder (future project3 integration endpoint).
        headers = {}
        if settings.KG_API_KEY:
            headers["Authorization"] = f"Bearer {settings.KG_API_KEY}"
        payload = {"query": query, "keywords": keywords, "top_k": top_k}
        try:
            response = requests.post(
                f"{settings.KG_API_BASE}/query",
                json=payload,
                headers=headers,
                timeout=settings.KG_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except Exception:
            # TODO: add Neo4j/Cypher execution mode:
            # - initialize Neo4j driver by URI/user/password
            # - map keywords to Cypher templates
            # - normalize graph records to schema-compatible dict
            return [{"source": "remote_mock_kg", "content": "KG remote placeholder response"}]


def get_kg_adapter() -> BaseKGAdapter:
    if settings.KG_PROVIDER.lower() == "remote":
        return RemoteKGAdapter()
    return LocalKGAdapter()
