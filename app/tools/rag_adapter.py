from typing import Any

import requests

from app.core.config import settings
from app.data_loader.loader import LocalDataStore
from app.tools.base import BaseRAGAdapter


class LocalRAGAdapter(BaseRAGAdapter):
    provider_name = "local"

    def __init__(self) -> None:
        self.store = LocalDataStore()

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        terms = keywords or [query]
        return self.store.search_rag(terms, limit=top_k)


class RemoteRAGAdapter(BaseRAGAdapter):
    provider_name = "remote"

    def search(self, query: str, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        headers = {}
        if settings.RAG_API_KEY:
            headers["Authorization"] = f"Bearer {settings.RAG_API_KEY}"
        payload = {"query": query, "keywords": keywords, "top_k": top_k}
        try:
            response = requests.post(
                f"{settings.RAG_API_BASE}/search",
                json=payload,
                headers=headers,
                timeout=settings.RAG_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except Exception:
            # Remote provider placeholder fallback for current local runnable mode.
            return [{"source": "remote_mock_rag", "content": "RAG remote placeholder response"}]


def get_rag_adapter() -> BaseRAGAdapter:
    if settings.RAG_PROVIDER.lower() == "remote":
        return RemoteRAGAdapter()
    return LocalRAGAdapter()
