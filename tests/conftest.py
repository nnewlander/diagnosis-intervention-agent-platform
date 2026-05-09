"""
单测中强制 LangGraph 节点使用本地 RAG/KG adapter，避免 .env 指向 remote 且服务未启动导致检索为空。
不修改业务代码，仅在 pytest 运行时替换 app.graph.nodes 中的工厂函数。
"""
from __future__ import annotations

import pytest

from app.tools.kg_adapter import LocalKGAdapter
from app.tools.rag_adapter import LocalRAGAdapter


@pytest.fixture(autouse=True)
def _force_local_rag_kg_adapters(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    # remote mapper 单测需走 RemoteRAGAdapter / RemoteKGAdapter，此处不再替换
    if request.path.name == "test_remote_mappers.py":
        return
    monkeypatch.setattr("app.graph.nodes.get_rag_adapter", lambda: LocalRAGAdapter())
    monkeypatch.setattr("app.graph.nodes.get_kg_adapter", lambda: LocalKGAdapter())
