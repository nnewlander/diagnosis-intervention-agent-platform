from fastapi import FastAPI
from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    query: str
    keywords: list[str] = Field(default_factory=list)
    top_k: int = 5


app = FastAPI(title="Mock RAG Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "mock_rag"}


@app.post("/search")
def search(query: RAGQuery) -> dict:
    # Use "hits" instead of "items" intentionally for mapper compatibility tests.
    return {
        "hits": [
            {
                "id": f"rag-{idx+1}",
                "name": f"RAG命中{idx+1}",
                "text": f"query={query.query}, keyword={query.keywords[:1]}",
                "similarity": 0.8 - idx * 0.1,
                "type": "mock_remote_rag",
            }
            for idx in range(min(query.top_k, 3))
        ]
    }
