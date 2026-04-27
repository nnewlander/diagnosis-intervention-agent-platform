from fastapi import FastAPI
from pydantic import BaseModel, Field


class KGQuery(BaseModel):
    query: str
    keywords: list[str] = Field(default_factory=list)
    top_k: int = 5


app = FastAPI(title="Mock KG Service", version="0.1.0")


@app.post("/graph_query")
def graph_query(query: KGQuery) -> dict:
    # Use "records" + subject/predicate/object keys intentionally.
    return {
        "records": [
            {
                "subject": query.keywords[0] if query.keywords else "函数",
                "subject_type": "knowledge_point",
                "predicate": "related_error",
                "object": "TypeError",
                "snippet": "常见于参数类型不匹配。",
                "confidence": 0.77,
            }
        ][: query.top_k]
    }
