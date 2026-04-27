from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import router as agent_router

app = FastAPI(title=settings.PROJECT_NAME, version="0.2.0")
app.include_router(agent_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
