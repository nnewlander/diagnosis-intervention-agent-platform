from app.data_loader.loader import LocalDataStore
from app.core.config import settings


store = LocalDataStore()


def retrieve_rag(keywords: list[str]) -> list[dict]:
    return store.search_rag(keywords=keywords, limit=settings.TOP_K_RAG)


def retrieve_kg(keywords: list[str]) -> list[dict]:
    return store.search_kg(keywords=keywords, limit=settings.TOP_K_KG)


def retrieve_mysql(student_id: str) -> dict:
    return {
        "profile": store.get_student_profile(student_id),
        "submissions": store.get_submissions(student_id, limit=settings.MAX_SUBMISSIONS),
    }


def retrieve_intervention_cases(limit: int = 5) -> list[dict]:
    return store.intervention_cases[:limit]


def retrieve_packages(knowledge_points: list[str]) -> list[dict]:
    return store.recommend_packages(knowledge_points=knowledge_points, limit=3)
