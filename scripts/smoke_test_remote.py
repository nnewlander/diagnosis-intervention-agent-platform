from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.graph.workflow import build_agent_graph


def main() -> None:
    settings.RAG_PROVIDER = "remote"
    settings.KG_PROVIDER = "remote"
    state = build_agent_graph().invoke(
        {"request_text": "请诊断 student_id:STU-0001 的函数报错并给出建议"}
    )
    print("[remote smoke] task_type:", state.get("task_type"))
    print("[remote smoke] rag provider:", settings.RAG_PROVIDER)
    print("[remote smoke] kg provider:", settings.KG_PROVIDER)
    print(
        "[remote smoke] rag evidence sample:",
        state.get("rag_evidence", [{}])[0] if state.get("rag_evidence") else {},
    )
    print(
        "[remote smoke] kg evidence sample:",
        state.get("kg_evidence", [{}])[0] if state.get("kg_evidence") else {},
    )


if __name__ == "__main__":
    main()
