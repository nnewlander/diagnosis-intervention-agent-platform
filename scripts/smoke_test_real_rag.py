from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.graph.workflow import build_agent_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="真实项目二 RAG 联调 smoke test")
    parser.add_argument("--force-remote", action="store_true", help="强制使用 RAG remote 模式")
    args = parser.parse_args()

    if args.force_remote:
        settings.RAG_PROVIDER = "remote"
        settings.KG_PROVIDER = "local"
    else:
        settings.RAG_PROVIDER = os.getenv("RAG_PROVIDER", settings.RAG_PROVIDER)
        settings.KG_PROVIDER = os.getenv("KG_PROVIDER", settings.KG_PROVIDER)

    print("[real_rag_smoke] 当前最终配置")
    print(f"- RAG_PROVIDER={settings.RAG_PROVIDER}")
    print(f"- RAG_API_BASE={settings.RAG_API_BASE}")
    print(f"- RAG_ENDPOINT={settings.RAG_ENDPOINT}")
    print(f"- KG_PROVIDER={settings.KG_PROVIDER}")

    if settings.RAG_PROVIDER.lower() != "remote":
        print("[real_rag_smoke][WARN] 当前 RAG_PROVIDER 不是 remote。")
        print("请在 .env 中设置 RAG_PROVIDER=remote，或使用 --force-remote。")
        return 1

    request_text = "课堂演示遇到 NameError，应该怎么给学生解释？"
    state = build_agent_graph().invoke({"request_text": request_text})

    trace = state.get("debug_trace", [])
    rag_trace = next((t for t in trace if t.get("node_name") == "fetch_rag_evidence"), {})
    output_summary = rag_trace.get("output_summary", {})

    print("[real_rag_smoke] 调用结果")
    print(f"- http_call_success={output_summary.get('validation_ok', False) or bool(state.get('rag_evidence'))}")
    print(f"- rag_response_style={settings.RAG_RESPONSE_STYLE}")
    print(f"- evidence_count={len(state.get('rag_evidence', []))}")
    print(f"- first_evidence={state.get('rag_evidence', [{}])[0] if state.get('rag_evidence') else {}}")
    print(f"- final_response_non_empty={bool(state.get('final_response', '').strip())}")
    print(f"- validation_ok={output_summary.get('validation_ok', False)}")
    print(f"- mapper_used={output_summary.get('mapper', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
