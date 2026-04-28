from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from app.core.config import settings
from app.graph.workflow import build_agent_graph


def _join_url(base: str, path: str) -> str:
    p = path if path.startswith("/") else f"/{path}"
    return f"{base.rstrip('/')}{p}"


def _get_json(url: str, timeout: int = 5) -> tuple[bool, dict]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return True, data if isinstance(data, dict) else {}
    except Exception:
        return False, {}


def _post_json(url: str, payload: dict | None = None, timeout: int = 15) -> tuple[bool, dict]:
    try:
        resp = requests.post(url, json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return True, data if isinstance(data, dict) else {}
    except Exception:
        return False, {}


def _evidence_flags(item: dict) -> dict:
    metadata = item.get("metadata") or {}
    fallback = bool(metadata.get("fallback")) or item.get("source_type") == "fallback_error_guide"
    route = metadata.get("route") or ""
    is_faq_bm25 = (item.get("source_type") == "faq") and (route == "bm25_faq")
    return {"fallback": fallback, "route": route, "is_faq_bm25": is_faq_bm25}


def _print_top_evidence(items: list[dict], top_k: int = 3) -> None:
    for idx, item in enumerate(items[:top_k], start=1):
        metadata = item.get("metadata") or {}
        flags = _evidence_flags(item)
        snippet = str(item.get("snippet") or "")[:120]
        print(f"[real_rag_smoke] top{idx}")
        print(f"- source_id={item.get('source_id','')}")
        print(f"- title={item.get('title','')}")
        print(f"- source_type={item.get('source_type','')}")
        print(f"- metadata.route={flags['route']}")
        print(f"- metadata.fallback={bool(metadata.get('fallback'))}")
        print(f"- snippet={snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="真实项目二 RAG 联调 smoke test")
    parser.add_argument("--force-remote", action="store_true", help="强制使用 RAG remote 模式")
    parser.add_argument("--fail-on-fallback", action="store_true", help="若命中 fallback evidence 则退出非0")
    parser.add_argument("--auto-warmup", action="store_true", help="自动检查ready并触发warmup")
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

    if args.auto_warmup:
        base = settings.RAG_API_BASE
        ok, health = _get_json(_join_url(base, "/health"), timeout=5)
        print(f"[real_rag_smoke] /health ok={ok} payload={health}")
        ok_ready1, ready1 = _get_json(_join_url(base, "/ready"), timeout=5)
        print(f"[real_rag_smoke] /ready ok={ok_ready1} payload={ready1}")
        faq_ready = bool(ready1.get("faq_ready")) if isinstance(ready1, dict) else False
        bm25_ready = bool(ready1.get("bm25_ready")) if isinstance(ready1, dict) else False
        if ok_ready1 and (not faq_ready or not bm25_ready):
            print("[real_rag_smoke] warmup required, calling /warmup ...")
            ok_warm, warm = _post_json(_join_url(base, "/warmup"), timeout=max(15, settings.RAG_TIMEOUT))
            print(f"[real_rag_smoke] /warmup ok={ok_warm} payload={warm}")
            ok_ready2, ready2 = _get_json(_join_url(base, "/ready"), timeout=5)
            print(f"[real_rag_smoke] /ready(after warmup) ok={ok_ready2} payload={ready2}")

    request_text = "课堂演示遇到 NameError，应该怎么给学生解释？"
    state = build_agent_graph().invoke({"request_text": request_text})

    trace = state.get("debug_trace", [])
    rag_trace = next((t for t in trace if t.get("node_name") == "fetch_rag_evidence"), {})
    output_summary = rag_trace.get("output_summary", {})

    parsed_slots = state.get("parsed_slots", {}) or {}
    rag_query = state.get("rag_query", "")
    print("[real_rag_smoke] 请求解析")
    print(f"- original_request_text={request_text}")
    print(f"- rag_query={rag_query}")
    print(f"- parsed_error_type={state.get('error_type','')}")
    print(f"- parsed_knowledge_points={parsed_slots.get('knowledge_points', [])}")

    print("[real_rag_smoke] 调用结果")
    print(f"- http_call_success={output_summary.get('validation_ok', False) or bool(state.get('rag_evidence'))}")
    print(f"- rag_response_style={settings.RAG_RESPONSE_STYLE}")
    print(f"- evidence_count={len(state.get('rag_evidence', []))}")
    rag_items = state.get("rag_evidence", []) or []
    first = rag_items[0] if rag_items else {}
    flags = _evidence_flags(first) if isinstance(first, dict) else {"fallback": False, "route": "", "is_faq_bm25": False}
    if flags.get("fallback"):
        print("[real_rag_smoke][WARN] 当前拿到的是 fallback evidence，不是真实 FAQ/BM25 evidence。")
    if flags.get("is_faq_bm25"):
        print("[real_rag_smoke] 当前拿到的是真实 FAQ/BM25 evidence。")
    _print_top_evidence(rag_items, top_k=3)
    print(f"- final_response_non_empty={bool(state.get('final_response', '').strip())}")
    print(f"- validation_ok={output_summary.get('validation_ok', False)}")
    print(f"- mapper_used={output_summary.get('mapper', '')}")
    if args.fail_on_fallback and flags.get("fallback"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
