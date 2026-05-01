from pathlib import Path
import argparse
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kg_smoke_probe import (
    REAL_KG_DEFAULT_BASE,
    base_points_to_mock_port,
    format_ready_payload_for_print,
    probe_kg_health_ready,
)
from rag_smoke_probe import (
    extract_rag_ready_fields,
    format_rag_ready_payload_for_print,
    joint_exit_code_for_rag_fallback,
    is_rag_fallback_item,
    probe_rag_health_ready,
    run_rag_warmup_and_reprobe,
)

from app.core.config import settings
from app.graph.workflow import build_agent_graph


def _print_rag_top_enhanced(items: list[dict], label: str) -> None:
    print(f"[rag_kg_smoke] {label} evidence_count={len(items)}")
    for idx, item in enumerate(items[:3], start=1):
        meta = item.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        snippet = str(item.get("snippet") or "")[:120]
        print(f"[rag_kg_smoke] {label} top{idx}")
        print(f"- source_id={item.get('source_id','')}")
        print(f"- title={item.get('title','')}")
        print(f"- source_type={item.get('source_type','')}")
        print(f"- metadata.route={meta.get('route','')}")
        print(f"- metadata.fallback={bool(meta.get('fallback'))}")
        print(f"- snippet={snippet}")


def _print_kg_top(items: list[dict]) -> None:
    print(f"[rag_kg_smoke] KG evidence_count={len(items)}")
    for idx, item in enumerate(items[:3], start=1):
        print(f"[rag_kg_smoke] KG top{idx}")
        print(f"- entity={item.get('entity','')}")
        print(f"- relation={item.get('relation','')}")
        print(f"- target={item.get('target','')}")
        print(f"- evidence={str(item.get('evidence',''))[:120]}")
        print(f"- metadata.source={(item.get('metadata') or {}).get('source')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="真实项目二 RAG + 项目三 KG 联合 smoke")
    parser.add_argument(
        "--rag-api-base",
        default=None,
        metavar="URL",
        help="覆盖 RAG_API_BASE（优先级高于环境变量 / settings）",
    )
    parser.add_argument(
        "--kg-api-base",
        default=None,
        metavar="URL",
        help="覆盖 KG_API_BASE（优先级高于环境变量 / settings）",
    )
    parser.add_argument(
        "--kg-endpoint",
        default=None,
        metavar="PATH",
        help="覆盖 KG_ENDPOINT（优先级高于环境变量 / settings）",
    )
    parser.add_argument(
        "--auto-warmup-rag",
        action="store_true",
        help="若 RAG /ready 显示检索索引未就绪，则自动 GET /warmup 并再次 /ready",
    )
    parser.add_argument(
        "--fail-on-rag-fallback",
        action="store_true",
        help="若 RAG top1 为 fallback evidence，则退出码为 1",
    )
    args = parser.parse_args()

    settings.RAG_PROVIDER = os.getenv("RAG_PROVIDER", "remote")
    settings.KG_PROVIDER = os.getenv("KG_PROVIDER", "remote")

    if args.rag_api_base is not None:
        settings.RAG_API_BASE = args.rag_api_base.strip().rstrip("/")
    else:
        settings.RAG_API_BASE = os.getenv("RAG_API_BASE", settings.RAG_API_BASE)

    if args.kg_api_base is not None:
        settings.KG_API_BASE = args.kg_api_base.strip().rstrip("/")
    else:
        settings.KG_API_BASE = os.getenv("KG_API_BASE", settings.KG_API_BASE)

    if args.kg_endpoint is not None:
        settings.KG_ENDPOINT = args.kg_endpoint.strip()

    effective_rag_api_base = settings.RAG_API_BASE.rstrip("/")
    effective_kg_api_base = settings.KG_API_BASE.rstrip("/")
    rag_to = min(float(settings.RAG_TIMEOUT), 10.0)

    print("[rag_kg_smoke] effective_rag_api_base=" + effective_rag_api_base)
    print("[rag_kg_smoke] effective_kg_api_base=" + effective_kg_api_base)

    if settings.KG_PROVIDER.lower() == "remote" and base_points_to_mock_port(effective_kg_api_base):
        print(
            "[rag_kg_smoke][WARN] 当前 KG_API_BASE 指向 mock 端口 9003，不是真实项目三默认端口 8002。"
        )
        print(
            f"[rag_kg_smoke][HINT] 请修正 .env 或使用：--kg-api-base {REAL_KG_DEFAULT_BASE}"
        )

    # --- RAG 就绪探测（workflow 之前）---
    rag_probe = probe_rag_health_ready(effective_rag_api_base, timeout=rag_to)
    rag_ready_payload = rag_probe.get("ready_payload") or {}
    rag_fields = extract_rag_ready_fields(rag_ready_payload if isinstance(rag_ready_payload, dict) else {})
    print("[rag_kg_smoke] RAG 就绪探测")
    print(f"- rag_health_ok={rag_probe.get('health_ok')}")
    print(f"- rag_ready_ok={rag_probe.get('ready_ok')}")
    print(f"- rag_ready_payload={format_rag_ready_payload_for_print(rag_ready_payload if isinstance(rag_ready_payload, dict) else {})}")
    print(f"- rag_serving_mode={rag_fields.get('rag_serving_mode')}")
    print(f"- rag_faq_ready={rag_fields.get('rag_faq_ready')}")
    print(f"- rag_bm25_ready={rag_fields.get('rag_bm25_ready')}")
    print(f"- rag_lightweight_search_ready={rag_fields.get('rag_lightweight_search_ready')}")
    print(f"- rag_faq_doc_count={rag_fields.get('rag_faq_doc_count')}")
    print(f"- rag_bm25_doc_count={rag_fields.get('rag_bm25_doc_count')}")
    if rag_probe.get("health_error"):
        print(f"- rag_health_error={rag_probe['health_error']}")
    if rag_probe.get("ready_error"):
        print(f"- rag_ready_error={rag_probe['ready_error']}")

    if not rag_probe.get("health_ok"):
        print(
            "[rag_kg_smoke][ERROR] 项目二 RAG 服务可能未启动，或 RAG_API_BASE 配置错误（/health 不可用）。"
        )
        return 1

    # --- 可选 RAG warmup（/ready 不通过索引时）---
    if args.auto_warmup_rag and isinstance(rag_ready_payload, dict):
        print("[rag_kg_smoke] RAG warmup 检查（--auto-warmup-rag）")
        print(f"- rag_ready_before_warmup={format_rag_ready_payload_for_print(rag_ready_payload)}")
        wu = run_rag_warmup_and_reprobe(
            effective_rag_api_base,
            rag_ready_payload,
            enabled=True,
            timeout=rag_to,
            warmup_timeout=max(15.0, rag_to),
        )
        if wu.get("warmup_called"):
            print(f"- rag_warmup_http_ok={wu.get('warmup_http_ok')}")
            print(f"- rag_warmup_payload={format_rag_ready_payload_for_print(wu.get('warmup_payload') or {})}")
            after_payload = wu.get("ready_after") or {}
            after_fields = extract_rag_ready_fields(after_payload if isinstance(after_payload, dict) else {})
            print(f"- rag_ready_after_warmup={format_rag_ready_payload_for_print(after_payload if isinstance(after_payload, dict) else {})}")
            print(f"- rag_faq_ready(after)={after_fields.get('rag_faq_ready')}")
            print(f"- rag_bm25_ready(after)={after_fields.get('rag_bm25_ready')}")
            print(f"- rag_lightweight_search_ready(after)={after_fields.get('rag_lightweight_search_ready')}")
            rag_probe = wu.get("probe_after") or rag_probe
            rag_ready_payload = rag_probe.get("ready_payload") or rag_ready_payload
        else:
            print("[rag_kg_smoke] RAG warmup 跳过（索引已就绪或 /ready 不可用）")

    # --- KG 就绪探测（保持原有行为）---
    probe = probe_kg_health_ready(effective_kg_api_base, timeout=min(float(settings.KG_TIMEOUT), 10.0))
    kg_health_ok = bool(probe["health_ok"])
    kg_ready_ok = bool(probe["ready_ok"])
    print("[rag_kg_smoke] KG 就绪探测")
    print(f"- kg_health_ok={kg_health_ok}")
    print(f"- kg_ready_ok={kg_ready_ok}")
    print(f"- ready_payload={format_ready_payload_for_print(probe['ready_payload'])}")
    print(f"- neo4j_connected={probe['neo4j_connected']}")
    print(f"- graph_node_count={probe['graph_node_count']}")
    print(f"- graph_relation_count={probe['graph_relation_count']}")

    if not kg_health_ok or not kg_ready_ok:
        print(
            "[rag_kg_smoke][ERROR] 项目三 KG 服务可能未启动，或 KG_API_BASE 配置错误。"
        )
        return 1

    request_text = "课堂演示遇到 NameError，应该怎么给学生解释？"
    state = build_agent_graph().invoke({"request_text": request_text})

    parsed_slots = state.get("parsed_slots", {}) or {}
    rag_query = state.get("rag_query", "")
    print("[rag_kg_smoke] 项目一发给 RAG 的检索上下文（可与项目二 /search 对齐）")
    print(f"- original_request_text={request_text}")
    print(f"- rag_query={rag_query}")
    print(f"- parsed_error_type={state.get('error_type','')}")
    print(f"- parsed_knowledge_points={parsed_slots.get('knowledge_points', [])}")

    rag_items = state.get("rag_evidence", []) or []
    kg_items = state.get("kg_evidence", []) or []

    _print_rag_top_enhanced(rag_items, label="RAG")
    _print_kg_top(kg_items)

    trace = state.get("debug_trace", [])
    rag_trace = next((t for t in trace if t.get("node_name") == "fetch_rag_evidence"), {})
    kg_trace = next((t for t in trace if t.get("node_name") == "fetch_kg_evidence"), {})

    kg_out = kg_trace.get("output_summary", {}) or {}
    kg_validation_ok = kg_out.get("validation_ok")
    kg_mapper_used = kg_out.get("mapper_used") or kg_out.get("mapper")

    print(f"[rag_kg_smoke] final_response_non_empty={bool(str(state.get('final_response','')).strip())}")
    print(f"[rag_kg_smoke] rag_provider={rag_trace.get('rag_provider')}")
    print(f"[rag_kg_smoke] kg_provider={kg_trace.get('kg_provider')}")
    print(f"[rag_kg_smoke] rag_validation_ok={rag_trace.get('output_summary',{}).get('validation_ok')}")
    print(f"[rag_kg_smoke] kg_validation_ok={kg_validation_ok}")
    print(f"[rag_kg_smoke] kg_mapper_used={kg_mapper_used}")
    print(f"[rag_kg_smoke] rag_mapper={rag_trace.get('output_summary',{}).get('mapper')}")

    if rag_items and is_rag_fallback_item(rag_items[0]):
        print(
            "[rag_kg_smoke][WARN] RAG 当前走 fallback，不是真实 FAQ/BM25 evidence。"
        )

    exit_extra = joint_exit_code_for_rag_fallback(rag_items, args.fail_on_rag_fallback)
    return exit_extra


if __name__ == "__main__":
    raise SystemExit(main())
