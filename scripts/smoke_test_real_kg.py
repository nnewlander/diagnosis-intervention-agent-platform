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

from app.core.config import settings
from app.tools.kg_adapter import get_kg_adapter


def _warn_if_empty(item: dict) -> None:
    for key in ["entity", "relation", "target", "evidence"]:
        if not str(item.get(key, "")).strip():
            print(f"[real_kg_smoke][WARN] evidence missing field: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="真实项目三 KG 联调 smoke test")
    parser.add_argument("--force-remote", action="store_true", help="强制使用 KG remote 模式")
    parser.add_argument(
        "--kg-api-base",
        default=None,
        metavar="URL",
        help="覆盖 KG_API_BASE（优先级高于 .env / settings）",
    )
    parser.add_argument(
        "--kg-endpoint",
        default=None,
        metavar="PATH",
        help="覆盖 KG_ENDPOINT（优先级高于 .env / settings）",
    )
    args = parser.parse_args()

    if args.force_remote:
        settings.KG_PROVIDER = "remote"
    else:
        settings.KG_PROVIDER = os.getenv("KG_PROVIDER", settings.KG_PROVIDER)

    if args.kg_api_base is not None:
        settings.KG_API_BASE = args.kg_api_base.strip().rstrip("/")
    if args.kg_endpoint is not None:
        settings.KG_ENDPOINT = args.kg_endpoint.strip()

    effective_base = settings.KG_API_BASE.rstrip("/")
    effective_endpoint = getattr(settings, "KG_ENDPOINT", "/graph_query")

    print("[real_kg_smoke] 当前最终配置")
    print(f"- KG_PROVIDER={settings.KG_PROVIDER}")
    print(f"- KG_API_BASE={effective_base}")
    print(f"- KG_ENDPOINT={effective_endpoint}")

    if settings.KG_PROVIDER.lower() != "remote":
        print("[real_kg_smoke][WARN] 当前 KG_PROVIDER 不是 remote。使用 --force-remote 或设置 .env KG_PROVIDER=remote")
        return 1

    if args.force_remote and base_points_to_mock_port(effective_base):
        print(
            "[real_kg_smoke][WARN] 当前 KG_API_BASE 指向 mock 端口 9003，不是真实项目三默认端口 8002。"
        )
        print(
            f"[real_kg_smoke][HINT] 请修正 .env 中 KG_API_BASE={REAL_KG_DEFAULT_BASE}，"
            f"或使用：--kg-api-base {REAL_KG_DEFAULT_BASE}"
        )

    probe = probe_kg_health_ready(effective_base, timeout=min(float(settings.KG_TIMEOUT), 10.0))
    print("[real_kg_smoke] 就绪探测（调用 /graph_query 之前）")
    print(f"- health_ok={probe['health_ok']}")
    print(f"- ready_ok={probe['ready_ok']}")
    print(f"- ready_payload={format_ready_payload_for_print(probe['ready_payload'])}")
    print(f"- neo4j_connected={probe['neo4j_connected']}")
    print(f"- graph_node_count={probe['graph_node_count']}")
    print(f"- graph_relation_count={probe['graph_relation_count']}")
    if probe.get("health_error"):
        print(f"- health_error={probe['health_error']}")
    if probe.get("ready_error"):
        print(f"- ready_error={probe['ready_error']}")

    if not probe["health_ok"] or not probe["ready_ok"]:
        print(
            "[real_kg_smoke][ERROR] 项目三 KG 服务可能未启动，或 KG_API_BASE 配置错误。"
        )
        return 1

    query = "NameError 变量未定义 函数名未定义"
    entity_terms = ["NameError", "变量", "报错排查"]
    adapter = get_kg_adapter()
    evidence = adapter.search(query=query, keywords=entity_terms, top_k=settings.TOP_K_KG)
    status = getattr(adapter, "last_status", {})

    print("[real_kg_smoke] 调用结果")
    print(f"- evidence_count={len(evidence)}")
    print(f"- validation_ok={status.get('validation_ok', True)}")
    print(f"- mapper_used={status.get('mapper', '')}")
    if not evidence:
        print("[real_kg_smoke][WARN] evidence_count=0，可能是服务未ready或实体未命中。")
        return 0

    for idx, item in enumerate(evidence[:3], start=1):
        print(f"[real_kg_smoke] top{idx}")
        print(f"- entity={item.get('entity','')}")
        print(f"- entity_type={item.get('entity_type','')}")
        print(f"- relation={item.get('relation','')}")
        print(f"- target={item.get('target','')}")
        print(f"- evidence={str(item.get('evidence',''))[:120]}")
        print(f"- score={item.get('score', 0.0)}")
        print(f"- metadata.source={(item.get('metadata') or {}).get('source')}")
        _warn_if_empty(item)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
