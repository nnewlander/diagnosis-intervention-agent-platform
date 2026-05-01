"""Shared KG health/ready probing for smoke scripts (avoid cross-import of script modules)."""

from __future__ import annotations

import json
from typing import Any

import requests

MOCK_KG_PORT = 9003
REAL_KG_DEFAULT_BASE = "http://127.0.0.1:8002"


def base_points_to_mock_port(base: str) -> bool:
    b = (base or "").strip().rstrip("/")
    return f":{MOCK_KG_PORT}" in b or b.endswith(str(MOCK_KG_PORT))


def extract_ready_metrics(ready_payload: dict[str, Any]) -> dict[str, Any]:
    neo = None
    nodes = None
    rels = None
    if isinstance(ready_payload, dict):
        neo = ready_payload.get("neo4j_connected")
        nodes = ready_payload.get("graph_node_count") or ready_payload.get("node_count")
        rels = ready_payload.get("graph_relation_count") or ready_payload.get("relation_count")
        for nested_key in ("detail", "data", "payload", "status"):
            inner = ready_payload.get(nested_key)
            if isinstance(inner, dict):
                neo = neo if neo is not None else inner.get("neo4j_connected")
                nodes = nodes if nodes is not None else inner.get("graph_node_count") or inner.get(
                    "node_count"
                )
                rels = rels if rels is not None else inner.get("graph_relation_count") or inner.get(
                    "relation_count"
                )
    return {
        "neo4j_connected": neo,
        "graph_node_count": nodes,
        "graph_relation_count": rels,
    }


def probe_kg_health_ready(base: str, timeout: float = 5.0) -> dict[str, Any]:
    root = base.rstrip("/")
    out: dict[str, Any] = {
        "health_ok": False,
        "ready_ok": False,
        "ready_payload": {},
        "neo4j_connected": None,
        "graph_node_count": None,
        "graph_relation_count": None,
        "health_error": "",
        "ready_error": "",
    }
    try:
        hr = requests.get(f"{root}/health", timeout=timeout)
        out["health_ok"] = hr.ok
        if not hr.ok:
            out["health_error"] = f"HTTP {hr.status_code}"
    except Exception as exc:
        out["health_error"] = str(exc)
        return out

    try:
        rr = requests.get(f"{root}/ready", timeout=timeout)
        out["ready_ok"] = rr.ok
        if not rr.ok:
            out["ready_error"] = f"HTTP {rr.status_code}"
            return out
        try:
            payload = rr.json()
        except Exception:
            payload = {"_non_json_body": rr.text[:800]}
        out["ready_payload"] = payload if isinstance(payload, dict) else {"value": payload}
        metrics = extract_ready_metrics(out["ready_payload"])
        out["neo4j_connected"] = metrics["neo4j_connected"]
        out["graph_node_count"] = metrics["graph_node_count"]
        out["graph_relation_count"] = metrics["graph_relation_count"]
    except Exception as exc:
        out["ready_error"] = str(exc)

    return out


def format_ready_payload_for_print(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
