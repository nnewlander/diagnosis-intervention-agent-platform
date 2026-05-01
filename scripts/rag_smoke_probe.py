"""RAG health/ready/warmup helpers for joint smoke scripts."""

from __future__ import annotations

import json
from typing import Any

import requests


def join_url(base: str, path: str) -> str:
    p = path if path.startswith("/") else f"/{path}"
    return f"{base.rstrip('/')}{p}"


def _merge_nested_ready(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge top-level and common nested shapes for field lookup."""
    if not isinstance(payload, dict):
        return {}
    merged: dict[str, Any] = dict(payload)
    for key in ("detail", "data", "status", "payload"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            for k, v in inner.items():
                if k not in merged or merged[k] is None:
                    merged[k] = v
    return merged


def extract_rag_ready_fields(ready_payload: dict[str, Any]) -> dict[str, Any]:
    p = _merge_nested_ready(ready_payload)
    return {
        "rag_serving_mode": p.get("serving_mode")
        or p.get("rag_serving_mode")
        or p.get("mode"),
        "rag_faq_ready": p.get("faq_ready"),
        "rag_bm25_ready": p.get("bm25_ready"),
        "rag_lightweight_search_ready": p.get("lightweight_search_ready"),
        "rag_faq_doc_count": p.get("faq_doc_count")
        or p.get("rag_faq_doc_count")
        or p.get("faq_docs"),
        "rag_bm25_doc_count": p.get("bm25_doc_count")
        or p.get("rag_bm25_doc_count")
        or p.get("bm25_docs"),
    }


def probe_rag_health_ready(base: str, timeout: float = 5.0) -> dict[str, Any]:
    root = base.rstrip("/")
    out: dict[str, Any] = {
        "health_ok": False,
        "ready_ok": False,
        "ready_payload": {},
        "health_error": "",
        "ready_error": "",
    }
    try:
        hr = requests.get(join_url(root, "/health"), timeout=timeout)
        out["health_ok"] = hr.ok
        if not hr.ok:
            out["health_error"] = f"HTTP {hr.status_code}"
    except Exception as exc:
        out["health_error"] = str(exc)
        return out

    try:
        rr = requests.get(join_url(root, "/ready"), timeout=timeout)
        out["ready_ok"] = rr.ok
        if not rr.ok:
            out["ready_error"] = f"HTTP {rr.status_code}"
            return out
        try:
            payload = rr.json()
        except Exception:
            payload = {"_non_json_body": rr.text[:800]}
        out["ready_payload"] = payload if isinstance(payload, dict) else {"value": payload}
    except Exception as exc:
        out["ready_error"] = str(exc)

    return out


def format_rag_ready_payload_for_print(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def rag_needs_warmup(ready_payload: dict[str, Any]) -> bool:
    """
    Warmup when none of faq_ready / bm25_ready / lightweight_search_ready is truthy.
    Missing keys count as not ready (cold start).
    """
    p = _merge_nested_ready(ready_payload)
    keys = ("faq_ready", "bm25_ready", "lightweight_search_ready")
    return not any(bool(p.get(k)) for k in keys)


def run_rag_warmup_and_reprobe(
    base: str,
    ready_before: dict[str, Any],
    *,
    enabled: bool,
    timeout: float,
    warmup_timeout: float | None = None,
) -> dict[str, Any]:
    """
    If enabled and ready_before indicates indexes not ready, GET /warmup then probe /health+/ready again.
    """
    result: dict[str, Any] = {
        "warmup_called": False,
        "ready_before": dict(ready_before) if isinstance(ready_before, dict) else {},
        "ready_after": {},
        "warmup_payload": {},
        "warmup_http_ok": False,
        "probe_after": {},
    }
    if not enabled:
        result["ready_after"] = dict(result["ready_before"])
        return result
    if not isinstance(ready_before, dict) or not rag_needs_warmup(ready_before):
        result["ready_after"] = dict(result["ready_before"])
        return result

    wt = warmup_timeout if warmup_timeout is not None else max(15.0, timeout)
    try:
        wr = requests.get(join_url(base, "/warmup"), timeout=wt)
        result["warmup_http_ok"] = wr.ok
        try:
            result["warmup_payload"] = wr.json() if wr.content else {}
            if not isinstance(result["warmup_payload"], dict):
                result["warmup_payload"] = {"value": result["warmup_payload"]}
        except Exception:
            result["warmup_payload"] = {"_non_json_body": wr.text[:800]}
    except Exception as exc:
        result["warmup_payload"] = {"error": str(exc)}

    result["warmup_called"] = True
    result["probe_after"] = probe_rag_health_ready(base, timeout=timeout)
    result["ready_after"] = result["probe_after"].get("ready_payload") or {}
    return result


def is_rag_fallback_item(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    sid = str(item.get("source_id") or "")
    if sid.upper().startswith("FALLBACK"):
        return True
    if item.get("source_type") == "fallback_error_guide":
        return True
    meta = item.get("metadata") or {}
    if isinstance(meta, dict) and bool(meta.get("fallback")):
        return True
    return False


def joint_exit_code_for_rag_fallback(rag_items: list[dict[str, Any]], fail_on_rag_fallback: bool) -> int:
    if not fail_on_rag_fallback or not rag_items:
        return 0
    if is_rag_fallback_item(rag_items[0]):
        return 1
    return 0
