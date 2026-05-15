"""
OpenAI-compatible vLLM 服务化验证脚本（独立 smoke，不接入 Agent / RAG / KG 主流程）。

用法示例：
  python scripts/smoke_test_vllm.py
  python scripts/smoke_test_vllm.py --base-url http://127.0.0.1:8008/v1 --timeout 120
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8008/v1"
DEFAULT_BASE_MODEL = "qwen2.5-14b-instruct"
DEFAULT_LORA_MODEL = "teaching_lora"
DEFAULT_TIMEOUT = 60

SMOKE_PROMPT = (
    "李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个3天干预建议。请只输出 JSON。"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vLLM OpenAI 兼容 API smoke 验证")
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API 根路径（含 /v1），默认 {DEFAULT_BASE_URL}",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help=f"基座模型名，默认 {DEFAULT_BASE_MODEL}")
    p.add_argument("--lora-model", default=DEFAULT_LORA_MODEL, help=f"LoRA 模型名，默认 {DEFAULT_LORA_MODEL}")
    p.add_argument("--timeout", type=float, default=float(DEFAULT_TIMEOUT), help=f"HTTP 超时秒数，默认 {DEFAULT_TIMEOUT}")
    return p


def normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _extract_json_candidate(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.I)
    if m:
        return m.group(1).strip()
    return t


def extract_message_content(completion_body: dict[str, Any]) -> str:
    """从 OpenAI chat.completions JSON 取出 assistant 文本。"""
    choices = completion_body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        return ""
    return str(content)


def extract_finish_reason(completion_body: dict[str, Any]) -> str | None:
    choices = completion_body.get("choices") or []
    if not choices:
        return None
    fr = choices[0].get("finish_reason")
    return None if fr is None else str(fr)


def list_model_ids_from_models_body(models_body: dict[str, Any]) -> list[str]:
    data = models_body.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def score_content_checks(content: str, preview_len: int = 400) -> dict[str, Any]:
    """轻量检查：可解析 JSON、是否含诊断/干预相关字段或字面。"""
    preview = (content or "")[:preview_len]
    empty = not (content or "").strip()
    out: dict[str, Any] = {
        "response_non_empty": not empty,
        "json_parse_ok": False,
        "has_diagnosis": False,
        "has_intervention_plan": False,
        "response_preview": preview,
    }
    if empty:
        return out

    candidate = _extract_json_candidate(content)
    parsed: Any = None
    try:
        parsed = json.loads(candidate)
        out["json_parse_ok"] = True
    except json.JSONDecodeError:
        out["json_parse_ok"] = False
        text_lower = content.lower()
        out["has_diagnosis"] = "诊断" in content or "diagnosis" in text_lower
        out["has_intervention_plan"] = (
            ("干预" in content and ("计划" in content or "天" in content or "day" in text_lower))
            or "intervention" in text_lower
        )
        return out

    def _collect_keys(obj: Any, acc: set[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                acc.add(str(k).lower())
                _collect_keys(v, acc)
        elif isinstance(obj, list):
            for it in obj[:50]:
                _collect_keys(it, acc)

    keys_lower: set[str] = set()
    _collect_keys(parsed, keys_lower)

    blob = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, (dict, list)) else str(parsed)
    blob_l = blob.lower()

    out["has_diagnosis"] = (
        "diagnosis" in keys_lower
        or any("诊断" in str(k) for k in (parsed.keys() if isinstance(parsed, dict) else []))
        or "诊断" in blob[:3000]
        or "diagnosis" in blob_l[:3000]
    )
    out["has_intervention_plan"] = (
        "intervention_plan" in keys_lower
        or "interventionplan" in keys_lower
        or any("干预" in str(k) for k in (parsed.keys() if isinstance(parsed, dict) else []))
        or ("干预" in blob and ("计划" in blob or "天" in blob or "day_" in blob_l))
        or "intervention" in keys_lower
    )
    return out


def summarize_completion_run(
    model: str,
    completion_body: dict[str, Any],
    latency_ms: float,
    preview_len: int = 400,
) -> dict[str, Any]:
    content = extract_message_content(completion_body)
    checks = score_content_checks(content, preview_len=preview_len)
    return {
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "finish_reason": extract_finish_reason(completion_body),
        **checks,
    }


def fetch_models(
    base_url: str,
    timeout: float,
) -> tuple[bool, int, dict[str, Any], list[str]]:
    """GET /models。返回 (models_ok, http_status, body_dict, model_ids)。"""
    url = f"{normalize_base_url(base_url)}/models"
    r = requests.get(url, timeout=timeout)
    status = r.status_code
    try:
        body = r.json()
    except ValueError:
        return False, status, {"parse_error": "response_not_json", "raw_preview": (r.text or "")[:300]}, []
    if not isinstance(body, dict):
        return False, status, {"parse_error": "json_not_object"}, []
    ids = list_model_ids_from_models_body(body)
    ok = status == 200 and isinstance(body.get("data"), list)
    return ok, status, body, ids


def post_chat_completion(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    """
    POST /chat/completions。
    返回 (http_status, response_json_or_error_dict, latency_ms)。
    """
    url = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.2,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        body = r.json()
    except ValueError:
        return r.status_code, {"error": "response_not_json", "text_preview": (r.text or "")[:500]}, latency_ms
    if not isinstance(body, dict):
        return r.status_code, {"error": "json_not_object"}, latency_ms
    if r.status_code != 200:
        return r.status_code, body, latency_ms
    return r.status_code, body, latency_ms


def run_single_model_chat(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        status, body, latency_ms = post_chat_completion(base_url, model, prompt, timeout=timeout)
    except requests.RequestException as e:
        return {
            "model": model,
            "http_status": None,
            "request_ok": False,
            "error": str(e)[:500],
            "latency_ms": 0.0,
            "response_non_empty": False,
            "finish_reason": None,
            "json_parse_ok": False,
            "has_diagnosis": False,
            "has_intervention_plan": False,
            "response_preview": "",
        }
    row: dict[str, Any] = {
        "http_status": status,
        "request_ok": status == 200,
    }
    if status != 200:
        row["error"] = body.get("error") if isinstance(body, dict) else str(body)
        row["latency_ms"] = round(latency_ms, 2)
        row["response_non_empty"] = False
        row["finish_reason"] = None
        row["json_parse_ok"] = False
        row["has_diagnosis"] = False
        row["has_intervention_plan"] = False
        row["response_preview"] = ""
        return row
    summary = summarize_completion_run(model, body, latency_ms)
    row.update(summary)
    return row


def build_report(
    base_url: str,
    base_model: str,
    lora_model: str,
    timeout: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "base_url": normalize_base_url(base_url),
        "base_model": base_model,
        "lora_model": lora_model,
        "timeout_sec": timeout,
        "models_ok": False,
        "available_models": [],
        "models_http_status": None,
        "base_model_chat": {},
        "lora_model_chat": {},
        "service_reachable": False,
        "user_message": "",
        "smoke_prompt": SMOKE_PROMPT,
    }

    try:
        ok, http_st, _models_body, ids = fetch_models(base_url, timeout=timeout)
        report["service_reachable"] = True
        report["models_ok"] = ok
        report["available_models"] = ids
        report["models_http_status"] = http_st
        if not ok:
            report["user_message"] = "已连通服务，但 /models 未返回预期 200 JSON，请检查 vLLM 是否正常启动。"
    except requests.RequestException as e:
        report["user_message"] = (
            f"无法连接 vLLM 服务（{normalize_base_url(base_url)}）。"
            "请确认进程已启动、端口与 --base-url 一致，且防火墙未拦截。"
        )
        report["connection_error"] = str(e)[:500]
        return report

    report["base_model_chat"] = run_single_model_chat(base_url, base_model, SMOKE_PROMPT, timeout=timeout)
    report["lora_model_chat"] = run_single_model_chat(base_url, lora_model, SMOKE_PROMPT, timeout=timeout)
    return report


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url
    try:
        report = build_report(
            base_url=base_url,
            base_model=args.base_model,
            lora_model=args.lora_model,
            timeout=float(args.timeout),
        )
    except requests.RequestException as e:
        report = {
            "base_url": normalize_base_url(base_url),
            "service_reachable": False,
            "user_message": (
                f"无法连接 vLLM 服务（{normalize_base_url(base_url)}）。"
                "请确认进程已启动、端口与 --base-url 一致。"
            ),
            "connection_error": str(e)[:500],
            "smoke_prompt": SMOKE_PROMPT,
        }
    except Exception as e:  # noqa: BLE001 — smoke 脚本避免长 traceback
        report = {
            "base_url": normalize_base_url(base_url),
            "service_reachable": False,
            "user_message": "验证过程出现意外错误，已抑制详细堆栈；请检查参数与服务日志。",
            "error_summary": f"{type(e).__name__}: {str(e)[:500]}",
            "smoke_prompt": SMOKE_PROMPT,
        }

    if report.get("user_message"):
        print(report["user_message"], file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 不要求全通过：仅在网络完全不可达时返回非 0
    if report.get("connection_error") and not report.get("service_reachable"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
