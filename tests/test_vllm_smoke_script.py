"""仅测试 scripts/smoke_test_vllm.py 的参数解析与响应解析逻辑，不启动真实 vLLM。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "smoke_test_vllm.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_test_vllm", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vllm_smoke():
    return _load_smoke_module()


def test_build_parser_defaults(vllm_smoke):
    p = vllm_smoke.build_parser()
    args = p.parse_args([])
    assert vllm_smoke.normalize_base_url(args.base_url) == "http://127.0.0.1:8008/v1"
    assert args.base_model == "qwen2.5-14b-instruct"
    assert args.lora_model == "teaching_lora"
    assert args.timeout == 60.0


def test_build_parser_overrides(vllm_smoke):
    p = vllm_smoke.build_parser()
    args = p.parse_args(
        [
            "--base-url",
            "http://10.0.0.1:9000/v1/",
            "--base-model",
            "m-base",
            "--lora-model",
            "m-lora",
            "--timeout",
            "30",
        ]
    )
    assert vllm_smoke.normalize_base_url(args.base_url) == "http://10.0.0.1:9000/v1"
    assert args.base_model == "m-base"
    assert args.lora_model == "m-lora"
    assert args.timeout == 30.0


def test_list_model_ids_from_models_body(vllm_smoke):
    ids = vllm_smoke.list_model_ids_from_models_body(
        {"data": [{"id": "a"}, {"id": "b"}, {"x": 1}]},
    )
    assert ids == ["a", "b"]


def test_extract_message_content_and_finish_reason(vllm_smoke):
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"diagnosis": "x"}'},
                "finish_reason": "stop",
            }
        ]
    }
    assert vllm_smoke.extract_message_content(body) == '{"diagnosis": "x"}'
    assert vllm_smoke.extract_finish_reason(body) == "stop"


def test_score_content_checks_valid_json(vllm_smoke):
    text = '{"diagnosis": {"observed": "1"}, "intervention_plan": {"day_1": "a"}}'
    r = vllm_smoke.score_content_checks(text)
    assert r["response_non_empty"] is True
    assert r["json_parse_ok"] is True
    assert r["has_diagnosis"] is True
    assert r["has_intervention_plan"] is True


def test_score_content_checks_markdown_fence(vllm_smoke):
    text = '```json\n{"诊断": "薄弱", "干预计划": {"d": 1}}\n```'
    r = vllm_smoke.score_content_checks(text)
    assert r["json_parse_ok"] is True
    assert r["has_diagnosis"] is True
    assert r["has_intervention_plan"] is True


def test_score_content_checks_invalid_json_heuristic(vllm_smoke):
    text = "诊断：学生 for 循环薄弱；三天干预计划：第一天复盘。"
    r = vllm_smoke.score_content_checks(text)
    assert r["response_non_empty"] is True
    assert r["json_parse_ok"] is False
    assert r["has_diagnosis"] is True
    assert r["has_intervention_plan"] is True


def test_summarize_completion_run(vllm_smoke):
    body = {
        "choices": [
            {"message": {"content": '{"diagnosis": {}}'}, "finish_reason": "length"},
        ]
    }
    s = vllm_smoke.summarize_completion_run("m1", body, 123.456)
    assert s["model"] == "m1"
    assert s["finish_reason"] == "length"
    assert s["latency_ms"] == 123.46
    assert s["json_parse_ok"] is True
