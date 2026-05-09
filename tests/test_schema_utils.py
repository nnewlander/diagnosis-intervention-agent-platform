"""schema_utils：JSON 抽取、别名归一化与 SFT 校验。"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SU_PATH = ROOT / "finetune_lora" / "scripts" / "schema_utils.py"
BUILD_SCRIPT = ROOT / "finetune_lora" / "scripts" / "build_sft_data.py"


def _load_su():
    import sys

    mod_name = "schema_utils_test"
    spec = importlib.util.spec_from_file_location(mod_name, SU_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_json_from_codeblock() -> None:
    su = _load_su()
    raw = """以下为结果：
```json
{"diagnosis": {"observed_problem": "x", "probable_cause": "y", "evidence_basis": "z", "confidence_level": "low"}, "intervention_plan": {"intervention_goal": "g", "day_1_action": "1", "day_2_action": "2", "day_3_action": "3", "optional_followup": "f"}}
```
谢谢
"""
    ext = su.extract_first_json_object(raw)
    assert ext.success and ext.obj is not None
    assert ext.obj["diagnosis"]["observed_problem"] == "x"


def test_normalize_day_alias_to_canonical() -> None:
    su = _load_su()
    blob = json.dumps(
        {
            "diagnosis": {
                "observed_problem": "a",
                "probable_cause": "b",
                "evidence_basis": "c",
                "confidence_level": "medium",
            },
            "intervention_plan": {
                "intervention_goal": "g",
                "day1_intervention": "d1",
                "day2_intervention": "d2",
                "day3_intervention": "d3",
                "day4_intervention": "rm",
                "optional_followup": "f",
            },
        },
        ensure_ascii=False,
    )
    ext = su.extract_first_json_object(blob)
    assert ext.success
    assert ext.schema_normalized
    ip = ext.obj["intervention_plan"]
    assert "day_1_action" in ip
    assert "day4_intervention" not in ip


def test_infer_constraints_forbids_day6() -> None:
    su = _load_su()
    t = su.infer_output_constraints_text()
    assert "day4_intervention" in t or "day4" in t
    assert "day_1_action" in t


def test_eval_semantic_flags_detects_forbidden_day_domain_overconfident() -> None:
    su = _load_su()
    inp = {
        "parsed_slots": {"user_mentioned_knowledge_points": ["for循环"]},
        "evidence_alignment_status": "mismatched",
    }
    obj_ok = {
        "diagnosis": {
            "observed_problem": "o",
            "probable_cause": "range 边界理解不稳",
            "evidence_basis": "e",
            "confidence_level": "medium",
        },
        "intervention_plan": {
            "intervention_goal": "g",
            "day_1_action": "1",
            "day_2_action": "2",
            "day_3_action": "3",
            "optional_followup": "f",
        },
    }
    t = json.dumps(obj_ok, ensure_ascii=False)
    f = su.eval_semantic_flags(inp, obj_ok, t + "\n第4天")
    assert f["forbidden_day_text"]
    assert not f["domain_cause_error"]
    assert not f["overconfident"]

    cause_bad = dict(obj_ok)
    cause_bad["diagnosis"] = dict(obj_ok["diagnosis"])
    cause_bad["diagnosis"]["probable_cause"] = "重复值导致问题"
    tb = json.dumps(cause_bad, ensure_ascii=False)
    f2 = su.eval_semantic_flags(inp, cause_bad, tb)
    assert f2["domain_cause_error"]

    high_bad = dict(obj_ok)
    high_bad["diagnosis"] = dict(obj_ok["diagnosis"])
    high_bad["diagnosis"]["confidence_level"] = "high"
    f3 = su.eval_semantic_flags(inp, high_bad, json.dumps(high_bad, ensure_ascii=False))
    assert f3["overconfident"]


def test_validate_sft_jsonl_line_rejects_forbidden_day_tokens() -> None:
    su = _load_su()
    out_obj = {
        "diagnosis": {
            "observed_problem": "观察：第6天仍出现同类错误。",
            "probable_cause": "p",
            "evidence_basis": "e",
            "confidence_level": "low",
        },
        "intervention_plan": {
            "intervention_goal": "g",
            "day_1_action": "第1天：巩固",
            "day_2_action": "第2天：练习",
            "day_3_action": "第3天：复盘",
            "optional_followup": "后续跟进",
        },
    }
    row = {
        "case_id": "BAD",
        "instruction": "i",
        "input": json.dumps({"evidence_alignment_status": "aligned"}, ensure_ascii=False),
        "output": json.dumps(out_obj, ensure_ascii=False),
    }
    with pytest.raises(ValueError, match="禁止"):
        su.validate_sft_jsonl_line(row, "BAD")


def test_validate_sft_jsonl_line_rejects_high_when_alignment_blocks() -> None:
    su = _load_su()
    out_obj = {
        "diagnosis": {
            "observed_problem": "o",
            "probable_cause": "p",
            "evidence_basis": "e",
            "confidence_level": "high",
        },
        "intervention_plan": {
            "intervention_goal": "g",
            "day_1_action": "1",
            "day_2_action": "2",
            "day_3_action": "3",
            "optional_followup": "f",
        },
    }
    row = {
        "case_id": "OC",
        "instruction": "i",
        "input": json.dumps({"evidence_alignment_status": "insufficient_data"}, ensure_ascii=False),
        "output": json.dumps(out_obj, ensure_ascii=False),
    }
    with pytest.raises(ValueError, match="high"):
        su.validate_sft_jsonl_line(row, "OC")


def test_eval_lora_report_includes_semantic_rates() -> None:
    text = (ROOT / "finetune_lora/scripts/eval_lora.py").read_text(encoding="utf-8")
    assert "forbidden_day_text_rate" in text
    assert "domain_cause_error_rate" in text
    assert "overconfident_rate" in text
    assert "forbidden_domain_term_rate" in text
    assert "after_json_extra_text_rate" in text
    assert "eval_lora_case_details.jsonl" in text
    assert "eval_lora_failed_cases.jsonl" in text


def test_eval_generation_schema_extra_top_level_and_missing_goal() -> None:
    su = _load_su()
    base_diag = {
        "observed_problem": "o",
        "probable_cause": "p",
        "evidence_basis": "e",
        "confidence_level": "low",
    }
    base_plan = {
        "intervention_goal": "g",
        "day_1_action": "1",
        "day_2_action": "2",
        "day_3_action": "3",
        "optional_followup": "f",
    }
    wrong_top = {
        "diagnosis": base_diag,
        "intervention_plan": dict(base_plan),
        "interaction_goal": "不应在顶层",
    }
    ok, reasons = su.eval_generation_schema_issues(wrong_top)
    assert not ok
    assert "extra_top_level_field" in reasons

    missing_goal = {"diagnosis": base_diag, "intervention_plan": {**base_plan, "intervention_goal": ""}}
    ok2, reasons2 = su.eval_generation_schema_issues(missing_goal)
    assert not ok2
    assert "missing_intervention_goal" in reasons2


def test_find_forbidden_education_terms() -> None:
    su = _load_su()
    assert "返现" in su.find_forbidden_education_terms("完成任务返现奖励")


def test_eval_semantic_flags_includes_domain_terms() -> None:
    su = _load_su()
    inp = {"parsed_slots": {}, "evidence_alignment_status": "aligned"}
    obj = {
        "diagnosis": {
            "observed_problem": "o",
            "probable_cause": "p",
            "evidence_basis": "e",
            "confidence_level": "medium",
        },
        "intervention_plan": {
            "intervention_goal": "g",
            "day_1_action": "1",
            "day_2_action": "2",
            "day_3_action": "3",
            "optional_followup": "采用治疗方案巩固",
        },
    }
    blob = json.dumps(obj, ensure_ascii=False)
    f = su.eval_semantic_flags(inp, obj, blob)
    assert f.get("forbidden_domain")
    assert "治疗方案" in (f.get("forbidden_domain_terms") or [])


def test_eval_skip_generation_fast(tmp_path: Path) -> None:
    dev = tmp_path / "dev.jsonl"
    row = {
        "case_id": "T1",
        "instruction": "i",
        "input": "{}",
        "output": json.dumps(
            {
                "diagnosis": {
                    "observed_problem": "o",
                    "probable_cause": "p",
                    "evidence_basis": "e",
                    "confidence_level": "low",
                },
                "intervention_plan": {
                    "intervention_goal": "g",
                    "day_1_action": "1",
                    "day_2_action": "2",
                    "day_3_action": "3",
                    "optional_followup": "f",
                },
            },
            ensure_ascii=False,
        ),
    }
    dev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "finetune_lora/scripts/eval_lora.py"),
            "--skip-generation",
            "--dev-file",
            str(dev),
            "--max-eval-samples",
            "3",
            "--report",
            str(tmp_path / "rep.json"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0
    rep = (tmp_path / "rep.json").read_text(encoding="utf-8")
    assert "gold_schema_ok_rate" in rep


def test_eval_help_has_max_eval_samples() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "finetune_lora/scripts/eval_lora.py"), "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--max-eval-samples" in r.stdout


def test_build_sft_outputs_match_schema(tmp_path: Path) -> None:
    out = tmp_path / "d"
    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--count",
            "12",
            "--dev-ratio",
            "0.25",
            "--output-dir",
            str(out),
            "--seed",
            "99",
        ],
        cwd=str(ROOT),
        check=True,
    )
    su = _load_su()
    for name in ("train.jsonl", "dev.jsonl"):
        p = out / name
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            su.validate_sft_jsonl_line(row, row.get("case_id", "?"))
