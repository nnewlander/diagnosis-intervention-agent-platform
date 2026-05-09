"""eval_lora 写出 case details / failed cases（mock 推理，不加载真实模型）。"""
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_LORA = ROOT / "finetune_lora" / "scripts" / "eval_lora.py"
README = ROOT / "finetune_lora" / "README.md"


def test_eval_main_writes_case_detail_and_failed_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = tmp_path / "dev.jsonl"
    row = {
        "case_id": "E1",
        "instruction": "i",
        "input": json.dumps(
            {
                "request_text": "测试",
                "parsed_slots": {"user_mentioned_knowledge_points": ["for循环"]},
                "evidence_alignment_status": "aligned",
            },
            ensure_ascii=False,
        ),
        "output": "{}",
    }
    dev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    details = tmp_path / "det.jsonl"
    failed = tmp_path / "fail.jsonl"
    rep = tmp_path / "rep.json"
    summary = tmp_path / "sum.md"

    spec = importlib.util.spec_from_file_location("eval_lora_case_test_mod", EVAL_LORA)
    assert spec and spec.loader
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    monkeypatch.setattr(ev, "adapter_ready", lambda _: True)

    mock_infer = MagicMock()
    mock_infer.load_model.return_value = (MagicMock(), MagicMock())
    mock_infer.generate_raw.return_value = "not-valid-json"

    monkeypatch.setattr(ev, "_load_infer_module", lambda: mock_infer)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_lora",
            "--dev-file",
            str(dev),
            "--case-details",
            str(details),
            "--failed-cases",
            str(failed),
            "--report",
            str(rep),
            "--summary-md",
            str(summary),
            "--max-eval-samples",
            "1",
        ],
    )
    ev.main()

    assert details.is_file()
    assert failed.is_file()
    assert summary.is_file()
    assert "LoRA 评估摘要" in summary.read_text(encoding="utf-8")
    line = details.read_text(encoding="utf-8").strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["case_id"] == "E1"
    assert obj["json_ok"] is False
    assert "json_parse_failed" in obj["error_reasons"]
    assert obj["raw_generation_tail_300_chars"] == "not-valid-json"
    assert "not-valid-json" in obj["raw_generation"]


def test_eval_after_json_extra_text_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """合法 JSON 后仍有文本时 after_json_extra_text_rate > 0。"""
    dev = tmp_path / "dev.jsonl"
    row = {
        "case_id": "E2",
        "instruction": "i",
        "input": json.dumps(
            {
                "request_text": "测试",
                "parsed_slots": {},
                "evidence_alignment_status": "aligned",
            },
            ensure_ascii=False,
        ),
        "output": "{}",
    }
    dev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    details = tmp_path / "det2.jsonl"
    failed = tmp_path / "fail2.jsonl"
    rep = tmp_path / "rep2.json"
    summary2 = tmp_path / "sum2.md"

    valid_json = (
        '{"diagnosis":{"observed_problem":"o","probable_cause":"p","evidence_basis":"e","confidence_level":"low"},'
        '"intervention_plan":{"intervention_goal":"g","day_1_action":"1","day_2_action":"2","day_3_action":"3","optional_followup":"f"}}'
    )

    spec = importlib.util.spec_from_file_location("eval_lora_case_test_mod2", EVAL_LORA)
    assert spec and spec.loader
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    monkeypatch.setattr(ev, "adapter_ready", lambda _: True)

    mock_infer = MagicMock()
    mock_infer.load_model.return_value = (MagicMock(), MagicMock())
    mock_infer.generate_raw.return_value = valid_json + "\n【诊断】尾随"

    monkeypatch.setattr(ev, "_load_infer_module", lambda: mock_infer)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_lora",
            "--dev-file",
            str(dev),
            "--case-details",
            str(details),
            "--failed-cases",
            str(failed),
            "--report",
            str(rep),
            "--summary-md",
            str(summary2),
            "--max-eval-samples",
            "1",
        ],
    )
    ev.main()

    rep_obj = json.loads(rep.read_text(encoding="utf-8"))
    assert rep_obj["after_json_extra_text_rate"] == 1.0
    line = json.loads(details.read_text(encoding="utf-8").strip().splitlines()[0])
    assert line["after_json_extra_text"] is True
    assert line["json_ok"] is True


def test_eval_cli_has_disable_adapter_and_repair_json_flags() -> None:
    r = subprocess.run(
        [sys.executable, str(EVAL_LORA), "-h"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(ROOT),
    )
    helptext = (r.stdout or "") + (r.stderr or "")
    assert "--disable-adapter" in helptext
    assert "--repair-json" in helptext


def test_finetune_readme_discourages_main_agent_replacement() -> None:
    text = README.read_text(encoding="utf-8")
    assert "不建议直接替代主 Agent" in text


def test_eval_disable_adapter_calls_load_base_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = tmp_path / "dev3.jsonl"
    row = {
        "case_id": "B1",
        "instruction": "i",
        "input": json.dumps(
            {
                "request_text": "x",
                "parsed_slots": {},
                "evidence_alignment_status": "aligned",
            },
            ensure_ascii=False,
        ),
        "output": "{}",
    }
    dev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rep = tmp_path / "r3.json"
    det = tmp_path / "d3.jsonl"
    fail = tmp_path / "f3.jsonl"
    sm = tmp_path / "s3.md"

    spec = importlib.util.spec_from_file_location("eval_lora_base_only", EVAL_LORA)
    assert spec and spec.loader
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    mock_infer = MagicMock()
    mock_infer.load_base_model_only.return_value = (MagicMock(), MagicMock())
    mock_infer.generate_raw.return_value = "{"

    monkeypatch.setattr(ev, "_load_infer_module", lambda: mock_infer)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_lora",
            "--dev-file",
            str(dev),
            "--report",
            str(rep),
            "--case-details",
            str(det),
            "--failed-cases",
            str(fail),
            "--summary-md",
            str(sm),
            "--disable-adapter",
            "--max-eval-samples",
            "1",
        ],
    )
    ev.main()
    mock_infer.load_base_model_only.assert_called_once()
    mock_infer.load_model.assert_not_called()
    rpt = json.loads(rep.read_text(encoding="utf-8"))
    assert rpt["adapter_used"] is False
    assert rpt["generation_used"] is True
