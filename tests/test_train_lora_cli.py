"""train_lora.py 命令行与 GPU 要求逻辑（不跑真实训练）。"""
from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LORA = ROOT / "finetune_lora" / "scripts" / "train_lora.py"


def _load_train_lora():
    spec = importlib.util.spec_from_file_location("train_lora_dyn", TRAIN_LORA)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load train_lora")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_line_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "instruction": "test",
        "input": json.dumps({"request_text": "x", "evidence_alignment_status": "aligned"}),
        "output": json.dumps(
            {
                "diagnosis": {
                    "observed_problem": "a",
                    "probable_cause": "b",
                    "evidence_basis": "c",
                    "confidence_level": "low",
                },
                "intervention_plan": {
                    "intervention_goal": "g",
                    "day_1_action": "1",
                    "day_2_action": "2",
                    "day_3_action": "3",
                    "optional_followup": "f",
                },
            }
        ),
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_help_includes_require_cuda() -> None:
    r = subprocess.run(
        [sys.executable, str(TRAIN_LORA), "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "--require-cuda" in r.stdout


def test_require_cuda_exits_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tf = tmp_path / "t.jsonl"
    _one_line_jsonl(tf)
    tl = _load_train_lora()
    with pytest.raises(SystemExit) as excinfo:
        tl.main(["--require-cuda", "--train-file", str(tf)])
    assert excinfo.value.code == 2


def test_require_cuda_logs_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tf = tmp_path / "t.jsonl"
    _one_line_jsonl(tf)
    tl = _load_train_lora()
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit):
        tl.main(["--require-cuda", "--train-file", str(tf)])
    assert "CUDA" in caplog.text or "cuda" in caplog.text.lower()


def test_dry_run_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tf = tmp_path / "t.jsonl"
    _one_line_jsonl(tf)
    tl = _load_train_lora()
    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token_id = 2
    with patch.object(tl, "AutoTokenizer") as AT:
        AT.from_pretrained.return_value = fake_tok
        with patch.object(tl, "dry_run") as m:
            tl.main(["--dry-run", "--train-file", str(tf), "--max-length", "256"])
            m.assert_called_once()


def test_help_lists_training_flags() -> None:
    r = subprocess.run(
        [sys.executable, str(TRAIN_LORA), "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    for flag in ("--max-samples", "--max-length", "--gradient-checkpointing", "--learning-rate"):
        assert flag in r.stdout
