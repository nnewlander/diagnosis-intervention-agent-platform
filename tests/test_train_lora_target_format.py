"""train_lora：compact JSON 标签与 eos 后缀。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LORA = ROOT / "finetune_lora" / "scripts" / "train_lora.py"


def _load_tl():
    spec = importlib.util.spec_from_file_location("train_lora_tgt", TRAIN_LORA)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compact_json_string_no_pretty_indent() -> None:
    tl = _load_tl()
    obj = {"diagnosis": {"x": "y"}, "intervention_plan": {"z": 1}}
    s = tl.compact_json_string(obj)
    parsed = json.loads(s)
    assert parsed == obj
    roundtrip = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    assert s == roundtrip
    assert "\n" not in s


def test_training_target_with_eos_appends_token() -> None:
    tl = _load_tl()
    tok = MagicMock()
    tok.eos_token = "</s>"
    tok.pad_token = None
    obj = {
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
    }
    s = tl.training_target_with_eos(obj, tok)
    assert s.endswith("</s>")
    body = s[: -len("</s>")]
    json.loads(body)
