"""schema_utils：JSON 后尾随文本检测与禁用天数扩展。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SU_PATH = ROOT / "finetune_lora" / "scripts" / "schema_utils.py"


def _load_su():
    import sys

    mod_name = "schema_utils_boundary_test"
    spec = importlib.util.spec_from_file_location(mod_name, SU_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_detect_extra_text_after_first_json() -> None:
    su = _load_su()
    blob = '{"diagnosis":{"observed_problem":"a","probable_cause":"b","evidence_basis":"c","confidence_level":"low"},"intervention_plan":{"intervention_goal":"g","day_1_action":"1","day_2_action":"2","day_3_action":"3","optional_followup":"f"}}'
    raw = blob + "\n\n【诊断】多余文本"
    extra, snip = su.detect_extra_text_after_first_json(raw)
    assert extra is True
    assert "【诊断】" in snip

    ok, _ = su.detect_extra_text_after_first_json(blob)
    assert ok is False


def test_forbidden_extended_day_markers_in_optional() -> None:
    su = _load_su()
    bad = '{"optional_followup":"延长至第4天巩固"}'
    assert su.output_string_has_forbidden_day_text(bad)
