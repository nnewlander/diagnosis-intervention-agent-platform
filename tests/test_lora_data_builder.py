"""LoRA SFT 数据构建脚本的基础校验。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "finetune_lora" / "scripts" / "build_sft_data.py"

VALID_CONF = {"high", "medium", "cautious_medium", "low"}

FORBIDDEN_DAY_MARKERS = ("第4天", "第5天", "第6天", "第4-5天", "day4", "day5", "day6")
BAD_ETL = ("重复值", "缺失值")
LOOP_COND_KP = {"for循环", "条件判断"}


def _forbidden_education_terms() -> tuple[str, ...]:
    import importlib.util
    import sys

    su_path = ROOT / "finetune_lora" / "scripts" / "schema_utils.py"
    mod_name = "schema_utils_forbidden_terms_test"
    spec = importlib.util.spec_from_file_location(mod_name, su_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return tuple(mod.FORBIDDEN_EDUCATION_DOMAIN_TERMS)


def test_build_sft_data_generates_train_dev(tmp_path: Path) -> None:
    out_dir = tmp_path / "lora_data"
    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--count",
        "40",
        "--dev-ratio",
        "0.25",
        "--output-dir",
        str(out_dir),
        "--seed",
        "7",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    forbidden_terms = _forbidden_education_terms()
    train_p = out_dir / "train.jsonl"
    dev_p = out_dir / "dev.jsonl"
    assert train_p.exists()
    assert dev_p.exists()
    train_lines = [ln for ln in train_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev_lines = [ln for ln in dev_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(train_lines) + len(dev_lines) == 40
    assert len(dev_lines) >= 1

    for path in (train_p, dev_p):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert "instruction" in row and "input" in row and "output" in row
            assert isinstance(row["instruction"], str) and row["instruction"].strip()
            out_obj = json.loads(row["output"])
            assert "diagnosis" in out_obj and "intervention_plan" in out_obj
            d = out_obj["diagnosis"]
            assert isinstance(d, dict)
            conf = str(d.get("confidence_level", "")).strip()
            assert conf in VALID_CONF
            assert str(d.get("observed_problem", "")).strip()
            assert str(d.get("probable_cause", "")).strip()
            ip = out_obj["intervention_plan"]
            for k in ("day_1_action", "day_2_action", "day_3_action"):
                assert k in ip
            blob = json.dumps(out_obj, ensure_ascii=False)
            assert "day1_intervention" not in blob
            assert "day4_intervention" not in blob
            blob_lower = blob.lower()
            for m in FORBIDDEN_DAY_MARKERS:
                if m.startswith("day"):
                    assert m.lower() not in blob_lower
                else:
                    assert m not in blob
            for term in forbidden_terms:
                assert term not in blob
            assert "三个月后" not in blob
            assert "第二天" not in str(out_obj["intervention_plan"].get("day_3_action", ""))

            inp_obj = json.loads(row["input"])
            slots = inp_obj.get("parsed_slots") or {}
            kps = []
            for key in ("user_mentioned_knowledge_points", "knowledge_points"):
                v = slots.get(key)
                if isinstance(v, list):
                    kps.extend(str(x) for x in v)
            cause = str(out_obj["diagnosis"].get("probable_cause", ""))
            if LOOP_COND_KP.intersection(set(kps)):
                for p in BAD_ETL:
                    assert p not in cause
            al = str(inp_obj.get("evidence_alignment_status") or "").strip()
            conf = str(out_obj["diagnosis"].get("confidence_level") or "").strip()
            if al in ("mismatched", "insufficient_data"):
                assert conf != "high"
            if al == "partially_aligned":
                assert conf != "high"


def test_build_records_confidence_levels() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_sft_data", ROOT / "finetune_lora/scripts/build_sft_data.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.build_records(count=64, seed=1)
    for r in rows:
        out = json.loads(r["output"])
        inp = json.loads(r["input"])
        al = str(inp.get("evidence_alignment_status") or "").strip()
        conf = out["diagnosis"]["confidence_level"]
        assert conf in VALID_CONF
        if al in ("mismatched", "insufficient_data"):
            assert conf != "high"
        if al == "partially_aligned":
            assert conf != "high"
