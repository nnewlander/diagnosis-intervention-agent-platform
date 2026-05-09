"""train_lora Trainer 初始化兼容性与 build_trainer_kwargs 单元测试（无 CUDA / 无真实训练）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LORA = ROOT / "finetune_lora" / "scripts" / "train_lora.py"


def _load_train_lora():
    spec = importlib.util.spec_from_file_location("train_lora_dyn2", TRAIN_LORA)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load train_lora")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_lora_source_has_no_trainer_tokenizer_equals_tokenizer() -> None:
    text = TRAIN_LORA.read_text(encoding="utf-8")
    assert "tokenizer=tokenizer" not in text


def test_build_trainer_kwargs_selects_processor_or_tokenizer() -> None:
    tl = _load_train_lora()
    m, a, d, c, tok = object(), object(), object(), object(), object()
    kw, label = tl.build_trainer_kwargs(
        model=m,
        args=a,
        train_dataset=d,
        data_collator=c,
        tokenizer=tok,
    )
    assert kw["model"] is m
    assert kw["args"] is a
    assert kw["train_dataset"] is d
    assert kw["data_collator"] is c
    assert label in ("processing_class", "tokenizer", "none")
    if label == "processing_class":
        assert kw.get("processing_class") is tok
        assert "tokenizer" not in kw
    elif label == "tokenizer":
        assert kw.get("tokenizer") is tok
        assert "processing_class" not in kw


def test_get_trainer_processor_param_name_matches_inspect() -> None:
    tl = _load_train_lora()
    from transformers import Trainer

    import inspect

    sig = inspect.signature(Trainer.__init__)
    name = tl.get_trainer_processor_param_name()
    if "processing_class" in sig.parameters:
        assert name == "processing_class"
    elif "tokenizer" in sig.parameters:
        assert name == "tokenizer"
    else:
        assert name is None
