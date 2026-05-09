"""infer_lora：build_generation_kwargs 与 generate_raw 的生成参数约束。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from transformers import GenerationConfig
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
INFER_LORA = ROOT / "finetune_lora" / "scripts" / "infer_lora.py"


def _load_infer():
    spec = importlib.util.spec_from_file_location("infer_lora_kw_test", INFER_LORA)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_generation_kwargs_greedy_has_no_temperature_top_p_top_k():
    mod = _load_infer()
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 2
    kw = mod.build_generation_kwargs(
        tokenizer=mock_tok,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
    )
    keys = set(kw.keys())
    assert "temperature" not in keys
    assert "top_p" not in keys
    assert "top_k" not in keys
    assert kw["do_sample"] is False


def test_build_generation_kwargs_sample_allows_sampling_keys():
    mod = _load_infer()
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 2
    kw = mod.build_generation_kwargs(
        tokenizer=mock_tok,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["top_k"] == 40


def test_generate_raw_greedy_uses_generation_config_without_sampling_in_kwargs():
    mod = _load_infer()
    captured: dict = {}

    mock_model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters.return_value = iter([param])
    mock_model.generation_config = GenerationConfig()

    def fake_generate(*args, **kwargs):
        captured.clear()
        captured["_args"] = args
        captured.update(kwargs)
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    mock_model.generate = fake_generate

    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 2
    mock_tok.return_value = {"input_ids": torch.tensor([[5]], dtype=torch.long), "attention_mask": torch.tensor([[1]])}

    mod.generate_raw(
        prompt="test",
        tokenizer=mock_tok,
        model=mock_model,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.9,
        top_p=0.95,
    )

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured
    gc = captured.get("generation_config")
    assert gc is not None
    assert gc.do_sample is False
    assert getattr(gc, "temperature", None) in (None, 1.0)  # None 或 HF 默认未用于 greedy


def test_generate_raw_sample_passes_sampling_via_generation_config():
    mod = _load_infer()
    captured: dict = {}

    mock_model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters.return_value = iter([param])
    mock_model.generation_config = GenerationConfig()

    def fake_generate(*args, **kwargs):
        captured.clear()
        captured["_args"] = args
        captured.update(kwargs)
        return torch.tensor([[1]], dtype=torch.long)

    mock_model.generate = fake_generate
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 2
    mock_tok.return_value = {"input_ids": torch.tensor([[5]], dtype=torch.long), "attention_mask": torch.tensor([[1]])}

    mod.generate_raw(
        prompt="test",
        tokenizer=mock_tok,
        model=mock_model,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
    )

    gc = captured["generation_config"]
    assert gc.do_sample is True
    assert gc.temperature == 0.7
    assert gc.top_p == 0.9
    assert gc.top_k == 20
