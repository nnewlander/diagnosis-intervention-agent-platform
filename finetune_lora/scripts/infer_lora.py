"""
加载基座模型 + LoRA adapter，对给定请求与证据生成结构化 JSON。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)

FINETUNE_ROOT = Path(__file__).resolve().parents[1]
BUILD_SFT_PATH = FINETUNE_ROOT / "scripts" / "build_sft_data.py"
SCHEMA_UTILS_PATH = FINETUNE_ROOT / "scripts" / "schema_utils.py"
DEFAULT_ADAPTER = FINETUNE_ROOT / "outputs" / "lora_adapter"


def _load_schema_utils() -> Any:
    import sys

    mod_name = "finetune_lora_schema_utils_infer"
    spec = importlib.util.spec_from_file_location(mod_name, SCHEMA_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load schema_utils")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_instruction() -> str:
    spec = importlib.util.spec_from_file_location("build_sft_data", BUILD_SFT_PATH)
    if spec is None or spec.loader is None:
        return (
            "你是面向中小学教师编程课堂的教学诊断与干预助手。"
            "请根据输入生成严格 JSON，顶层仅含 diagnosis 与 intervention_plan。"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(getattr(mod, "INSTRUCTION", "")) or (
        "你是面向中小学教师编程课堂的教学诊断与干预助手。"
        "请根据输入生成严格 JSON，顶层仅含 diagnosis 与 intervention_plan。"
    )


def _build_inference_prompt(instruction: str, input_text: str, su: Any) -> str:
    rules = su.infer_output_constraints_text()
    return f"{instruction}\n\n{rules}\n\n### 输入\n{input_text}\n\n### 输出（仅 JSON，单行或多行均可）\n"


def adapter_ready(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").exists() or (adapter_dir / "adapter_model.safetensors").exists()


def _model_load_kw(*, bf16: bool) -> dict[str, Any]:
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    has_dtype = "dtype" in sig.parameters
    kw: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        dt = torch.bfloat16 if bf16 else torch.float16
        if has_dtype:
            kw["dtype"] = dt
        else:
            kw["torch_dtype"] = dt
        kw["device_map"] = "auto"
    else:
        dt = torch.float32
        if has_dtype:
            kw["dtype"] = dt
        else:
            kw["torch_dtype"] = dt
    return kw


def load_model(base_model: str, adapter_dir: Path) -> tuple[Any, Any]:
    from peft import PeftModel

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    kw = _model_load_kw(bf16=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, **kw)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return tokenizer, model


def load_base_model_only(base_model: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kw = _model_load_kw(bf16=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, **kw)
    base.eval()
    return tokenizer, base


def build_generation_kwargs(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
    }
    if do_sample:
        if temperature is not None:
            kw["temperature"] = temperature
        if top_p is not None:
            kw["top_p"] = top_p
        if top_k is not None:
            kw["top_k"] = top_k
    return kw


def _merge_generation_config(model: Any, gen_kw: dict[str, Any]) -> GenerationConfig:
    """深拷贝模型 generation_config，greedy 时清空 temperature/top_p/top_k，避免无效参数告警。"""
    try:
        base_gc = getattr(model, "generation_config", None)
        gc = copy.deepcopy(base_gc) if base_gc is not None else GenerationConfig()
    except Exception:
        gc = GenerationConfig()
    gc.max_new_tokens = gen_kw["max_new_tokens"]
    gc.do_sample = gen_kw["do_sample"]
    gc.pad_token_id = gen_kw["pad_token_id"]
    gc.eos_token_id = gen_kw["eos_token_id"]
    if gen_kw["do_sample"]:
        if "temperature" in gen_kw:
            gc.temperature = gen_kw["temperature"]
        if "top_p" in gen_kw:
            gc.top_p = gen_kw["top_p"]
        if "top_k" in gen_kw:
            gc.top_k = gen_kw["top_k"]
    else:
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gc, attr):
                setattr(gc, attr, None)
    return gc


def generate_raw(
    *,
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kw = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    logger.info("generate: generate_kwargs keys (build_generation_kwargs): %s", sorted(gen_kw.keys()))
    gc = _merge_generation_config(model, gen_kw)
    attn = inputs.get("attention_mask")
    with torch.no_grad():
        if attn is not None:
            out_ids = model.generate(inputs["input_ids"], attention_mask=attn, generation_config=gc)
        else:
            out_ids = model.generate(inputs["input_ids"], generation_config=gc)
    gen = out_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 推理")
    parser.add_argument("--request-text", type=str, default="", help="教师请求原文")
    parser.add_argument(
        "--evidence-json",
        type=str,
        default="",
        help='证据 JSON 字符串，或 @ 前缀指向文件',
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--adapter-path", type=Path, default=None, help="同 --adapter-dir")
    parser.add_argument("--base-model", type=str, default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="默认 256；若 JSON 截断可提高到 512",
    )
    parser.add_argument("--do-sample", action="store_true", help="采样生成（默认贪心）")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="采样时可选 top_k")
    args = parser.parse_args()

    adapter_dir = (args.adapter_path or args.adapter_dir).resolve()
    if not adapter_ready(adapter_dir):
        print("请先运行 train_lora.py 完成微调并生成 adapter（预期路径：finetune_lora/outputs/lora_adapter）。")
        sys.exit(2)

    ev_raw = args.evidence_json.strip()
    if ev_raw.startswith("@"):
        path = Path(ev_raw[1:]).resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
    elif ev_raw:
        evidence = json.loads(ev_raw)
    else:
        evidence = {
            "request_text": args.request_text or "课堂演示遇到 NameError，应该怎么给学生解释？",
            "parsed_slots": {"task_type": "technical_qa", "knowledge_points": ["变量定义"]},
            "student_evidence": {},
            "rag_evidence": [],
            "kg_evidence": [],
            "evidence_alignment_status": "insufficient_data",
        }

    req = args.request_text.strip() or str(evidence.get("request_text", ""))
    evidence.setdefault("request_text", req)

    su = _load_schema_utils()
    instruction = _load_instruction()
    input_text = json.dumps(evidence if isinstance(evidence, dict) else {"payload": evidence}, ensure_ascii=False)
    prompt = _build_inference_prompt(instruction, input_text, su)

    tokenizer, model = load_model(args.base_model, adapter_dir)
    do_sample = bool(args.do_sample)
    temperature = args.temperature if do_sample else None
    top_p = args.top_p if do_sample else None
    top_k = args.top_k if do_sample else None

    raw = generate_raw(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    ext = su.extract_first_json_object(raw)
    if ext.success and ext.obj is not None:
        out_obj = ext.obj
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))
        if ext.schema_normalized:
            print("[infer] schema_normalized=true", file=sys.stderr)
        if ext.extra_fields_removed:
            print(f"[infer] extra_fields_removed={ext.extra_fields_removed}", file=sys.stderr)
    else:
        print(raw)
        if ext.parse_error:
            print(f"[infer] parse_error: {ext.parse_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
