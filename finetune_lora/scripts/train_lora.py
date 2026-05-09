"""
LoRA 微调：Transformers + PEFT。
无 CUDA 且未指定 --force-train-cpu 时默认 dry-run；--require-cuda 在无 GPU 时非零退出。
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

FINETUNE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = FINETUNE_ROOT / "configs" / "lora_config.yaml"
DEFAULT_TRAIN = FINETUNE_ROOT / "data" / "train.jsonl"

# 适合约 6GB～8GB 显存的默认训练超参（可被 CLI 覆盖）
DEFAULT_NUM_TRAIN_EPOCHS = 2
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_STEPS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lora")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction}\n\n### 输入\n{input_text}\n\n### 输出（仅 JSON）\n"


def compact_json_string(output_field: Any) -> str:
    """将标签序列化为紧凑单行 JSON（训练 target，不含换行缩进）。"""
    if isinstance(output_field, str):
        try:
            parsed = json.loads(output_field)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            return output_field.strip()
    return json.dumps(output_field, ensure_ascii=False, separators=(",", ":"))


def training_target_with_eos(output_field: Any, tokenizer: AutoTokenizer) -> str:
    """compact JSON + eos_token 字符串（assistant 部分）；eos 为空则用 pad_token。"""
    body = compact_json_string(output_field)
    eos_str = tokenizer.eos_token or tokenizer.pad_token
    if eos_str and not body.endswith(eos_str):
        body = body + eos_str
    return body


def _tokenize_one(
    instruction: str,
    inp: str,
    out: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """out 已含 eos 文本；labels 仅监督 output 段（含 eos token）。"""
    prompt = _build_prompt(instruction, inp)
    p_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    r_ids = tokenizer(out, add_special_tokens=False, truncation=False)["input_ids"]
    if len(p_ids) >= max_length:
        p_ids = p_ids[: max(max_length // 2, 32)]
    remain = max_length - len(p_ids)
    r_ids = r_ids[: max(1, remain)]
    pair = p_ids + r_ids
    p_len = len(p_ids)
    labels = [-100] * p_len + pair[p_len:]
    return pair, labels


def _tokenize_batch(
    examples: dict[str, list[Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, list[Any]]:
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    for instruction, inp, out in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
        strict=False,
    ):
        ids, lab = _tokenize_one(instruction, inp, out, tokenizer, max_length)
        input_ids_list.append(ids)
        labels_list.append(lab)
    return {"input_ids": input_ids_list, "labels": labels_list}


def _pad_sequences(
    seqs: list[list[int]],
    pad_id: int,
    max_len: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    lens = max_len or max((len(s) for s in seqs), default=0)
    padded: list[list[int]] = []
    mask: list[list[int]] = []
    for s in seqs:
        pad_n = lens - len(s)
        padded.append(s + [pad_id] * pad_n if pad_n > 0 else s[:lens])
        mask.append([1] * len(s) + [0] * max(pad_n, 0) if pad_n > 0 else [1] * lens)
    return padded, mask


class LoraCausalCollator:
    """Pad input_ids / labels / attention_mask for causal LM with masked labels."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(x) for x in input_ids)
        batch_input, attn = _pad_sequences(input_ids, self.pad_id, max_len)
        batch_labels: list[list[int]] = []
        for lab in labels:
            pad_n = max_len - len(lab)
            batch_labels.append(lab + [-100] * pad_n if pad_n > 0 else lab[:max_len])
        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def dry_run(tokenizer: AutoTokenizer, train_rows: list[dict[str, Any]], max_length: int) -> None:
    logger.info("Dry-run 模式：校验 tokenizer 与样本编码（不加载完整模型权重）。")
    logger.info(
        "tokenizer eos_token=%r eos_token_id=%s pad_token=%r pad_token_id=%s",
        tokenizer.eos_token,
        tokenizer.eos_token_id,
        tokenizer.pad_token,
        tokenizer.pad_token_id,
    )
    train_rows = [{**r, "output": training_target_with_eos(r["output"], tokenizer)} for r in train_rows]
    tail = train_rows[0]["output"]
    logger.info("首条训练 target 尾部 repr（确认含 eos）: %r", tail[-min(160, len(tail)) :])
    sample = train_rows[0]
    prompt = _build_prompt(sample["instruction"], sample["input"])
    enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
    logger.info("首条 prompt token 长度: %d", len(enc["input_ids"]))
    joined = _tokenize_batch(
        {
            "instruction": [r["instruction"] for r in train_rows[:3]],
            "input": [r["input"] for r in train_rows[:3]],
            "output": [r["output"] for r in train_rows[:3]],
        },
        tokenizer,
        max_length,
    )
    i0, l0 = joined["input_ids"][0], joined["labels"][0]
    logger.info("批量编码前 3 条成功；首条 seq_len=%d labels non-mask=%d", len(i0), sum(1 for x in l0 if x != -100))
    logger.info("Dry-run 完成。")


def _oom_hint() -> None:
    logger.error(
        "CUDA 显存不足（OOM）。可尝试：\n"
        "  · 降低 --max-length（例如 384 或 256）\n"
        "  · 保持 --per-device-train-batch-size=1\n"
        "  · 增大 --gradient-accumulation-steps\n"
        "  · 使用 --max-samples 先在小集合上验证\n"
        "  · 换更小的 BASE_MODEL\n"
    )


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def log_transformers_trainer_diagnostics() -> None:
    """打印 transformers 版本及 Trainer 使用的 processor 参数名。"""
    import transformers
    from transformers import Trainer

    logger.info("transformers.__version__=%s", transformers.__version__)
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        logger.info("Trainer processor arg: processing_class")
    elif "tokenizer" in sig.parameters:
        logger.info("Trainer processor arg: tokenizer")
    else:
        logger.info("Trainer processor arg: none")


def get_trainer_processor_param_name() -> str | None:
    """返回 Trainer.__init__ 应传入的处理器参数名；旧版为 tokenizer，新版为 processing_class。"""
    from transformers import Trainer

    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return "processing_class"
    if "tokenizer" in sig.parameters:
        return "tokenizer"
    return None


def build_trainer_kwargs(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    eval_dataset: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """
    按当前 transformers 中 Trainer 签名组装 kwargs，避免 tokenizer= 与 processing_class= 不兼容。
    返回 (kwargs, label)，label 为 processing_class | tokenizer | none。
    """
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    name = get_trainer_processor_param_name()
    if name == "processing_class":
        trainer_kwargs["processing_class"] = tokenizer
        return trainer_kwargs, "processing_class"
    if name == "tokenizer":
        trainer_kwargs["tokenizer"] = tokenizer
        return trainer_kwargs, "tokenizer"
    return trainer_kwargs, "none"


def _model_from_pretrained_dtype_kwargs(*, bf16: bool) -> dict[str, Any]:
    """兼容 transformers：优先 dtype=，旧版使用 torch_dtype=。"""
    from transformers import AutoModelForCausalLM

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    has_dtype = "dtype" in sig.parameters
    kw: dict[str, Any] = {}
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


def log_training_plan(
    *,
    base_model: str,
    train_file: Path,
    output_adapter_dir: Path,
    train_sample_count: int,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    learning_rate: float,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
) -> None:
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    logger.info("======== 训练配置 ========")
    logger.info("BASE_MODEL=%s", base_model)
    logger.info("train_file=%s", train_file)
    logger.info("output_adapter_dir=%s", output_adapter_dir)
    logger.info("device=%s", device)
    logger.info("cuda_available=%s", cuda_ok)
    if cuda_ok:
        logger.info("cuda_device=%s", torch.cuda.get_device_name(0))
    logger.info("train_sample_count=%s", train_sample_count)
    logger.info("num_train_epochs=%s", num_train_epochs)
    logger.info("per_device_train_batch_size=%s", per_device_train_batch_size)
    logger.info("gradient_accumulation_steps=%s", gradient_accumulation_steps)
    logger.info("max_length=%s", max_length)
    logger.info("learning_rate=%s", learning_rate)
    logger.info("fp16=%s bf16=%s", fp16, bf16)
    logger.info("gradient_checkpointing=%s", gradient_checkpointing)
    logger.info("==========================")


def run_train(
    base_model: str,
    train_rows: list[dict[str, Any]],
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    max_length: int,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    train_file_display: Path,
) -> None:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    if not train_rows:
        raise SystemExit("训练数据为空")

    log_transformers_trainer_diagnostics()

    log_training_plan(
        base_model=base_model,
        train_file=train_file_display,
        output_adapter_dir=output_dir,
        train_sample_count=len(train_rows),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
    )

    logger.info("加载 tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "tokenizer eos_token=%r eos_token_id=%s pad_token=%r pad_token_id=%s",
        tokenizer.eos_token,
        tokenizer.eos_token_id,
        tokenizer.pad_token,
        tokenizer.pad_token_id,
    )
    train_rows = [{**r, "output": training_target_with_eos(r["output"], tokenizer)} for r in train_rows]
    if train_rows:
        t0 = train_rows[0]["output"]
        logger.info("首条训练 target 尾部 repr（确认含 eos）: %r", t0[-min(160, len(t0)) :])

    ds = Dataset.from_dict(
        {
            "instruction": [r["instruction"] for r in train_rows],
            "input": [r["input"] for r in train_rows],
            "output": [r["output"] for r in train_rows],
        }
    )

    def _tok(examples: dict[str, list[Any]]) -> dict[str, Any]:
        return _tokenize_batch(examples, tokenizer, max_length)

    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="python")
    bad = [i for i, row in enumerate(tokenized) if len(row["input_ids"]) != len(row["labels"])]
    if bad:
        raise RuntimeError(f"编码长度不一致样本索引: {bad[:10]}")

    logger.info("加载基座模型…")
    load_kw: dict[str, Any] = {"trust_remote_code": True}
    load_kw.update(_model_from_pretrained_dtype_kwargs(bf16=bf16))
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kw)

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )
    model = get_peft_model(model, peft_config)
    if gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_yaml = cfg.get("training", {})
    warmup_steps = int(t_yaml.get("warmup_steps", DEFAULT_WARMUP_STEPS))
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(num_train_epochs),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        learning_rate=float(learning_rate),
        warmup_steps=warmup_steps,
        logging_steps=int(t_yaml.get("logging_steps", 5)),
        save_strategy=str(t_yaml.get("save_strategy", "epoch")),
        fp16=bool(fp16) and torch.cuda.is_available(),
        bf16=bool(bf16) and torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = LoraCausalCollator(tokenizer)
    tok_ref = tokenizer
    trainer_kwargs, _proc_label = build_trainer_kwargs(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tok_ref,
    )
    trainer = Trainer(**trainer_kwargs)

    logger.info("开始 LoRA 训练，样本数=%s …", len(tokenized))
    try:
        trainer.train()
    except Exception as exc:
        if torch.cuda.is_available() and _is_oom(exc):
            _oom_hint()
        raise
    logger.info("保存 adapter 与 tokenizer 到 %s", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("训练完成。")


def _resolve_precision(args: argparse.Namespace) -> tuple[bool, bool]:
    """返回 (fp16, bf16)。显式 --bf16 / --fp16 优先，否则 CUDA 下默认 fp16。"""
    if args.bf16:
        return False, True
    if getattr(args, "fp16_flag", False):
        return True, False
    if torch.cuda.is_available():
        return True, False
    return False, False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA 微调（PEFT）")
    p.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    p.add_argument(
        "--output-adapter-dir",
        type=Path,
        default=None,
        help="默认 finetune_lora/outputs/lora_adapter",
    )
    p.add_argument("--dry-run", action="store_true", help="仅校验数据与 tokenizer，不训练")
    p.add_argument("--force-train-cpu", action="store_true", help="强制 CPU 训练（极慢）")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="必须在 GPU 上训练；若未检测到 CUDA 则报错退出",
    )
    p.add_argument("--max-samples", type=int, default=None, help="最多使用前 N 条样本（调试用）")
    p.add_argument(
        "--num-train-epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument(
        "--fp16",
        dest="fp16_flag",
        action="store_true",
        help="启用 fp16（未指定且 CUDA 可用时默认开启）",
    )
    p.add_argument("--bf16", action="store_true", help="启用 bf16（与 fp16 互斥，优先于 fp16）")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="梯度检查点（默认开启，省显存；可用 --no-gradient-checkpointing 关闭）",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = _load_yaml(CONFIG_PATH)
    base_model = os.environ.get(
        str(cfg.get("base_model_env", "BASE_MODEL")),
        cfg.get("default_base_model", "Qwen/Qwen2.5-0.5B-Instruct"),
    )

    train_path = args.train_file.resolve()
    if not train_path.exists():
        raise SystemExit(
            f"未找到训练数据 {train_path}。请先运行：python finetune_lora/scripts/build_sft_data.py"
        )

    train_rows = _read_jsonl(train_path)
    if args.max_samples is not None and args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]

    max_length = int(args.max_length)
    logger.info("读取训练样本 %d 条：%s", len(train_rows), train_path)

    # --dry-run：任意环境下只做校验（优先于 --require-cuda）
    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        dry_run(tokenizer, train_rows, max_length)
        return

    if args.require_cuda and not torch.cuda.is_available():
        logger.error(
            "当前未检测到 CUDA，无法执行 GPU LoRA 训练。请检查 NVIDIA 驱动和 CUDA 版 PyTorch。"
        )
        sys.exit(2)

    # 无 CUDA 且未强制 CPU：保持原行为，仅 dry-run（下载 tokenizer 用于校验）
    if not torch.cuda.is_available() and not args.force_train_cpu:
        logger.info("未检测到 CUDA 且未使用 --force-train-cpu，执行 dry-run。")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        dry_run(tokenizer, train_rows, max_length)
        return

    out_adapter = args.output_adapter_dir
    if out_adapter is None:
        out_adapter = FINETUNE_ROOT / str(cfg.get("training", {}).get("output_dir", "outputs/lora_adapter"))

    fp16, bf16 = _resolve_precision(args)
    if not torch.cuda.is_available():
        fp16, bf16 = False, False

    try:
        run_train(
            base_model,
            train_rows,
            out_adapter,
            cfg,
            max_length=max_length,
            num_train_epochs=float(args.num_train_epochs),
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            learning_rate=float(args.learning_rate),
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=bool(args.gradient_checkpointing),
            train_file_display=train_path,
        )
    except Exception as exc:
        if torch.cuda.is_available() and _is_oom(exc):
            _oom_hint()
        raise


if __name__ == "__main__":
    main()
