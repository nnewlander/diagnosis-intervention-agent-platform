"""
对 dev 集做轻量推理评估；支持进度打印、跳过生成、adapter/base-model 参数。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

FINETUNE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV = FINETUNE_ROOT / "data" / "dev.jsonl"
DEFAULT_ADAPTER = FINETUNE_ROOT / "outputs" / "lora_adapter"
DEFAULT_REPORT = FINETUNE_ROOT / "outputs" / "eval_lora_report.json"
DEFAULT_CASE_DETAILS = FINETUNE_ROOT / "outputs" / "eval_lora_case_details.jsonl"
DEFAULT_FAILED_CASES = FINETUNE_ROOT / "outputs" / "eval_lora_failed_cases.jsonl"
DEFAULT_SUMMARY_MD = FINETUNE_ROOT / "outputs" / "eval_lora_summary.md"

VALID_CONF = {"high", "medium", "cautious_medium", "low"}


def _load_infer_module() -> Any:
    p = FINETUNE_ROOT / "scripts" / "infer_lora.py"
    spec = importlib.util.spec_from_file_location("infer_lora", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 infer_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_schema_utils() -> Any:
    import sys

    p = FINETUNE_ROOT / "scripts" / "schema_utils.py"
    mod_name = "finetune_lora_schema_utils_eval"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 schema_utils.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_build_instruction() -> str:
    p = FINETUNE_ROOT / "scripts" / "build_sft_data.py"
    spec = importlib.util.spec_from_file_location("build_sft_data", p)
    if spec is None or spec.loader is None:
        return "你是教学诊断助手，输出 JSON。"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "INSTRUCTION", "你是教学诊断助手，输出 JSON。")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def adapter_ready(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").exists() or (adapter_dir / "adapter_model.safetensors").exists()


def _has_extra_day_fields_before_normalize(gen_text: str) -> bool:
    """生成原文中是否出现 day4+ 类键名（粗检）。"""
    lower = gen_text.lower()
    for token in ("day4", "day5", "day6", "day_4", "day_5", "day_6"):
        if token in lower.replace(" ", ""):
            if "day4_intervention" in lower or "day5_intervention" in lower or "day6_intervention" in lower:
                return True
            if "day_4_action" in lower or "day_5_action" in lower or "day_6_action" in lower:
                return True
    return False


def _flush_print(msg: str) -> None:
    print(msg, flush=True)


def _apply_base_only_default_paths(args: argparse.Namespace) -> None:
    """--disable-adapter 且仍使用默认输出路径时，改用 *_base_only 文件名以便对比。"""
    if not args.disable_adapter:
        return
    if args.report.resolve() == DEFAULT_REPORT.resolve():
        args.report = FINETUNE_ROOT / "outputs" / "eval_lora_report_base_only.json"
    if args.case_details.resolve() == DEFAULT_CASE_DETAILS.resolve():
        args.case_details = FINETUNE_ROOT / "outputs" / "eval_lora_case_details_base_only.jsonl"
    if args.failed_cases.resolve() == DEFAULT_FAILED_CASES.resolve():
        args.failed_cases = FINETUNE_ROOT / "outputs" / "eval_lora_failed_cases_base_only.jsonl"
    if args.summary_md.resolve() == DEFAULT_SUMMARY_MD.resolve():
        args.summary_md = FINETUNE_ROOT / "outputs" / "eval_lora_summary_base_only.md"


def write_eval_summary_md(
    path: Path,
    *,
    report: dict[str, Any],
    failed_samples: list[dict[str, Any]],
) -> None:
    """写入 Markdown 汇总（评估配置、指标、失败简表、结论）。"""
    lines: list[str] = []
    lines.append("# LoRA 评估摘要\n")
    lines.append("## 本次评估配置\n")
    lines.append(f"- **base_model**: `{report.get('base_model', '')}`")
    lines.append(f"- **adapter_dir**: `{report.get('adapter_dir', '')}`")
    lines.append(f"- **adapter_used**: `{report.get('adapter_used', '')}`")
    lines.append(f"- **eval_sample_count**: {report.get('eval_sample_count', 0)}")
    lines.append(f"- **max_new_tokens**: {report.get('max_new_tokens', '')}")
    lines.append(f"- **repair_json**: `{report.get('repair_json', False)}`\n")
    lines.append("## 指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    for key in (
        "json_parse_success_rate",
        "required_fields_complete_rate",
        "forbidden_domain_term_rate",
        "after_json_extra_text_rate",
        "avg_generation_seconds",
    ):
        if key in report:
            lines.append(f"| `{key}` | {report[key]} |")
    if "repaired_json_parse_success_rate" in report:
        lines.append(f"| `repaired_json_parse_success_rate` | {report['repaired_json_parse_success_rate']} |")
    lines.append("")
    lines.append("## top_error_reasons\n")
    ter = report.get("top_error_reasons") or {}
    if isinstance(ter, dict):
        for k, v in ter.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append(str(ter))
    lines.append("")
    lines.append("## 失败样本简表\n")
    lines.append("| case_id | error_reasons | forbidden_terms | raw tail 片段 |")
    lines.append("|---------|---------------|-----------------|---------------|")
    for row in failed_samples[:50]:
        cid = str(row.get("case_id", ""))
        reasons = ",".join(row.get("error_reasons") or [])
        terms = ",".join(row.get("forbidden_domain_terms") or [])
        tail = str(row.get("raw_generation_tail_300_chars", ""))[:80].replace("|", "\\|")
        lines.append(f"| {cid} | {reasons} | {terms} | {tail} |")
    lines.append("")
    lines.append("## 当前结论\n")
    lines.append("- **是否推荐接入主流程**：否，仅作为 LoRA 复现模块。")
    lines.append("- **主要短板**：JSON 停止边界不稳定、禁用领域词、JSON 后额外文本。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 轻量评估")
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--adapter-path", type=Path, default=None, help="同 --adapter-dir")
    parser.add_argument("--base-model", type=str, default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-eval-samples", type=int, default=5, help="最多评估 dev 前 N 条")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="默认 384；快速试跑可用 256，正式小样本评估建议 384～512",
    )
    parser.add_argument(
        "--case-details",
        type=Path,
        default=DEFAULT_CASE_DETAILS,
        help="逐条评估详情 jsonl，默认 finetune_lora/outputs/eval_lora_case_details.jsonl",
    )
    parser.add_argument(
        "--failed-cases",
        type=Path,
        default=DEFAULT_FAILED_CASES,
        help="仅失败样本 jsonl，默认 finetune_lora/outputs/eval_lora_failed_cases.jsonl",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=DEFAULT_SUMMARY_MD,
        help="Markdown 汇总，默认 finetune_lora/outputs/eval_lora_summary.md",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="每几条打印进度")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="只做 dev 集标注 output 的 schema 校验，不加载模型",
    )
    parser.add_argument(
        "--disable-adapter",
        action="store_true",
        help="仅加载基座模型（不加载 LoRA），用于与 adapter 结果对比",
    )
    parser.add_argument(
        "--repair-json",
        action="store_true",
        help="对 json_parse_failed 样本尝试从原文抽取首个完整 JSON（不修改原始 json_parse_success_rate）",
    )
    args = parser.parse_args()
    _apply_base_only_default_paths(args)

    dev_path = args.dev_file.resolve()
    adapter_dir = (args.adapter_path or args.adapter_dir).resolve()
    report_path = args.report.resolve()
    summary_path = args.summary_md.resolve()
    base_model = args.base_model

    su = _load_schema_utils()
    rows_all = _read_jsonl(dev_path)
    n_take = min(len(rows_all), max(0, args.max_eval_samples))
    rows = rows_all[:n_take]

    adapter_ok = adapter_ready(adapter_dir)

    report: dict[str, Any] = {
        "adapter_available": adapter_ok,
        "adapter_used": not args.disable_adapter,
        "base_model": base_model,
        "dev_file": str(dev_path),
        "adapter_dir": str(adapter_dir),
        "eval_sample_count": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "repair_json": bool(args.repair_json),
        "generation_used": False,
        "json_parse_success_rate": 0.0,
        "required_fields_complete_rate": 0.0,
        "confidence_level_valid_rate": 0.0,
        "user_points_covered_rate": 0.0,
        "evidence_terms_mentioned_rate": 0.0,
        "extra_day_fields_rate": 0.0,
        "schema_normalized_count": 0,
        "avg_generation_seconds": 0.0,
        "forbidden_day_text_rate": 0.0,
        "domain_cause_error_rate": 0.0,
        "overconfident_rate": 0.0,
        "forbidden_domain_term_rate": 0.0,
        "extra_top_level_field_rate": 0.0,
        "after_json_extra_text_rate": 0.0,
        "repaired_json_parse_success_rate": 0.0,
    }

    if not rows_all:
        report["note"] = "dev 集为空，请先运行 build_sft_data.py"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_eval_summary_md(summary_path, report=report, failed_samples=[])
        _flush_print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.skip_generation:
        bad = 0
        for i, row in enumerate(rows, start=1):
            try:
                su.validate_sft_jsonl_line(row, str(row.get("case_id", f"L{i}")))
            except ValueError as e:
                bad += 1
                _flush_print(f"[eval] schema fail {i}/{len(rows)}: {e}")
        report["generation_used"] = False
        report["skip_generation"] = True
        report["gold_schema_fail_count"] = bad
        report["gold_schema_ok_rate"] = round((len(rows) - bad) / len(rows), 6) if rows else 0.0
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_eval_summary_md(summary_path, report=report, failed_samples=[])
        _flush_print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not args.disable_adapter and not adapter_ok:
        report["note"] = "adapter_available=false，跳过真实生成"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_eval_summary_md(summary_path, report=report, failed_samples=[])
        _flush_print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    infer = _load_infer_module()
    instruction = _load_build_instruction()
    input_rules = su.infer_output_constraints_text()
    if args.disable_adapter:
        tokenizer, model = infer.load_base_model_only(base_model)
    else:
        tokenizer, model = infer.load_model(base_model, adapter_dir)

    n = len(rows)
    ok_parse = 0
    ok_fields = 0
    ok_conf = 0
    ok_points = 0
    ok_ev = 0
    extra_day_hits = 0
    norm_count = 0
    gen_secs: list[float] = []
    forbidden_day_hits = 0
    domain_cause_errors = 0
    overconfident_hits = 0
    forbidden_domain_hits = 0
    extra_top_level_hits = 0
    after_json_extra_hits = 0
    repaired_parse_ok = 0
    error_counter: Counter[str] = Counter()
    detail_lines: list[dict[str, Any]] = []
    failed_lines: list[dict[str, Any]] = []

    details_path = args.case_details.resolve()
    failed_path = args.failed_cases.resolve()

    def _user_points(input_obj: dict[str, Any]) -> list[str]:
        slots = input_obj.get("parsed_slots") or {}
        pts: list[str] = []
        for key in ("user_mentioned_knowledge_points", "knowledge_points"):
            v = slots.get(key)
            if isinstance(v, list):
                pts.extend(str(x) for x in v if x)
        return list(dict.fromkeys(pts))

    def _evidence_terms(input_obj: dict[str, Any]) -> list[str]:
        import re

        terms: list[str] = []
        rag = input_obj.get("rag_evidence") or []
        if rag and isinstance(rag[0], dict):
            snip = str(rag[0].get("snippet") or "")[:40]
            terms.extend([t for t in re.split(r"\s+|，|。|；", snip) if len(t) >= 2][:3])
        kg = input_obj.get("kg_evidence") or []
        if kg and isinstance(kg[0], dict):
            terms.append(str(kg[0].get("entity") or "")[:20])
        return [t for t in terms if t]

    report["generation_used"] = True

    for idx, row in enumerate(rows, start=1):
        inp_text = row.get("input", "")
        try:
            input_obj = json.loads(inp_text)
        except json.JSONDecodeError:
            input_obj = {}

        input_blob = json.dumps(input_obj, ensure_ascii=False)
        prompt = f"{instruction}\n\n{input_rules}\n\n### 输入\n{input_blob}\n\n### 输出（仅 JSON）\n"

        t0 = time.perf_counter()
        gen_text = infer.generate_raw(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        gen_secs.append(time.perf_counter() - t0)

        if _has_extra_day_fields_before_normalize(gen_text):
            extra_day_hits += 1

        ext = su.extract_first_json_object(gen_text)
        obj = ext.obj if ext.success else None
        if ext.schema_normalized:
            norm_count += 1

        json_ok = obj is not None
        repaired_ok = False
        repaired_by_extract = False
        if not json_ok and args.repair_json:
            ext_rep = su.extract_first_json_object_repair(gen_text)
            if ext_rep.success and ext_rep.obj is not None:
                repaired_ok = True
                repaired_by_extract = True
                repaired_parse_ok += 1

        schema_ok, schema_reasons = su.eval_generation_schema_issues(obj)
        fields_ok = schema_ok

        conf = ""
        confidence_valid = False
        if json_ok and isinstance(obj, dict):
            dd = obj.get("diagnosis")
            if isinstance(dd, dict):
                conf = str(dd.get("confidence_level", "") or "").strip()
                confidence_valid = conf in VALID_CONF

        if json_ok:
            ok_parse += 1
        if fields_ok:
            ok_fields += 1
        if confidence_valid:
            ok_conf += 1

        gen_flat = json.dumps(obj, ensure_ascii=False) if obj else gen_text
        combined_for_semantic = f"{gen_text}\n{gen_flat}"
        flags = su.eval_semantic_flags(input_obj, obj, combined_for_semantic)
        if flags["forbidden_day_text"]:
            forbidden_day_hits += 1
        if flags["domain_cause_error"]:
            domain_cause_errors += 1
        if flags["overconfident"]:
            overconfident_hits += 1
        if flags.get("forbidden_domain"):
            forbidden_domain_hits += 1
        if schema_reasons and "extra_top_level_field" in schema_reasons:
            extra_top_level_hits += 1

        extra_after, extra_snip = su.detect_extra_text_after_first_json(gen_text)
        if extra_after:
            after_json_extra_hits += 1
            error_counter["after_json_extra_text"] += 1

        pts = _user_points(input_obj)
        user_cov = bool(pts and any(p and p in gen_flat for p in pts))
        if user_cov:
            ok_points += 1
        ev_terms = _evidence_terms(input_obj)
        ev_cov = bool(ev_terms and any(t and t in gen_flat for t in ev_terms))
        if ev_cov:
            ok_ev += 1

        error_reasons: list[str] = []
        if not json_ok:
            error_reasons.append("json_parse_failed")
        error_reasons.extend(schema_reasons)
        if json_ok and not confidence_valid:
            error_reasons.append("confidence_invalid")
        if flags["forbidden_day_text"]:
            error_reasons.append("forbidden_day_text")
        if flags["domain_cause_error"]:
            error_reasons.append("domain_cause_error")
        if flags["overconfident"]:
            error_reasons.append("overconfident")
        if flags.get("forbidden_domain"):
            error_reasons.append("forbidden_domain_term")
        if obj and isinstance(obj.get("intervention_plan"), dict):
            d3t = str((obj["intervention_plan"] or {}).get("day_3_action") or "")
            if "第二天" in d3t:
                error_reasons.append("day3_mentions_second_day")
        error_reasons = list(dict.fromkeys(error_reasons))
        for er in error_reasons:
            error_counter[er] += 1

        case_id = str(row.get("case_id") or f"ROW-{idx}")
        warn_list: list[str] = []
        if extra_after:
            warn_list.append("after_json_extra_text")

        tail300 = gen_text[-300:] if len(gen_text) > 300 else gen_text
        forbidden_terms = list(flags.get("forbidden_domain_terms") or [])

        detail: dict[str, Any] = {
            "case_id": case_id,
            "request_text": str(input_obj.get("request_text") or ""),
            "expected_output": row.get("output", ""),
            "raw_generation": gen_text,
            "raw_generation_length": len(gen_text),
            "raw_generation_tail_300_chars": tail300,
            "parsed_output": obj,
            "json_parse_error": ext.parse_error if not json_ok else None,
            "json_ok": json_ok,
            "fields_ok": fields_ok,
            "confidence_valid": confidence_valid,
            "user_points_covered": user_cov,
            "evidence_terms_mentioned": ev_cov,
            "forbidden_day_text_found": bool(flags["forbidden_day_text"]),
            "domain_cause_error": bool(flags["domain_cause_error"]),
            "overconfident": bool(flags["overconfident"]),
            "forbidden_domain_terms": forbidden_terms,
            "after_json_extra_text": bool(extra_after),
            "after_json_extra_text_snippet": extra_snip,
            "warnings": warn_list,
            "error_reasons": error_reasons,
            "repaired_json_ok": repaired_ok,
            "repaired_by_extract_first_json": repaired_by_extract,
        }
        detail_lines.append(detail)
        if error_reasons:
            failed_lines.append(detail)
            tail_show = tail300.replace("\n", "\\n")
            if len(tail_show) > 200:
                tail_show = tail_show[-200:]
            _flush_print(
                f"[failed] {case_id} reasons={','.join(error_reasons)} "
                f"forbidden_terms={forbidden_terms} tail={tail_show!r}"
            )

        if args.progress_every > 0 and idx % args.progress_every == 0:
            _flush_print(f"[eval] {idx}/{n} done, json_ok={json_ok}, fields_ok={fields_ok}")

    def rate(x: int) -> float:
        return round(x / n, 6) if n else 0.0

    report.update(
        {
            "case_details_file": str(details_path),
            "failed_cases_file": str(failed_path),
            "summary_md_file": str(summary_path),
            "top_error_reasons": dict(error_counter.most_common(50)),
            "json_parse_success_rate": rate(ok_parse),
            "required_fields_complete_rate": rate(ok_fields),
            "confidence_level_valid_rate": rate(ok_conf),
            "user_points_covered_rate": rate(ok_points),
            "evidence_terms_mentioned_rate": rate(ok_ev),
            "extra_day_fields_rate": rate(extra_day_hits),
            "schema_normalized_count": norm_count,
            "avg_generation_seconds": round(sum(gen_secs) / len(gen_secs), 6) if gen_secs else 0.0,
            "forbidden_day_text_rate": rate(forbidden_day_hits),
            "domain_cause_error_rate": rate(domain_cause_errors),
            "overconfident_rate": rate(overconfident_hits),
            "forbidden_domain_term_rate": rate(forbidden_domain_hits),
            "extra_top_level_field_rate": rate(extra_top_level_hits),
            "after_json_extra_text_rate": rate(after_json_extra_hits),
            "repaired_json_parse_success_rate": rate(repaired_parse_ok),
        }
    )

    details_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as df:
        for drow in detail_lines:
            df.write(json.dumps(drow, ensure_ascii=False) + "\n")
    with failed_path.open("w", encoding="utf-8") as ff:
        for frow in failed_lines:
            ff.write(json.dumps(frow, ensure_ascii=False) + "\n")

    _flush_print("[eval] top error reasons (aggregated counts; deduped per sample):")
    for rk, rv in error_counter.most_common(30):
        _flush_print(f"  {rk}: {rv}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_eval_summary_md(summary_path, report=report, failed_samples=failed_lines)
    _flush_print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
