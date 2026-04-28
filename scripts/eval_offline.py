import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.graph.workflow import build_agent_graph


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_expected_task(row: dict[str, Any]) -> str:
    expected_primary_node = str(row.get("expected_primary_node", "")).lower()
    task_hint = str(row.get("task_hint", "")).lower()
    blob = f"{expected_primary_node} {task_hint}"
    if any(k in blob for k in ["diagnosis", "学情", "诊断"]):
        return "diagnosis"
    if any(k in blob for k in ["intervention", "干预"]):
        return "intervention"
    if any(k in blob for k in ["assignment", "练习", "下发", "dispatch"]):
        return "dispatch"
    if any(k in blob for k in ["technical", "技术", "报错", "qa"]):
        return "technical_qa"
    if "mixed" in blob:
        return "mixed"
    return ""


def get_request_text(row: dict[str, Any]) -> str:
    for key in ["request_text", "query", "teacher_request", "input"]:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _as_bool(v: Any) -> bool:
    return bool(v)


def _dict_non_empty(d: Any) -> bool:
    return isinstance(d, dict) and len(d) > 0


def _list_non_empty(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0


def determine_target_task(expected_task: str, actual_primary_task: str) -> str:
    return expected_task or actual_primary_task or "unknown"


def evaluate_task_aware_case(
    state: dict[str, Any], expected_task: str = "", actual_primary_task: str = ""
) -> dict[str, Any]:
    target_task = determine_target_task(expected_task, actual_primary_task)
    diagnosis_non_empty = _dict_non_empty(state.get("diagnosis", {}))
    intervention_non_empty = _dict_non_empty(state.get("intervention_plan", {}))
    packages_non_empty = _list_non_empty(state.get("recommended_packages", []))
    final_response_non_empty = bool(str(state.get("final_response", "")).strip())
    need_clarify = _as_bool(state.get("need_clarify", False))
    routing_mode = str(state.get("routing_mode", ""))
    rag_hit_count = len(state.get("rag_evidence", []) or [])
    rag_summary_hit = (
        state.get("evidence_summary", {}).get("rag_summary", {}).get("hit_count", 0) or 0
    )
    rag_hit = (rag_hit_count > 0) or (rag_summary_hit > 0)
    mysql_non_empty = _dict_non_empty(state.get("mysql_evidence", {}))
    has_student_context = bool(
        state.get("student_id") or state.get("student_mention") or state.get("parsed_slots", {}).get("student_id")
    )
    secondary = state.get("secondary_task_types", []) or []

    checks: list[tuple[bool, str]] = []

    if target_task == "technical_qa":
        checks = [
            (final_response_non_empty, "technical_qa_final_response_empty"),
            (not need_clarify, "technical_qa_need_clarify_true"),
            (routing_mode == "technical_qa_short_path", "technical_qa_not_short_path"),
            (rag_hit, "technical_qa_rag_miss"),
        ]
    elif target_task == "diagnosis":
        checks = [(diagnosis_non_empty, "diagnosis_empty")]
        if has_student_context:
            checks.append((mysql_non_empty, "diagnosis_mysql_evidence_empty"))
    elif target_task == "intervention":
        checks = [(intervention_non_empty, "intervention_plan_empty")]
    elif target_task == "dispatch":
        checks = [(packages_non_empty, "dispatch_packages_empty")]
    elif target_task == "mixed":
        # Mixed coverage by secondary tasks.
        sec_checks: list[tuple[bool, str]] = []
        if "diagnosis" in secondary:
            sec_checks.append((diagnosis_non_empty, "mixed_missing_diagnosis"))
        if "intervention" in secondary:
            sec_checks.append((intervention_non_empty, "mixed_missing_intervention"))
        if "dispatch" in secondary:
            sec_checks.append((packages_non_empty, "mixed_missing_dispatch_packages"))
        # if secondary is empty, fallback to requiring final response
        if not sec_checks:
            sec_checks.append((final_response_non_empty, "mixed_final_response_empty"))
        checks = sec_checks
    else:
        checks = [(final_response_non_empty, "unknown_final_response_empty")]

    failed_reasons = [reason for ok, reason in checks if not ok]
    return {
        "target_task": target_task,
        "task_aware_structure_ok": len(failed_reasons) == 0,
        "failed_reasons": failed_reasons,
        "diagnosis_non_empty": diagnosis_non_empty,
        "intervention_non_empty": intervention_non_empty,
        "packages_non_empty": packages_non_empty,
        "final_response_non_empty": final_response_non_empty,
        "need_clarify": need_clarify,
        "routing_mode": routing_mode,
        "rag_hit": rag_hit,
    }


def _safe_div(n: int | float, d: int | float) -> float | None:
    if d == 0:
        return None
    return n / d


def main() -> None:
    graph = build_agent_graph()
    eval_rows = read_jsonl(settings.EVAL_DIR / "agent_eval_requests_10pct.jsonl")
    settings.output_dir_path.mkdir(parents=True, exist_ok=True)
    if not eval_rows:
        print("[eval] 未找到评测数据或数据为空。")
        return

    details: list[dict[str, Any]] = []
    technical_qa_error_cases: list[dict[str, Any]] = []
    routing_error_cases: list[dict[str, Any]] = []

    # Legacy counters (kept for backward compatibility).
    legacy_structure_hits = 0
    legacy_diagnosis_hits = 0
    legacy_final_hits = 0
    legacy_package_non_empty_hits = 0
    legacy_task_match_hits = 0
    legacy_task_match_total = 0

    # Task-aware counters.
    task_aware_ok_hits = 0
    technical_qa_total = 0
    technical_qa_match_hits = 0
    technical_qa_final_ok = 0
    technical_qa_rag_hit = 0
    technical_qa_need_clarify_false = 0
    technical_qa_short_path = 0
    diagnosis_total = 0
    diagnosis_non_empty_hits = 0
    intervention_total = 0
    intervention_non_empty_hits = 0
    dispatch_total = 0
    dispatch_package_non_empty_hits = 0
    mixed_total = 0
    mixed_secondary_coverage_hits = 0

    for idx, row in enumerate(eval_rows, start=1):
        request_text = get_request_text(row)
        if not request_text:
            continue
        state = graph.invoke({"request_text": request_text})
        expected_task = extract_expected_task(row)
        actual_primary_task = state.get("primary_task_type", "unknown")

        # Legacy metrics.
        legacy_required = [
            "task_type",
            "primary_task_type",
            "secondary_task_types",
            "diagnosis",
            "intervention_plan",
            "recommended_packages",
            "evidence_summary",
            "final_response",
        ]
        legacy_structure_ok = all(k in state for k in legacy_required)
        legacy_diagnosis_ok = _dict_non_empty(state.get("diagnosis", {}))
        legacy_final_ok = bool(str(state.get("final_response", "")).strip())
        legacy_package_ok = _list_non_empty(state.get("recommended_packages", []))
        legacy_structure_hits += int(legacy_structure_ok)
        legacy_diagnosis_hits += int(legacy_diagnosis_ok)
        legacy_final_hits += int(legacy_final_ok)
        legacy_package_non_empty_hits += int(legacy_package_ok)
        if expected_task:
            legacy_task_match_total += 1
            if expected_task == actual_primary_task:
                legacy_task_match_hits += 1

        # Task-aware evaluation.
        eval_result = evaluate_task_aware_case(
            state=state,
            expected_task=expected_task,
            actual_primary_task=actual_primary_task,
        )
        target_task = eval_result["target_task"]
        task_aware_ok = eval_result["task_aware_structure_ok"]
        task_aware_ok_hits += int(task_aware_ok)

        if target_task == "technical_qa":
            technical_qa_total += 1
            if expected_task == "technical_qa" and actual_primary_task == "technical_qa":
                technical_qa_match_hits += 1
            technical_qa_final_ok += int(eval_result["final_response_non_empty"])
            technical_qa_rag_hit += int(eval_result["rag_hit"])
            technical_qa_need_clarify_false += int(not eval_result["need_clarify"])
            technical_qa_short_path += int(eval_result["routing_mode"] == "technical_qa_short_path")
            if not task_aware_ok:
                technical_qa_error_cases.append(
                    {
                        "eval_id": idx,
                        "case_id": row.get("id", idx),
                        "request_text": request_text,
                        "expected_task": expected_task,
                        "actual_primary_task": actual_primary_task,
                        "need_clarify": state.get("need_clarify", False),
                        "routing_mode": state.get("routing_mode", ""),
                        "rag_hit_count": len(state.get("rag_evidence", []) or []),
                        "final_response": state.get("final_response", ""),
                        "error_reason": ";".join(eval_result["failed_reasons"]),
                    }
                )
        elif target_task == "diagnosis":
            diagnosis_total += 1
            diagnosis_non_empty_hits += int(eval_result["diagnosis_non_empty"])
        elif target_task == "intervention":
            intervention_total += 1
            intervention_non_empty_hits += int(eval_result["intervention_non_empty"])
        elif target_task == "dispatch":
            dispatch_total += 1
            dispatch_package_non_empty_hits += int(eval_result["packages_non_empty"])
        elif target_task == "mixed":
            mixed_total += 1
            mixed_secondary_coverage_hits += int(task_aware_ok)

        if expected_task and expected_task != actual_primary_task:
            routing_error_cases.append(
                {
                    "request_text": request_text,
                    "expected_task": expected_task,
                    "actual_primary_task": actual_primary_task,
                    "detected_task_types": json.dumps(
                        state.get("parsed_slots", {}).get("detected_task_types", []), ensure_ascii=False
                    ),
                    "parsed_slots": json.dumps(state.get("parsed_slots", {}), ensure_ascii=False),
                    "error_reason": "task_mismatch",
                }
            )

        details.append(
            {
                "case_id": row.get("id", idx),
                "request_text": request_text,
                "expected_task": expected_task,
                "actual_primary_task": actual_primary_task,
                "target_task": target_task,
                "task_aware_structure_ok": task_aware_ok,
                "task_aware_failed_reasons": ";".join(eval_result["failed_reasons"]),
                "need_clarify": eval_result["need_clarify"],
                "routing_mode": eval_result["routing_mode"],
                "rag_hit": eval_result["rag_hit"],
                # Keep legacy fields for compatibility.
                "legacy_structure_ok": legacy_structure_ok,
                "legacy_diagnosis_non_empty": legacy_diagnosis_ok,
                "legacy_final_response_non_empty": legacy_final_ok,
                "legacy_recommended_packages_non_empty": legacy_package_ok,
            }
        )

    total = len(details) if details else 1

    legacy_metrics = {
        "legacy_task_type_hit_rate": _safe_div(legacy_task_match_hits, legacy_task_match_total),
        "legacy_json_structure_completeness_rate": _safe_div(legacy_structure_hits, total),
        "legacy_diagnosis_non_empty_rate": _safe_div(legacy_diagnosis_hits, total),
        "legacy_final_response_non_empty_rate": _safe_div(legacy_final_hits, total),
        "legacy_recommended_packages_non_empty_rate": _safe_div(legacy_package_non_empty_hits, total),
    }

    task_aware_metrics = {
        "task_aware_structure_completeness_rate": _safe_div(task_aware_ok_hits, total),
        "technical_qa_hit_rate": _safe_div(technical_qa_match_hits, technical_qa_total),
        "technical_qa_final_response_rate": _safe_div(technical_qa_final_ok, technical_qa_total),
        "technical_qa_rag_hit_rate": _safe_div(technical_qa_rag_hit, technical_qa_total),
        "technical_qa_need_clarify_false_rate": _safe_div(
            technical_qa_need_clarify_false, technical_qa_total
        ),
        "technical_qa_short_path_rate": _safe_div(technical_qa_short_path, technical_qa_total),
        "diagnosis_task_diagnosis_non_empty_rate": _safe_div(
            diagnosis_non_empty_hits, diagnosis_total
        ),
        "intervention_task_plan_non_empty_rate": _safe_div(
            intervention_non_empty_hits, intervention_total
        ),
        "dispatch_task_package_non_empty_rate": _safe_div(
            dispatch_package_non_empty_hits, dispatch_total
        ),
        "mixed_task_secondary_coverage_rate": _safe_div(
            mixed_secondary_coverage_hits, mixed_total
        ),
    }

    report = {"total_cases": len(details), **task_aware_metrics, **legacy_metrics, "legacy": legacy_metrics}

    report_json = settings.output_dir_path / "eval_report.json"
    report_csv = settings.output_dir_path / "eval_report.csv"
    detail_csv = settings.output_dir_path / "eval_case_details.csv"
    technical_qa_error_csv = settings.output_dir_path / "technical_qa_error_cases.csv"
    routing_error_csv = settings.output_dir_path / "routing_error_cases.csv"

    report_json.write_text(
        json.dumps(
            {
                "summary": report,
                "task_aware_metrics": task_aware_metrics,
                "legacy_metrics": legacy_metrics,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)

    with detail_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "request_text",
                "expected_task",
                "actual_primary_task",
                "target_task",
                "task_aware_structure_ok",
                "task_aware_failed_reasons",
                "need_clarify",
                "routing_mode",
                "rag_hit",
                "legacy_structure_ok",
                "legacy_diagnosis_non_empty",
                "legacy_final_response_non_empty",
                "legacy_recommended_packages_non_empty",
            ],
        )
        writer.writeheader()
        for row in details:
            writer.writerow(row)

    with technical_qa_error_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_id",
                "case_id",
                "request_text",
                "expected_task",
                "actual_primary_task",
                "need_clarify",
                "routing_mode",
                "rag_hit_count",
                "final_response",
                "error_reason",
            ],
        )
        writer.writeheader()
        for row in technical_qa_error_cases:
            writer.writerow(row)

    with routing_error_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_text",
                "expected_task",
                "actual_primary_task",
                "detected_task_types",
                "parsed_slots",
                "error_reason",
            ],
        )
        writer.writeheader()
        for row in routing_error_cases:
            writer.writerow(row)

    print("[eval] 汇总结果：")
    for key, value in report.items():
        if key == "legacy":
            continue
        print(f"- {key}: {value}")
    print(f"[eval] 输出文件: {report_json}")
    print(f"[eval] 输出文件: {report_csv}")
    print(f"[eval] 详情文件: {detail_csv}")
    print(f"[eval] technical_qa 错误样本: {technical_qa_error_csv}")
    print(f"[eval] routing 错误样本: {routing_error_csv}")


if __name__ == "__main__":
    main()
