import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any

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
    if any(k in blob for k in ["assignment", "练习", "下发"]):
        return "assignment"
    if any(k in blob for k in ["technical", "技术", "报错", "qa"]):
        return "technical_qa"
    return ""


def get_request_text(row: dict[str, Any]) -> str:
    for key in ["request_text", "query", "teacher_request", "input"]:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _normalized_score(hit_count: int, total_checks: int) -> float:
    if total_checks <= 0:
        return 0.0
    return hit_count / total_checks


def calc_diagnosis_quality_score(state: dict[str, Any]) -> float:
    diagnosis = state.get("diagnosis", {}) or {}
    slots = state.get("parsed_slots", {}) or {}
    knowledge_points = slots.get("knowledge_points", []) or []
    observed_problem = str(diagnosis.get("observed_problem", ""))
    probable_cause = str(diagnosis.get("probable_cause", ""))
    evidence_basis = diagnosis.get("evidence_basis", {})

    hit = 0
    # 规则1：提到 knowledge_points（文本包含任一知识点即命中）
    if knowledge_points and any(kp in observed_problem for kp in knowledge_points if kp):
        hit += 1
    # 规则2：包含 probable_cause
    if bool(probable_cause.strip()):
        hit += 1
    # 规则3：引用 evidence_basis
    if isinstance(evidence_basis, dict) and len(evidence_basis) > 0:
        hit += 1
    return _normalized_score(hit, 3)


def calc_intervention_quality_score(state: dict[str, Any]) -> float:
    plan = state.get("intervention_plan", {}) or {}
    slots = state.get("parsed_slots", {}) or {}
    knowledge_points = slots.get("knowledge_points", []) or []

    intervention_goal = str(plan.get("intervention_goal", ""))
    day_1_action = str(plan.get("day_1_action", ""))
    day_2_action = str(plan.get("day_2_action", ""))
    day_3_action = str(plan.get("day_3_action", ""))

    hit = 0
    if bool(intervention_goal.strip()):
        hit += 1
    if bool(day_1_action.strip()):
        hit += 1
    # 规则3：包含多天结构（至少 day_1 + day_2，或 day_3）
    if (bool(day_1_action.strip()) and bool(day_2_action.strip())) or bool(day_3_action.strip()):
        hit += 1
    # 规则4：与 knowledge_points 有重合
    merged_plan_text = " ".join([intervention_goal, day_1_action, day_2_action, day_3_action])
    if knowledge_points and any(kp in merged_plan_text for kp in knowledge_points if kp):
        hit += 1
    return _normalized_score(hit, 4)


def calc_evidence_usage_score(state: dict[str, Any]) -> float:
    mysql_evidence = state.get("mysql_evidence", {}) or {}
    rag_evidence = state.get("rag_evidence", []) or []
    kg_evidence = state.get("kg_evidence", []) or []
    hit = 0
    if isinstance(mysql_evidence, dict) and len(mysql_evidence) > 0:
        hit += 1
    if isinstance(rag_evidence, list) and len(rag_evidence) > 0:
        hit += 1
    if isinstance(kg_evidence, list) and len(kg_evidence) > 0:
        hit += 1
    return _normalized_score(hit, 3)


def check_conservative_mode_reasonable(state: dict[str, Any]) -> bool:
    """need_clarify=True 时不应给出完整诊断（否则判为不合理）。"""
    need_clarify = bool(state.get("need_clarify", False))
    if not need_clarify:
        return True
    diagnosis = state.get("diagnosis", {}) or {}
    # 认为“完整诊断”至少包含这三项且非空
    has_full = all(
        bool(str(diagnosis.get(key, "")).strip())
        for key in ["observed_problem", "probable_cause", "evidence_basis"]
    )
    return not has_full


def main() -> None:
    graph = build_agent_graph()
    eval_rows = read_jsonl(settings.EVAL_DIR / "agent_eval_requests_10pct.jsonl")
    settings.output_dir_path.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, Any]] = []
    if not eval_rows:
        print("[eval] 未找到评测数据或数据为空。")
        return

    structure_hits = 0
    diagnosis_hits = 0
    final_hits = 0
    task_match_hits = 0
    task_match_total = 0
    primary_task_match_hits = 0
    primary_task_match_total = 0
    slot_student_id_hits = 0
    slot_kp_hits = 0
    clarify_trigger_hits = 0
    package_non_empty_hits = 0
    diagnosis_quality_sum = 0.0
    intervention_quality_sum = 0.0
    evidence_usage_sum = 0.0
    conservative_reasonable_hits = 0

    for idx, row in enumerate(eval_rows, start=1):
        request_text = get_request_text(row)
        if not request_text:
            continue
        state = graph.invoke({"request_text": request_text})
        expected_task = extract_expected_task(row)
        actual_task = state.get("primary_task_type", "unknown")

        if expected_task:
            task_match_total += 1
            primary_task_match_total += 1
            if expected_task == actual_task:
                task_match_hits += 1
                primary_task_match_hits += 1

        required = [
            "task_type",
            "primary_task_type",
            "secondary_task_types",
            "diagnosis",
            "intervention_plan",
            "recommended_packages",
            "evidence_summary",
            "final_response",
        ]
        structure_ok = all(k in state for k in required)
        diagnosis_obj = state.get("diagnosis", {})
        diagnosis_ok = isinstance(diagnosis_obj, dict) and bool(
            diagnosis_obj.get("observed_problem", "")
        )
        final_ok = bool(state.get("final_response", "").strip())
        slots = state.get("parsed_slots", {})
        student_slot_ok = bool(slots.get("student_id"))
        kp_slot_ok = bool(slots.get("knowledge_points", []))
        clarify_ok = bool(state.get("need_clarify", False))
        package_ok = len(state.get("recommended_packages", [])) > 0
        diagnosis_quality_score = calc_diagnosis_quality_score(state)
        intervention_quality_score = calc_intervention_quality_score(state)
        evidence_usage_score = calc_evidence_usage_score(state)
        conservative_reasonable = check_conservative_mode_reasonable(state)

        structure_hits += int(structure_ok)
        diagnosis_hits += int(diagnosis_ok)
        final_hits += int(final_ok)
        slot_student_id_hits += int(student_slot_ok)
        slot_kp_hits += int(kp_slot_ok)
        clarify_trigger_hits += int(clarify_ok)
        package_non_empty_hits += int(package_ok)
        diagnosis_quality_sum += diagnosis_quality_score
        intervention_quality_sum += intervention_quality_score
        evidence_usage_sum += evidence_usage_score
        conservative_reasonable_hits += int(conservative_reasonable)

        details.append(
            {
                "case_id": row.get("id", idx),
                "request_text": request_text,
                "expected_task": expected_task,
                "actual_primary_task": actual_task,
                "task_match": expected_task == actual_task if expected_task else None,
                "structure_ok": structure_ok,
                "diagnosis_non_empty": diagnosis_ok,
                "final_response_non_empty": final_ok,
                "slot_student_id_hit": student_slot_ok,
                "slot_knowledge_points_hit": kp_slot_ok,
                "need_clarify_triggered": clarify_ok,
                "recommended_packages_non_empty": package_ok,
                "diagnosis_quality_score": diagnosis_quality_score,
                "intervention_quality_score": intervention_quality_score,
                "evidence_usage_score": evidence_usage_score,
                "conservative_mode_reasonable": conservative_reasonable,
            }
        )

    total = len(details) if details else 1
    report = {
        "total_cases": len(details),
        "task_type_hit_rate": (task_match_hits / task_match_total) if task_match_total else None,
        "primary_task_type_hit_rate": (
            primary_task_match_hits / primary_task_match_total
        )
        if primary_task_match_total
        else None,
        "json_structure_completeness_rate": structure_hits / total,
        "slot_student_id_coverage_rate": slot_student_id_hits / total,
        "slot_knowledge_points_coverage_rate": slot_kp_hits / total,
        "need_clarify_trigger_rate": clarify_trigger_hits / total,
        "diagnosis_non_empty_rate": diagnosis_hits / total,
        "final_response_non_empty_rate": final_hits / total,
        "recommended_packages_non_empty_rate": package_non_empty_hits / total,
        "diagnosis_quality_avg": diagnosis_quality_sum / total,
        "intervention_quality_avg": intervention_quality_sum / total,
        "evidence_usage_avg": evidence_usage_sum / total,
        "conservative_mode_reasonable_rate": conservative_reasonable_hits / total,
        "task_type_eval_cases": task_match_total,
    }

    report_json = settings.output_dir_path / "eval_report.json"
    report_csv = settings.output_dir_path / "eval_report.csv"
    detail_csv = settings.output_dir_path / "eval_case_details.csv"

    report_json.write_text(
        json.dumps({"summary": report, "details": details}, ensure_ascii=False, indent=2),
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
                "task_match",
                "structure_ok",
                "diagnosis_non_empty",
                "final_response_non_empty",
                "slot_student_id_hit",
                "slot_knowledge_points_hit",
                "need_clarify_triggered",
                "recommended_packages_non_empty",
                "diagnosis_quality_score",
                "intervention_quality_score",
                "evidence_usage_score",
                "conservative_mode_reasonable",
            ],
        )
        writer.writeheader()
        for row in details:
            writer.writerow(row)

    print("[eval] 汇总结果：")
    for key, value in report.items():
        print(f"- {key}: {value}")
    print(f"[eval] 输出文件: {report_json}")
    print(f"[eval] 输出文件: {report_csv}")
    print(f"[eval] 详情文件: {detail_csv}")


if __name__ == "__main__":
    main()
