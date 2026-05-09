"""
从项目一样本请求与评测请求构造教学诊断/干预 SFT 数据（instruction / input / output）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "你是面向中小学教师编程课堂的教学诊断与干预助手。"
    "请根据输入中的教师请求（request_text）、解析槽位（parsed_slots）、学情证据（student_evidence）、"
    "RAG 证据（rag_evidence）、知识图谱证据（kg_evidence）以及证据对齐状态（evidence_alignment_status），"
    "生成严格的 JSON，且顶层仅包含 diagnosis 与 intervention_plan 两个键。"
    "diagnosis 必须包含 observed_problem、probable_cause、evidence_basis、confidence_level；不得省略 observed_problem 或 probable_cause。"
    "diagnosis.evidence_basis 用简短中文概括证据依据；diagnosis.confidence_level 只能是 high、medium、cautious_medium、low 之一。"
    "confidence_level 规则：evidence_alignment_status=aligned 且学情证据直接支持教师关注点时可 high；"
    "partially_aligned 时仅 medium 或 cautious_medium；mismatched 时 cautious_medium 或 low；insufficient_data 时 low；"
    "仅有 request_text、缺少可信学情/RAG/KG 命中时不应轻易 high。"
    "for循环与条件判断场景的 probable_cause 必须围绕控制流与语法理解（如 range 边界、缩进、分支结构），不要使用「重复值」「缺失值」等数据清洗话术。"
    "intervention_plan 只能使用字段：intervention_goal、day_1_action、day_2_action、day_3_action、optional_followup。"
    "干预目标必须写在 intervention_plan.intervention_goal；禁止把 intervention_goal、interaction_goal 放在 JSON 顶层。"
    "禁止使用 day1_intervention、day2_intervention、day3_intervention、day4_intervention、day5_intervention、day6_intervention；"
    "三日军方案只写 day_1_action、day_2_action、day_3_action；正文不要出现「第4天」「第5天」「第6天」「第4-5天」或 day4/day5/day6。"
    "不要使用「治疗」「返现」「下班前」「延期复训」等不适合少儿编程课堂的表述；optional_followup 不要写「三个月后」。"
    "day_1_action 只写第1天、day_2_action 只写第2天、day_3_action 只写第3天；day_3_action 不要出现「第二天」。"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = FINETUNE_ROOT / "data"

# 训练标签紧凑 JSON（无缩进换行，降低输出 token）
_OUTPUT_JSON_SEPARATORS = (",", ":")


def _dumps_output(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=_OUTPUT_JSON_SEPARATORS)


SAMPLE_REQUESTS_PATH = PROJECT_ROOT / "data" / "sample_requests.json"
EVAL_JSONL_PATH = (
    PROJECT_ROOT / "project1_agent_raw_data_10pct" / "data" / "agent_eval_requests_10pct.jsonl"
)

KNOWLEDGE_POINTS = [
    "for循环",
    "条件判断",
    "变量定义",
    "NameError",
    "TypeError",
    "SyntaxError",
    "列表索引",
    "字典操作",
    "函数定义",
]

ALIGNMENTS = ["aligned", "partially_aligned", "mismatched", "insufficient_data"]

# 任务分布：侧重 diagnosis / mixed（周期 100，便于扩展到 300～500 条）
TASK_TYPES_CYCLE = (
    ["diagnosis"] * 40
    + ["mixed"] * 35
    + ["intervention"] * 15
    + ["technical_qa"] * 10
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval_request_texts(path: Path) -> list[str]:
    rows: list[str] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("request_text") or obj.get("query")
                if t:
                    rows.append(str(t))
            except json.JSONDecodeError:
                continue
    return rows


def _pick_student_id(seed: int) -> str:
    return f"STU-{1000 + (seed % 900)}"


def _mock_rag_evidence(seed: int, kp: str, error_hint: str) -> list[dict[str, Any]]:
    sid = f"DOC-RAG-{seed % 10000:04d}"
    if kp == "for循环":
        sn_a = (
            f"示例：核对 range 起止是否覆盖需求、循环体缩进是否一致、循环变量是否在循环体内正确更新；知识点：{kp}。"
        )
        sn_b = "让学生用手指跟踪迭代次数，口述每次进入循环体时变量如何变化。"
    elif kp == "条件判断":
        sn_a = f"示例：把条件拆成比较表达式与布尔组合，检查 if/elif/else 缩进是否界定分支范围；知识点：{kp}。"
        sn_b = "用红绿灯类比 True/False，再映射到代码条件行。"
    else:
        sn_a = f"示例：当学生遇到 {error_hint} 时，可引导学生逐步定位语法与类型问题；知识点：{kp}。"
        sn_b = "先用生活类比再映射到代码行号，让学生复述报错含义。"
    return [
        {
            "source_id": sid,
            "title": f"课堂讲义摘录：{kp} 常见误区",
            "snippet": sn_a,
            "score": round(0.55 + (seed % 40) / 100, 3),
            "source_type": "faq",
            "metadata": {"chunk_id": sid},
        },
        {
            "source_id": f"{sid}-b",
            "title": "低龄班级讲解话术",
            "snippet": sn_b,
            "score": round(0.45 + (seed % 30) / 100, 3),
            "source_type": "rag_local",
            "metadata": {},
        },
    ]


def _mock_kg_evidence(seed: int, kp: str, error_hint: str) -> list[dict[str, Any]]:
    return [
        {
            "entity": kp,
            "entity_type": "KnowledgePoint",
            "relation": "HAS_PREREQUISITE",
            "target": "报错排查",
            "evidence": f"图谱路径提示：掌握 {kp} 可减少 {error_hint} 类错误。",
            "score": round(0.5 + (seed % 35) / 100, 3),
            "metadata": {"source": "neo4j_mock"},
        },
        {
            "entity": error_hint or "运行错误",
            "entity_type": "ErrorType",
            "relation": "RELATED_ERROR",
            "target": kp,
            "evidence": "错误类型与知识点弱相关时的迁移训练建议。",
            "score": 0.42,
            "metadata": {},
        },
    ]


def _mock_student_evidence(
    seed: int,
    alignment: str,
    kp: str,
    student_id: str,
) -> dict[str, Any]:
    base_profile = {
        "grade_band": "G5-G6",
        "learning_style_hint": "需要可视化脚手架",
    }
    if alignment == "insufficient_data":
        return {
            "profile_summary": {},
            "recent_submission_summary": {"submissions": [], "total": 0},
            "weak_point_summary": {"weak_knowledge_points": []},
            "recent_error_summary": {"error_distribution": {}},
            "intervention_feedback_summary": {},
            "alignment_summary": {
                "matched_user_mentioned_points": [],
                "unmatched_user_mentioned_points": [kp],
                "evidence_alignment_status": "insufficient_data",
            },
        }

    weak = [kp, "报错排查"]
    err_dist = {"NameError": 3, "SyntaxError": 1} if "Name" in kp or seed % 2 == 0 else {"TypeError": 2, "IndexError": 2}

    if alignment == "aligned":
        matched, unmatched = [kp], []
        status = "aligned"
    elif alignment == "partially_aligned":
        other = KNOWLEDGE_POINTS[(seed + 3) % len(KNOWLEDGE_POINTS)]
        other = other if other != kp else KNOWLEDGE_POINTS[(seed + 4) % len(KNOWLEDGE_POINTS)]
        matched = [kp]
        unmatched = [other] if other not in matched else []
        status = "partially_aligned"
    else:  # mismatched
        matched, unmatched = [], [kp]
        status = "mismatched"

    submissions = [
        {
            "knowledge_point": kp,
            "error_type": list(err_dist.keys())[0],
            "score": 0.55 + (seed % 10) / 100,
        },
        {
            "knowledge_point": "报错排查",
            "error_type": list(err_dist.keys())[-1],
            "score": 0.62,
        },
    ]

    return {
        "profile_summary": {"student_id": student_id, **base_profile},
        "recent_submission_summary": {"submissions": submissions, "total": len(submissions)},
        "weak_point_summary": {"weak_knowledge_points": weak[:3]},
        "recent_error_summary": {"error_distribution": err_dist},
        "intervention_feedback_summary": {"teacher_acceptance": "partial"},
        "alignment_summary": {
            "matched_user_mentioned_points": matched,
            "unmatched_user_mentioned_points": unmatched,
            "data_weak_points": weak,
            "evidence_alignment_status": status,
        },
    }


def _student_evidence_supports_high(st_ev: dict[str, Any]) -> bool:
    """aligned 且学情有直接命中与提交记录时，标签允许 high。"""
    summ = st_ev.get("alignment_summary")
    if not isinstance(summ, dict):
        return False
    if summ.get("evidence_alignment_status") != "aligned":
        return False
    matched = summ.get("matched_user_mentioned_points") or []
    if not matched:
        return False
    rss = st_ev.get("recent_submission_summary")
    if not isinstance(rss, dict):
        return False
    total = int(rss.get("total") or 0)
    return total > 0


def _confidence_for_case(alignment: str, idx: int, st_ev: dict[str, Any]) -> str:
    if alignment == "insufficient_data":
        return "low"
    if alignment == "mismatched":
        return ["cautious_medium", "low"][idx % 2]
    if alignment == "partially_aligned":
        return ["medium", "cautious_medium"][idx % 2]
    # aligned
    if _student_evidence_supports_high(st_ev):
        return "high"
    return "medium"


def _probable_cause_for_kp(kp: str, request_text: str, idx: int) -> str:
    """少儿编程课语境下的原因表述；for/条件分支禁止数据清洗套话。"""
    if kp == "for循环":
        variants = (
            (
                "常见原因包括对 range 起止边界（含 stop 是否取到）理解不稳、循环体缩进层级混乱、"
                "循环变量未按预期更新，或对循环体执行次数与迭代顺序认识不清。"
            ),
            (
                "多见于循环次数与 range 参数不一致、嵌套循环时内层缩进错位，或对每次迭代应执行的操作理解不完整。"
            ),
            (
                "常见原因包括对循环控制变量何时递增理解不足，或循环体内误改迭代序列导致执行路径与预期不符。"
            ),
        )
        return variants[idx % len(variants)]
    if kp == "条件判断":
        variants = (
            (
                "常见原因包括对条件表达式书写不熟、比较运算符（==、!=、>、< 等）混用、"
                "if/elif/else 分支结构不清晰，或对布尔值 True/False 与条件真假判断不稳。"
            ),
            (
                "多见于条件分支缩进错误导致 else 绑定到错误的 if、或elif链条遗漏，从而执行了不该执行的分支。"
            ),
            (
                "常见原因包括对复合条件（and/or）求值顺序不熟，或把赋值（=）误写为比较（==）导致分支走向异常。"
            ),
        )
        return variants[idx % len(variants)]
    if kp == "变量定义":
        return "常见原因是赋值与命名规则不熟，或变量作用域理解不足，导致后续引用未绑定对象。"
    if kp == "NameError" or "NameError" in request_text:
        return "典型原因是变量或函数在使用前未定义，或命名拼写/作用域不一致导致 NameError。"
    if kp == "TypeError":
        return "典型原因是运算或索引两侧对象类型不匹配，或对函数返回值类型假设错误引发 TypeError。"
    if kp == "SyntaxError":
        return "常见原因是冒号/括号/引号配对不当，或缩进层级不符合 Python 语法引发 SyntaxError。"
    if kp == "列表索引":
        return "常见原因是有效下标范围理解不足，或负数索引与切片边界混用引发越界类错误。"
    if kp == "字典操作":
        return "常见原因是对键是否存在缺少判断，或 KeyError 处理与默认值用法不熟。"
    if kp == "函数定义":
        return "常见原因是对 def 语法、参数列表与返回值路径不熟，或在调用前未完成函数定义。"
    return (
        f"可能原因包括对「{kp}」的概念理解与排错路径仍不稳定；"
        "需结合课堂示范与个别反馈巩固。"
    )


def _build_output_json(
    *,
    task_type: str,
    alignment: str,
    kp: str,
    request_text: str,
    student_id: str,
    idx: int,
    st_ev: dict[str, Any],
) -> dict[str, Any]:
    conf = _confidence_for_case(alignment, idx, st_ev)
    observed = (
        f"教师关注学生在「{kp}」相关任务上的稳定性；结合学情与外部证据，"
        f"当前请求类型判定为 {task_type}，对齐状态为 {alignment}。"
    )
    if alignment == "insufficient_data":
        observed = (
            f"教师描述重点在「{kp}」，但学情提交不足或仅有 request_text，难以形成稳定模式判断。"
        )

    cause = _probable_cause_for_kp(kp, request_text, idx)

    evidence_basis = (
        f"对齐状态={alignment}；学情命中教师关注点的情况见 alignment_summary；"
        f"结合 RAG/KG 中与「{kp}」相关的证据片段形成结论。"
    )
    if alignment == "insufficient_data":
        evidence_basis = (
            f"对齐状态={alignment}；当前主要依据 request_text 与教师表述，学情/RAG/KG 证据不足，结论偏保守。"
        )

    goal = f"围绕「{kp}」降低同类错误并提升独立排错与迁移应用能力。"
    day1 = f"第1天：用最小复现场景复盘「{kp}」的关键规则与典型报错含义。"
    day2 = f"第2天：分层练习与当堂反馈，对照错因标签观察是否改善。"
    day3 = f"第3天：口述当日错题排查步骤，巩固「{kp}」的心智模型（仅聚焦当天复盘与迁移）。"
    follow_pool = (
        "若两周内同类错误仍高，可安排一次专项复盘。",
        "若后续作业仍出现同类错误，可补充一次针对性讲解。",
        "后续只做复核与观察，不扩展新的天数安排。",
        "若后续两周内同类错误占比仍高，可与家长同步关注点并安排一次课堂专项复盘（不扩展更多天数安排）。",
    )
    follow = follow_pool[idx % len(follow_pool)]

    if task_type == "technical_qa":
        goal = f"帮助教师用低龄友好话术解释「{kp}」，并给出课堂演示顺序。"
        day1 = f"第1天：先用生活类比解释报错含义，再回到代码定位「{kp}」相关行。"
        day2 = "第2天：让学生口述报错行含义并完成一道同类小题。"
        day3 = "第3天：迁移到相邻知识点的一道综合小题并当堂复盘。"
        follow = "若班级共性明显，可沉淀一页讲义话术卡片以便复用。"

    if alignment == "insufficient_data":
        goal = "优先补齐学生标识与近期提交样本，再制定个性化干预。"
        day1 = "第1天：收集 student_id、最近提交与典型报错截图。"
        day2 = "第2天：补齐数据后进行短诊断复盘。"
        day3 = "第3天：证据充分后再展开分层练习。"
        follow = "证据不足时避免过度推断，明确提示需补数据；不在此承诺额外天数。"

    out: dict[str, Any] = {
        "diagnosis": {
            "observed_problem": observed,
            "probable_cause": cause,
            "evidence_basis": evidence_basis,
            "confidence_level": conf,
        },
        "intervention_plan": {
            "intervention_goal": goal,
            "day_1_action": day1,
            "day_2_action": day2,
            "day_3_action": day3,
            "optional_followup": follow,
        },
    }
    _ = student_id
    return out


def _synthesize_request_text(
    seed: int,
    task_type: str,
    kp: str,
    seeds: list[str],
) -> str:
    rid = seed % max(len(seeds), 1)
    base = seeds[rid]
    student = ["李同学", "王同学", "张同学", "陈同学", "刘同学"][seed % 5]
    if task_type == "diagnosis":
        return f"{student} 最近在「{kp}」相关练习里反复出错，student_id:{_pick_student_id(seed)}，帮我做学情诊断并说明证据。"
    if task_type == "intervention":
        return (
            f"请给 {student} 一个 3 天干预计划，重点关注 {kp}；"
            f"student_id:{_pick_student_id(seed)}；请求原文参考：{base[:60]}…"
        )
    if task_type == "mixed":
        if kp in ("for循环", "条件判断"):
            return (
                f"先诊断 {student} 在「{kp}」上的问题并给出证据说明，再给 3 天课堂干预建议。"
                f"student_id:{_pick_student_id(seed)}"
            )
        return (
            f"先诊断 {student} 在 {kp} 上的问题，再给干预建议；"
            f"同时解释课堂上学生常见的 NameError 应该怎么讲。student_id:{_pick_student_id(seed)}"
        )
    # technical_qa
    return f"课堂演示遇到 {kp if kp != 'NameError' else 'NameError'}，应该怎么给学生解释？参考：{base[:50]}"


def _parsed_slots_for(task_type: str, kp: str, seed: int, student_id: str) -> dict[str, Any]:
    error_type = "NameError" if kp == "NameError" or seed % 7 == 0 else ""
    if kp in ("列表索引",):
        error_type = error_type or "IndexError"
    if kp == "字典操作":
        error_type = error_type or "KeyError"

    detected = {
        "diagnosis": ["diagnosis"],
        "intervention": ["intervention"],
        "mixed": ["diagnosis", "technical_qa"],
        "technical_qa": ["technical_qa"],
    }[task_type]

    return {
        "task_type": task_type if task_type != "mixed" else "mixed",
        "detected_task_types": detected,
        "student_id": student_id,
        "class_id": f"CLS-A{(seed % 3) + 1:02d}",
        "knowledge_points": [kp] if kp != "NameError" else ["变量定义", "报错排查"],
        "user_mentioned_knowledge_points": [kp],
        "desired_days": 3 if task_type in ("intervention", "mixed") else 0,
        "error_type": error_type,
        "task_priority": ["high", "medium", "low"][seed % 3],
        "student_mention": "",
    }


def build_records(count: int = 350, seed: int = 42) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    samples = _load_json(SAMPLE_REQUESTS_PATH)
    seed_texts = [str(x.get("request_text", "")) for x in samples if x.get("request_text")]
    seed_texts.extend(_load_eval_request_texts(EVAL_JSONL_PATH))
    seed_texts = [t for t in seed_texts if t.strip()]
    if len(seed_texts) < 8:
        seed_texts = [
            "帮我看看学生最近作业为什么总出错。",
            "想要一个三天的干预计划。",
            "课堂上 NameError 怎么讲？",
        ]

    records: list[dict[str, Any]] = []
    boost_n = min(max(8, count // 8), max(0, count - 16))
    core_n = count - boost_n
    task_list = (TASK_TYPES_CYCLE * ((core_n // len(TASK_TYPES_CYCLE)) + 1))[:core_n]

    for i in range(core_n):
        task_type = task_list[i]
        kp = KNOWLEDGE_POINTS[i % len(KNOWLEDGE_POINTS)]
        alignment = ALIGNMENTS[i % len(ALIGNMENTS)]
        student_id = _pick_student_id(i + rnd.randint(0, 5))

        error_hint = "NameError" if kp == "NameError" else "常见运行错误"
        req = _synthesize_request_text(i, task_type, kp, seed_texts)
        parsed = _parsed_slots_for(task_type, kp, i, student_id)

        st_ev = _mock_student_evidence(i, alignment, kp, student_id)
        # ensure top-level alignment matches narrative control
        st_ev.setdefault("alignment_summary", {})
        if isinstance(st_ev["alignment_summary"], dict):
            st_ev["alignment_summary"]["evidence_alignment_status"] = alignment

        rag = _mock_rag_evidence(i, kp, error_hint)
        kg = _mock_kg_evidence(i, kp, error_hint)
        # 证据不足：削弱外部证据，强调不应仅凭 request_text 给出过高置信度
        if alignment == "insufficient_data":
            rag = []
            kg = []
        elif alignment == "mismatched" and (i % 11 == 0):
            rag = rag[:1]
            kg = kg[:1]

        input_obj: dict[str, Any] = {
            "request_text": req,
            "parsed_slots": parsed,
            "student_evidence": st_ev,
            "rag_evidence": rag,
            "kg_evidence": kg,
            "evidence_alignment_status": alignment,
        }

        output_obj = _build_output_json(
            task_type=task_type,
            alignment=alignment,
            kp=kp,
            request_text=req,
            student_id=student_id,
            idx=i,
            st_ev=st_ev,
        )

        records.append(
            {
                "case_id": f"SFT-{i:05d}",
                "instruction": INSTRUCTION,
                "input": json.dumps(input_obj, ensure_ascii=False),
                "output": _dumps_output(output_obj),
                "meta": {"task_type": task_type, "alignment": alignment, "kp": kp, "idx": i},
            }
        )

    # 强化 for循环 / 条件判断 高质量样本（避免与 NameError 提示串味）
    for j in range(boost_n):
        i = core_n + j
        kp = ["for循环", "条件判断"][j % 2]
        alignment = ["aligned", "partially_aligned"][j % 2]
        task_type = ["diagnosis", "intervention", "mixed"][j % 3]
        student_id = _pick_student_id(i + rnd.randint(0, 5))
        error_hint = "边界与分支"
        req = _synthesize_request_text(i, task_type, kp, seed_texts)
        parsed = _parsed_slots_for(task_type, kp, i, student_id)
        st_ev = _mock_student_evidence(i, alignment, kp, student_id)
        st_ev.setdefault("alignment_summary", {})
        if isinstance(st_ev["alignment_summary"], dict):
            st_ev["alignment_summary"]["evidence_alignment_status"] = alignment
        rag = _mock_rag_evidence(i, kp, error_hint)
        kg = _mock_kg_evidence(i, kp, error_hint)
        if alignment == "insufficient_data":
            rag = []
            kg = []
        elif alignment == "mismatched" and (i % 11 == 0):
            rag = rag[:1]
            kg = kg[:1]
        input_obj = {
            "request_text": req,
            "parsed_slots": parsed,
            "student_evidence": st_ev,
            "rag_evidence": rag,
            "kg_evidence": kg,
            "evidence_alignment_status": alignment,
        }
        output_obj = _build_output_json(
            task_type=task_type,
            alignment=alignment,
            kp=kp,
            request_text=req,
            student_id=student_id,
            idx=i,
            st_ev=st_ev,
        )
        records.append(
            {
                "case_id": f"SFT-{i:05d}",
                "instruction": INSTRUCTION,
                "input": json.dumps(input_obj, ensure_ascii=False),
                "output": _dumps_output(output_obj),
                "meta": {"task_type": task_type, "alignment": alignment, "kp": kp, "idx": i, "boost": True},
            }
        )

    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]], strip_meta: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if strip_meta:
                row = {k: v for k, v in row.items() if k != "meta"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_jsonl_outputs(train_path: Path, dev_path: Path) -> None:
    import importlib.util
    import sys

    su_p = FINETUNE_ROOT / "scripts" / "schema_utils.py"
    mod_name = "finetune_lora_schema_utils"
    spec = importlib.util.spec_from_file_location(mod_name, su_p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 schema_utils.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    mod.validate_sft_jsonl_file(train_path)
    mod.validate_sft_jsonl_file(dev_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 LoRA SFT 数据 train/dev jsonl")
    parser.add_argument("--count", type=int, default=350, help="总样本数，默认 350（建议 300～500）")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="验证集比例，默认 0.15（约30条）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="输出目录，默认 finetune_lora/data",
    )
    args = parser.parse_args()

    records = build_records(count=args.count, seed=args.seed)
    rnd = random.Random(args.seed)
    rnd.shuffle(records)

    dev_n = max(1, int(round(args.count * args.dev_ratio)))
    dev_n = min(dev_n, args.count - 1)
    dev_rows = records[:dev_n]
    train_rows = records[dev_n:]

    out_dir = args.output_dir.resolve()
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)

    print(f"[build_sft_data] wrote {len(train_rows)} train -> {train_path}")
    print(f"[build_sft_data] wrote {len(dev_rows)} dev   -> {dev_path}")
    try:
        _validate_jsonl_outputs(train_path, dev_path)
        print("[build_sft_data] schema validation OK (train + dev)")
    except ValueError as e:
        print(f"[build_sft_data] SCHEMA ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
