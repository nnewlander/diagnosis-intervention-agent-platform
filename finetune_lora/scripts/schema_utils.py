"""
LoRA 推理 JSON 抽取、干预字段别名归一化与 SFT schema 校验。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_CONFIDENCE = frozenset({"high", "medium", "cautious_medium", "low"})

# 顶层仅允许 diagnosis + intervention_plan（禁止 interaction_goal 等旁路字段）
ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"diagnosis", "intervention_plan"})

REQUIRED_DIAGNOSIS_KEYS = frozenset(
    {"observed_problem", "probable_cause", "evidence_basis", "confidence_level"}
)
REQUIRED_INTERVENTION_KEYS = frozenset(
    {"intervention_goal", "day_1_action", "day_2_action", "day_3_action", "optional_followup"}
)

# 禁止出现在合法 schema 中的字段名（训练与后处理）
FORBIDDEN_INTERVENTION_ALIASES = frozenset(
    {
        "day1_intervention",
        "day2_intervention",
        "day3_intervention",
        "day4_intervention",
        "day5_intervention",
        "day6_intervention",
    }
)

# day4+ 扩展字段：归一化时移除并记入 extra_fields_removed
EXTRA_DAY_KEYS = frozenset(
    {
        "day4_intervention",
        "day5_intervention",
        "day6_intervention",
        "day_4_action",
        "day_5_action",
        "day_6_action",
    }
)

# 文本内容级：禁止在整条 output JSON 字符串中出现以下片段（含 intervention 叙事越界）
FORBIDDEN_DAY_TEXT_MARKERS: tuple[str, ...] = (
    "第4天",
    "第5天",
    "第五天",
    "第6天",
    "第六天",
    "第4-5天",
    "延长至第4天",
    "扩展到第4天",
    "day4",
    "day5",
    "day6",
)

# for / 条件判断 场景下，probable_cause 不应套用数据清洗类表述
DOMAIN_LOOP_CONDITION_KNOWLEDGE: frozenset[str] = frozenset({"for循环", "条件判断"})
BAD_ETL_CAUSE_PHRASES: tuple[str, ...] = ("重复值", "缺失值")
# for/条件场景下易把原因带偏的表述（教学诊断中应避免）
BAD_LOOP_COND_DISTRACTOR_PHRASES: tuple[str, ...] = (
    "变量定义",
    "函数名称",
    "函数名",
    "数据缺失",
    "重复值",
    "缺失值",
    "日志排查",
)

# 教育场景不宜出现的商业化/医疗化/职场化措辞（SFT 与 eval 共用）
FORBIDDEN_EDUCATION_DOMAIN_TERMS: tuple[str, ...] = (
    "治疗",
    "治疗方案",
    "返现",
    "下班前",
    "三个月内仍不出错",
    "家长或教育专家咨询",
    "延期复训",
)

# confidence 与对齐状态冲突时校验失败
ALIGNMENT_BLOCKING_HIGH: frozenset[str] = frozenset({"mismatched", "insufficient_data"})

ALIAS_TO_CANONICAL = {
    "day1_intervention": "day_1_action",
    "day2_intervention": "day_2_action",
    "day3_intervention": "day_3_action",
}


def infer_output_constraints_text() -> str:
    """拼接到推理 prompt，约束仅输出合法 JSON 字段。"""
    return (
        "你必须只输出一个 JSON 对象。\n"
        "输出到 JSON 对象的最后一个字符必须是「}」，随后立刻停止生成；不要在 JSON 后追加任何字符。\n"
        "不要输出 Markdown、标题、【诊断】、【干预】、alignment_summary、自然语言总结或学情字段复述。\n"
        "不要输出 ```json 代码块。\n"
        "不要输出解释文字。\n"
        "顶层只能有两个键：diagnosis 与 intervention_plan。干预目标只能写在 intervention_plan.intervention_goal，禁止顶层 interaction_goal 等额外字段。\n"
        "字段名必须严格使用：\n"
        "diagnosis.observed_problem\n"
        "diagnosis.probable_cause\n"
        "diagnosis.evidence_basis\n"
        "diagnosis.confidence_level\n"
        "intervention_plan.intervention_goal\n"
        "intervention_plan.day_1_action\n"
        "intervention_plan.day_2_action\n"
        "intervention_plan.day_3_action\n"
        "intervention_plan.optional_followup\n"
        "禁止输出：day1_intervention、day2_intervention、day3_intervention、"
        "day4_intervention、day5_intervention、day6_intervention。\n"
        "若用户要求 3 天干预，只能输出 day_1_action、day_2_action、day_3_action，不要输出 day4/day5/day6。\n"
        "正文叙述中也不要出现「第4天」「第5天」「第6天」「第4-5天」等超出三天的表述。\n"
        "不要使用医疗化、促销或职场话术（如「治疗」「返现」「下班前」）；干预对象是少儿编程课堂。\n"
        "day_3_action 中不要出现「第二天」；optional_followup 不要写「三个月后」类超长承诺。\n"
    )


def normalize_intervention_plan_keys(plan: dict[str, Any]) -> tuple[dict[str, Any], bool, list[str]]:
    """别名 -> day_*_action；移除 day4+ 字段。返回 (新 dict, 是否做过别名归一, 移除的键名列表)。"""
    if not isinstance(plan, dict):
        return {}, False, []
    out = dict(plan)
    normalized = False
    removed: list[str] = []

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if alias in out:
            if canonical not in out or not str(out.get(canonical, "")).strip():
                out[canonical] = out.pop(alias)
                normalized = True
            else:
                del out[alias]
                removed.append(alias)

    for ek in EXTRA_DAY_KEYS:
        if ek in out:
            del out[ek]
            removed.append(ek)

    return out, normalized, removed


def normalize_parsed_object(obj: dict[str, Any]) -> tuple[dict[str, Any], bool, list[str]]:
    """对顶层 diagnosis/intervention_plan 做干预字段归一化。"""
    if not isinstance(obj, dict):
        return obj, False, []
    out = dict(obj)
    extra_all: list[str] = []
    norm_any = False
    ip = out.get("intervention_plan")
    if isinstance(ip, dict):
        new_ip, n, rem = normalize_intervention_plan_keys(ip)
        out["intervention_plan"] = new_ip
        if n:
            norm_any = True
        extra_all.extend(rem)
    return out, norm_any, extra_all


@dataclass
class ExtractResult:
    success: bool
    obj: dict[str, Any] | None
    raw_text: str
    parse_error: str | None
    schema_normalized: bool
    extra_fields_removed: list[str] = field(default_factory=list)


def extract_first_json_object(text: str) -> ExtractResult:
    """
    从模型输出中提取第一个 JSON 对象；支持 ```json 代码块；失败时返回 parse_error。
    成功后对 intervention_plan 做别名归一化与 day4+ 剔除。
    """
    raw = text.strip()
    if not raw:
        return ExtractResult(False, None, raw, "empty output", False, [])

    candidate = raw
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if m:
            candidate = m.group(1).strip()

    first_err: str | None = None
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            obj, norm, extra = normalize_parsed_object(parsed)
            return ExtractResult(True, obj, raw, None, norm, extra)
    except json.JSONDecodeError as e:
        first_err = str(e)

    # 括号截取第一个完整对象（简化：从第一个 { 到最后一个 }）
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        chunk = candidate[start : end + 1]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                obj, norm, extra = normalize_parsed_object(parsed)
                return ExtractResult(True, obj, raw, None, norm, extra)
        except json.JSONDecodeError as e:
            return ExtractResult(False, None, raw, str(e), False, [])

    return ExtractResult(False, None, raw, first_err or "json parse failed", False, [])


def try_extract_dict_via_raw_decode_scan(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    从任意偏移扫描首个可 raw_decode 的 JSON object（修复 rfind('}') 截断错误）。
    成功返回 (dict, None)，失败返回 (None, error_reason)。
    """
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            continue
    return None, "scan_raw_decode_failed"


def extract_first_json_object_repair(text: str) -> ExtractResult:
    """
    先走 extract_first_json_object；失败时再尝试逐起点 raw_decode（用于 eval --repair-json）。
    """
    r = extract_first_json_object(text)
    if r.success:
        return r
    scanned, err = try_extract_dict_via_raw_decode_scan(text.strip())
    if scanned is None:
        return ExtractResult(False, None, text.strip(), r.parse_error or err, False, [])
    obj, norm, extra = normalize_parsed_object(scanned)
    return ExtractResult(True, obj, text.strip(), None, norm, extra)


def detect_extra_text_after_first_json(text: str) -> tuple[bool, str]:
    """
    若文本中从首个「完整 JSON 对象」解析结束后仍有非空白后缀，返回 (True, snippet)。
    用于检测模型在合法 JSON 后继续生成解释性废话。
    """
    s = text.strip()
    if not s:
        return False, ""
    decoder = json.JSONDecoder()
    i = 0
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s):
        return False, ""
    if s[i] != "{":
        j = s.find("{")
        if j < 0:
            return False, ""
        i = j
    try:
        _, end = decoder.raw_decode(s, i)
        tail = s[end:].strip()
        snippet = tail[:1200]
        return bool(tail), snippet
    except json.JSONDecodeError:
        return False, ""


def validate_sft_output_object(obj: dict[str, Any], case_id: str) -> None:
    """校验单条 SFT 标签 JSON；不符合则 ValueError。"""
    if not isinstance(obj, dict):
        raise ValueError(f"[{case_id}] output 顶层不是 JSON object")

    top_keys = set(obj.keys())
    if top_keys != ALLOWED_TOP_LEVEL_KEYS:
        extra = sorted(top_keys - ALLOWED_TOP_LEVEL_KEYS)
        missing = sorted(ALLOWED_TOP_LEVEL_KEYS - top_keys)
        raise ValueError(
            f"[{case_id}] 顶层键必须为且仅为 diagnosis 与 intervention_plan；多余={extra!r} 缺失={missing!r}"
        )

    d = obj.get("diagnosis")
    p = obj.get("intervention_plan")
    if not isinstance(d, dict):
        raise ValueError(f"[{case_id}] diagnosis 必须是 object")
    if not isinstance(p, dict):
        raise ValueError(f"[{case_id}] intervention_plan 必须是 object")

    dk = set(d.keys())
    pk = set(p.keys())
    if not REQUIRED_DIAGNOSIS_KEYS.issubset(dk):
        raise ValueError(f"[{case_id}] diagnosis 缺少键: {REQUIRED_DIAGNOSIS_KEYS - dk}")
    if not REQUIRED_INTERVENTION_KEYS.issubset(pk):
        raise ValueError(f"[{case_id}] intervention_plan 缺少键: {REQUIRED_INTERVENTION_KEYS - pk}")

    for key in REQUIRED_DIAGNOSIS_KEYS:
        if key == "confidence_level":
            v = str(d.get(key, "")).strip()
            if v not in VALID_CONFIDENCE:
                raise ValueError(f"[{case_id}] confidence_level 非法: {v!r}")
        else:
            if not str(d.get(key, "")).strip():
                raise ValueError(f"[{case_id}] diagnosis.{key} 不能为空")

    for key in REQUIRED_INTERVENTION_KEYS:
        if not str(p.get(key, "")).strip():
            raise ValueError(f"[{case_id}] intervention_plan.{key} 不能为空")

    forbidden_in_plan = FORBIDDEN_INTERVENTION_ALIASES.intersection(pk)
    if forbidden_in_plan:
        raise ValueError(f"[{case_id}] intervention_plan 含非法键: {forbidden_in_plan}")

    extra = EXTRA_DAY_KEYS.intersection(pk)
    if extra:
        raise ValueError(f"[{case_id}] intervention_plan 含不允许的扩展日字段: {extra}")


def output_string_has_forbidden_day_text(output_raw: str) -> bool:
    """检查 output 原始字符串是否含越界「第4天+」或 day4/day5/day6（大小写不敏感）。"""
    lower = output_raw.lower()
    for m in FORBIDDEN_DAY_TEXT_MARKERS:
        if m.lower().startswith("day"):
            if m.lower() in lower:
                return True
        elif m in output_raw:
            return True
    return False


def _knowledge_points_from_input(input_obj: dict[str, Any]) -> list[str]:
    slots = input_obj.get("parsed_slots") if isinstance(input_obj.get("parsed_slots"), dict) else {}
    pts: list[str] = []
    for key in ("user_mentioned_knowledge_points", "knowledge_points"):
        v = slots.get(key)
        if isinstance(v, list):
            pts.extend(str(x) for x in v if x)
    return pts


def _alignment_from_input(input_obj: dict[str, Any]) -> str | None:
    top = input_obj.get("evidence_alignment_status")
    if isinstance(top, str) and top.strip():
        return top.strip()
    st = input_obj.get("student_evidence")
    if isinstance(st, dict):
        summ = st.get("alignment_summary")
        if isinstance(summ, dict):
            s = summ.get("evidence_alignment_status")
            if isinstance(s, str) and s.strip():
                return s.strip()
    return None


def find_forbidden_education_terms(text: str) -> list[str]:
    """返回文本中出现的禁用词（按常量表顺序命中即收录）。"""
    found: list[str] = []
    for term in FORBIDDEN_EDUCATION_DOMAIN_TERMS:
        if term in text:
            found.append(term)
    return found


def validate_sft_output_content(
    obj: dict[str, Any],
    *,
    case_id: str,
    output_raw: str,
    input_obj: dict[str, Any] | None,
) -> None:
    """文本语义级校验（字段结构已由 validate_sft_output_object 保证）。"""
    if output_string_has_forbidden_day_text(output_raw):
        raise ValueError(f"[{case_id}] output 文本含禁止的「第4天+」或 day4/day5/day6 表述")

    bad_terms = find_forbidden_education_terms(output_raw)
    if bad_terms:
        raise ValueError(f"[{case_id}] output 含教育场景禁用词: {bad_terms}")

    if "三个月后" in output_raw:
        raise ValueError(f"[{case_id}] output 含不宜表述「三个月后」")

    ip_check = obj.get("intervention_plan")
    if isinstance(ip_check, dict):
        d3 = str(ip_check.get("day_3_action", "") or "")
        if "第二天" in d3:
            raise ValueError(f"[{case_id}] day_3_action 不应包含「第二天」")
        fo = str(ip_check.get("optional_followup", "") or "")
        if "三个月后" in fo:
            raise ValueError(f"[{case_id}] optional_followup 不应包含「三个月后」")

    d = obj.get("diagnosis")
    if not isinstance(d, dict):
        return
    probable = str(d.get("probable_cause", "") or "")
    conf = str(d.get("confidence_level", "") or "").strip()

    kps = _knowledge_points_from_input(input_obj) if input_obj else []
    if kps and DOMAIN_LOOP_CONDITION_KNOWLEDGE.intersection(kps):
        for phrase in BAD_ETL_CAUSE_PHRASES:
            if phrase in probable:
                raise ValueError(
                    f"[{case_id}] for循环/条件判断场景 probable_cause 不应包含数据清洗类表述「{phrase}」"
                )
        for phrase in BAD_LOOP_COND_DISTRACTOR_PHRASES:
            if phrase in probable:
                raise ValueError(
                    f"[{case_id}] for循环/条件判断场景 probable_cause 不应包含偏离控制流的表述「{phrase}」"
                )

    al = _alignment_from_input(input_obj) if input_obj else None
    if conf == "high" and al in ALIGNMENT_BLOCKING_HIGH:
        raise ValueError(
            f"[{case_id}] confidence_level=high 与 evidence_alignment_status={al!r} 不允许同时出现"
        )


def eval_generation_schema_issues(obj: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """
    评估用：解析后的 JSON 是否符合顶层与 intervention_plan 结构。
    返回 (fields_ok, error_reason_codes)。
    """
    reasons: list[str] = []
    if obj is None:
        return False, ["json_parse_failed"]
    if not isinstance(obj, dict):
        return False, ["json_parse_failed"]

    keys = set(obj.keys())
    extra = sorted(keys - ALLOWED_TOP_LEVEL_KEYS)
    missing_sec = sorted(ALLOWED_TOP_LEVEL_KEYS - keys)
    if extra:
        reasons.append("extra_top_level_field")
    if missing_sec:
        reasons.append("missing_top_level_section")

    d = obj.get("diagnosis")
    ip = obj.get("intervention_plan")
    if not isinstance(d, dict):
        reasons.append("invalid_diagnosis_object")
    else:
        if not REQUIRED_DIAGNOSIS_KEYS.issubset(set(d.keys())):
            reasons.append("missing_diagnosis_keys")
        for k in ("observed_problem", "probable_cause", "evidence_basis"):
            if not str(d.get(k, "") or "").strip():
                reasons.append(f"empty_diagnosis_{k}")
        cv = str(d.get("confidence_level", "") or "").strip()
        if cv not in VALID_CONFIDENCE:
            reasons.append("confidence_invalid")

    if not isinstance(ip, dict):
        reasons.append("invalid_intervention_plan_object")
    else:
        pk = set(ip.keys())
        if not REQUIRED_INTERVENTION_KEYS.issubset(pk):
            reasons.append("missing_intervention_plan_keys")
        for k in REQUIRED_INTERVENTION_KEYS:
            if not str(ip.get(k, "") or "").strip():
                if k == "intervention_goal":
                    reasons.append("missing_intervention_goal")
                elif k == "optional_followup":
                    reasons.append("empty_optional_followup")
                else:
                    reasons.append(f"empty_{k}")

    ok = len(reasons) == 0
    reasons = list(dict.fromkeys(reasons))
    return ok, reasons


def eval_semantic_flags(
    input_obj: dict[str, Any],
    obj: dict[str, Any] | None,
    generation_text: str,
) -> dict[str, Any]:
    """
    评估用：检测语义层违规（与 SFT 校验规则一致）。
    generation_text 为模型原始输出或解析后的 JSON 字符串。
    """
    forbidden_day = output_string_has_forbidden_day_text(generation_text)
    domain_terms = find_forbidden_education_terms(generation_text)
    forbidden_domain = len(domain_terms) > 0
    domain_cause_error = False
    overconfident = False

    if obj is None:
        return {
            "forbidden_day_text": forbidden_day,
            "domain_cause_error": False,
            "overconfident": False,
            "forbidden_domain_terms": domain_terms,
            "forbidden_domain": forbidden_domain,
        }

    d = obj.get("diagnosis")
    probable = ""
    conf = ""
    if isinstance(d, dict):
        probable = str(d.get("probable_cause", "") or "")
        conf = str(d.get("confidence_level", "") or "").strip()
    kps = _knowledge_points_from_input(input_obj)
    if kps and DOMAIN_LOOP_CONDITION_KNOWLEDGE.intersection(kps):
        for phrase in BAD_ETL_CAUSE_PHRASES:
            if phrase in probable:
                domain_cause_error = True
                break
        if not domain_cause_error:
            for phrase in BAD_LOOP_COND_DISTRACTOR_PHRASES:
                if phrase in probable:
                    domain_cause_error = True
                    break
    al = _alignment_from_input(input_obj)
    if conf == "high" and al in ALIGNMENT_BLOCKING_HIGH:
        overconfident = True
    return {
        "forbidden_day_text": forbidden_day,
        "domain_cause_error": domain_cause_error,
        "overconfident": overconfident,
        "forbidden_domain_terms": domain_terms,
        "forbidden_domain": forbidden_domain,
    }


def validate_sft_jsonl_line(row: dict[str, Any], line_label: str) -> None:
    """校验 jsonl 一行（含 output 字符串）。"""
    cid = str(row.get("case_id") or line_label)
    out_raw = row.get("output")
    if not isinstance(out_raw, str):
        raise ValueError(f"[{cid}] 缺少 output 字符串")
    try:
        obj = json.loads(out_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"[{cid}] output 非合法 JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"[{cid}] output JSON 顶层不是 object")
    validate_sft_output_object(obj, cid)
    inp_raw = row.get("input")
    input_obj: dict[str, Any] | None = None
    if isinstance(inp_raw, str) and inp_raw.strip():
        try:
            parsed_in = json.loads(inp_raw)
            input_obj = parsed_in if isinstance(parsed_in, dict) else None
        except json.JSONDecodeError:
            input_obj = None
    validate_sft_output_content(obj, case_id=cid, output_raw=out_raw, input_obj=input_obj)


def validate_sft_jsonl_file(path: Path) -> None:
    p = path
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            validate_sft_jsonl_line(row, f"{p.name}:L{idx}")
