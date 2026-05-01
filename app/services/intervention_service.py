from typing import Any


def _clean_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("。。", "。").replace("；；", "；").replace("，，", "，")
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def build_intervention_plan(
    diagnosis: dict[str, Any],
    knowledge_points: list[str],
    intervention_cases: list[dict[str, Any]],
    desired_days: int = 3,
) -> dict[str, Any]:
    if diagnosis.get("mode") == "conservative":
        return {
            "intervention_goal": "先补全学生信息与近期作业证据，再制定个性化干预方案",
            "day_1_action": "收集 student_id、最近3次提交与典型报错截图",
            "day_2_action": "根据补齐数据进行一次短诊断复盘",
            "optional_followup": "若证据充分，再生成3-5天分层干预计划",
            "mode": "conservative",
        }

    kp_text = "、".join(knowledge_points[:3]) if knowledge_points else "基础语法与排错流程"
    has_case = bool(intervention_cases)
    plan: dict[str, Any] = {
        "intervention_goal": f"围绕{kp_text}降低重复错误并提升独立排错能力",
        "day_1_action": f"针对{kp_text}做错因讲解 + 最小复现演示。",
        "day_2_action": "安排分层练习并现场反馈，记录错因类型变化。",
        "optional_followup": "第7天回看错误分布，若未下降则加一轮针对性讲练。",
        "mode": "normal",
    }
    if desired_days >= 3:
        plan["day_3_action"] = "组织错题复盘与迁移训练，要求学生口述排查步骤。"
    if has_case:
        plan["case_hint"] = "已参考历史干预案例进行动作编排。"
    for k, v in list(plan.items()):
        if isinstance(v, str):
            plan[k] = _clean_text(v)
    return plan
