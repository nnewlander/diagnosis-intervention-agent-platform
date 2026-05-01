from typing import Any


def select_kg_reference(kg_items: list[dict[str, Any]]) -> str:
    if not kg_items:
        return "-"
    best_solution = None
    best_related = None
    for item in kg_items:
        relation = str(item.get("relation", ""))
        entity = str(item.get("entity", "-"))
        target = str(item.get("target", "-"))
        text = f"{entity} → {relation or '-'} → {target}"
        if relation == "HAS_SOLUTION" and not best_solution:
            best_solution = text
        elif relation == "RELATED_ERROR" and not best_related:
            best_related = text
    return best_solution or best_related or (
        f"{kg_items[0].get('entity', '-')} → {kg_items[0].get('relation', '-')} → {kg_items[0].get('target', '-')}"
    )


def build_demo_final_response(
    raw_final_response: str,
    *,
    task_type: str,
    primary_task_type: str,
    parsed_error_type: str,
    rag_items: list[dict[str, Any]],
    kg_items: list[dict[str, Any]],
) -> str:
    if task_type != "technical_qa" and primary_task_type != "technical_qa":
        text = raw_final_response or ""
        text = text.replace(
            "for循环方向在学情记录中直接证据不足，主要由 KG 结构化知识补充，RAG可作为泛化参考；",
            "变量定义方向由学情记录和 RAG FAQ 共同支持；for循环方向在学情记录中直接证据不足，本次主要由 KG 结构化知识补充；",
        )
        return text
    error_type = parsed_error_type or "常见运行错误"
    rag_ref = "-"
    if rag_items:
        first = rag_items[0]
        rag_ref = str(first.get("source_id") or first.get("title") or "-")
    kg_ref = select_kg_reference(kg_items)
    return (
        f"**问题判断：** 这是一个 {error_type} 类问题，可以理解为“程序找不到这个名字”。\n\n"
        "**原因解释：** NameError 通常表示变量或函数名在使用前未定义，或者命名与实际定义不一致。\n\n"
        "**课堂讲法：**\n"
        "1. 可以先告诉学生：你在代码里喊了一个名字，但电脑还不知道这个名字是谁；\n"
        "2. 让学生定位报错行，找到报错的变量名或函数名；\n"
        "3. 检查它是否先定义后使用；\n"
        "4. 再检查大小写、拼写和作用域。\n\n"
        "**参考证据：**\n"
        f"- RAG: {rag_ref}\n"
        f"- KG: {kg_ref}"
    )


def build_kg_conclusion(kg_items: list[dict[str, Any]]) -> str:
    if not kg_items:
        return "暂无 KG 结构化结论。"
    entity_misuse: dict[str, list[str]] = {}
    suggestions: list[str] = []
    for item in kg_items:
        entity = str(item.get("entity") or "该知识点")
        relation = str(item.get("relation") or "")
        target = str(item.get("target") or "")
        if relation == "COMMON_MISUSE" and target:
            entity_misuse.setdefault(entity, [])
            if target not in entity_misuse[entity]:
                entity_misuse[entity].append(target)
        if relation == "HAS_SOLUTION" and target and target not in suggestions:
            suggestions.append(target)
    if "for循环" in entity_misuse and "条件判断" in entity_misuse:
        f_text = "、".join(entity_misuse.get("for循环", [])[:2]) or "循环边界与变量更新理解不稳"
        c_text = "、".join(entity_misuse.get("条件判断", [])[:2]) or "条件表达式与分支结构理解不稳"
        s_text = "和".join(suggestions[:2]) if suggestions else "通过逐行跟踪变量变化和流程图辅助学生理解执行路径"
        return (
            "KG 结构化结论：for循环常见问题包括"
            f"{f_text}；条件判断常见问题包括{c_text}。建议{s_text}。"
        )
    entity = str(kg_items[0].get("entity") or "该问题")
    common_misuse = entity_misuse.get(entity, [])
    if not common_misuse:
        common_misuse = [str(x.get("target") or "") for x in kg_items if str(x.get("target") or "").strip()][:2]
    misuse_text = "、".join([x for x in common_misuse if x][:2]) if common_misuse else "关键概念理解不稳"
    suggestion_text = suggestions[0] if suggestions else "通过逐行跟踪变量变化帮助学生理解执行过程"
    return f"KG 结构化结论：{entity} 常见问题包括{misuse_text}，建议{suggestion_text}。"

