import re
from typing import Any

# Task keywords are grouped by business intent.
TASK_KEYWORDS = {
    "technical_qa": [
        "报错",
        "报错排查",
        "技术答疑",
        "代码",
        "编译",
        "调试",
        "运行失败",
        "异常处理",
        "错误信息",
    ],
    "diagnosis": [
        "诊断",
        "学情",
        "掌握",
        "薄弱",
        "错误率",
        "表现",
        "最近几次作业",
        "最近提交",
        "最近总报错",
    ],
    "intervention": [
        "干预",
        "改进",
        "提升",
        "辅导",
        "计划",
        "策略",
        "先诊断再干预",
    ],
    "dispatch": [
        "练习下发",
        "下发",
        "推荐题",
        "题包",
        "补练",
        "作业包",
    ],
}

KNOWLEDGE_POINT_ALIASES = {
    "for循环": ["for循环", "循环", "循环结构"],
    "条件判断": ["条件判断", "if", "分支判断"],
    "函数": ["函数", "方法", "函数定义"],
    "列表": ["列表", "list", "数组"],
    "字典": ["字典", "dict", "映射"],
    "字符串": ["字符串", "string"],
    "异常处理": ["异常处理", "try", "except"],
    "报错排查": ["报错排查", "错误排查", "调试排错"],
}

ERROR_PATTERNS = [
    r"(TypeError|ValueError|IndexError|KeyError|SyntaxError|NameError|ModuleNotFoundError)",
    r"(编译错误|语法错误|超时|运行时错误|空指针)",
]

PRIORITY_PATTERNS = {
    "high": ["优先", "尽快", "紧急", "马上", "今天就要"],
    "medium": ["本周", "尽量", "安排一下"],
    "low": ["有空", "后续", "先不急"],
}


def parse_student_id(text: str) -> str:
    patterns = [
        r"(?:student[_\s-]?id|学号)\s*[:：]?\s*([A-Za-z0-9_-]+)",
        r"\b(STU-\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def parse_class_id(text: str) -> str:
    patterns = [
        r"(?:class[_\s-]?id|班级)\s*[:：]?\s*([A-Za-z0-9_-]+)",
        r"\b(CLS-[A-Za-z0-9-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def parse_knowledge_points(text: str) -> list[str]:
    found: list[str] = []
    text_lower = text.lower()
    for canonical, aliases in KNOWLEDGE_POINT_ALIASES.items():
        if any(alias.lower() in text_lower for alias in aliases):
            found.append(canonical)
    return found


def parse_desired_days(text: str) -> int:
    match = re.search(r"(\d+)\s*天", text)
    if match:
        return int(match.group(1))
    if "本周计划" in text:
        return 5
    return 0


def parse_error_type(text: str) -> str:
    for pattern in ERROR_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def parse_task_priority(text: str) -> str:
    for level, keywords in PRIORITY_PATTERNS.items():
        if any(keyword in text for keyword in keywords):
            return level
    return "medium"


def extract_student_mention(text: str) -> str:
    # Support "李同学 / 王同学 / 这个学生 / 这个孩子" style mentions.
    match = re.search(r"([\u4e00-\u9fa5]{1,3})同学", text)
    if match:
        return f"{match.group(1)}同学"
    for phrase in ["这个学生", "这个孩子", "该学生", "这位同学"]:
        if phrase in text:
            return phrase
    return ""


def detect_task_types(text: str) -> list[str]:
    text_lower = text.lower()
    detected: list[str] = []
    for task, keywords in TASK_KEYWORDS.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            detected.append(task)
    return detected


def detect_task_type(text: str) -> str:
    tasks = detect_task_types(text)
    if len(tasks) > 1:
        return "mixed"
    if len(tasks) == 1:
        return tasks[0]
    return "unknown"


def parse_request_slots(text: str) -> dict[str, Any]:
    task_types = detect_task_types(text)
    task_type = "mixed" if len(task_types) > 1 else (task_types[0] if task_types else "unknown")
    return {
        "task_type": task_type,
        "detected_task_types": task_types,
        "student_id": parse_student_id(text),
        "class_id": parse_class_id(text),
        "knowledge_points": parse_knowledge_points(text),
        "desired_days": parse_desired_days(text),
        "error_type": parse_error_type(text),
        "task_priority": parse_task_priority(text),
        "student_mention": extract_student_mention(text),
        "request_context_flags": {
            "recent_submissions": any(k in text for k in ["最近几次作业", "最近提交", "最近总报错"]),
            "stepwise_plan": any(k in text for k in ["先诊断再干预", "先诊断", "再干预"]),
        },
    }
import re


TASK_KEYWORDS = {
    "technical_qa": ["报错", "报错信息", "技术", "代码", "编译", "调试", "运行失败"],
    "diagnosis": ["诊断", "学情", "掌握", "薄弱", "错误率", "表现"],
    "intervention": ["干预", "改进", "提升", "辅导", "计划", "策略"],
    "assignment": ["练习", "作业", "下发", "推荐题", "补练", "题包"],
}


def parse_student_id(text: str) -> str:
    match = re.search(r"(?:student[_\s-]?id|学生)\s*[:：]?\s*([A-Za-z0-9_-]+)", text, re.I)
    return match.group(1) if match else ""


def parse_class_id(text: str) -> str:
    match = re.search(r"(?:class[_\s-]?id|班级)\s*[:：]?\s*([A-Za-z0-9_-]+)", text, re.I)
    return match.group(1) if match else ""


def parse_knowledge_points(text: str) -> list[str]:
    candidates = []
    known_terms = ["循环", "函数", "数组", "递归", "语法", "变量", "条件判断", "指针", "字符串"]
    for term in known_terms:
        if term in text:
            candidates.append(term)
    return candidates


def detect_task_type(text: str) -> str:
    tasks = detect_task_types(text)
    if len(tasks) > 1:
        return "mixed"
    if len(tasks) == 1:
        return tasks[0]
    return "unknown"


def detect_task_types(text: str) -> list[str]:
    text_lower = text.lower()
    hit: list[str] = []
    for task, keywords in TASK_KEYWORDS.items():
        if any(k.lower() in text_lower for k in keywords):
            hit.append(task)
    return hit
