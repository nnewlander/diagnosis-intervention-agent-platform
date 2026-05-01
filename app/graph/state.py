from typing import TypedDict


class AgentState(TypedDict, total=False):
    request_text: str
    task_type: str
    primary_task_type: str
    secondary_task_types: list[str]
    student_id: str
    class_id: str
    knowledge_points: list[str]
    user_mentioned_knowledge_points: list[str]
    desired_days: int
    error_type: str
    task_priority: str
    parsed_slots: dict
    student_mention: str
    resolver_result: dict
    need_clarify: bool
    clarify_questions: list[str]
    rag_evidence: list[dict]
    rag_query: str
    kg_evidence: list[dict]
    mysql_evidence: dict
    diagnosis: str
    intervention_plan: str
    recommended_packages: list[dict]
    intervention_case_evidence: list[dict]
    evidence_summary: dict
    routing_mode: str
    final_response: str
    debug_trace: list[dict]
