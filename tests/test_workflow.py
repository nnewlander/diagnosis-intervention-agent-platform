from app.core.config import settings
from app.graph.workflow import build_agent_graph
from app.tools.kg_adapter import get_kg_adapter
from app.tools.rag_adapter import get_rag_adapter
from app.tools.student_data_adapter import get_student_data_adapter
from scripts.build_local_sqlite import build_local_sqlite


def _run(request_text: str) -> dict:
    graph = build_agent_graph()
    return graph.invoke({"request_text": request_text})


def test_local_provider_mode_workflow_runs(monkeypatch):
    monkeypatch.setattr(settings, "RAG_PROVIDER", "local")
    monkeypatch.setattr(settings, "KG_PROVIDER", "local")
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "local_csv_jsonl")
    state = _run("请做学情诊断：student_id:STU-0001 最近几次作业表现如何？")
    assert state["primary_task_type"] == "diagnosis"
    assert "debug_trace" in state
    assert "evidence_summary" in state


def test_sqlite_provider_mode_student_query_runs(monkeypatch):
    db_path = build_local_sqlite()
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(db_path.relative_to(settings.PROJECT_ROOT)))
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "sqlite")
    state = _run("请诊断 student_id:STU-0001 的函数问题")
    assert "mysql_evidence" in state
    assert "profile_summary" in state["mysql_evidence"]


def test_provider_switch_logic(monkeypatch):
    monkeypatch.setattr(settings, "RAG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "local_csv_jsonl")
    assert get_rag_adapter().provider_name == "remote"
    assert get_kg_adapter().provider_name == "remote"
    assert get_student_data_adapter().provider_name == "local_csv_jsonl"
from app.core.config import settings
from app.graph.workflow import build_agent_graph
from app.tools.kg_adapter import RemoteKGAdapter, get_kg_adapter
from app.tools.rag_adapter import RemoteRAGAdapter, get_rag_adapter
from app.tools.student_data_adapter import get_student_data_adapter
from scripts.build_local_sqlite import build_local_sqlite


def _run(request_text: str) -> dict:
    graph = build_agent_graph()
    return graph.invoke({"request_text": request_text})


def test_local_provider_workflow_runs(monkeypatch):
    monkeypatch.setattr(settings, "RAG_PROVIDER", "local")
    monkeypatch.setattr(settings, "KG_PROVIDER", "local")
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "local_csv_jsonl")
    state = _run("请做学情诊断：student_id:STU-0001 最近几次作业表现如何？")
    assert state["primary_task_type"] == "diagnosis"
    assert "debug_trace" in state


def test_sqlite_provider_student_query_runs(monkeypatch):
    db_path = build_local_sqlite()
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(db_path.relative_to(settings.PROJECT_ROOT)))
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "sqlite")
    state = _run("请诊断 student_id:STU-0001 的函数问题")
    assert "mysql_evidence" in state
    assert "profile_summary" in state["mysql_evidence"]


def test_remote_provider_placeholder_init_no_error():
    rag = RemoteRAGAdapter()
    kg = RemoteKGAdapter()
    assert rag.provider_name == "remote"
    assert kg.provider_name == "remote"


def test_provider_switch_logic(monkeypatch):
    monkeypatch.setattr(settings, "RAG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "KG_PROVIDER", "remote")
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "local_csv_jsonl")
    assert get_rag_adapter().provider_name == "remote"
    assert get_kg_adapter().provider_name == "remote"
    assert get_student_data_adapter().provider_name == "local_csv_jsonl"
from app.graph.workflow import build_agent_graph


def _run(request_text: str) -> dict:
    graph = build_agent_graph()
    return graph.invoke({"request_text": request_text})


def test_technical_qa_only():
    state = _run("学生最近总报 SyntaxError，我想做技术答疑，怎么讲异常处理？")
    assert state["primary_task_type"] == "technical_qa"


def test_diagnosis_only():
    state = _run("请做学情诊断：student_id:STU-0001 最近几次作业表现如何？")
    assert state["primary_task_type"] == "diagnosis"
    assert "observed_problem" in state["diagnosis"]


def test_mixed_diagnosis_intervention():
    state = _run("请先诊断 student_id:STU-0002 的函数问题，再做3天干预计划。")
    assert state["task_type"] == "mixed"
    assert state["primary_task_type"] in {"diagnosis", "intervention"}


def test_name_only_without_student_id():
    state = _run("李同学最近总报错，帮我看下要不要干预。")
    assert "resolver_result" in state
    assert "student_id" in state


def test_need_clarify_when_evidence_missing():
    state = _run("帮我做一下学情分析。")
    assert state["need_clarify"] is True
    assert len(state["clarify_questions"]) > 0


def test_knowledge_points_without_student_info():
    state = _run("请给我 for循环 和 条件判断 的补练下发建议。")
    assert state["parsed_slots"]["knowledge_points"]
    assert "recommended_packages" in state
from app.graph.workflow import build_agent_graph


def test_graph_runs_end_to_end():
    graph = build_agent_graph()
    state = graph.invoke({"request_text": "请诊断 student_id:1001 的循环薄弱点并推荐练习"})
    assert "task_type" in state
    assert "primary_task_type" in state
    assert "evidence_summary" in state
    assert "debug_trace" in state
    assert "final_response" in state
