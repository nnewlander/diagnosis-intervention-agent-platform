from app.core.config import settings
from app.graph.workflow import build_agent_graph
from app.tools.kg_adapter import get_kg_adapter
from app.tools.rag_adapter import get_rag_adapter
from app.tools.student_data_adapter import get_student_data_adapter
from scripts.build_local_sqlite import build_local_sqlite


def _run(request_text: str) -> dict:
    return build_agent_graph().invoke({"request_text": request_text})


def test_local_provider_mode_workflow_runs(monkeypatch):
    monkeypatch.setattr(settings, "RAG_PROVIDER", "local")
    monkeypatch.setattr(settings, "KG_PROVIDER", "local")
    monkeypatch.setattr(settings, "STUDENT_DATA_PROVIDER", "local_csv_jsonl")
    state = _run("请做学情诊断：student_id:STU-0001 最近几次作业表现如何？")
    assert state["primary_task_type"] == "diagnosis"
    assert "debug_trace" in state


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


def test_nameerror_explain_should_be_technical_qa():
    state = _run("课堂演示遇到 NameError，应该怎么给学生解释？")
    assert state["primary_task_type"] == "technical_qa"
    assert state["routing_mode"] == "technical_qa_short_path"
    assert state["need_clarify"] is False


def test_typical_error_case_pure_explain():
    state = _run("这个报错是什么意思，学生问这个怎么解释？")
    assert state["primary_task_type"] == "technical_qa"
    assert state["need_clarify"] is False


def test_typical_error_case_classroom_teaching():
    state = _run("课堂上怎么讲这个异常，怎么带学生定位错误？")
    assert state["primary_task_type"] == "technical_qa"
    assert state["routing_mode"] == "technical_qa_short_path"


def test_typical_error_case_low_age_explain():
    state = _run("怎么解释给低龄学生，变量未定义怎么讲？")
    assert state["primary_task_type"] == "technical_qa"


def test_typical_error_case_runtime_fail():
    state = _run("这段代码为什么跑不起来，运行失败怎么办？")
    assert state["primary_task_type"] == "technical_qa"


def test_with_student_context_should_not_be_pure_technical_qa():
    state = _run("李同学最近几次作业一直报错，先帮我做诊断再给干预建议")
    assert state["primary_task_type"] in {"diagnosis", "intervention"}
    assert state["routing_mode"] == "task_based_routing"
from app.core.config import settings
from app.graph.workflow import build_agent_graph
from app.tools.kg_adapter import get_kg_adapter
from app.tools.rag_adapter import get_rag_adapter
from app.tools.student_data_adapter import get_student_data_adapter
from scripts.build_local_sqlite import build_local_sqlite


def _run(request_text: str) -> dict:
    return build_agent_graph().invoke({"request_text": request_text})


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


def test_nameerror_explain_should_be_technical_qa():
    state = _run("课堂演示遇到 NameError，应该怎么给学生解释？")
    assert state["primary_task_type"] == "technical_qa"
    assert state["error_type"] == "NameError"
    assert state["need_clarify"] is False
    assert "technical_qa" in state.get("parsed_slots", {}).get("detected_task_types", [])
    assert isinstance(state.get("rag_evidence", []), list)
    assert "请补充 student_id" not in state.get("final_response", "")
    assert state.get("routing_mode") == "technical_qa_short_path"
    assert state.get("diagnosis", {}) in ({}, None)
    assert state.get("intervention_plan", {}) in ({}, None)
    assert state.get("recommended_packages", []) == []
    rag_query = state.get("rag_query", "")
    assert "课堂演示遇到 NameError" in rag_query
    assert "NameError" in rag_query
    assert "变量" in rag_query


def test_nameerror_with_student_should_be_diagnosis():
    state = _run("李同学最近总是 NameError，帮我诊断一下")
    assert state["primary_task_type"] == "diagnosis"
    assert state["need_clarify"] is True


def test_mixed_request_still_runs_diagnosis_branch():
    state = _run("先诊断 student_id:STU-0001 的 NameError 问题，再给我3天干预计划")
    assert state["task_type"] == "mixed"
    assert state.get("routing_mode") == "task_based_routing"
    assert isinstance(state.get("diagnosis", {}), dict)
    assert isinstance(state.get("intervention_plan", {}), dict)
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
