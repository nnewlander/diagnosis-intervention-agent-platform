from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_final_response,
    clarify_if_needed,
    fetch_kg_evidence,
    fetch_mysql_evidence,
    fetch_rag_evidence,
    generate_diagnosis,
    generate_intervention,
    parse_request,
    recommend_package,
    route_task,
)
from app.graph.state import AgentState


def _after_kg_route(state: AgentState) -> str:
    if state.get("routing_mode") == "technical_qa_short_path":
        return "build_final_response"
    return "fetch_mysql_evidence"


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("route_task", route_task)
    graph.add_node("clarify_if_needed", clarify_if_needed)
    graph.add_node("fetch_rag_evidence", fetch_rag_evidence)
    graph.add_node("fetch_kg_evidence", fetch_kg_evidence)
    graph.add_node("fetch_mysql_evidence", fetch_mysql_evidence)
    graph.add_node("generate_diagnosis", generate_diagnosis)
    graph.add_node("generate_intervention", generate_intervention)
    graph.add_node("recommend_package", recommend_package)
    graph.add_node("build_final_response", build_final_response)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "route_task")
    graph.add_edge("route_task", "clarify_if_needed")
    graph.add_edge("clarify_if_needed", "fetch_rag_evidence")
    graph.add_edge("fetch_rag_evidence", "fetch_kg_evidence")
    graph.add_conditional_edges(
        "fetch_kg_evidence",
        _after_kg_route,
        {
            "build_final_response": "build_final_response",
            "fetch_mysql_evidence": "fetch_mysql_evidence",
        },
    )
    graph.add_edge("fetch_mysql_evidence", "generate_diagnosis")
    graph.add_edge("generate_diagnosis", "generate_intervention")
    graph.add_edge("generate_intervention", "recommend_package")
    graph.add_edge("recommend_package", "build_final_response")
    graph.add_edge("build_final_response", END)

    return graph.compile()
