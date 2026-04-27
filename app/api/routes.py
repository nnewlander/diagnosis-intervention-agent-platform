from fastapi import APIRouter

from app.graph.workflow import build_agent_graph
from app.models.schemas import AgentRequest, AgentResponse, EvidenceSummary

router = APIRouter(prefix="/agent", tags=["agent"])
agent_graph = build_agent_graph()


@router.post("/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    state = agent_graph.invoke({"request_text": request.request_text})

    evidence_summary = EvidenceSummary(**state.get("evidence_summary", {}))

    return AgentResponse(
        task_type=state.get("task_type", "unknown"),
        primary_task_type=state.get("primary_task_type", "unknown"),
        secondary_task_types=state.get("secondary_task_types", []),
        parsed_slots=state.get("parsed_slots", {}),
        evidence_summary=evidence_summary,
        diagnosis=state.get("diagnosis", {}),
        intervention_plan=state.get("intervention_plan", {}),
        recommended_packages=state.get("recommended_packages", []),
        final_response=state.get("final_response", ""),
        debug_trace=state.get("debug_trace", []),
        debug_state=state,
    )
