"""LangGraph workflow wiring the Phase 9 agents together.

    router -> retrieval (document_qa / summary / comparison)
           -> research  (research)
    retrieval / research -> answer -> validation -> finalize (END)
                                            |
                                            +-> increment_retry -> retrieval / research (bounded)

This module only wires together already-tested, independent functions
(classify_route, run_research, generate_answer, validate_answer,
RetrievalTools.search_documents) -- none of their internal logic is
duplicated or reimplemented here. The only new logic is:

- the deterministic retrieval node, a thin wrapper around
  RetrievalTools.search_documents mirroring the same pure-function /
  AgentTraceEntry pattern the other stages already use (no LLM call), and
- the routing/retry glue LangGraph requires (_route_after_router,
  _route_after_validation, increment_retry, finalize).

The hard retry ceiling is enforced by the graph itself, reading
state.retry_count against validation.MAX_VALIDATION_RETRIES in
_route_after_validation -- not by LangGraph's generic recursion limit,
which remains an unrelated, unmodified safety net.
"""

import logging
import time

from langgraph.graph import END, StateGraph

from app.agents.answer import generate_answer
from app.agents.research import run_research
from app.agents.router import classify_route
from app.agents.state import AgentState, AgentTraceEntry, RouteType
from app.agents.tools import RetrievalTools
from app.agents.validation import MAX_VALIDATION_RETRIES, validate_answer
from app.models.generator import Generator
from app.observability import configure_logging, log_event, new_trace_id

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 3


def _extract_update(before: AgentState, after: AgentState) -> dict:
    """Diff two AgentState instances into a partial-update dict for
    LangGraph's node contract.

    Every agent function already returns a full new AgentState without
    mutating its input, so this just narrows that to the fields that
    actually changed. Comparing/returning the live attribute values (not a
    serialized form) means nested models (ValidationResult, AgentTraceEntry,
    the RouteType enum) are handed back as already-valid instances of their
    real types.
    """
    return {
        field_name: after_value
        for field_name in AgentState.model_fields
        if (after_value := getattr(after, field_name)) != getattr(before, field_name)
    }


def _run_retrieval(state: AgentState, tools: RetrievalTools) -> dict:
    """Deterministic Retrieval node: one search_documents call, no LLM."""
    trace_id = state.trace_id
    query = state.rewritten_query or state.user_query
    start = time.perf_counter()

    evidence = tools.search_documents(query, top_k=RETRIEVAL_TOP_K, trace_id=trace_id)
    duration = time.perf_counter() - start

    log_event(
        trace_id,
        "retrieval_stage_completed",
        evidence_count=len(evidence),
        duration_seconds=duration,
    )
    trace_entry = AgentTraceEntry(
        stage="retrieval",
        status="ok",
        duration_seconds=duration,
        decision=f"{len(evidence)} evidence item(s)",
        tools_used=["search_documents"],
    )
    return {"retrieved_evidence": evidence, "agent_trace": [*state.agent_trace, trace_entry]}


def _route_after_router(state: AgentState) -> str:
    return "research" if state.selected_route == RouteType.RESEARCH else "retrieval"


def _route_after_validation(state: AgentState) -> str:
    if state.validation_result is not None and state.validation_result.is_valid:
        return "finalize"
    if state.retry_count >= MAX_VALIDATION_RETRIES:
        return "finalize"
    return "increment_retry"


def _increment_retry(state: AgentState) -> dict:
    return {"retry_count": state.retry_count + 1}


def _finalize(state: AgentState) -> dict:
    return {"final_answer": state.draft_answer}


def build_agent_graph(generator: Generator, tools: RetrievalTools):
    """Build and compile the LangGraph workflow.

    `generator` and `tools` are injected rather than constructed here, so
    tests (and callers) can supply mocks or real instances -- the graph
    definition itself makes no Gemini or retrieval calls of its own.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", lambda state: _extract_update(state, classify_route(state, generator)))
    graph.add_node("retrieval", lambda state: _run_retrieval(state, tools))
    graph.add_node("research", lambda state: _extract_update(state, run_research(state, generator, tools)))
    graph.add_node("answer", lambda state: _extract_update(state, generate_answer(state, generator)))
    graph.add_node("validation", lambda state: _extract_update(state, validate_answer(state, generator)))
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", _route_after_router, {"retrieval": "retrieval", "research": "research"}
    )
    graph.add_edge("retrieval", "answer")
    graph.add_edge("research", "answer")
    graph.add_edge("answer", "validation")
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"finalize": "finalize", "increment_retry": "increment_retry"},
    )
    graph.add_conditional_edges(
        "increment_retry", _route_after_router, {"retrieval": "retrieval", "research": "research"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent_workflow(
    user_query: str,
    generator: Generator,
    tools: RetrievalTools,
    conversation_context: list[dict] | None = None,
    trace_id: str | None = None,
) -> AgentState:
    """Run the full Router -> Retrieval/Research -> Answer -> Validation
    workflow for one user query and return the final AgentState."""
    configure_logging()
    trace_id = trace_id or new_trace_id()

    initial_state = AgentState(
        user_query=user_query,
        conversation_context=conversation_context or [],
        trace_id=trace_id,
    )

    compiled_graph = build_agent_graph(generator, tools)
    result = compiled_graph.invoke(initial_state)

    return AgentState.model_validate(result)
