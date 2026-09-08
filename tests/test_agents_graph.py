"""Unit and integration tests for the LangGraph workflow (app/agents/graph.py).

Generator and RetrievalTools are always mocked -- no real Gemini call, no
real retrieval. The integration tests run the *real* compiled LangGraph
graph (real classify_route/run_research/generate_answer/validate_answer),
only mocking the leaf I/O dependencies, so they actually exercise LangGraph
routing/state-merging rather than asserting on our assumptions about it.
"""

from unittest.mock import Mock

from app.agents.graph import (
    _extract_update,
    _finalize,
    _increment_retry,
    _route_after_router,
    _route_after_validation,
    _run_retrieval,
    run_agent_workflow,
)
from app.agents.state import AgentState, AgentTraceEntry, RouteType, ValidationResult
from app.agents.tools import RetrievalTools
from app.agents.validation import MAX_VALIDATION_RETRIES


def make_generator(*responses) -> Mock:
    generator = Mock()
    generator.generate.side_effect = list(responses)
    return generator


def make_tools(search_results=None) -> Mock:
    tools = Mock(spec=RetrievalTools)
    if search_results is not None:
        tools.search_documents.side_effect = search_results
    else:
        tools.search_documents.return_value = []
    return tools


ROUTE_DOCUMENT_QA = '{"route": "document_qa", "reasoning": "simple lookup"}'
ROUTE_RESEARCH = '{"route": "research", "reasoning": "multi-part"}'
VALID_OK = (
    '{"claims_supported": true, "addresses_question": true, '
    '"citation_correct": true, "unsupported_claims": [], "reasoning": "ok"}'
)
INVALID_UNSUPPORTED = (
    '{"claims_supported": false, "addresses_question": true, '
    '"citation_correct": true, "unsupported_claims": ["x"], "reasoning": "bad"}'
)


def make_evidence(document_id="docA", chunk_id=0):
    return {
        "document_id": document_id,
        "filename": "a.pdf",
        "page_number": 1,
        "chunk_id": chunk_id,
        "rerank_score": 0.9,
        "document": "X is true.",
    }


# ---------------------------------------------------------------------------
# _extract_update
# ---------------------------------------------------------------------------

def test_extract_update_returns_only_changed_fields():
    before = AgentState(user_query="q")
    after = before.model_copy(update={"selected_route": RouteType.DOCUMENT_QA})

    update = _extract_update(before, after)

    assert update == {"selected_route": RouteType.DOCUMENT_QA}


def test_extract_update_returns_empty_dict_when_nothing_changed():
    before = AgentState(user_query="q")
    after = before.model_copy()

    assert _extract_update(before, after) == {}


def test_extract_update_preserves_real_nested_types_not_serialized_dicts():
    before = AgentState(user_query="q")
    trace_entry = AgentTraceEntry(stage="router", status="ok")
    after = before.model_copy(update={"agent_trace": [trace_entry]})

    update = _extract_update(before, after)

    assert update["agent_trace"][0] is trace_entry


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def test_route_after_router_returns_research_for_research_route():
    state = AgentState(user_query="q", selected_route=RouteType.RESEARCH)
    assert _route_after_router(state) == "research"


def test_route_after_router_returns_retrieval_for_document_qa():
    state = AgentState(user_query="q", selected_route=RouteType.DOCUMENT_QA)
    assert _route_after_router(state) == "retrieval"


def test_route_after_router_returns_retrieval_for_summary_and_comparison():
    for route in (RouteType.SUMMARY, RouteType.COMPARISON):
        state = AgentState(user_query="q", selected_route=route)
        assert _route_after_router(state) == "retrieval"


def test_route_after_validation_finalizes_when_valid():
    state = AgentState(
        user_query="q",
        retry_count=0,
        validation_result=ValidationResult(
            is_valid=True, claims_supported=True, citation_correct=True, addresses_question=True
        ),
    )
    assert _route_after_validation(state) == "finalize"


def test_route_after_validation_retries_when_invalid_and_under_cap():
    state = AgentState(
        user_query="q",
        retry_count=0,
        validation_result=ValidationResult(
            is_valid=False, claims_supported=False, citation_correct=True, addresses_question=True
        ),
    )
    assert _route_after_validation(state) == "increment_retry"
    assert MAX_VALIDATION_RETRIES > 0  # sanity: the cap actually allows this case to exist


def test_route_after_validation_finalizes_when_invalid_but_at_cap():
    state = AgentState(
        user_query="q",
        retry_count=MAX_VALIDATION_RETRIES,
        validation_result=ValidationResult(
            is_valid=False, claims_supported=False, citation_correct=True, addresses_question=True
        ),
    )
    assert _route_after_validation(state) == "finalize"


# ---------------------------------------------------------------------------
# Deterministic nodes
# ---------------------------------------------------------------------------

def test_increment_retry_increments_count():
    state = AgentState(user_query="q", retry_count=0)
    assert _increment_retry(state) == {"retry_count": 1}


def test_finalize_copies_draft_answer_to_final_answer():
    state = AgentState(user_query="q", draft_answer="An answer.")
    assert _finalize(state) == {"final_answer": "An answer."}


def test_run_retrieval_uses_user_query_when_no_rewritten_query():
    state = AgentState(user_query="original question")
    tools = make_tools(search_results=[[make_evidence()]])

    _run_retrieval(state, tools)

    assert tools.search_documents.call_args.args[0] == "original question"


def test_run_retrieval_prefers_rewritten_query_when_present():
    state = AgentState(user_query="original", rewritten_query="standalone rewritten")
    tools = make_tools(search_results=[[make_evidence()]])

    _run_retrieval(state, tools)

    assert tools.search_documents.call_args.args[0] == "standalone rewritten"


def test_run_retrieval_never_touches_tools_it_was_not_given():
    """Only search_documents is called -- retrieval never reaches for other
    RetrievalTools methods or ChromaDB internals."""
    state = AgentState(user_query="q")
    tools = make_tools(search_results=[[make_evidence()]])

    _run_retrieval(state, tools)

    tools.list_available_documents.assert_not_called()
    tools.get_document_metadata.assert_not_called()
    tools.get_document_page.assert_not_called()


# ---------------------------------------------------------------------------
# Full workflow integration -- real graph, real agent functions, mocked I/O
# ---------------------------------------------------------------------------

def test_workflow_document_qa_end_to_end():
    generator = make_generator(ROUTE_DOCUMENT_QA, "The answer is X [1].", VALID_OK)
    tools = make_tools(search_results=[[make_evidence()]])

    result = run_agent_workflow("What is X?", generator, tools)

    assert result.selected_route == RouteType.DOCUMENT_QA
    assert result.final_answer == "The answer is X [1]."
    assert result.validation_result.is_valid is True
    assert result.retry_count == 0
    assert [entry.stage for entry in result.agent_trace] == [
        "router",
        "retrieval",
        "answer",
        "validation",
    ]


def test_workflow_research_end_to_end():
    generator = make_generator(
        ROUTE_RESEARCH,
        '{"subquestions": ["sub1", "sub2"]}',
        '{"sufficient": true, "additional_subquestions": [], "reasoning": "enough"}',
        "The answer is X [1].",
        VALID_OK,
    )
    tools = make_tools(search_results=[[make_evidence()], [make_evidence(chunk_id=1)]])

    result = run_agent_workflow("Compare A and B", generator, tools)

    assert result.selected_route == RouteType.RESEARCH
    assert result.research_subquestions == ["sub1", "sub2"]
    assert result.final_answer == "The answer is X [1]."
    assert [entry.stage for entry in result.agent_trace] == [
        "router",
        "research_decompose",
        "research_retrieve",
        "research_sufficiency",
        "answer",
        "validation",
    ]


def test_workflow_retries_exactly_once_then_finalizes_regardless_of_second_verdict():
    generator = make_generator(
        ROUTE_DOCUMENT_QA,
        "Answer attempt 1 [1].",
        INVALID_UNSUPPORTED,
        "Answer attempt 2 [1].",
        INVALID_UNSUPPORTED,
    )
    tools = make_tools(search_results=[[make_evidence()], [make_evidence()]])

    result = run_agent_workflow("What is X?", generator, tools)

    # Bounded to exactly one retry: router + (answer, validation) x2 = 5 LLM calls.
    assert generator.generate.call_count == 5
    assert tools.search_documents.call_count == 2
    assert result.retry_count == MAX_VALIDATION_RETRIES
    assert result.validation_result.is_valid is False
    # Finalized anyway once the cap is hit, even though validation never passed.
    assert result.final_answer == "Answer attempt 2 [1]."


def test_workflow_stops_retrying_as_soon_as_validation_passes():
    generator = make_generator(
        ROUTE_DOCUMENT_QA,
        "Answer attempt 1 [1].",
        INVALID_UNSUPPORTED,
        "Answer attempt 2 [1].",
        VALID_OK,
    )
    tools = make_tools(search_results=[[make_evidence()], [make_evidence()]])

    result = run_agent_workflow("What is X?", generator, tools)

    assert result.retry_count == 1
    assert result.validation_result.is_valid is True
    assert result.final_answer == "Answer attempt 2 [1]."
    assert generator.generate.call_count == 5


def test_workflow_trace_accumulates_across_retry():
    generator = make_generator(
        ROUTE_DOCUMENT_QA,
        "Answer attempt 1 [1].",
        INVALID_UNSUPPORTED,
        "Answer attempt 2 [1].",
        VALID_OK,
    )
    tools = make_tools(search_results=[[make_evidence()], [make_evidence()]])

    result = run_agent_workflow("What is X?", generator, tools)

    stages = [entry.stage for entry in result.agent_trace]
    assert stages == [
        "router",
        "retrieval",
        "answer",
        "validation",
        "retrieval",
        "answer",
        "validation",
    ]


def test_workflow_no_evidence_abstains_without_extra_llm_calls():
    generator = make_generator(ROUTE_DOCUMENT_QA)
    tools = make_tools(search_results=[[]])

    result = run_agent_workflow("What is X?", generator, tools)

    # Router (1 call) + no answer-generation call (abstention short-circuit)
    # + no validation call (abstention short-circuit) = 1 call total.
    assert generator.generate.call_count == 1
    assert result.retry_count == 0
    assert result.validation_result.is_valid is True
    assert "couldn't find enough information" in result.final_answer


def test_workflow_propagates_trace_id_to_final_state():
    generator = make_generator(ROUTE_DOCUMENT_QA, "An answer [1].", VALID_OK)
    tools = make_tools(search_results=[[make_evidence()]])

    result = run_agent_workflow("What is X?", generator, tools, trace_id="my-trace-id")

    assert result.trace_id == "my-trace-id"
