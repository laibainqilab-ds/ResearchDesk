"""Unit tests for the Phase 9 agent state/structured-output models.

These are pure Pydantic model tests: no LLM calls, no LangGraph, no
retrieval collaborators. They check that the schemas actually constrain
what they claim to -- invalid routes, extra fields, and out-of-range
values must fail validation rather than silently pass through.
"""

import pytest
from pydantic import ValidationError

from app.agents.state import (
    AgentState,
    AgentTraceEntry,
    ResearchFindings,
    RouteType,
    RouterDecision,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# RouteType / RouterDecision
# ---------------------------------------------------------------------------

def test_route_type_accepts_all_documented_routes():
    assert RouteType("document_qa") is RouteType.DOCUMENT_QA
    assert RouteType("research") is RouteType.RESEARCH
    assert RouteType("summary") is RouteType.SUMMARY
    assert RouteType("comparison") is RouteType.COMPARISON


def test_router_decision_accepts_valid_route():
    decision = RouterDecision(route="research", reasoning="Needs multiple lookups.")

    assert decision.route is RouteType.RESEARCH
    assert decision.reasoning == "Needs multiple lookups."


def test_router_decision_defaults_reasoning_to_empty_string():
    decision = RouterDecision(route="document_qa")

    assert decision.reasoning == ""


def test_router_decision_rejects_invalid_route():
    with pytest.raises(ValidationError):
        RouterDecision(route="not_a_real_route")


def test_router_decision_requires_route():
    with pytest.raises(ValidationError):
        RouterDecision()


def test_router_decision_rejects_unknown_fields():
    """A hallucinated extra field from a structured LLM response must fail
    validation rather than be silently dropped or accepted."""
    with pytest.raises(ValidationError):
        RouterDecision(route="document_qa", confidence=0.9)


# ---------------------------------------------------------------------------
# ResearchFindings
# ---------------------------------------------------------------------------

def test_research_findings_defaults():
    findings = ResearchFindings()

    assert findings.subquestions == []
    assert findings.evidence == []
    assert findings.sufficient is False
    assert findings.reasoning == ""


def test_research_findings_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ResearchFindings(subquestions=["a"], extra_field="nope")


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

def test_validation_result_requires_core_verdict_fields():
    with pytest.raises(ValidationError):
        ValidationResult(is_valid=True)


def test_validation_result_accepts_full_payload():
    result = ValidationResult(
        is_valid=False,
        claims_supported=False,
        citation_correct=True,
        addresses_question=True,
        unsupported_claims=["The system costs $50/month."],
        citation_issues=[],
        reasoning="One claim has no supporting evidence.",
    )

    assert result.is_valid is False
    assert result.unsupported_claims == ["The system costs $50/month."]


def test_validation_result_defaults_lists_to_empty():
    result = ValidationResult(
        is_valid=True,
        claims_supported=True,
        citation_correct=True,
        addresses_question=True,
    )

    assert result.unsupported_claims == []
    assert result.citation_issues == []


def test_validation_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ValidationResult(
            is_valid=True,
            claims_supported=True,
            citation_correct=True,
            addresses_question=True,
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# AgentTraceEntry
# ---------------------------------------------------------------------------

def test_agent_trace_entry_requires_stage_and_status():
    with pytest.raises(ValidationError):
        AgentTraceEntry()


def test_agent_trace_entry_defaults():
    entry = AgentTraceEntry(stage="router", status="ok")

    assert entry.duration_seconds is None
    assert entry.decision is None
    assert entry.tools_used == []


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

def test_agent_state_requires_user_query():
    with pytest.raises(ValidationError):
        AgentState()


def test_agent_state_defaults():
    state = AgentState(user_query="What is Evo 2?")

    assert state.conversation_context == []
    assert state.selected_route is None
    assert state.rewritten_query is None
    assert state.research_subquestions == []
    assert state.retrieved_evidence == []
    assert state.research_findings is None
    assert state.draft_answer is None
    assert state.validation_result is None
    assert state.retry_count == 0
    assert state.final_answer is None
    assert state.trace_id is None
    assert state.agent_trace == []


def test_agent_state_retry_count_rejects_negative_values():
    with pytest.raises(ValidationError):
        AgentState(user_query="q", retry_count=-1)


def test_agent_state_accepts_nested_structured_outputs():
    state = AgentState(
        user_query="Compare Evo 1 and Evo 2.",
        selected_route=RouteType.COMPARISON,
        validation_result=ValidationResult(
            is_valid=True,
            claims_supported=True,
            citation_correct=True,
            addresses_question=True,
        ),
        research_findings=ResearchFindings(sufficient=True),
        agent_trace=[AgentTraceEntry(stage="router", status="ok")],
    )

    assert state.selected_route is RouteType.COMPARISON
    assert state.validation_result.is_valid is True
    assert state.research_findings.sufficient is True
    assert state.agent_trace[0].stage == "router"


def test_agent_state_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AgentState(user_query="q", made_up_field=True)


def test_agent_state_round_trips_through_dict_serialization():
    original = AgentState(
        user_query="What is Evo 2?",
        conversation_context=[{"role": "user", "content": "earlier question"}],
        retrieved_evidence=[{"document_id": "docA", "chunk_id": 0}],
        retry_count=1,
    )

    restored = AgentState.model_validate(original.model_dump())

    assert restored == original
