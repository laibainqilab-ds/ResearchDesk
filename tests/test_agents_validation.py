"""Unit tests for the Validation agent (app/agents/validation.py).

The Generator is always mocked -- no real Gemini call. These tests check
the no-LLM short-circuit paths (no answer, abstention message), that a
structurally invalid citation forces citation_correct=False regardless of
what the LLM claims, the fail-safe fallback on LLM failure, and trace
recording.
"""

from unittest.mock import Mock

from app.agents.state import AgentState, AgentTraceEntry
from app.agents.validation import MAX_VALIDATION_RETRIES, validate_answer
from app.models.generator import GenerationUnavailableError
from app.rag import NO_EVIDENCE_ANSWER


def make_generator(response_text=None, side_effect=None) -> Mock:
    generator = Mock()
    if side_effect is not None:
        generator.generate.side_effect = side_effect
    else:
        generator.generate.return_value = response_text
    return generator


def make_evidence(document_id="docA", chunk_id=0, page_number=3, filename="a.pdf", document="chunk text"):
    return {
        "document_id": document_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_id": chunk_id,
        "rerank_score": 0.9,
        "document": document,
    }


VALID_OK = (
    '{"claims_supported": true, "addresses_question": true, '
    '"citation_correct": true, "unsupported_claims": [], "reasoning": "Well grounded."}'
)


# ---------------------------------------------------------------------------
# No-LLM short-circuits
# ---------------------------------------------------------------------------

def test_validate_answer_skips_llm_when_no_draft_answer():
    state = AgentState(user_query="a question", retrieved_evidence=[make_evidence()])
    generator = make_generator()

    result = validate_answer(state, generator)

    generator.generate.assert_not_called()
    assert result.validation_result.is_valid is False
    assert result.agent_trace[-1].status == "skipped_no_answer"


def test_validate_answer_skips_llm_when_answer_is_abstention_message():
    state = AgentState(user_query="a question", draft_answer=NO_EVIDENCE_ANSWER)
    generator = make_generator()

    result = validate_answer(state, generator)

    generator.generate.assert_not_called()
    assert result.validation_result.is_valid is True
    assert result.agent_trace[-1].status == "skipped_abstention"


# ---------------------------------------------------------------------------
# Successful validation
# ---------------------------------------------------------------------------

def test_validate_answer_calls_generator_with_question_evidence_and_answer():
    state = AgentState(
        user_query="What does the document say?",
        draft_answer="The system supports X [1].",
        retrieved_evidence=[make_evidence(document="The system supports X.")],
    )
    generator = make_generator(VALID_OK)

    validate_answer(state, generator)

    prompt = generator.generate.call_args.args[0]
    assert "What does the document say?" in prompt
    assert "The system supports X." in prompt
    assert "The system supports X [1]." in prompt


def test_validate_answer_returns_valid_result_from_well_formed_response():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(VALID_OK)

    result = validate_answer(state, generator)

    assert result.validation_result.is_valid is True
    assert result.validation_result.claims_supported is True
    assert result.validation_result.citation_correct is True
    assert result.validation_result.addresses_question is True
    assert result.agent_trace[-1].status == "ok"


def test_validate_answer_is_valid_false_when_any_check_fails():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(
        '{"claims_supported": true, "addresses_question": true, '
        '"citation_correct": false, "unsupported_claims": [], "reasoning": "Wrong evidence cited."}'
    )

    result = validate_answer(state, generator)

    assert result.validation_result.is_valid is False
    assert result.validation_result.citation_correct is False


def test_validate_answer_populates_unsupported_claims_from_llm():
    state = AgentState(
        user_query="q",
        draft_answer="The system costs $50/month [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(
        '{"claims_supported": false, "addresses_question": true, "citation_correct": true, '
        '"unsupported_claims": ["The system costs $50/month."], "reasoning": "Price is not in the evidence."}'
    )

    result = validate_answer(state, generator)

    assert result.validation_result.claims_supported is False
    assert result.validation_result.unsupported_claims == ["The system costs $50/month."]
    assert result.validation_result.is_valid is False


# ---------------------------------------------------------------------------
# Deterministic citation override
# ---------------------------------------------------------------------------

def test_validate_answer_forces_citation_incorrect_on_out_of_range_marker_even_if_llm_says_correct():
    """The deterministic structural check must win over a wrong LLM verdict
    -- citation_correct can only be made stricter, never looser."""
    state = AgentState(
        user_query="q",
        draft_answer="This is supported by evidence [9].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(
        '{"claims_supported": true, "addresses_question": true, '
        '"citation_correct": true, "unsupported_claims": [], "reasoning": "Looks fine."}'
    )

    result = validate_answer(state, generator)

    assert result.validation_result.citation_correct is False
    assert result.validation_result.is_valid is False
    assert "[9]" in result.validation_result.citation_issues[0]


def test_validate_answer_citation_issues_empty_when_all_citations_in_range():
    state = AgentState(
        user_query="q",
        draft_answer="This is supported by evidence [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(VALID_OK)

    result = validate_answer(state, generator)

    assert result.validation_result.citation_issues == []


# ---------------------------------------------------------------------------
# Fail-safe fallback
# ---------------------------------------------------------------------------

def test_validate_answer_falls_back_on_generation_unavailable():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(side_effect=GenerationUnavailableError("Gemini is down"))

    result = validate_answer(state, generator)

    assert result.validation_result.is_valid is False
    assert result.validation_result.claims_supported is False
    assert result.validation_result.addresses_question is False
    assert result.agent_trace[-1].status == "fallback"


def test_validate_answer_fallback_still_uses_deterministic_citation_check():
    """Even in the LLM-failure fallback path, citation_correct is computed
    deterministically, not blindly set to False alongside everything else."""
    state = AgentState(
        user_query="q",
        draft_answer="This is supported by evidence [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(side_effect=GenerationUnavailableError("down"))

    result = validate_answer(state, generator)

    assert result.validation_result.citation_correct is True
    assert result.validation_result.is_valid is False  # still False overall (claims_supported is False)


def test_validate_answer_falls_back_on_malformed_json():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator("not json at all")

    result = validate_answer(state, generator)

    assert result.validation_result.is_valid is False
    assert result.agent_trace[-1].status == "fallback"


def test_validate_answer_falls_back_on_unexpected_extra_field():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(
        '{"claims_supported": true, "addresses_question": true, "citation_correct": true, '
        '"unsupported_claims": [], "reasoning": "ok", "confidence": 0.9}'
    )

    result = validate_answer(state, generator)

    assert result.agent_trace[-1].status == "fallback"


# ---------------------------------------------------------------------------
# Trace / retry_count representation / state handling
# ---------------------------------------------------------------------------

def test_validate_answer_trace_decision_includes_retry_count():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
        retry_count=1,
    )
    generator = make_generator(VALID_OK)

    result = validate_answer(state, generator)

    assert f"retry_count=1/{MAX_VALIDATION_RETRIES}" in result.agent_trace[-1].decision


def test_validate_answer_does_not_mutate_retry_count():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
        retry_count=1,
    )
    generator = make_generator(VALID_OK)

    result = validate_answer(state, generator)

    assert result.retry_count == 1


def test_validate_answer_does_not_mutate_input_state():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator(VALID_OK)

    validate_answer(state, generator)

    assert state.validation_result is None
    assert state.agent_trace == []


def test_validate_answer_preserves_existing_trace_entries():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
        agent_trace=[AgentTraceEntry(stage="answer", status="ok")],
    )
    generator = make_generator(VALID_OK)

    result = validate_answer(state, generator)

    assert len(result.agent_trace) == 2
    assert result.agent_trace[0].stage == "answer"
    assert result.agent_trace[1].stage == "validation"


def test_validate_answer_uses_state_trace_id_when_not_passed_explicitly():
    state = AgentState(
        user_query="q",
        draft_answer="An answer [1].",
        retrieved_evidence=[make_evidence()],
        trace_id="state-trace-id",
    )
    generator = make_generator(VALID_OK)

    validate_answer(state, generator)

    assert generator.generate.call_args.kwargs["trace_id"] == "state-trace-id"
