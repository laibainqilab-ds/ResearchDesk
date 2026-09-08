"""Unit tests for the Answer agent (app/agents/answer.py).

The Generator is always mocked -- no real Gemini call. These tests check
the no-evidence abstention path, that generation reuses app.rag's exact
prompt/citation logic (not a duplicate), failure handling, and trace
recording.
"""

from unittest.mock import Mock

from app.agents.answer import generate_answer
from app.agents.state import AgentState, AgentTraceEntry
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


# ---------------------------------------------------------------------------
# No-evidence abstention path
# ---------------------------------------------------------------------------

def test_generate_answer_returns_no_evidence_message_when_no_evidence():
    state = AgentState(user_query="a question")
    generator = make_generator()

    result = generate_answer(state, generator)

    assert result.draft_answer == NO_EVIDENCE_ANSWER


def test_generate_answer_skips_llm_call_when_no_evidence():
    state = AgentState(user_query="a question")
    generator = make_generator()

    generate_answer(state, generator)

    generator.generate.assert_not_called()


def test_generate_answer_records_no_evidence_trace_status():
    state = AgentState(user_query="a question")
    generator = make_generator()

    result = generate_answer(state, generator)

    assert result.agent_trace[-1].stage == "answer"
    assert result.agent_trace[-1].status == "no_evidence"


# ---------------------------------------------------------------------------
# Successful generation -- reuses app.rag's exact prompt/citation logic
# ---------------------------------------------------------------------------

def test_generate_answer_calls_generator_with_evidence_in_prompt():
    state = AgentState(
        user_query="What does the document say?",
        retrieved_evidence=[make_evidence(document="The system supports X.")],
    )
    generator = make_generator("The system supports X [1].")

    generate_answer(state, generator)

    prompt = generator.generate.call_args.args[0]
    assert "The system supports X." in prompt
    assert "What does the document say?" in prompt


def test_generate_answer_uses_original_user_query_not_rewritten_query():
    """Matches RAG.answer()'s existing behavior: the final prompt uses the
    user's original phrasing, not the retrieval-rewritten standalone query."""
    state = AgentState(
        user_query="What about it?",
        rewritten_query="What is Evo 2's context window?",
        retrieved_evidence=[make_evidence()],
    )
    generator = make_generator("An answer [1].")

    generate_answer(state, generator)

    prompt = generator.generate.call_args.args[0]
    assert "What about it?" in prompt
    assert "Evo 2's context window" not in prompt


def test_generate_answer_sets_draft_answer_on_success():
    state = AgentState(user_query="a question", retrieved_evidence=[make_evidence()])
    generator = make_generator("The generated answer [1].")

    result = generate_answer(state, generator)

    assert result.draft_answer == "The generated answer [1]."


def test_generate_answer_records_ok_trace_status_with_citation_summary():
    state = AgentState(user_query="a question", retrieved_evidence=[make_evidence()])
    generator = make_generator("The generated answer [1].")

    result = generate_answer(state, generator)

    entry = result.agent_trace[-1]
    assert entry.status == "ok"
    assert "1 valid" in entry.decision
    assert "0 invalid" in entry.decision


def test_generate_answer_citation_metadata_is_never_taken_from_generated_text():
    """Same guarantee as app.rag.extract_citations: citation metadata comes
    from the real evidence, never from whatever the model's text claims."""
    state = AgentState(
        user_query="a question",
        retrieved_evidence=[make_evidence(filename="real.pdf", page_number=1)],
    )
    generator = make_generator("According to fake-document.pdf, page 999 [1], the answer is X.")

    result = generate_answer(state, generator)

    assert result.draft_answer == "According to fake-document.pdf, page 999 [1], the answer is X."
    assert "fake-document.pdf" not in "".join(
        entry.decision or "" for entry in result.agent_trace
    )


# ---------------------------------------------------------------------------
# Generation failure
# ---------------------------------------------------------------------------

def test_generate_answer_handles_generation_unavailable():
    state = AgentState(user_query="a question", retrieved_evidence=[make_evidence()])
    generator = make_generator(side_effect=GenerationUnavailableError("Gemini is down"))

    result = generate_answer(state, generator)

    assert result.draft_answer is None
    assert result.agent_trace[-1].status == "failed"


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------

def test_generate_answer_does_not_mutate_input_state():
    state = AgentState(user_query="a question", retrieved_evidence=[make_evidence()])
    generator = make_generator("An answer.")

    generate_answer(state, generator)

    assert state.draft_answer is None
    assert state.agent_trace == []


def test_generate_answer_preserves_existing_trace_entries():
    state = AgentState(
        user_query="a question",
        retrieved_evidence=[make_evidence()],
        agent_trace=[AgentTraceEntry(stage="router", status="ok")],
    )
    generator = make_generator("An answer.")

    result = generate_answer(state, generator)

    assert len(result.agent_trace) == 2
    assert result.agent_trace[0].stage == "router"
    assert result.agent_trace[1].stage == "answer"


def test_generate_answer_uses_state_trace_id_when_not_passed_explicitly():
    state = AgentState(
        user_query="a question",
        retrieved_evidence=[make_evidence()],
        trace_id="state-trace-id",
    )
    generator = make_generator("An answer.")

    generate_answer(state, generator)

    assert generator.generate.call_args.kwargs["trace_id"] == "state-trace-id"
