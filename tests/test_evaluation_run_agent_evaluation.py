"""Unit tests for evaluation/run_agent_evaluation.py's pure helper logic.

run_agent_workflow (the only entry point into the Phase 9 agent pipeline
this runner is allowed to use) is always monkeypatched to a canned fake --
no real Gemini call, no real retrieval, no real graph execution happens in
this file. These tests check the *shaping* logic: does
run_question_via_agent() correctly translate an AgentState into the
existing per-question result-dict shape, and does it honestly mark the
retrieval-candidates limitation rather than inventing data.
"""

import time
from unittest.mock import Mock

import pytest

import evaluation.run_agent_evaluation as run_agent_evaluation
from app.agents.state import AgentState, AgentTraceEntry, RouteType, ValidationResult
from evaluation.run_agent_evaluation import (
    DEFAULT_PACING_SECONDS,
    run_question_via_agent,
    run_questions,
    select_questions,
)


def make_question(question_id="q001", category="direct", should_abstain=False):
    return {
        "id": question_id,
        "question": f"Question text for {question_id}?",
        "category": category,
        "should_abstain": should_abstain,
    }


def make_rag():
    rag = Mock()
    rag.generator = Mock()
    rag.generator.model_name = "gemini-test"
    return rag


def make_evidence(document_id="docA", chunk_id=0):
    return {
        "document_id": document_id,
        "filename": "a.pdf",
        "page_number": 1,
        "chunk_id": chunk_id,
        "rerank_score": 0.9,
        "document": "Evidence text.",
    }


def make_state(**overrides) -> AgentState:
    defaults = dict(
        user_query="Question text?",
        selected_route=RouteType.DOCUMENT_QA,
        retrieved_evidence=[make_evidence()],
        draft_answer="An answer [1].",
        final_answer="An answer [1].",
        retry_count=0,
        trace_id="trace-abc",
        validation_result=ValidationResult(
            is_valid=True,
            claims_supported=True,
            citation_correct=True,
            addresses_question=True,
            reasoning="Well grounded.",
        ),
        agent_trace=[
            AgentTraceEntry(stage="router", status="ok"),
            AgentTraceEntry(stage="retrieval", status="ok", tools_used=["search_documents"]),
            AgentTraceEntry(stage="answer", status="ok", decision="1 valid, 0 invalid citation(s)."),
            AgentTraceEntry(stage="validation", status="ok"),
        ],
    )
    defaults.update(overrides)
    return AgentState(**defaults)


# ---------------------------------------------------------------------------
# select_questions
# ---------------------------------------------------------------------------

def test_select_questions_returns_full_dataset_when_no_filter():
    dataset = [make_question("q001"), make_question("q002")]

    assert select_questions(dataset, None) == dataset


def test_select_questions_returns_full_dataset_when_empty_string_filter():
    dataset = [make_question("q001"), make_question("q002")]

    assert select_questions(dataset, "") == dataset


def test_select_questions_filters_and_preserves_requested_order():
    dataset = [make_question("q001"), make_question("q002"), make_question("q003")]

    result = select_questions(dataset, "q003,q001")

    assert [q["id"] for q in result] == ["q003", "q001"]


def test_select_questions_raises_on_unknown_id():
    dataset = [make_question("q001")]

    with pytest.raises(ValueError, match="q999"):
        select_questions(dataset, "q001,q999")


# ---------------------------------------------------------------------------
# run_question_via_agent -- successful run
# ---------------------------------------------------------------------------

def test_run_question_via_agent_preserves_question_metadata(monkeypatch):
    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", lambda **kwargs: make_state())
    question = make_question("q001", category="direct", should_abstain=False)

    result = run_question_via_agent(make_rag(), Mock(), question, [])

    assert result["id"] == "q001"
    assert result["question"] == question["question"]
    assert result["category"] == "direct"
    assert result["should_abstain"] is False


def test_run_question_via_agent_shapes_successful_result(monkeypatch):
    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", lambda **kwargs: make_state())

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert result["actual_answer"] == "An answer [1]."
    assert result["selected_route"] == "document_qa"
    assert result["retry_count"] == 0
    assert result["trace_id"] == "trace-abc"
    assert result["run_error"] is None
    assert result["generation_error"] is None
    assert len(result["agent_trace"]) == 4
    assert result["agent_trace"][0]["stage"] == "router"


def test_run_question_via_agent_validation_result_serialized_as_plain_dict(monkeypatch):
    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", lambda **kwargs: make_state())

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert isinstance(result["validation_result"], dict)
    assert result["validation_result"]["is_valid"] is True
    assert result["validation_result"]["reasoning"] == "Well grounded."


def test_run_question_via_agent_actual_sources_built_from_retrieved_evidence(monkeypatch):
    evidence = make_evidence(document_id="docXYZ", chunk_id=7)
    monkeypatch.setattr(
        run_agent_evaluation, "run_agent_workflow", lambda **kwargs: make_state(retrieved_evidence=[evidence])
    )

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert result["final_evidence"] == [evidence]
    assert len(result["actual_sources"]) == 1
    assert result["actual_sources"][0]["document_id"] == "docXYZ"
    assert result["actual_sources"][0]["chunk_id"] == 7
    assert result["actual_sources"][0]["citation_id"] == 1


# ---------------------------------------------------------------------------
# The retrieval-candidates honesty requirement
# ---------------------------------------------------------------------------

def test_run_question_via_agent_never_fabricates_retrieval_candidates(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation,
        "run_agent_workflow",
        lambda **kwargs: make_state(retrieved_evidence=[make_evidence(), make_evidence(chunk_id=1)]),
    )

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert result["retrieval_candidates"] is None
    assert result["retrieval_candidates_available"] is False


# ---------------------------------------------------------------------------
# Generation-error derivation from agent_trace
# ---------------------------------------------------------------------------

def test_run_question_via_agent_derives_generation_error_from_failed_answer_stage(monkeypatch):
    failing_state = make_state(
        draft_answer=None,
        final_answer=None,
        validation_result=None,
        agent_trace=[
            AgentTraceEntry(stage="router", status="ok"),
            AgentTraceEntry(stage="retrieval", status="ok"),
            AgentTraceEntry(stage="answer", status="failed", decision="Gemini is down."),
            AgentTraceEntry(stage="validation", status="skipped_no_answer"),
        ],
    )
    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", lambda **kwargs: failing_state)

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert result["generation_error"] == {"message": "Gemini is down."}
    assert result["actual_answer"] is None


def test_run_question_via_agent_no_generation_error_when_answer_stage_ok(monkeypatch):
    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", lambda **kwargs: make_state())

    result = run_question_via_agent(make_rag(), Mock(), make_question(), [])

    assert result["generation_error"] is None


# ---------------------------------------------------------------------------
# Workflow-level exception handling
# ---------------------------------------------------------------------------

def test_run_question_via_agent_handles_workflow_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", boom)

    result = run_question_via_agent(make_rag(), Mock(), make_question("q999"), [])

    assert result["run_error"] == "graph exploded"
    assert result["actual_answer"] is None
    assert result["actual_sources"] == []
    assert result["final_evidence"] == []
    assert result["selected_route"] is None
    assert result["retry_count"] is None
    assert result["validation_result"] is None
    assert result["agent_trace"] == []
    assert result["retrieval_candidates"] is None
    assert result["retrieval_candidates_available"] is False
    assert result["id"] == "q999"


# ---------------------------------------------------------------------------
# Dependency pass-through (to run_agent_workflow)
# ---------------------------------------------------------------------------

def test_run_question_via_agent_passes_generator_tools_and_history_through(monkeypatch):
    captured = {}

    def fake_run_agent_workflow(**kwargs):
        captured.update(kwargs)
        return make_state()

    monkeypatch.setattr(run_agent_evaluation, "run_agent_workflow", fake_run_agent_workflow)

    rag = make_rag()
    tools = Mock()
    history = [{"role": "user", "content": "earlier question"}]

    run_question_via_agent(rag, tools, make_question("q002"), history)

    assert captured["generator"] is rag.generator
    assert captured["tools"] is tools
    assert captured["conversation_context"] == history
    assert captured["user_query"] == "Question text for q002?"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_parse_args_defaults_to_none(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_agent_evaluation.py"])
    args = run_agent_evaluation.parse_args()

    assert args.question_ids is None


def test_parse_args_captures_question_ids(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_agent_evaluation.py", "--question-ids", "q001,q002"])
    args = run_agent_evaluation.parse_args()

    assert args.question_ids == "q001,q002"


def test_parse_args_pacing_defaults_to_default_constant(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_agent_evaluation.py"])
    args = run_agent_evaluation.parse_args()

    assert args.pacing_seconds == DEFAULT_PACING_SECONDS


def test_parse_args_pacing_can_be_overridden(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_agent_evaluation.py", "--pacing-seconds", "3.5"])
    args = run_agent_evaluation.parse_args()

    assert args.pacing_seconds == 3.5


def test_parse_args_pacing_can_be_disabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_agent_evaluation.py", "--pacing-seconds", "0"])
    args = run_agent_evaluation.parse_args()

    assert args.pacing_seconds == 0


# ---------------------------------------------------------------------------
# run_questions -- pacing behavior (time.sleep is always mocked: these tests
# must run instantly regardless of the pacing value under test)
# ---------------------------------------------------------------------------

def _fake_result(question_id: str) -> dict:
    return {
        "id": question_id,
        "question": f"Question text for {question_id}?",
        "category": "direct",
        "should_abstain": False,
        "actual_answer": "An answer.",
        "actual_sources": [],
        "final_evidence": [],
        "retrieval_candidates": None,
        "retrieval_candidates_available": False,
        "selected_route": "document_qa",
        "retry_count": 0,
        "validation_result": {"is_valid": True},
        "agent_trace": [],
        "generation_error": None,
        "run_error": None,
        "total_latency_seconds": 1.23,
        "model_name": "gemini-test",
        "trace_id": "trace-x",
    }


def test_run_questions_sleeps_between_but_not_after_last_question(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation, "run_question_via_agent", lambda rag, tools, q, hist: _fake_result(q["id"])
    )
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    questions = [make_question("q001"), make_question("q002"), make_question("q003")]

    run_questions(make_rag(), Mock(), questions, pacing_seconds=7.0)

    assert sleep_calls == [7.0, 7.0]  # exactly 2 sleeps for 3 questions, never after the last


def test_run_questions_does_not_sleep_when_pacing_is_zero(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation, "run_question_via_agent", lambda rag, tools, q, hist: _fake_result(q["id"])
    )
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    questions = [make_question("q001"), make_question("q002")]

    run_questions(make_rag(), Mock(), questions, pacing_seconds=0)

    assert sleep_calls == []


def test_run_questions_does_not_sleep_for_a_single_question(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation, "run_question_via_agent", lambda rag, tools, q, hist: _fake_result(q["id"])
    )
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    run_questions(make_rag(), Mock(), [make_question("q001")], pacing_seconds=DEFAULT_PACING_SECONDS)

    assert sleep_calls == []


def test_run_questions_default_pacing_used_when_not_specified(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation, "run_question_via_agent", lambda rag, tools, q, hist: _fake_result(q["id"])
    )
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    questions = [make_question("q001"), make_question("q002")]

    run_questions(make_rag(), Mock(), questions)  # pacing_seconds not passed

    assert sleep_calls == [DEFAULT_PACING_SECONDS]


def test_run_questions_returns_results_in_order(monkeypatch):
    monkeypatch.setattr(
        run_agent_evaluation, "run_question_via_agent", lambda rag, tools, q, hist: _fake_result(q["id"])
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    questions = [make_question("q003"), make_question("q001")]

    results = run_questions(make_rag(), Mock(), questions, pacing_seconds=0)

    assert [r["id"] for r in results] == ["q003", "q001"]


def test_run_questions_builds_conversation_history_for_dependent_questions(monkeypatch):
    captured_histories = {}

    def fake_run_question_via_agent(rag, tools, question, conversation_history):
        captured_histories[question["id"]] = conversation_history
        return _fake_result(question["id"])

    monkeypatch.setattr(run_agent_evaluation, "run_question_via_agent", fake_run_question_via_agent)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    base_question = make_question("q010")
    follow_up_question = {**make_question("q010b"), "depends_on": "q010"}

    run_questions(make_rag(), Mock(), [base_question, follow_up_question], pacing_seconds=0)

    assert captured_histories["q010"] == []
    assert captured_histories["q010b"] == [
        {"role": "user", "content": base_question["question"]},
        {"role": "assistant", "content": "An answer."},
    ]
