"""Tests for the Streamlit-free chat-progress helpers (app/chat_ui_helpers.py).

Kept separate from streamlit_app.py specifically so these can be imported
and tested with plain pytest, no Streamlit runtime required.
"""

from app.chat_ui_helpers import agent_progress_outcome, find_run_by_trace_id


# ---------------------------------------------------------------------------
# find_run_by_trace_id
# ---------------------------------------------------------------------------

def test_find_run_by_trace_id_matches_correct_run():
    runs = [{"trace_id": "a", "route": "document_qa"}, {"trace_id": "b", "route": "research"}]
    assert find_run_by_trace_id(runs, "b") == {"trace_id": "b", "route": "research"}


def test_find_run_by_trace_id_returns_none_when_not_found():
    runs = [{"trace_id": "a"}]
    assert find_run_by_trace_id(runs, "missing") is None


def test_find_run_by_trace_id_returns_none_for_missing_trace_id():
    runs = [{"trace_id": "a"}]
    assert find_run_by_trace_id(runs, None) is None
    assert find_run_by_trace_id(runs, "") is None


def test_find_run_by_trace_id_returns_none_for_empty_runs_list():
    assert find_run_by_trace_id([], "a") is None


# ---------------------------------------------------------------------------
# agent_progress_outcome
# ---------------------------------------------------------------------------

def test_agent_progress_outcome_error_when_result_is_error():
    state, label = agent_progress_outcome({"is_error": True}, None)
    assert state == "error"
    assert "could not produce" in label


def test_agent_progress_outcome_error_when_no_result_at_all():
    # Covers the timeout/connection-failure path, where no result was ever received.
    state, label = agent_progress_outcome(None, None)
    assert state == "error"


def test_agent_progress_outcome_complete_with_route_and_no_retry():
    run = {"route": "document_qa", "retry_count": 0, "is_valid": True}
    state, label = agent_progress_outcome({"is_error": False}, run)
    assert state == "complete"
    assert "document_qa" in label
    assert "retry" not in label


def test_agent_progress_outcome_complete_mentions_retry_count():
    run = {"route": "research", "retry_count": 1, "is_valid": True}
    state, label = agent_progress_outcome({"is_error": False}, run)
    assert state == "complete"
    assert "1 retry" in label


def test_agent_progress_outcome_never_reports_error_for_a_produced_answer():
    # A run that finalized after exhausting its retry budget without passing
    # validation is still is_error=False in the chat bubble today (see
    # app/services/chat_service.py::_run_agent) -- this must stay "complete",
    # not silently become stricter than the rest of the app.
    run = {"route": "document_qa", "retry_count": 1, "is_valid": False}
    state, label = agent_progress_outcome({"is_error": False}, run)
    assert state == "complete"
    assert "validation flagged" in label


def test_agent_progress_outcome_leads_with_the_validation_caveat_not_a_trailing_note():
    # The caveat must be the start of the sentence, not an easily-missed
    # trailing parenthetical, so it survives label truncation and isn't
    # overshadowed by the green "complete" checkmark.
    run = {"route": "document_qa", "retry_count": 1, "is_valid": False}
    _, label = agent_progress_outcome({"is_error": False}, run)
    assert label.startswith("Answered, but validation flagged")


def test_agent_progress_outcome_handles_missing_run_record_gracefully():
    # The secondary GET /chats/{id}/runs lookup is best-effort -- if it
    # fails or finds nothing, the core outcome must still be reportable.
    state, label = agent_progress_outcome({"is_error": False}, None)
    assert state == "complete"
    assert "unknown" in label
