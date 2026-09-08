"""Unit tests for app/agents/trace_view.py -- pure formatting/summarization
helpers for the Streamlit Agent Trace page. No Streamlit import anywhere in
this module or the one under test, so these run like any other unit test.
"""

from app.agents.state import AgentState, AgentTraceEntry, RouteType, ValidationResult
from app.agents.trace_view import (
    build_trace_rows,
    format_duration,
    format_status,
    format_trace_entry,
    summarize_workflow_run,
)


# ---------------------------------------------------------------------------
# format_status
# ---------------------------------------------------------------------------

def test_format_status_known_statuses():
    assert format_status("ok") == "✅ ok"
    assert format_status("fallback") == "⚠️ fallback"
    assert format_status("failed") == "❌ failed"
    assert format_status("no_evidence") == "ℹ️ no evidence"
    assert format_status("skipped_no_answer") == "ℹ️ skipped (no answer)"
    assert format_status("skipped_abstention") == "ℹ️ abstained"


def test_format_status_unknown_status_falls_back_to_raw_text():
    assert format_status("something_new") == "• something_new"


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------

def test_format_duration_none_is_not_available():
    assert format_duration(None) == "n/a"


def test_format_duration_sub_second_shown_in_milliseconds():
    assert format_duration(0.1234) == "123 ms"


def test_format_duration_one_second_or_more_shown_in_seconds():
    assert format_duration(1.5) == "1.50 s"


def test_format_duration_zero_shown_in_milliseconds():
    assert format_duration(0.0) == "0 ms"


# ---------------------------------------------------------------------------
# format_trace_entry / build_trace_rows
# ---------------------------------------------------------------------------

def test_format_trace_entry_includes_all_fields():
    entry = AgentTraceEntry(
        stage="retrieval",
        status="ok",
        duration_seconds=0.25,
        decision="3 evidence item(s)",
        tools_used=["search_documents"],
    )

    row = format_trace_entry(entry)

    assert row == {
        "stage": "retrieval",
        "status": "✅ ok",
        "duration": "250 ms",
        "decision": "3 evidence item(s)",
        "tools_used": "search_documents",
    }


def test_format_trace_entry_handles_missing_decision_and_tools():
    entry = AgentTraceEntry(stage="router", status="ok")

    row = format_trace_entry(entry)

    assert row["decision"] == ""
    assert row["tools_used"] == "none"


def test_format_trace_entry_joins_multiple_tools():
    entry = AgentTraceEntry(stage="research_retrieve", status="ok", tools_used=["search_documents", "get_document_page"])

    row = format_trace_entry(entry)

    assert row["tools_used"] == "search_documents, get_document_page"


def test_build_trace_rows_preserves_order():
    trace = [
        AgentTraceEntry(stage="router", status="ok"),
        AgentTraceEntry(stage="retrieval", status="ok"),
        AgentTraceEntry(stage="answer", status="failed"),
        AgentTraceEntry(stage="validation", status="skipped_no_answer"),
    ]

    rows = build_trace_rows(trace)

    assert [row["stage"] for row in rows] == ["router", "retrieval", "answer", "validation"]
    assert rows[2]["status"] == "❌ failed"


def test_build_trace_rows_empty_trace_returns_empty_list():
    assert build_trace_rows([]) == []


# ---------------------------------------------------------------------------
# summarize_workflow_run
# ---------------------------------------------------------------------------

def test_summarize_workflow_run_with_no_validation_result_yet():
    state = AgentState(user_query="q")

    summary = summarize_workflow_run(state)

    assert summary["route"] == "n/a"
    assert summary["retry_count"] == 0
    assert summary["is_valid"] is None
    assert summary["unsupported_claims"] == []
    assert summary["citation_issues"] == []


def test_summarize_workflow_run_with_valid_result():
    state = AgentState(
        user_query="q",
        selected_route=RouteType.RESEARCH,
        retry_count=1,
        validation_result=ValidationResult(
            is_valid=True,
            claims_supported=True,
            citation_correct=True,
            addresses_question=True,
            reasoning="Well grounded.",
        ),
    )

    summary = summarize_workflow_run(state)

    assert summary["route"] == "research"
    assert summary["retry_count"] == 1
    assert summary["is_valid"] is True
    assert summary["validation_reasoning"] == "Well grounded."


def test_summarize_workflow_run_surfaces_unsupported_claims_and_citation_issues():
    state = AgentState(
        user_query="q",
        validation_result=ValidationResult(
            is_valid=False,
            claims_supported=False,
            citation_correct=False,
            addresses_question=True,
            unsupported_claims=["The price is $50/month."],
            citation_issues=["Citation [9] does not correspond to any retrieved evidence."],
        ),
    )

    summary = summarize_workflow_run(state)

    assert summary["unsupported_claims"] == ["The price is $50/month."]
    assert summary["citation_issues"] == ["Citation [9] does not correspond to any retrieved evidence."]
