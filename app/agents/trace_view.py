"""Pure formatting/summarization helpers for displaying one agent workflow
run (an AgentState produced by app.agents.graph.run_agent_workflow) in
Streamlit.

Deliberately has no `streamlit` import: these are plain functions over
AgentState/AgentTraceEntry, unit-testable without a running Streamlit
script context. The Streamlit page itself only calls these and renders the
result -- it stays thin rendering glue, the same separation of pure logic
from UI already used elsewhere in app/streamlit_app.py (format_score,
location_label).
"""

from app.agents.state import AgentState, AgentTraceEntry

STATUS_LABELS = {
    "ok": "✅ ok",
    "fallback": "⚠️ fallback",
    "failed": "❌ failed",
    "no_evidence": "ℹ️ no evidence",
    "skipped_no_answer": "ℹ️ skipped (no answer)",
    "skipped_abstention": "ℹ️ abstained",
}


def format_status(status: str) -> str:
    return STATUS_LABELS.get(status, f"• {status}")


def format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "n/a"
    if duration_seconds < 1:
        return f"{duration_seconds * 1000:.0f} ms"
    return f"{duration_seconds:.2f} s"


def format_trace_entry(entry: AgentTraceEntry) -> dict:
    """One display-ready row for a single agent_trace entry."""
    return {
        "stage": entry.stage,
        "status": format_status(entry.status),
        "duration": format_duration(entry.duration_seconds),
        "decision": entry.decision or "",
        "tools_used": ", ".join(entry.tools_used) if entry.tools_used else "none",
    }


def build_trace_rows(agent_trace: list[AgentTraceEntry]) -> list[dict]:
    return [format_trace_entry(entry) for entry in agent_trace]


def summarize_workflow_run(state: AgentState) -> dict:
    """A compact summary of one full workflow run, for the top-of-page
    metrics: route, retries used, and the final validation verdict."""
    validation = state.validation_result

    return {
        "route": state.selected_route.value if state.selected_route else "n/a",
        "retry_count": state.retry_count,
        "is_valid": validation.is_valid if validation else None,
        "validation_reasoning": validation.reasoning if validation else "",
        "unsupported_claims": validation.unsupported_claims if validation else [],
        "citation_issues": validation.citation_issues if validation else [],
    }
