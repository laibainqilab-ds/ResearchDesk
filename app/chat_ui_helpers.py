"""Pure, Streamlit-free helpers for showing agent-run progress in the chat
UI. Kept separate from app/streamlit_app.py so they're unit-testable
without a Streamlit runtime (importing app.streamlit_app directly isn't
possible in a plain pytest process -- it executes top-level st.* calls that
require a real Streamlit script context).
"""


def find_run_by_trace_id(runs: list[dict], trace_id: str | None) -> dict | None:
    """Find the persisted run record (from GET /chats/{chat_id}/runs)
    matching a message's trace_id, or None if not found."""
    if not trace_id:
        return None

    for run in runs:
        if run.get("trace_id") == trace_id:
            return run

    return None


def agent_progress_outcome(result: dict | None, run: dict | None) -> tuple[str, str]:
    """Determine the st.status() state ("complete" or "error") and a short
    summary label for a finished agent-mode chat request.

    Mirrors the chat bubble's own is_error semantics
    (app/services/chat_service.py): only an actual generation failure
    (is_error=True, or no result at all) is ever reported as "error" here --
    an answer that finalized after exhausting the retry budget without
    passing validation is still "complete", same as how it's already shown
    as a normal (non-warning) assistant message elsewhere in this app.
    """
    if result is None or result.get("is_error"):
        return "error", "The agent could not produce an answer."

    route = (run or {}).get("route") or "unknown"
    retry_count = (run or {}).get("retry_count") or 0
    is_valid = (run or {}).get("is_valid")

    retry_note = f" after {retry_count} retry" if retry_count else ""

    if is_valid is False:
        # Leads with the caveat, not a trailing parenthetical, so it can't
        # be missed by skimming past a green "complete" checkmark or by a
        # label getting truncated in a narrow layout.
        return "complete", f"Answered, but validation flagged this response (route: {route}{retry_note})."

    return "complete", f"Finished via the {route} route{retry_note}."
