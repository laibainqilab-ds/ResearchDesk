"""Unit tests for the Router agent (app/agents/router.py).

The Generator is always mocked -- no real Gemini call in any test. These
tests check classification parsing, the deterministic fallback path, trace
recording, and that classify_route never mutates its input state.
"""

import json
import logging
from unittest.mock import Mock

from app.agents.router import FALLBACK_ROUTE, classify_route
from app.agents.state import AgentState, RouteType
from app.models.generator import GenerationUnavailableError

RESEARCHDESK_LOGGER = "researchdesk"


def _log_payloads(caplog):
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == RESEARCHDESK_LOGGER
    ]


def make_generator(response_text: str | None = None, side_effect=None) -> Mock:
    generator = Mock()
    if side_effect is not None:
        generator.generate.side_effect = side_effect
    else:
        generator.generate.return_value = response_text
    return generator


# ---------------------------------------------------------------------------
# Successful classification
# ---------------------------------------------------------------------------

def test_classify_route_returns_valid_route_from_well_formed_json():
    state = AgentState(user_query="How does Evo 2 handle long sequences?")
    generator = make_generator('{"route": "document_qa", "reasoning": "Single fact lookup."}')

    result = classify_route(state, generator)

    assert result.selected_route == RouteType.DOCUMENT_QA


def test_classify_route_parses_json_wrapped_in_markdown_fence():
    state = AgentState(user_query="Compare Evo 1 and Evo 2.")
    generator = make_generator(
        '```json\n{"route": "comparison", "reasoning": "Two entities compared."}\n```'
    )

    result = classify_route(state, generator)

    assert result.selected_route == RouteType.COMPARISON


def test_classify_route_parses_response_with_surrounding_prose():
    state = AgentState(user_query="Summarize the FYDP proposal.")
    generator = make_generator(
        'Sure, here is the classification: {"route": "summary", "reasoning": "Overview requested."} Thanks.'
    )

    result = classify_route(state, generator)

    assert result.selected_route == RouteType.SUMMARY


def test_classify_route_trace_entry_status_ok_on_success():
    state = AgentState(user_query="What is Evo 2?")
    generator = make_generator('{"route": "document_qa", "reasoning": "Direct question."}')

    result = classify_route(state, generator)

    assert result.agent_trace[-1].status == "ok"
    assert result.agent_trace[-1].stage == "router"
    assert result.agent_trace[-1].decision == "document_qa"


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_classify_route_falls_back_on_invalid_route_value():
    state = AgentState(user_query="a question")
    generator = make_generator('{"route": "not_a_real_route", "reasoning": "n/a"}')

    result = classify_route(state, generator)

    assert result.selected_route == FALLBACK_ROUTE


def test_classify_route_falls_back_on_malformed_json():
    state = AgentState(user_query="a question")
    generator = make_generator("this is not json at all")

    result = classify_route(state, generator)

    assert result.selected_route == FALLBACK_ROUTE


def test_classify_route_falls_back_on_missing_route_field():
    state = AgentState(user_query="a question")
    generator = make_generator('{"reasoning": "forgot the route"}')

    result = classify_route(state, generator)

    assert result.selected_route == FALLBACK_ROUTE


def test_classify_route_falls_back_on_unexpected_extra_field():
    """A hallucinated extra field must not be silently accepted alongside a
    valid-looking route -- the whole response is untrusted, not partially
    trusted."""
    state = AgentState(user_query="a question")
    generator = make_generator(
        '{"route": "research", "reasoning": "n/a", "confidence": 0.9}'
    )

    result = classify_route(state, generator)

    assert result.selected_route == FALLBACK_ROUTE


def test_classify_route_falls_back_on_generation_unavailable():
    state = AgentState(user_query="a question")
    generator = make_generator(side_effect=GenerationUnavailableError("Gemini is down"))

    result = classify_route(state, generator)

    assert result.selected_route == FALLBACK_ROUTE


def test_classify_route_trace_entry_status_fallback_on_failure():
    state = AgentState(user_query="a question")
    generator = make_generator("garbage")

    result = classify_route(state, generator)

    assert result.agent_trace[-1].status == "fallback"
    assert result.agent_trace[-1].decision == FALLBACK_ROUTE.value


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------

def test_classify_route_does_not_mutate_input_state():
    state = AgentState(user_query="a question")
    generator = make_generator('{"route": "research", "reasoning": "n/a"}')

    classify_route(state, generator)

    assert state.selected_route is None
    assert state.agent_trace == []


def test_classify_route_preserves_existing_trace_entries():
    from app.agents.state import AgentTraceEntry

    state = AgentState(
        user_query="a question",
        agent_trace=[AgentTraceEntry(stage="previous_stage", status="ok")],
    )
    generator = make_generator('{"route": "document_qa", "reasoning": "n/a"}')

    result = classify_route(state, generator)

    assert len(result.agent_trace) == 2
    assert result.agent_trace[0].stage == "previous_stage"
    assert result.agent_trace[1].stage == "router"


def test_classify_route_includes_conversation_context_in_prompt():
    state = AgentState(
        user_query="What about the second one?",
        conversation_context=[
            {"role": "user", "content": "Tell me about Evo 1 and Evo 2."},
        ],
    )
    generator = make_generator('{"route": "document_qa", "reasoning": "n/a"}')

    classify_route(state, generator)

    prompt = generator.generate.call_args.args[0]
    assert "Evo 1 and Evo 2" in prompt


def test_classify_route_uses_state_trace_id_when_not_passed_explicitly():
    state = AgentState(user_query="a question", trace_id="state-trace-id")
    generator = make_generator('{"route": "document_qa", "reasoning": "n/a"}')

    classify_route(state, generator)

    assert generator.generate.call_args.kwargs["trace_id"] == "state-trace-id"


def test_classify_route_prefers_explicit_trace_id_over_state_trace_id():
    state = AgentState(user_query="a question", trace_id="state-trace-id")
    generator = make_generator('{"route": "document_qa", "reasoning": "n/a"}')

    classify_route(state, generator, trace_id="explicit-trace-id")

    assert generator.generate.call_args.kwargs["trace_id"] == "explicit-trace-id"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_classify_route_logs_completion_as_info_on_success(caplog):
    state = AgentState(user_query="a question", trace_id="log-trace")
    generator = make_generator('{"route": "document_qa", "reasoning": "n/a"}')

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        classify_route(state, generator)

    events = [p for p in _log_payloads(caplog) if p["event"] == "router_classification_completed"]
    assert len(events) == 1
    assert events[0]["level"] == "INFO"
    assert events[0]["route"] == "document_qa"


def test_classify_route_logs_failure_and_fallback_as_warning(caplog):
    state = AgentState(user_query="a question", trace_id="log-trace")
    generator = make_generator("garbage")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        classify_route(state, generator)

    failed_events = [p for p in _log_payloads(caplog) if p["event"] == "router_classification_failed"]
    completed_events = [p for p in _log_payloads(caplog) if p["event"] == "router_classification_completed"]

    assert len(failed_events) == 1
    assert len(completed_events) == 1
    assert completed_events[0]["level"] == "WARNING"
    assert completed_events[0]["route"] == FALLBACK_ROUTE.value
