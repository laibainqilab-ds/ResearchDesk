"""Router agent: classifies a user query into one of the RouteType values.

Uses exactly one Gemini call per request. The raw response is never trusted
at face value -- it's parsed as JSON and validated through RouterDecision
(Pydantic, extra="forbid"), the same "never trust the model's output
directly" pattern app.rag already applies to citations (see
extract_citations in app/rag.py). Any parsing or validation failure falls
back deterministically to FALLBACK_ROUTE (document_qa) -- the same
single-pass retrieve-then-answer behavior Phases 1-8 already run, so a
misclassification degrades to already-battle-tested behavior rather than to
an untested code path.

Only depends on a Generator, not the full RAG object -- routing needs no
retrieval or vector-store access.
"""

import json
import logging
import re
import time

from pydantic import ValidationError

from app.agents.state import AgentState, AgentTraceEntry, RouteType, RouterDecision
from app.models.generator import Generator, GenerationUnavailableError
from app.observability import log_event

logger = logging.getLogger(__name__)

FALLBACK_ROUTE = RouteType.DOCUMENT_QA

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_ROUTE_DESCRIPTIONS = (
    "- document_qa: a single, direct question answerable from the documents "
    "in one retrieval pass.\n"
    "- research: the question requires breaking it into multiple "
    "sub-questions and gathering evidence for each before it can be "
    "answered.\n"
    "- summary: the user wants an overview or summary of a document or "
    "topic, not one specific fact.\n"
    "- comparison: the user wants two or more things (documents, concepts, "
    "entities) compared against each other."
)


def _format_conversation_context(conversation_context: list[dict]) -> str:
    if not conversation_context:
        return "(no prior conversation)"
    return "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in conversation_context
    )


def _parse_router_response(raw_text: str) -> RouterDecision:
    match = _JSON_BLOCK_PATTERN.search(raw_text or "")
    if not match:
        raise ValueError("No JSON object found in router response.")
    payload = json.loads(match.group(0))
    return RouterDecision.model_validate(payload)


def classify_route(
    state: AgentState,
    generator: Generator,
    trace_id: str | None = None,
) -> AgentState:
    """Classify state.user_query into a RouteType using one Gemini call.

    Returns a new AgentState (the input state is left unmodified) with
    `selected_route` set and one AgentTraceEntry appended to `agent_trace`.
    Never raises -- any failure to obtain a usable classification falls
    back to FALLBACK_ROUTE.
    """
    trace_id = trace_id or state.trace_id
    start = time.perf_counter()

    prompt = f"""
Classify the user's request into exactly one of these route types:
{_ROUTE_DESCRIPTIONS}

Conversation so far:
{_format_conversation_context(state.conversation_context)}

User's request:
{state.user_query}

Respond with only a JSON object in this exact shape, no other text:
{{"route": "<one of document_qa, research, summary, comparison>", "reasoning": "<one short sentence>"}}
"""

    log_event(trace_id, "router_classification_started", query=state.user_query)

    try:
        raw_response = generator.generate(prompt, trace_id=trace_id)
        decision = _parse_router_response(raw_response)
        status = "ok"
    except (GenerationUnavailableError, ValueError, ValidationError) as error:
        logger.warning(
            "Router classification failed, falling back to %s: %s",
            FALLBACK_ROUTE.value,
            error,
        )
        log_event(
            trace_id,
            "router_classification_failed",
            level=logging.WARNING,
            error=str(error),
            fallback_route=FALLBACK_ROUTE.value,
        )
        decision = RouterDecision(
            route=FALLBACK_ROUTE,
            reasoning=f"Fallback after classification failure: {error}",
        )
        status = "fallback"

    duration = time.perf_counter() - start

    log_event(
        trace_id,
        "router_classification_completed",
        level=logging.WARNING if status == "fallback" else logging.INFO,
        route=decision.route.value,
        status=status,
        duration_seconds=duration,
    )

    trace_entry = AgentTraceEntry(
        stage="router",
        status=status,
        duration_seconds=duration,
        decision=decision.route.value,
        tools_used=[],
    )

    return state.model_copy(
        update={
            "selected_route": decision.route,
            "agent_trace": [*state.agent_trace, trace_entry],
        }
    )
