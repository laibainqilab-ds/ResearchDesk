"""Validation agent: checks whether the draft answer is actually supported
by the evidence, cites it correctly, and addresses the question.

The Answer agent already runs extract_citations() to check that citation
markers are structurally in range. This agent goes further: it makes the
semantic judgment call extract_citations() cannot -- whether the claims in
the text are actually grounded in the evidence, whether a citation is
attached to the *right* evidence (not just a valid-looking number), and
whether the answer addresses what was asked.

The structural citation check is deterministic and unmodified (reused
directly from app.rag) and always runs; it can only make citation_correct
stricter than the LLM's verdict, never looser. `is_valid` is always
computed by this module, never trusted as a field the LLM reports about
itself.

Performs no retrieval and decides nothing about what to search for next.
Does not implement retries or loop back to any other stage -- retry_count
is read only to record it in the trace; incrementing it and routing back to
Retrieval/Research/Answer is the LangGraph workflow's job (not built yet).
"""

import json
import logging
import re
import time

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.state import AgentState, AgentTraceEntry, ValidationResult
from app.models.generator import Generator, GenerationUnavailableError
from app.observability import log_event
from app.rag import NO_EVIDENCE_ANSWER, build_evidence_context, extract_citations

logger = logging.getLogger(__name__)

# The retry ceiling the future LangGraph conditional edge will enforce when
# deciding whether to loop back to Retrieval/Research/Answer after a failed
# validation. This module does not perform retries itself.
MAX_VALIDATION_RETRIES = 1

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class _ValidationResponse(BaseModel):
    """Internal parse target for the validation call. `is_valid` is
    deliberately not part of this schema -- it's always computed by
    validate_answer(), never taken as a self-reported field from the LLM."""

    model_config = ConfigDict(extra="forbid")

    claims_supported: bool
    addresses_question: bool
    citation_correct: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _parse_validation_response(raw_text: str) -> _ValidationResponse:
    match = _JSON_BLOCK_PATTERN.search(raw_text or "")
    if not match:
        raise ValueError("No JSON object found in validation response.")
    payload = json.loads(match.group(0))
    return _ValidationResponse.model_validate(payload)


def _decision_summary(result: ValidationResult, retry_count: int) -> str:
    return (
        f"is_valid={result.is_valid} (claims_supported={result.claims_supported}, "
        f"citation_correct={result.citation_correct}, "
        f"addresses_question={result.addresses_question}); "
        f"retry_count={retry_count}/{MAX_VALIDATION_RETRIES}"
    )


def _finalize(
    state: AgentState,
    result: ValidationResult,
    status: str,
    duration: float,
    trace_id: str | None,
) -> AgentState:
    log_event(
        trace_id,
        "validation_completed",
        level=logging.WARNING if not result.is_valid else logging.INFO,
        status=status,
        is_valid=result.is_valid,
        duration_seconds=duration,
    )
    trace_entry = AgentTraceEntry(
        stage="validation",
        status=status,
        duration_seconds=duration,
        decision=_decision_summary(result, state.retry_count),
        tools_used=[],
    )
    return state.model_copy(
        update={
            "validation_result": result,
            "agent_trace": [*state.agent_trace, trace_entry],
        }
    )


def validate_answer(
    state: AgentState,
    generator: Generator,
    trace_id: str | None = None,
) -> AgentState:
    """Validate state.draft_answer against state.retrieved_evidence.

    Returns a new AgentState (input is left unmodified) with
    validation_result set and one AgentTraceEntry appended to agent_trace.
    Never raises. Makes at most one LLM call, and none at all when there is
    no answer to check or the answer is the deterministic abstention
    message.
    """
    trace_id = trace_id or state.trace_id
    start = time.perf_counter()

    if state.draft_answer is None:
        result = ValidationResult(
            is_valid=False,
            claims_supported=False,
            citation_correct=False,
            addresses_question=False,
            reasoning="No draft answer to validate.",
        )
        return _finalize(state, result, "skipped_no_answer", time.perf_counter() - start, trace_id)

    if state.draft_answer == NO_EVIDENCE_ANSWER:
        result = ValidationResult(
            is_valid=True,
            claims_supported=True,
            citation_correct=True,
            addresses_question=True,
            reasoning="Model correctly abstained due to insufficient evidence.",
        )
        return _finalize(state, result, "skipped_abstention", time.perf_counter() - start, trace_id)

    context, sources = build_evidence_context(state.retrieved_evidence)
    citation_check = extract_citations(state.draft_answer, sources, trace_id=trace_id)
    citation_issues = [
        f"Citation [{number}] does not correspond to any retrieved evidence."
        for number in citation_check["invalid"]
    ]
    structurally_valid_citations = not citation_issues

    prompt = f"""
Given the question, the answer that was generated, and the evidence it was
supposed to be grounded in, evaluate the answer.

Question:
{state.user_query}

Evidence:
{context}

Answer:
{state.draft_answer}

Check:
- claims_supported: does every factual claim in the answer come from the evidence above (no outside knowledge, no invented details)?
- addresses_question: does the answer actually answer the question that was asked?
- citation_correct: are the evidence numbers used in the answer actually the right evidence for the claims next to them (not just present, but relevant)?

Respond with only a JSON object in this exact shape, no other text:
{{"claims_supported": true, "addresses_question": true, "citation_correct": true, "unsupported_claims": [], "reasoning": "<one short sentence>"}}
"""

    try:
        raw_response = generator.generate(prompt, trace_id=trace_id)
        parsed = _parse_validation_response(raw_response)
        claims_supported = parsed.claims_supported
        addresses_question = parsed.addresses_question
        citation_correct = parsed.citation_correct and structurally_valid_citations
        unsupported_claims = parsed.unsupported_claims
        reasoning = parsed.reasoning
        status = "ok"
    except (GenerationUnavailableError, ValueError, ValidationError) as error:
        logger.warning("Validation check failed, failing safe: %s", error)
        log_event(trace_id, "validation_check_failed", level=logging.WARNING, error=str(error))
        claims_supported = False
        addresses_question = False
        citation_correct = structurally_valid_citations
        unsupported_claims = []
        reasoning = f"Fallback after validation-check failure: {error}"
        status = "fallback"

    is_valid = claims_supported and citation_correct and addresses_question

    result = ValidationResult(
        is_valid=is_valid,
        claims_supported=claims_supported,
        citation_correct=citation_correct,
        addresses_question=addresses_question,
        unsupported_claims=unsupported_claims,
        citation_issues=citation_issues,
        reasoning=reasoning,
    )

    return _finalize(state, result, status, time.perf_counter() - start, trace_id)
