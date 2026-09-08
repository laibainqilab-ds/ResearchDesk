"""Research agent: handles questions that need more than one retrieval pass.

Where the Router makes one classification decision and RetrievalTools
execute one deterministic search, this agent makes two decisions a fixed
function can't reasonably make: how to break a multi-part question into
subquestions worth searching separately, and whether the evidence gathered
so far actually covers the original question -- or whether one more
retrieval round is worth it.

Both decisions are single, bounded LLM calls, and every actual document
lookup goes through the existing deterministic RetrievalTools -- this agent
never calls RAG.retrieve, the vector store, or the embedder directly. Hard
caps make the loop impossible to run away, regardless of what the model
returns:

- at most one subquestion-decomposition call
- at most one sufficiency-check call (MAX_RESEARCH_ITERATIONS retrieval
  rounds total)
- at most MAX_INITIAL_SUBQUESTIONS + MAX_ADDITIONAL_SUBQUESTIONS search
  calls total

Any parse/generation failure degrades deterministically rather than
retrying or looping: a failed decomposition falls back to treating the
whole question as one subquestion; a failed sufficiency check stops the
loop and reports "sufficient" (hand off whatever evidence exists) rather
than looping on a broken judgment call.
"""

import json
import logging
import re
import time

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.state import AgentState, AgentTraceEntry, ResearchFindings
from app.agents.tools import RetrievalTools
from app.models.generator import Generator, GenerationUnavailableError
from app.observability import log_event

logger = logging.getLogger(__name__)

MAX_INITIAL_SUBQUESTIONS = 3
MAX_ADDITIONAL_SUBQUESTIONS = 2
MAX_RESEARCH_ITERATIONS = 2
SEARCH_TOP_K = 3

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class _SubquestionResponse(BaseModel):
    """Internal parse target for the decomposition call -- not part of
    AgentState; the final structured result stored on state is
    ResearchFindings."""

    model_config = ConfigDict(extra="forbid")

    subquestions: list[str] = Field(default_factory=list)


class _SufficiencyResponse(BaseModel):
    """Internal parse target for the sufficiency-check call."""

    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    additional_subquestions: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _parse_json_block(raw_text: str, model: type[BaseModel]):
    match = _JSON_BLOCK_PATTERN.search(raw_text or "")
    if not match:
        raise ValueError("No JSON object found in response.")
    payload = json.loads(match.group(0))
    return model.model_validate(payload)


def _dedupe_evidence(evidence_items: list[dict]) -> list[dict]:
    """Cross-subquestion dedupe by (document_id, chunk_id).

    Each RetrievalTools.search_documents() call already dedupes within
    itself (via RAG.retrieve), but different subquestions can independently
    surface the same chunk -- this collapses those across the whole
    research run.
    """
    seen = set()
    deduped = []
    for item in evidence_items:
        key = (item.get("document_id"), item.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _decompose(question: str, generator: Generator, trace_id: str | None) -> list[str]:
    prompt = f"""
Break the following question into at most {MAX_INITIAL_SUBQUESTIONS} smaller
search-friendly subquestions that together would help answer it. If the
question is already a single simple lookup, return it unchanged as the only
subquestion.

Question:
{question}

Respond with only a JSON object in this exact shape, no other text:
{{"subquestions": ["<subquestion 1>", "<subquestion 2>"]}}
"""
    raw_response = generator.generate(prompt, trace_id=trace_id)
    parsed = _parse_json_block(raw_response, _SubquestionResponse)

    subquestions = [q for q in parsed.subquestions if q.strip()][:MAX_INITIAL_SUBQUESTIONS]

    if not subquestions:
        raise ValueError("Decomposition returned no usable subquestions.")

    return subquestions


def _check_sufficiency(
    question: str, evidence: list[dict], generator: Generator, trace_id: str | None
) -> _SufficiencyResponse:
    evidence_text = "\n\n".join(
        f"[{index + 1}] {item.get('document')}" for index, item in enumerate(evidence)
    ) or "(no evidence retrieved)"

    prompt = f"""
Given the original question and the evidence gathered so far, decide whether
there is enough evidence to answer the question. If not, suggest at most
{MAX_ADDITIONAL_SUBQUESTIONS} additional search queries that would help.

Question:
{question}

Evidence gathered so far:
{evidence_text}

Respond with only a JSON object in this exact shape, no other text:
{{"sufficient": true, "additional_subquestions": [], "reasoning": "<one short sentence>"}}
"""
    raw_response = generator.generate(prompt, trace_id=trace_id)
    return _parse_json_block(raw_response, _SufficiencyResponse)


def run_research(
    state: AgentState,
    generator: Generator,
    tools: RetrievalTools,
    trace_id: str | None = None,
) -> AgentState:
    """Decompose the question, retrieve evidence for each subquestion via
    RetrievalTools, and iterate (bounded) if the evidence looks insufficient.

    Returns a new AgentState (input is left unmodified) with
    research_subquestions, retrieved_evidence, and research_findings set,
    and the research stages appended to agent_trace.
    """
    trace_id = trace_id or state.trace_id
    question = state.rewritten_query or state.user_query
    trace_entries: list[AgentTraceEntry] = []

    start = time.perf_counter()
    try:
        subquestions = _decompose(question, generator, trace_id)
        decompose_status = "ok"
    except (GenerationUnavailableError, ValueError, ValidationError) as error:
        logger.warning("Research decomposition failed, using original question: %s", error)
        log_event(trace_id, "research_decomposition_failed", level=logging.WARNING, error=str(error))
        subquestions = [question]
        decompose_status = "fallback"
    duration = time.perf_counter() - start

    log_event(
        trace_id,
        "research_decomposition_completed",
        level=logging.WARNING if decompose_status == "fallback" else logging.INFO,
        status=decompose_status,
        subquestion_count=len(subquestions),
        duration_seconds=duration,
    )
    trace_entries.append(
        AgentTraceEntry(
            stage="research_decompose",
            status=decompose_status,
            duration_seconds=duration,
            decision=f"{len(subquestions)} subquestion(s)",
            tools_used=[],
        )
    )

    all_subquestions = list(subquestions)
    all_evidence: list[dict] = []
    pending = subquestions
    iteration = 0
    sufficient = True
    reasoning = ""

    while pending:
        iteration += 1
        start = time.perf_counter()

        new_evidence = []
        for subquestion in pending:
            new_evidence.extend(
                tools.search_documents(subquestion, top_k=SEARCH_TOP_K, trace_id=trace_id)
            )
        all_evidence = _dedupe_evidence(all_evidence + new_evidence)
        duration = time.perf_counter() - start

        log_event(
            trace_id,
            "research_retrieval_completed",
            iteration=iteration,
            subquestion_count=len(pending),
            evidence_count=len(all_evidence),
            duration_seconds=duration,
        )
        trace_entries.append(
            AgentTraceEntry(
                stage="research_retrieve",
                status="ok",
                duration_seconds=duration,
                decision=f"{len(all_evidence)} evidence item(s) total",
                tools_used=["search_documents"],
            )
        )

        if iteration >= MAX_RESEARCH_ITERATIONS:
            reasoning = "Reached maximum research iterations; proceeding with the evidence gathered so far."
            break

        start = time.perf_counter()
        try:
            decision = _check_sufficiency(question, all_evidence, generator, trace_id)
            sufficiency_status = "ok"
            sufficient = decision.sufficient
            reasoning = decision.reasoning
        except (GenerationUnavailableError, ValueError, ValidationError) as error:
            logger.warning("Research sufficiency check failed, stopping research: %s", error)
            log_event(trace_id, "research_sufficiency_check_failed", level=logging.WARNING, error=str(error))
            sufficiency_status = "fallback"
            sufficient = True
            reasoning = f"Fallback after sufficiency-check failure: {error}"
            decision = None
        duration = time.perf_counter() - start

        log_event(
            trace_id,
            "research_sufficiency_completed",
            level=logging.WARNING if sufficiency_status == "fallback" else logging.INFO,
            status=sufficiency_status,
            sufficient=sufficient,
            duration_seconds=duration,
        )
        trace_entries.append(
            AgentTraceEntry(
                stage="research_sufficiency",
                status=sufficiency_status,
                duration_seconds=duration,
                decision="sufficient" if sufficient else "insufficient",
                tools_used=[],
            )
        )

        if sufficient or decision is None:
            break

        pending = [q for q in decision.additional_subquestions if q.strip()][:MAX_ADDITIONAL_SUBQUESTIONS]

        if not pending:
            sufficient = True
            break

        all_subquestions.extend(pending)

    findings = ResearchFindings(
        subquestions=all_subquestions,
        evidence=all_evidence,
        sufficient=sufficient,
        reasoning=reasoning,
    )

    return state.model_copy(
        update={
            "research_subquestions": all_subquestions,
            "retrieved_evidence": all_evidence,
            "research_findings": findings,
            "agent_trace": [*state.agent_trace, *trace_entries],
        }
    )
