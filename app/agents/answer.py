"""Answer agent: turns already-collected evidence into a grounded, cited answer.

Unlike the Router (classifies intent) and Research agent (decides what to
search for and whether it's enough), Answer makes no retrieval or strategy
decisions -- it only formats state.retrieved_evidence and generates one
answer from it. It never calls RetrievalTools, RAG.retrieve, or the vector
store, and makes at most one LLM call.

Reuses the exact Phase 1-8 evidence-formatting/prompt/citation logic from
app.rag (build_evidence_context, build_answer_prompt, extract_citations,
NO_EVIDENCE_ANSWER) rather than a second copy of it, so the answer this
produces is generated and validated identically to RAG.answer().
"""

import logging
import time

from app.agents.state import AgentState, AgentTraceEntry
from app.models.generator import Generator, GenerationUnavailableError
from app.observability import log_event
from app.rag import NO_EVIDENCE_ANSWER, build_answer_prompt, build_evidence_context, extract_citations

logger = logging.getLogger(__name__)


def generate_answer(
    state: AgentState,
    generator: Generator,
    trace_id: str | None = None,
) -> AgentState:
    """Generate state.draft_answer from state.retrieved_evidence.

    Returns a new AgentState (input is left unmodified) with draft_answer
    set and one AgentTraceEntry (stage="answer") appended to agent_trace.
    Never raises.
    """
    trace_id = trace_id or state.trace_id
    start = time.perf_counter()

    if not state.retrieved_evidence:
        log_event(trace_id, "answer_skipped_no_evidence", level=logging.WARNING)
        trace_entry = AgentTraceEntry(
            stage="answer",
            status="no_evidence",
            duration_seconds=time.perf_counter() - start,
            decision="No evidence available; abstained without a generation call.",
            tools_used=[],
        )
        return state.model_copy(
            update={
                "draft_answer": NO_EVIDENCE_ANSWER,
                "agent_trace": [*state.agent_trace, trace_entry],
            }
        )

    context, sources = build_evidence_context(state.retrieved_evidence)
    prompt = build_answer_prompt(state.user_query, context)

    try:
        answer = generator.generate(prompt, trace_id=trace_id)
        status = "ok"
    except GenerationUnavailableError as error:
        logger.warning("Answer generation unavailable: %s", error)
        log_event(trace_id, "answer_generation_failed", level=logging.WARNING, error=str(error))
        answer = None
        status = "failed"

    duration = time.perf_counter() - start

    if answer is not None:
        citations = extract_citations(answer, sources, trace_id=trace_id)
        decision = (
            f"{len(citations['valid'])} valid, {len(citations['invalid'])} invalid citation(s)."
        )
    else:
        decision = "Answer generation failed."

    log_event(
        trace_id,
        "answer_stage_completed",
        level=logging.WARNING if status == "failed" else logging.INFO,
        status=status,
        evidence_count=len(sources),
        duration_seconds=duration,
    )

    trace_entry = AgentTraceEntry(
        stage="answer",
        status=status,
        duration_seconds=duration,
        decision=decision,
        tools_used=[],
    )

    return state.model_copy(
        update={
            "draft_answer": answer,
            "agent_trace": [*state.agent_trace, trace_entry],
        }
    )
