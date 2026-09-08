"""Run a subset of the Phase 5 evaluation dataset through the Phase 9
multi-agent workflow (`app.agents.graph.run_agent_workflow`), instead of the
baseline `RAG.answer()` chain that `evaluation/run_evaluation.py` exercises.

Usage:
    python -m evaluation.run_agent_evaluation
    python -m evaluation.run_agent_evaluation --question-ids q001,q045a,q045b,q032,q048

This reuses, unmodified:
    - evaluation/rag_evaluation.json (the dataset)
    - evaluation.run_evaluation.load_dataset / build_conversation_history /
      check_corpus_coverage
    - app.agents.graph.run_agent_workflow as the ONLY way this script drives
      the agent pipeline -- Router/Research/Answer/Validation/RetrievalTools
      are never called directly here.

KNOWN LIMITATION -- read before interpreting any results this produces:

    AgentState.retrieved_evidence only contains the already-selected top-k
    evidence actually used for generation (currently top_k=3, hardcoded as
    RETRIEVAL_TOP_K in app/agents/graph.py), not the full ranked candidate
    pool that RAG.retrieve() returns and that evaluation/metrics.py's
    Recall/Precision/HitRate@K and MRR are designed to score against
    multiple K cuts.

    This runner does NOT populate `retrieval_candidates` from that evidence
    and does NOT call metrics.evaluate_retrieval() / report.build_retrieval_
    report() on these results. Feeding a top-3-only list into those
    functions would silently produce Recall@5/Precision@5 numbers that look
    like real measurements but actually only ever had 3 items to find
    anything in -- a false, not just incomplete, measurement. Every result
    record instead carries `retrieval_candidates: None` and
    `retrieval_candidates_available: False` so this is impossible to miss
    downstream.

    `final_evidence` IS populated correctly (it is exactly the evidence the
    answer was generated from), so metrics that only need evidence *text*
    (faithfulness) or the system's actual returned sources (citation
    correctness) remain valid and are computed the normal way by
    evaluation.report's existing functions.
"""

import argparse
import json
import time
from pathlib import Path

from app.agents.graph import run_agent_workflow
from app.agents.state import AgentState
from app.agents.tools import RetrievalTools
from app.rag import RAG, build_evidence_context
from evaluation.run_evaluation import build_conversation_history, check_corpus_coverage, load_dataset

RESULTS_FILE = Path("evaluation/rag_results_multi_agent.json")

# Paces the START of each new question's run_agent_workflow() call, not
# calls *within* a single run (Router/Research/Answer/Validation are never
# individually paced -- see the module docstring). A single question can
# itself issue several sequential Gemini calls (router + retrieval/research
# + answer + validation, up to roughly double that if one validation retry
# happens), so this is a partial mitigation for Gemini's free-tier
# rate limit, not a guarantee against every possible burst within one
# question. 12s keeps the *start* of consecutive questions under 5
# requests/minute even in the worst case -- the free-tier quota this
# runner has actually hit in practice (RESOURCE_EXHAUSTED, quotaValue: 5,
# GenerateRequestsPerMinutePerProjectPerModel-FreeTier). Configurable via
# --pacing-seconds; set to 0 to disable entirely for a faster/paid-tier run.
DEFAULT_PACING_SECONDS = 12.0


def select_questions(dataset: list[dict], question_ids: str | None) -> list[dict]:
    """Filter the dataset to an explicit, ordered list of question IDs, or
    return the whole dataset unchanged if no filter is given.

    Raises ValueError (not a silent empty result) if an unknown ID is
    requested, so a typo in --question-ids fails loudly rather than quietly
    running fewer questions than intended.
    """
    if not question_ids:
        return dataset

    wanted_ids = [question_id.strip() for question_id in question_ids.split(",") if question_id.strip()]
    dataset_by_id = {question["id"]: question for question in dataset}

    missing_ids = [question_id for question_id in wanted_ids if question_id not in dataset_by_id]
    if missing_ids:
        raise ValueError(f"Unknown question id(s): {missing_ids}")

    return [dataset_by_id[question_id] for question_id in wanted_ids]


def _generation_error_from_trace(state: AgentState) -> dict | None:
    """Derive a RAG.answer()-shaped {"message": ...} error dict from the
    answer stage's trace entry, since AgentState (unlike RAG.answer()'s
    return value) does not carry a dedicated error field of its own."""
    for entry in state.agent_trace:
        if entry.stage == "answer" and entry.status == "failed":
            return {"message": entry.decision or "Answer generation failed."}
    return None


def run_question_via_agent(
    rag: RAG,
    tools: RetrievalTools,
    question: dict,
    conversation_history: list[dict],
) -> dict:
    """Run one dataset question through run_agent_workflow() and shape the
    result into the same per-question result-dict keys
    evaluation/run_evaluation.py's run_question() already produces, plus
    additive Phase-9-only fields (selected_route, retry_count,
    validation_result, agent_trace, trace_id).

    `actual_sources` is reconstructed via build_evidence_context() (the
    same pure function the Answer/Validation agents already use) since
    AgentState does not store a separate sources/citations field -- this is
    the "smallest adapter necessary", not new evidence-formatting logic.
    """
    start = time.perf_counter()

    try:
        state = run_agent_workflow(
            user_query=question["question"],
            generator=rag.generator,
            tools=tools,
            conversation_context=conversation_history,
        )
        run_error = None
    except Exception as error:  # noqa: BLE001 - record any failure, don't crash the run
        state = None
        run_error = str(error)

    total_latency_seconds = time.perf_counter() - start

    base_result = {
        "id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "should_abstain": question["should_abstain"],
        "retrieval_candidates": None,
        "retrieval_candidates_available": False,
        "run_error": run_error,
        "total_latency_seconds": total_latency_seconds,
        "model_name": rag.generator.model_name,
    }

    if state is None:
        return {
            **base_result,
            "actual_answer": None,
            "actual_sources": [],
            "final_evidence": [],
            "selected_route": None,
            "retry_count": None,
            "validation_result": None,
            "agent_trace": [],
            "generation_error": None,
            "trace_id": None,
        }

    _, actual_sources = build_evidence_context(state.retrieved_evidence)

    return {
        **base_result,
        "actual_answer": state.final_answer,
        "actual_sources": actual_sources,
        "final_evidence": state.retrieved_evidence,
        "selected_route": state.selected_route.value if state.selected_route else None,
        "retry_count": state.retry_count,
        "validation_result": state.validation_result.model_dump() if state.validation_result else None,
        "agent_trace": [entry.model_dump() for entry in state.agent_trace],
        "generation_error": _generation_error_from_trace(state),
        "trace_id": state.trace_id,
    }


def run_questions(
    rag: RAG,
    tools: RetrievalTools,
    questions: list[dict],
    pacing_seconds: float = DEFAULT_PACING_SECONDS,
) -> list[dict]:
    """Run each question in order via run_question_via_agent(), printing
    per-question progress, and sleeping `pacing_seconds` between the START
    of each question's run (never after the last one, never when
    pacing_seconds <= 0).

    The sleep happens strictly between questions here, outside
    run_question_via_agent()'s own timed region, so it never contaminates
    any individual question's total_latency_seconds.
    """
    results: list[dict] = []
    results_by_id: dict[str, dict] = {}

    for index, question in enumerate(questions, start=1):
        print(f"\n[{index}/{len(questions)}] ({question['category']}) {question['question']}")

        conversation_history = build_conversation_history(question, results_by_id)
        result = run_question_via_agent(rag, tools, question, conversation_history)

        results.append(result)
        results_by_id[question["id"]] = result

        if result["run_error"]:
            print(f"  RUN ERROR: {result['run_error']}")
        else:
            print(f"  Route: {result['selected_route']}")
            print(f"  Retry count: {result['retry_count']}")
            if result["validation_result"]:
                print(f"  Validation is_valid: {result['validation_result']['is_valid']}")
            print(f"  Answer: {result['actual_answer']}")
            print(f"  Latency: {result['total_latency_seconds']:.2f}s")

        if index < len(questions) and pacing_seconds > 0:
            time.sleep(pacing_seconds)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset question IDs to run, in the given "
            "order (default: the whole dataset). Use this to validate on a "
            "tiny subset before a full run."
        ),
    )
    parser.add_argument(
        "--pacing-seconds",
        type=float,
        default=DEFAULT_PACING_SECONDS,
        help=(
            f"Seconds to wait between questions, to stay under Gemini's "
            f"free-tier rate limit (default: {DEFAULT_PACING_SECONDS}). Each "
            "question can make several sequential Gemini calls internally "
            "(router, answer, validation, and possibly a retry of those), so "
            "this paces the start of each new question, not calls within one. "
            "Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset()
    questions = select_questions(dataset, args.question_ids)

    print(f"Loaded {len(dataset)} evaluation questions from evaluation/rag_evaluation.json")
    print(f"Running {len(questions)} question(s) through the Phase 9 agent workflow (run_agent_workflow).")
    if args.question_ids:
        print(f"Subset: {args.question_ids}")
    if args.pacing_seconds > 0:
        print(f"Pacing {args.pacing_seconds}s between questions (Gemini free-tier rate-limit mitigation).")
    else:
        print("Pacing disabled (--pacing-seconds 0).")

    rag = RAG()
    check_corpus_coverage(rag, dataset)
    tools = RetrievalTools(rag)

    results = run_questions(rag, tools, questions, pacing_seconds=args.pacing_seconds)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pipeline": "multi_agent",
                "question_ids": [question["id"] for question in questions],
                "full_dataset_size": len(dataset),
                "pacing_seconds": args.pacing_seconds,
                "retrieval_candidates_available": False,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nAgent evaluation run complete. Raw results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
