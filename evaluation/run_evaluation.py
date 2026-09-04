"""Run the Phase 5 evaluation dataset against the live RAG pipeline.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --experiment basic_vector_retrieval
    python -m evaluation.run_evaluation --experiment multi_query_retrieval --retrieval-only

This is READ-ONLY with respect to the vector store: it only calls
RAG.answer()/RAG.retrieve() (query + retrieve + generate), and never calls
store.reset(), store.delete_document(), or any ingestion function. It does
not modify app/rag.py's default behavior — RAG.answer()'s enable_* flags
default to the same always-on configuration as before, and this runner just
passes the selected experiment's flags through explicitly.

All five registered experiments in evaluation/experiments.py are runnable.
Each non-default experiment writes to its own results file
(evaluation/rag_results_<experiment>.json) so comparisons don't overwrite
each other; the default "final_pipeline" experiment keeps writing to
evaluation/rag_results.json, unchanged from before this flag existed.

--retrieval-only skips the final Gemini answer-generation call (query
rewriting, multi-query generation, retrieval, reranking, and source/citation
construction still run in full) so retrieval metrics -- and citation
correctness, which only needs retrieved source metadata -- can be measured
without spending an answer-generation call per question. It writes to a
separate *_retrieval_only.json file so it never overwrites a full run's
results, and is independent of --experiment: any of the five configurations
can be run in either mode.
"""

import argparse
import json
import time
from pathlib import Path

from app.rag import RAG
from evaluation.experiments import FINAL_PIPELINE, experiment_names, flags_for

EVALUATION_FILE = Path("evaluation/rag_evaluation.json")


def results_file_for(experiment_name: str, retrieval_only: bool = False) -> Path:
    suffix = "_retrieval_only" if retrieval_only else ""

    if experiment_name == FINAL_PIPELINE:
        return Path(f"evaluation/rag_results{suffix}.json")
    return Path(f"evaluation/rag_results_{experiment_name}{suffix}.json")


def build_experiment_flags(experiment_name: str, retrieval_only: bool) -> dict:
    """The retrieval-stage flags for `experiment_name`, plus
    enable_answer_generation set from --retrieval-only. Experiment selection
    (flags_for) is untouched -- retrieval-only is a separate, orthogonal
    concern merged in afterward."""
    experiment_flags = flags_for(experiment_name)
    experiment_flags["enable_answer_generation"] = not retrieval_only
    return experiment_flags

# Deliberately different from Chat/Retrieval Inspector's hardcoded top_k=3:
# this gives us a ranked candidate pool large enough to compute Recall/
# Precision/HitRate/MRR at both K=3 and K=5 from a single retrieve call,
# without a second pipeline invocation. The FINAL ANSWER and its citations
# are still generated from this same top_k's final_evidence, so this is a
# real (if slightly wider) top_k, not a synthetic one -- documented here so
# it isn't mistaken for a change to production Chat behavior, which is
# untouched and still hardcoded to top_k=3.
EVALUATION_TOP_K = 5
RETRIEVAL_METRIC_K_VALUES = [3, 5]

# Paces requests at ~12 RPM, safely under Gemini's 15 RPM free-tier limit.
# Applied between questions in main()'s loop only -- never inside
# run_question()'s timed region -- so it never contaminates
# total_latency_seconds or any retrieval/scoring behavior.
REQUEST_PACING_SECONDS = 5.0


def load_dataset() -> list[dict]:
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_corpus_coverage(rag: RAG, dataset: list[dict]) -> None:
    """Warn (don't fail) if the dataset references documents that are not
    currently indexed -- e.g. the corpus was reset since the dataset was
    written. We still run the evaluation; missing-document questions will
    simply score as retrieval misses, which is an honest result."""
    indexed_ids = {document["document_id"] for document in rag.store.list_documents()}
    referenced_ids = {
        source["document_id"]
        for question in dataset
        for source in question.get("supporting_sources", [])
    }
    missing = referenced_ids - indexed_ids

    if missing:
        print(
            "WARNING: the following document_ids are referenced by the evaluation "
            f"dataset but are not currently indexed in the vector store: {sorted(missing)}. "
            "Questions depending on them will score as retrieval/answer failures, not because "
            "the system is broken, but because the corpus doesn't match the dataset right now."
        )


def build_conversation_history(question: dict, results_by_id: dict) -> list[dict]:
    depends_on = question.get("depends_on")

    if not depends_on:
        return []

    prior_result = results_by_id.get(depends_on)

    if prior_result is None:
        print(
            f"WARNING: {question['id']} depends_on '{depends_on}', which hasn't been "
            "run yet (dataset ordering issue). Running with empty conversation history."
        )
        return []

    return [
        {"role": "user", "content": prior_result["question"]},
        {"role": "assistant", "content": prior_result["actual_answer"] or ""},
    ]


def run_question(
    rag: RAG,
    question: dict,
    conversation_history: list[dict],
    experiment_flags: dict,
) -> dict:
    start = time.perf_counter()

    try:
        result = rag.answer(
            question=question["question"],
            top_k=EVALUATION_TOP_K,
            conversation_history=conversation_history,
            **experiment_flags,
        )
        run_error = None
    except Exception as error:  # noqa: BLE001 - we want to record any failure, not crash the run
        result = None
        run_error = str(error)

    total_latency_seconds = time.perf_counter() - start

    if result is None:
        return {
            "id": question["id"],
            "question": question["question"],
            "category": question["category"],
            "should_abstain": question["should_abstain"],
            "actual_answer": None,
            "actual_sources": [],
            "retrieval_candidates": [],
            "final_evidence": [],
            "generation_error": None,
            "run_error": run_error,
            "total_latency_seconds": total_latency_seconds,
            "model_name": rag.generator.model_name,
        }

    return {
        "id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "should_abstain": question["should_abstain"],
        "actual_answer": result["answer"],
        "actual_sources": result["sources"],
        "retrieval_candidates": result["retrieval"]["candidates"],
        "final_evidence": result["retrieval"]["final_evidence"],
        "rewritten_question": result["retrieval"]["rewritten_question"],
        "search_queries": result["retrieval"]["search_queries"],
        "generation_error": result["error"],
        "run_error": run_error,
        "total_latency_seconds": total_latency_seconds,
        "model_name": rag.generator.model_name,
        "trace_id": result.get("trace_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=experiment_names(),
        default=FINAL_PIPELINE,
        help=f"Which Phase 5 experiment configuration to run (default: {FINAL_PIPELINE}).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Skip the final Gemini answer-generation call so only retrieval "
            "(and citation metadata) is evaluated, at a fraction of the Gemini "
            "call cost. Independent of --experiment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment
    retrieval_only = args.retrieval_only
    experiment_flags = build_experiment_flags(experiment_name, retrieval_only)
    results_file = results_file_for(experiment_name, retrieval_only=retrieval_only)

    dataset = load_dataset()

    print(f"Loaded {len(dataset)} evaluation questions from {EVALUATION_FILE}")
    print(f"Running experiment: {experiment_name} (flags: {experiment_flags})")
    if retrieval_only:
        print("Retrieval-only mode: the final Gemini answer-generation call will be skipped.")
    print(f"top_k={EVALUATION_TOP_K}, retrieval metrics computed at K={RETRIEVAL_METRIC_K_VALUES}")

    rag = RAG()
    check_corpus_coverage(rag, dataset)

    results = []
    results_by_id = {}

    for index, question in enumerate(dataset, start=1):
        print(f"\n[{index}/{len(dataset)}] ({question['category']}) {question['question']}")

        conversation_history = build_conversation_history(question, results_by_id)
        result = run_question(rag, question, conversation_history, experiment_flags)

        results.append(result)
        results_by_id[question["id"]] = result

        if result["run_error"]:
            print(f"  RUN ERROR: {result['run_error']}")
        else:
            print(f"  Answer: {result['actual_answer']}")
            print(f"  Latency: {result['total_latency_seconds']:.2f}s")

        # Pace requests between questions (not inside run_question()'s timed
        # region), and skip the wait after the last question.
        if index < len(dataset):
            time.sleep(REQUEST_PACING_SECONDS)

    with results_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": experiment_name,
                "experiment_flags": experiment_flags,
                "retrieval_only": retrieval_only,
                "evaluation_top_k": EVALUATION_TOP_K,
                "retrieval_metric_k_values": RETRIEVAL_METRIC_K_VALUES,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    report_flag = " --retrieval-only" if retrieval_only else ""
    print(f"\nEvaluation run complete. Raw results saved to: {results_file}")
    print(
        f"Run `python -m evaluation.report --experiment {experiment_name}{report_flag}` "
        "to compute metrics and produce the report."
    )


if __name__ == "__main__":
    main()
