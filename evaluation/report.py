"""Aggregate an evaluation run into metrics and a human-readable report.

Usage:
    python -m evaluation.report
    python -m evaluation.report --experiment basic_vector_retrieval

Reads the raw run produced by `python -m evaluation.run_evaluation
[--experiment ...]` and the dataset it was run against, computes
retrieval/answer/citation/abstention metrics via evaluation/metrics.py (no
LLM judge, fully deterministic), and writes a report JSON plus a printed
summary. The default experiment ("final_pipeline") reads/writes the same
evaluation/rag_results.json and evaluation/rag_report.json paths as before
this flag existed; other experiments read/write their own
rag_results_<experiment>.json / rag_report_<experiment>.json files so
comparisons don't overwrite each other.
"""

import argparse
import json
import statistics
from pathlib import Path

from evaluation import metrics
from evaluation.experiments import FINAL_PIPELINE, experiment_names

EVALUATION_FILE = Path("evaluation/rag_evaluation.json")

ANSWER_QUALITY_NA = "N/A — retrieval-only run"


def results_file_for(experiment_name: str, retrieval_only: bool = False) -> Path:
    suffix = "_retrieval_only" if retrieval_only else ""

    if experiment_name == FINAL_PIPELINE:
        return Path(f"evaluation/rag_results{suffix}.json")
    return Path(f"evaluation/rag_results_{experiment_name}{suffix}.json")


def report_file_for(experiment_name: str, retrieval_only: bool = False) -> Path:
    suffix = "_retrieval_only" if retrieval_only else ""

    if experiment_name == FINAL_PIPELINE:
        return Path(f"evaluation/rag_report{suffix}.json")
    return Path(f"evaluation/rag_report_{experiment_name}{suffix}.json")


def load_dataset() -> dict:
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return {question["id"]: question for question in json.load(file)}


def load_results(results_file: Path) -> dict:
    with results_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dataset_summary(dataset_by_id: dict) -> dict:
    questions = list(dataset_by_id.values())

    by_category: dict[str, int] = {}
    for question in questions:
        by_category[question["category"]] = by_category.get(question["category"], 0) + 1

    documents = sorted({
        source["filename"]
        for question in questions
        for source in question.get("supporting_sources", [])
    })

    answerable = sum(1 for question in questions if not question["should_abstain"])

    return {
        "total_questions": len(questions),
        "by_category": by_category,
        "answerable_questions": answerable,
        "unanswerable_questions": len(questions) - answerable,
        "documents_represented": documents,
    }


def build_retrieval_report(dataset_by_id: dict, results: list[dict], k_values: list[int]) -> dict:
    per_k = {k: {"recall": [], "precision": [], "hit_rate": []} for k in k_values}
    mrr_values = []
    per_question = {}

    for result in results:
        question = dataset_by_id[result["id"]]
        supporting_sources = question.get("supporting_sources", [])

        if not supporting_sources:
            continue  # unanswerable questions have no retrieval ground truth

        candidates = result.get("retrieval_candidates", [])
        question_metrics = metrics.evaluate_retrieval(candidates, supporting_sources, k_values)
        per_question[result["id"]] = question_metrics

        if question_metrics["mrr"] is not None:
            mrr_values.append(question_metrics["mrr"])

        for k in k_values:
            for metric_name in ("recall", "precision", "hit_rate"):
                value = question_metrics[f"{metric_name}_at_{k}"]
                if value is not None:
                    per_k[k][metric_name].append(value)

    aggregate = {
        f"{metric_name}_at_{k}": metrics.average_metric(values) if values else None
        for k, metric_group in per_k.items()
        for metric_name, values in metric_group.items()
    }
    aggregate["mrr"] = metrics.average_metric(mrr_values)
    aggregate["questions_evaluated"] = len(per_question)

    return {"aggregate": aggregate, "per_question": per_question}


def build_answer_report(dataset_by_id: dict, results: list[dict], retrieval_only: bool = False) -> dict:
    """Citation correctness only needs `actual_sources` (built from retrieved
    chunk metadata, independent of whether an answer was generated), so it is
    still scored in retrieval-only mode. Correctness and faithfulness both
    require real generated answer text to mean anything -- in retrieval-only
    mode `actual_answer` is always None, so scoring them would just label
    every question "incorrect"/"unscored" for a reason that has nothing to do
    with system quality. They're reported as ANSWER_QUALITY_NA instead."""
    correctness_counts = {"correct": 0, "partial": 0, "incorrect": 0, "unscored": 0}
    faithfulness_counts = {"grounded": 0, "unsupported": 0, "unscored": 0}
    citation_counts = {"correct": 0, "incorrect": 0, "no_citations": 0, "not_applicable": 0}
    per_question = {}

    for result in results:
        question = dataset_by_id[result["id"]]

        if question["should_abstain"]:
            continue  # correctness/faithfulness/citations are scored for answerable questions only

        key_facts = question.get("key_facts", [])
        actual_answer = result.get("actual_answer")

        citation_status = metrics.score_citations(
            result.get("actual_sources", []), question.get("supporting_sources", [])
        )
        citation_counts[citation_status] += 1

        entry = {"citation_status": citation_status}

        if retrieval_only:
            entry["correctness"] = None
            entry["faithfulness"] = None
            entry["key_facts_found"] = []
        else:
            correctness = metrics.score_correctness(actual_answer, key_facts)
            correctness_counts[correctness] += 1

            evidence_texts = [chunk.get("document", "") for chunk in result.get("final_evidence", [])]
            faithfulness = metrics.score_faithfulness(actual_answer, evidence_texts, key_facts)
            faithfulness_counts[faithfulness] += 1

            entry["correctness"] = correctness
            entry["faithfulness"] = faithfulness
            entry["key_facts_found"] = metrics.key_facts_found(actual_answer, key_facts)

        per_question[result["id"]] = entry

    return {
        "correctness_counts": ANSWER_QUALITY_NA if retrieval_only else correctness_counts,
        "faithfulness_counts": ANSWER_QUALITY_NA if retrieval_only else faithfulness_counts,
        "citation_counts": citation_counts,
        "per_question": per_question,
    }


def build_abstention_report(dataset_by_id: dict, results: list[dict], retrieval_only: bool = False) -> dict:
    """Abstention is entirely inferred from actual_answer's text (via
    is_abstention()'s phrase heuristic), which is always None in
    retrieval-only mode -- scoring it there would count every question as a
    "correct_abstention" for a reason that has nothing to do with whether the
    system would actually abstain, so it's skipped entirely."""
    if retrieval_only:
        return {
            "expected_abstentions": None,
            "correct_abstentions": None,
            "missed_abstentions": None,
            "unexpected_abstentions": None,
            "correct_answers": None,
            "abstention_accuracy": None,
            "per_question": {},
        }

    counts = {
        "correct_abstention": 0,
        "missed_abstention": 0,
        "unexpected_abstention": 0,
        "correct_answer": 0,
    }
    per_question = {}

    for result in results:
        question = dataset_by_id[result["id"]]
        status = metrics.score_abstention(question["should_abstain"], result.get("actual_answer"))
        counts[status] += 1
        per_question[result["id"]] = status

    expected_abstentions = counts["correct_abstention"] + counts["missed_abstention"]
    accuracy = (
        counts["correct_abstention"] / expected_abstentions if expected_abstentions else None
    )

    return {
        "expected_abstentions": expected_abstentions,
        "correct_abstentions": counts["correct_abstention"],
        "missed_abstentions": counts["missed_abstention"],
        "unexpected_abstentions": counts["unexpected_abstention"],
        "correct_answers": counts["correct_answer"],
        "abstention_accuracy": accuracy,
        "per_question": per_question,
    }


def build_performance_report(results: list[dict]) -> dict:
    latencies = [
        result["total_latency_seconds"]
        for result in results
        if result.get("total_latency_seconds") is not None
    ]
    model_names = {result.get("model_name") for result in results if result.get("model_name")}

    return {
        "average_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "min_latency_seconds": min(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "model_names_used": sorted(model_names),
        "retrieval_latency_seconds": None,
        "generation_latency_seconds": None,
        "token_usage": None,
        "unavailable_metrics_reason": (
            "RAG.answer() performs query rewriting, multi-query generation, retrieval, and "
            "final generation as one call with no internal timing hooks exposed, so only "
            "total per-question latency can be measured without modifying app/rag.py (out of "
            "scope for this task). Token usage is unavailable because Generator.generate() in "
            "app/models/generator.py only returns the response text -- the Gemini response "
            "object exposes usage metadata (e.g. token counts) via response.usage_metadata, "
            "but capturing them would require changing the Generator return interface, which "
            "risks changing LLM generation behavior and was out of scope here."
        ),
    }


def build_failures(dataset_by_id: dict, results: list[dict], abstention_by_id: dict, answer_report: dict) -> list[dict]:
    failures = []

    for result in results:
        question_id = result["id"]
        question = dataset_by_id[question_id]
        reasons = []

        if result.get("run_error"):
            reasons.append(f"run_error: {result['run_error']}")

        if result.get("generation_error"):
            reasons.append(f"generation_unavailable: {result['generation_error'].get('message')}")

        abstention_status = abstention_by_id.get(question_id)
        if abstention_status in ("missed_abstention", "unexpected_abstention"):
            reasons.append(abstention_status)

        answer_info = answer_report["per_question"].get(question_id)
        if answer_info:
            if answer_info["correctness"] == "incorrect":
                reasons.append("incorrect_answer")
            if answer_info["citation_status"] == "incorrect":
                reasons.append("incorrect_citation")
            if answer_info["faithfulness"] == "unsupported":
                reasons.append("unsupported_claim")

        if reasons:
            failures.append({
                "id": question_id,
                "category": question["category"],
                "question": question["question"],
                "reasons": reasons,
            })

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=experiment_names(),
        default=FINAL_PIPELINE,
        help=f"Which Phase 5 experiment's results to report on (default: {FINAL_PIPELINE}).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Load/report the retrieval-only results file for this experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment
    results_file = results_file_for(experiment_name, retrieval_only=args.retrieval_only)
    report_file = report_file_for(experiment_name, retrieval_only=args.retrieval_only)

    dataset_by_id = load_dataset()
    run = load_results(results_file)
    results = run["results"]
    k_values = run.get("retrieval_metric_k_values", [3, 5])
    # Report content is driven by what the results file itself says, not the
    # CLI flag used to locate it, in case the two ever disagree.
    retrieval_only = run.get("retrieval_only", False)

    dataset_summary = build_dataset_summary(dataset_by_id)
    retrieval_report = build_retrieval_report(dataset_by_id, results, k_values)
    answer_report = build_answer_report(dataset_by_id, results, retrieval_only=retrieval_only)
    abstention_report = build_abstention_report(dataset_by_id, results, retrieval_only=retrieval_only)
    performance_report = build_performance_report(results)
    failures = build_failures(dataset_by_id, results, abstention_report["per_question"], answer_report)

    report = {
        "experiment": run.get("experiment"),
        "retrieval_only": retrieval_only,
        "evaluation_top_k": run.get("evaluation_top_k"),
        "dataset": dataset_summary,
        "retrieval": retrieval_report["aggregate"],
        "answers": {
            "correctness_counts": answer_report["correctness_counts"],
            "faithfulness_counts": answer_report["faithfulness_counts"],
            "citation_counts": answer_report["citation_counts"],
        },
        "abstention": (
            ANSWER_QUALITY_NA
            if retrieval_only
            else {key: value for key, value in abstention_report.items() if key != "per_question"}
        ),
        "performance": performance_report,
        "failures": failures,
    }

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    _print_summary(report)
    print(f"\nFull report saved to: {report_file}")


def _print_summary(report: dict) -> None:
    print("=== ResearchDesk Phase 5 Evaluation Report ===\n")

    dataset = report["dataset"]
    print(f"Dataset: {dataset['total_questions']} questions "
          f"({dataset['answerable_questions']} answerable, "
          f"{dataset['unanswerable_questions']} unanswerable)")
    print(f"Documents represented: {', '.join(dataset['documents_represented'])}")
    print(f"By category: {dataset['by_category']}")

    print("\n--- Retrieval ---")
    for key, value in report["retrieval"].items():
        if key == "questions_evaluated":
            continue
        formatted = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key}: {formatted}")
    print(f"  (computed over {report['retrieval']['questions_evaluated']} answerable questions)")

    print("\n--- Answers ---")
    print(f"  Correctness: {report['answers']['correctness_counts']}")
    print(f"  Faithfulness: {report['answers']['faithfulness_counts']}")
    print(f"  Citations: {report['answers']['citation_counts']}")

    print("\n--- Abstention ---")
    abstention = report["abstention"]
    if isinstance(abstention, str):
        print(f"  {abstention}")
    else:
        accuracy = abstention["abstention_accuracy"]
        accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(f"  Expected abstentions: {abstention['expected_abstentions']}")
        print(f"  Correct abstentions: {abstention['correct_abstentions']}")
        print(f"  Missed abstentions: {abstention['missed_abstentions']}")
        print(f"  Unexpected abstentions: {abstention['unexpected_abstentions']}")
        print(f"  Abstention accuracy: {accuracy_str}")

    print("\n--- Performance ---")
    performance = report["performance"]
    avg = performance["average_latency_seconds"]
    median = performance["median_latency_seconds"]
    print(f"  Average latency: {avg:.2f}s" if avg is not None else "  Average latency: N/A")
    print(f"  Median latency: {median:.2f}s" if median is not None else "  Median latency: N/A")
    print(f"  Model(s) used: {performance['model_names_used']}")
    print(f"  Token usage: {performance['token_usage']} ({performance['unavailable_metrics_reason']})")

    print(f"\n--- Failures ({len(report['failures'])}) ---")
    for failure in report["failures"]:
        print(f"  {failure['id']} [{failure['category']}]: {', '.join(failure['reasons'])}")


if __name__ == "__main__":
    main()
