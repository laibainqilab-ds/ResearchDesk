
import json
from pathlib import Path


EVALUATION_FILE = Path("evaluation/rag_evaluation.json")
RESULTS_FILE = Path("evaluation/rag_results.json")
REPORT_FILE = Path("evaluation/rag_report.json")


def is_abstention(answer: str | None) -> bool:
    if not answer:
        return True

    answer_lower = answer.lower()

    abstention_phrases = [
        "does not contain",
        "doesn't contain",
        "not contain",
        "not mentioned",
        "no information",
        "couldn't find",
        "cannot answer",
        "can't answer",
        "not provided",
        "not available",
    ]

    return any(
        phrase in answer_lower
        for phrase in abstention_phrases
    )


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    evaluation_by_id = {
        item["id"]: item
        for item in evaluation
    }

    report = {
        "total_questions": len(results),
        "questions": [],
    }

    for result in results:
        question_id = result["id"]
        expected = evaluation_by_id[question_id]

        actual_answer = result["actual_answer"]
        should_abstain = expected.get("should_abstain", False)
        actual_abstention = is_abstention(actual_answer)

        if should_abstain:
            abstention_status = (
                "PASS" if actual_abstention else "FAIL"
            )
        else:
            abstention_status = (
                "FAIL" if actual_abstention else "PASS"
            )

        report["questions"].append(
            {
                "id": question_id,
                "question": result["question"],
                "expected_answer": expected.get("expected_answer"),
                "actual_answer": actual_answer,
                "should_abstain": should_abstain,
                "actual_abstention": actual_abstention,
                "abstention_status": abstention_status,
                "sources": result["sources"],
                "error": result["error"],
            }
        )

    expected_abstentions = sum(
        item["should_abstain"]
        for item in report["questions"]
    )

    correct_abstentions = sum(
        item["should_abstain"] and item["actual_abstention"]
        for item in report["questions"]
    )

    failed_abstentions = sum(
        item["should_abstain"] and not item["actual_abstention"]
        for item in report["questions"]
    )

    unexpected_abstentions = sum(
        not item["should_abstain"] and item["actual_abstention"]
        for item in report["questions"]
    )

    abstention_accuracy = (
        correct_abstentions / expected_abstentions
        if expected_abstentions
        else 0
    )

    report["abstention_metrics"] = {
        "expected_abstentions": expected_abstentions,
        "correct_abstentions": correct_abstentions,
        "failed_abstentions": failed_abstentions,
        "unexpected_abstentions": unexpected_abstentions,
        "abstention_accuracy": abstention_accuracy,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=== RAG Evaluation Report ===")
    print(f"Total questions: {len(results)}")
    print(f"Expected abstentions: {expected_abstentions}")
    print(f"Correct abstentions: {correct_abstentions}")
    print(f"Failed abstentions: {failed_abstentions}")
    print(f"Unexpected abstentions: {unexpected_abstentions}")
    print(f"Abstention accuracy: {abstention_accuracy:.2%}")

    print("\nReport saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()