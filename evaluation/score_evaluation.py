import json
from pathlib import Path


EVALUATION_FILE = Path("evaluation/rag_evaluation.json")
RESULTS_FILE = Path("evaluation/rag_results.json")


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

    total = len(results)
    expected_abstentions = 0
    correct_abstentions = 0
    unexpected_abstentions = 0

    print("\n=== RAG Evaluation Summary ===\n")

    for result in results:
        question_id = result["id"]
        expected = evaluation_by_id[question_id]

        should_abstain = expected.get("should_abstain", False)
        actual_abstain = is_abstention(result["actual_answer"])

        if should_abstain:
            expected_abstentions += 1

            if actual_abstain:
                correct_abstentions += 1
                status = "PASS"
            else:
                status = "FAIL"
        else:
            if actual_abstain:
                unexpected_abstentions += 1
                status = "FAIL"
            else:
                status = "PASS"

        print(
            f"{question_id}: {status} | "
            f"expected_abstain={should_abstain} | "
            f"actual_abstain={actual_abstain}"
        )

    abstention_accuracy = (
        correct_abstentions / expected_abstentions
        if expected_abstentions
        else 0
    )

    print("\n=== Abstention Metrics ===")
    print(f"Total questions: {total}")
    print(f"Expected abstentions: {expected_abstentions}")
    print(f"Correct abstentions: {correct_abstentions}")
    print(f"Unexpected abstentions: {unexpected_abstentions}")
    print(f"Abstention accuracy: {abstention_accuracy:.2%}")


if __name__ == "__main__":
    main()