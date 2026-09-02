import json
from pathlib import Path


EVALUATION_FILE = Path("evaluation/rag_evaluation.json")
RESULTS_FILE = Path("evaluation/rag_results.json")


# Manual baseline assessment based on the actual generated answers.
# This avoids falsely marking correct answers as wrong because of
# differences in wording.
ASSESSMENTS = {
    "q1": "correct",
    "q2": "correct",
    "q3": "correct",
    "q4": "correct",
    "q5": "correct",
    "q6": "incorrect",
    "q7": "correct",
    "q8": "correct",
    "q9": "correct",
    "q10": "correct",
    "q11": "correct",
    "q12": "partial",
    "q13": "correct",
    "q14": "incorrect",
    "q15": "correct",
}


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    evaluation_by_id = {
        item["id"]: item
        for item in evaluation
    }

    correct = 0
    partial = 0
    incorrect = 0

    print("\n=== Answer Correctness Evaluation ===\n")

    for result in results:
        question_id = result["id"]

        # Abstention questions are evaluated separately.
        if evaluation_by_id[question_id].get("should_abstain", False):
            continue

        assessment = ASSESSMENTS.get(question_id, "incorrect")

        if assessment == "correct":
            correct += 1
            status = "PASS"
        elif assessment == "partial":
            partial += 1
            status = "PARTIAL"
        else:
            incorrect += 1
            status = "FAIL"

        print(f"{question_id}: {status}")

    total_answerable = correct + partial + incorrect

    strict_accuracy = (
        correct / total_answerable
        if total_answerable
        else 0
    )

    # Partial answers receive half credit.
    adjusted_accuracy = (
        (correct + (partial * 0.5)) / total_answerable
        if total_answerable
        else 0
    )

    print("\n=== Answer Correctness Metrics ===")
    print(f"Answerable questions: {total_answerable}")
    print(f"Correct answers: {correct}")
    print(f"Partial answers: {partial}")
    print(f"Incorrect answers: {incorrect}")
    print(f"Strict correctness: {strict_accuracy:.2%}")
    print(f"Adjusted correctness: {adjusted_accuracy:.2%}")


if __name__ == "__main__":
    main()