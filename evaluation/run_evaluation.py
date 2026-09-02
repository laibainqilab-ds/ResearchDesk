import json
from pathlib import Path

from app.rag import RAG


EVALUATION_FILE = Path("evaluation/rag_evaluation.json")
RESULTS_FILE = Path("evaluation/rag_results.json")


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        evaluation_questions = json.load(file)

    rag = RAG()
    results = []

    for index, item in enumerate(evaluation_questions, start=1):
        question = item["question"]

        print(f"\n[{index}/{len(evaluation_questions)}] {question}")

        result = rag.answer(
            question=question,
            conversation_history=[],
        )

        results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_answer": item.get("expected_answer"),
                "should_abstain": item.get("should_abstain", False),
                "actual_answer": result["answer"],
                "sources": result["sources"],
                "error": result["error"],
            }
        )

        print(f"Answer: {result['answer']}")

        if result["sources"]:
            print("Sources:")
            for source in result["sources"]:
                print(
                    f"  - {source['filename']}, "
                    f"page {source['page_number']}, "
                    f"chunk {source['chunk_id']}"
                )

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()