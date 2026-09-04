from evaluation import report


def make_dataset_by_id():
    return {
        "q1": {
            "id": "q1",
            "category": "direct",
            "question": "What score did the model get?",
            "should_abstain": False,
            "key_facts": ["0.95", "BRCA1"],
            "supporting_sources": [
                {"document_id": "docA", "filename": "docA.pdf", "page_number": 8, "chunk_id": 13},
            ],
        },
        "q2": {
            "id": "q2",
            "category": "unanswerable",
            "question": "What is the capital of France?",
            "should_abstain": True,
            "key_facts": [],
            "supporting_sources": [],
        },
    }


def make_results():
    return [
        {
            "id": "q1",
            "actual_answer": "The model scored 0.95 AUROC on BRCA1.",
            "actual_sources": [{"document_id": "docA", "page_number": 8}],
            "retrieval_candidates": [
                {"document_id": "docA", "page_number": 8, "filename": "docA.pdf"},
                {"document_id": "docB", "page_number": 1, "filename": "docB.pdf"},
            ],
            "final_evidence": [
                {"document": "The model scored 0.95 AUROC on BRCA1 mutation classification."}
            ],
            "run_error": None,
            "generation_error": None,
            "total_latency_seconds": 1.5,
            "model_name": "test-model",
        },
        {
            "id": "q2",
            "actual_answer": "I don't have enough information to answer that.",
            "actual_sources": [],
            "retrieval_candidates": [],
            "final_evidence": [],
            "run_error": None,
            "generation_error": None,
            "total_latency_seconds": 0.5,
            "model_name": "test-model",
        },
    ]


def test_dataset_summary_counts_categories_and_documents():
    summary = report.build_dataset_summary(make_dataset_by_id())

    assert summary["total_questions"] == 2
    assert summary["answerable_questions"] == 1
    assert summary["unanswerable_questions"] == 1
    assert summary["documents_represented"] == ["docA.pdf"]
    assert summary["by_category"] == {"direct": 1, "unanswerable": 1}


def test_retrieval_report_skips_unanswerable_questions():
    dataset_by_id = make_dataset_by_id()
    results = make_results()

    retrieval = report.build_retrieval_report(dataset_by_id, results, k_values=[1, 2])

    assert retrieval["aggregate"]["questions_evaluated"] == 1
    assert retrieval["aggregate"]["recall_at_1"] == 1.0
    assert "q2" not in retrieval["per_question"]


def test_answer_report_scores_only_answerable_questions():
    dataset_by_id = make_dataset_by_id()
    results = make_results()

    answers = report.build_answer_report(dataset_by_id, results)

    assert answers["correctness_counts"]["correct"] == 1
    assert answers["faithfulness_counts"]["grounded"] == 1
    assert answers["citation_counts"]["correct"] == 1
    assert "q2" not in answers["per_question"]


def test_abstention_report_scores_all_questions():
    dataset_by_id = make_dataset_by_id()
    results = make_results()

    abstention = report.build_abstention_report(dataset_by_id, results)

    assert abstention["correct_answers"] == 1
    assert abstention["correct_abstentions"] == 1
    assert abstention["abstention_accuracy"] == 1.0


def test_performance_report_computes_latency_stats():
    performance = report.build_performance_report(make_results())

    assert performance["average_latency_seconds"] == 1.0
    assert performance["median_latency_seconds"] == 1.0
    assert performance["model_names_used"] == ["test-model"]
    assert performance["token_usage"] is None


def test_failures_empty_when_everything_correct():
    dataset_by_id = make_dataset_by_id()
    results = make_results()

    answer_report = report.build_answer_report(dataset_by_id, results)
    abstention_report = report.build_abstention_report(dataset_by_id, results)

    failures = report.build_failures(
        dataset_by_id, results, abstention_report["per_question"], answer_report
    )

    assert failures == []


def test_failures_flags_incorrect_answer():
    dataset_by_id = make_dataset_by_id()
    results = make_results()
    results[0]["actual_answer"] = "I have no idea."

    answer_report = report.build_answer_report(dataset_by_id, results)
    abstention_report = report.build_abstention_report(dataset_by_id, results)

    failures = report.build_failures(
        dataset_by_id, results, abstention_report["per_question"], answer_report
    )

    failure_ids = [failure["id"] for failure in failures]
    assert "q1" in failure_ids


# ---------------------------------------------------------------------------
# retrieval_only reporting: answer=None must not be scored as an incorrect
# answer or as a correct/expected abstention -- both would be misleading,
# since no answer was ever generated to judge.
# ---------------------------------------------------------------------------

def make_retrieval_only_results():
    """Same shape run_evaluation.py produces with --retrieval-only: answer
    generation was skipped, so actual_answer is None, but sources/candidates
    are still real (retrieval ran in full)."""
    results = make_results()
    for result in results:
        result["actual_answer"] = None
    return results


def test_answer_report_retrieval_only_does_not_score_correctness_or_faithfulness():
    dataset_by_id = make_dataset_by_id()
    results = make_retrieval_only_results()

    answers = report.build_answer_report(dataset_by_id, results, retrieval_only=True)

    assert answers["correctness_counts"] == report.ANSWER_QUALITY_NA
    assert answers["faithfulness_counts"] == report.ANSWER_QUALITY_NA
    assert answers["per_question"]["q1"]["correctness"] is None
    assert answers["per_question"]["q1"]["faithfulness"] is None


def test_answer_report_retrieval_only_still_scores_citations():
    dataset_by_id = make_dataset_by_id()
    results = make_retrieval_only_results()

    answers = report.build_answer_report(dataset_by_id, results, retrieval_only=True)

    # q1's actual_sources ({"document_id": "docA", "page_number": 8}) matches
    # its supporting_sources -- citation scoring only needs source metadata,
    # not the generated answer text, so it must still work.
    assert answers["citation_counts"]["correct"] == 1
    assert answers["per_question"]["q1"]["citation_status"] == "correct"


def test_abstention_report_retrieval_only_returns_na_not_correct_abstention():
    dataset_by_id = make_dataset_by_id()
    results = make_retrieval_only_results()

    abstention = report.build_abstention_report(dataset_by_id, results, retrieval_only=True)

    assert abstention["correct_abstentions"] is None
    assert abstention["expected_abstentions"] is None
    assert abstention["abstention_accuracy"] is None
    assert abstention["per_question"] == {}


def test_retrieval_metrics_still_computed_in_retrieval_only_mode():
    dataset_by_id = make_dataset_by_id()
    results = make_retrieval_only_results()

    retrieval = report.build_retrieval_report(dataset_by_id, results, k_values=[1, 2])

    # Retrieval metrics never depend on actual_answer, so a None answer must
    # not change them at all.
    assert retrieval["aggregate"]["questions_evaluated"] == 1
    assert retrieval["aggregate"]["recall_at_1"] == 1.0


def test_failures_do_not_flag_none_answer_as_incorrect_in_retrieval_only_mode():
    dataset_by_id = make_dataset_by_id()
    results = make_retrieval_only_results()

    answer_report = report.build_answer_report(dataset_by_id, results, retrieval_only=True)
    abstention_report = report.build_abstention_report(dataset_by_id, results, retrieval_only=True)

    failures = report.build_failures(
        dataset_by_id, results, abstention_report["per_question"], answer_report
    )

    reasons = [reason for failure in failures for reason in failure["reasons"]]
    assert "incorrect_answer" not in reasons
    assert "unsupported_claim" not in reasons
    assert "missed_abstention" not in reasons
    assert "unexpected_abstention" not in reasons
