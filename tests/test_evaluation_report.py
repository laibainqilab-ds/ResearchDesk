from pathlib import Path

from evaluation import experiments, report


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


# ---------------------------------------------------------------------------
# Evaluation UI helpers (Phase 7): report discovery, safe loading, labeling,
# availability checks, and the five-experiment comparison.
# ---------------------------------------------------------------------------

def test_discover_reports_finds_only_rag_report_files(tmp_path):
    (tmp_path / "rag_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rag_report_basic_vector_retrieval.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rag_results.json").write_text("{}", encoding="utf-8")  # not a report file
    (tmp_path / "rag_evaluation.json").write_text("{}", encoding="utf-8")  # dataset, not a report

    found = report.discover_reports(reports_dir=tmp_path)
    found_names = {path.name for path in found}

    assert found_names == {"rag_report.json", "rag_report_basic_vector_retrieval.json"}


def test_load_report_safely_returns_none_for_missing_file(tmp_path):
    assert report.load_report_safely(tmp_path / "does_not_exist.json") is None


def test_load_report_safely_returns_none_for_malformed_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    assert report.load_report_safely(bad_file) is None


def test_load_report_safely_returns_parsed_dict_for_valid_json(tmp_path):
    good_file = tmp_path / "good.json"
    good_file.write_text('{"experiment": "test"}', encoding="utf-8")

    assert report.load_report_safely(good_file) == {"experiment": "test"}


def test_report_label_indicates_retrieval_only_mode():
    label = report.report_label(
        Path("rag_report_x.json"),
        {"experiment": "multi_query_retrieval", "retrieval_only": True},
    )

    assert "multi_query_retrieval" in label
    assert "retrieval-only" in label


def test_report_label_indicates_full_generation_mode():
    label = report.report_label(
        Path("rag_report_x.json"),
        {"experiment": "final_pipeline", "retrieval_only": False},
    )

    assert "full generation" in label


def test_has_dict_metric_true_for_real_dict():
    assert report.has_dict_metric({"abstention": {"correct_abstentions": 4}}, "abstention") is True


def test_has_dict_metric_false_for_na_sentinel_string():
    assert report.has_dict_metric({"abstention": "N/A — retrieval-only run"}, "abstention") is False


def test_has_dict_metric_checks_nested_answers_key():
    report_data = {
        "answers": {
            "correctness_counts": {"correct": 1},
            "faithfulness_counts": "N/A — retrieval-only run",
        }
    }

    assert report.has_dict_metric(report_data, "answers", "correctness_counts") is True
    assert report.has_dict_metric(report_data, "answers", "faithfulness_counts") is False


def test_has_dict_metric_false_when_section_missing():
    assert report.has_dict_metric({}, "abstention") is False


def test_token_usage_message_reports_unavailable_without_ollama_wording():
    message = report.token_usage_message({
        "token_usage": None,
        "unavailable_metrics_reason": "...Ollama's /api/generate...",
    })

    assert "Ollama" not in message
    assert message == (
        "Token usage unavailable: the current generator does not expose token usage metadata."
    )


def test_token_usage_message_shows_real_value_when_available():
    message = report.token_usage_message({"token_usage": {"total_tokens": 123}})

    assert "123" in message


def test_build_experiment_comparison_rows_covers_all_five_experiments_in_order():
    rows = report.build_experiment_comparison_rows()

    assert [row["experiment"] for row in rows] == experiments.experiment_names()
    assert len(rows) == 5


def test_build_experiment_comparison_rows_uses_real_existing_reports():
    # These retrieval-only report files already exist in evaluation/ from
    # completed Phase 5 runs -- this exercises the real reuse path, not a
    # synthetic fixture, per "use the existing report JSONs as source of truth".
    rows = report.build_experiment_comparison_rows()
    by_name = {row["experiment"]: row for row in rows}

    assert by_name["multi_query_plus_reranking"]["available"] is True
    assert by_name["multi_query_plus_reranking"]["retrieval_only"] is True
    assert isinstance(by_name["multi_query_plus_reranking"]["mrr"], float)


def test_summarize_best_retrieval_configuration_returns_none_when_data_missing():
    rows = [
        {
            "experiment": "multi_query_plus_reranking",
            "available": False,
            "mrr": None,
            "average_latency_seconds": None,
        },
        {
            "experiment": "final_pipeline",
            "available": False,
            "mrr": None,
            "average_latency_seconds": None,
        },
    ]

    assert report.summarize_best_retrieval_configuration(rows) is None


def test_summarize_best_retrieval_configuration_uses_real_reports():
    rows = report.build_experiment_comparison_rows()
    summary = report.summarize_best_retrieval_configuration(rows)

    assert summary is not None
    assert "multi_query_plus_reranking" in summary
    assert "retrieval-quality" in summary
