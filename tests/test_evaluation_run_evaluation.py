from pathlib import Path

from evaluation.run_evaluation import build_experiment_flags, parse_args, results_file_for


# ---------------------------------------------------------------------------
# results_file_for -- retrieval-only file naming
# ---------------------------------------------------------------------------

def test_results_file_for_final_pipeline_default_unchanged():
    assert results_file_for("final_pipeline") == Path("evaluation/rag_results.json")
    assert results_file_for("final_pipeline", retrieval_only=False) == Path("evaluation/rag_results.json")


def test_results_file_for_final_pipeline_retrieval_only_uses_separate_file():
    assert results_file_for("final_pipeline", retrieval_only=True) == Path(
        "evaluation/rag_results_retrieval_only.json"
    )


def test_results_file_for_other_experiment_default_unchanged():
    assert results_file_for("basic_vector_retrieval") == Path(
        "evaluation/rag_results_basic_vector_retrieval.json"
    )


def test_results_file_for_other_experiment_retrieval_only_uses_separate_file():
    assert results_file_for("basic_vector_retrieval", retrieval_only=True) == Path(
        "evaluation/rag_results_basic_vector_retrieval_retrieval_only.json"
    )


# ---------------------------------------------------------------------------
# build_experiment_flags -- retrieval-only is independent of experiment choice
# ---------------------------------------------------------------------------

def test_build_experiment_flags_default_enables_answer_generation():
    flags = build_experiment_flags("final_pipeline", retrieval_only=False)

    assert flags["enable_answer_generation"] is True
    assert flags["enable_query_rewrite"] is True
    assert flags["enable_multi_query"] is True
    assert flags["enable_reranking"] is True


def test_build_experiment_flags_retrieval_only_disables_answer_generation_only():
    flags = build_experiment_flags("basic_vector_retrieval", retrieval_only=True)

    # Retrieval-only must not change which retrieval stages the experiment enables.
    assert flags["enable_query_rewrite"] is False
    assert flags["enable_multi_query"] is False
    assert flags["enable_reranking"] is False
    assert flags["enable_answer_generation"] is False


def test_build_experiment_flags_retrieval_only_is_independent_across_experiments():
    for experiment_name in ("multi_query_retrieval", "multi_query_plus_reranking", "final_pipeline"):
        flags = build_experiment_flags(experiment_name, retrieval_only=True)
        assert flags["enable_answer_generation"] is False


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_parse_args_defaults_to_full_generation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_evaluation.py"])
    args = parse_args()

    assert args.experiment == "final_pipeline"
    assert args.retrieval_only is False


def test_parse_args_retrieval_only_flag_is_recorded(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_evaluation.py", "--experiment", "multi_query_retrieval", "--retrieval-only"],
    )
    args = parse_args()

    assert args.experiment == "multi_query_retrieval"
    assert args.retrieval_only is True
