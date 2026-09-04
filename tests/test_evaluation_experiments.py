import pytest

from evaluation import experiments


EXPECTED_FLAGS = {
    "basic_vector_retrieval": {
        "enable_query_rewrite": False,
        "enable_multi_query": False,
        "enable_reranking": False,
    },
    "vector_plus_query_rewriting": {
        "enable_query_rewrite": True,
        "enable_multi_query": False,
        "enable_reranking": False,
    },
    "multi_query_retrieval": {
        "enable_query_rewrite": False,
        "enable_multi_query": True,
        "enable_reranking": False,
    },
    "multi_query_plus_reranking": {
        "enable_query_rewrite": False,
        "enable_multi_query": True,
        "enable_reranking": True,
    },
    "final_pipeline": {
        "enable_query_rewrite": True,
        "enable_multi_query": True,
        "enable_reranking": True,
    },
}


def test_all_five_required_experiments_are_registered():
    assert set(experiments.experiment_names()) == set(EXPECTED_FLAGS)


@pytest.mark.parametrize("experiment_name", EXPECTED_FLAGS.keys())
def test_experiment_flags_match_required_configuration(experiment_name):
    assert experiments.flags_for(experiment_name) == EXPECTED_FLAGS[experiment_name]


def test_flags_for_unknown_experiment_raises_key_error():
    with pytest.raises(KeyError):
        experiments.flags_for("not_a_real_experiment")


def test_final_pipeline_constant_matches_registry_key():
    assert experiments.FINAL_PIPELINE == "final_pipeline"
    assert experiments.FINAL_PIPELINE in experiments.EXPERIMENTS
