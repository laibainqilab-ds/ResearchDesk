"""Registry of the Phase 5 comparison experiments.

Each experiment maps to the three independent retrieval-stage flags on
RAG.answer() (`enable_query_rewrite`, `enable_multi_query`,
`enable_reranking`), added specifically so these five configurations can be
run and compared against each other rather than only ever exercising the
system's always-on production pipeline.
"""

BASIC_VECTOR_RETRIEVAL = "basic_vector_retrieval"
VECTOR_PLUS_QUERY_REWRITING = "vector_plus_query_rewriting"
MULTI_QUERY_RETRIEVAL = "multi_query_retrieval"
MULTI_QUERY_PLUS_RERANKING = "multi_query_plus_reranking"
FINAL_PIPELINE = "final_pipeline"

EXPERIMENTS = {
    BASIC_VECTOR_RETRIEVAL: {
        "description": (
            "Single embedded query, no rewriting, no multi-query, no reranking. "
            "The retrieval control: raw vector-distance ranking only."
        ),
        "enable_query_rewrite": False,
        "enable_multi_query": False,
        "enable_reranking": False,
    },
    VECTOR_PLUS_QUERY_REWRITING: {
        "description": (
            "Query rewriting enabled, single-query retrieval, no reranking. "
            "Isolates the effect of rewriting follow-up questions into standalone queries."
        ),
        "enable_query_rewrite": True,
        "enable_multi_query": False,
        "enable_reranking": False,
    },
    MULTI_QUERY_RETRIEVAL: {
        "description": (
            "Multi-query generation enabled, no query rewriting, no reranking "
            "(raw vector-distance ranking over the merged candidate pool)."
        ),
        "enable_query_rewrite": False,
        "enable_multi_query": True,
        "enable_reranking": False,
    },
    MULTI_QUERY_PLUS_RERANKING: {
        "description": (
            "Multi-query generation + cross-encoder reranking, no query rewriting."
        ),
        "enable_query_rewrite": False,
        "enable_multi_query": True,
        "enable_reranking": True,
    },
    FINAL_PIPELINE: {
        "description": (
            "Query rewriting (when there is conversation history) + multi-query "
            "generation + vector retrieval + reranking. This is exactly what Chat "
            "and the Retrieval Inspector already run -- the system's always-on "
            "production configuration."
        ),
        "enable_query_rewrite": True,
        "enable_multi_query": True,
        "enable_reranking": True,
    },
}


def experiment_names() -> list[str]:
    return list(EXPERIMENTS.keys())


def flags_for(experiment_name: str) -> dict:
    """The three RAG.answer() kwargs for a given experiment name.

    Raises KeyError with the valid options listed if the name is unknown.
    """
    if experiment_name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment '{experiment_name}'. "
            f"Valid experiments: {', '.join(EXPERIMENTS)}"
        )

    config = EXPERIMENTS[experiment_name]

    return {
        "enable_query_rewrite": config["enable_query_rewrite"],
        "enable_multi_query": config["enable_multi_query"],
        "enable_reranking": config["enable_reranking"],
    }
