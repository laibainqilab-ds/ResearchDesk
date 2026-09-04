from evaluation import metrics


def candidate(document_id, page_number=None, document_text=""):
    return {"document_id": document_id, "page_number": page_number, "document": document_text}


def source(document_id, page_number=None):
    return {"document_id": document_id, "filename": f"{document_id}.pdf", "page_number": page_number, "chunk_id": None}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_recall_at_k_full_hit():
    candidates = [candidate("docA", 1), candidate("docA", 2), candidate("docB", 5)]
    sources = [source("docA", 1)]

    assert metrics.recall_at_k(candidates, sources, k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    candidates = [candidate("docB", 5), candidate("docB", 6), candidate("docA", 1)]
    sources = [source("docA", 1)]

    assert metrics.recall_at_k(candidates, sources, k=2) == 0.0
    assert metrics.recall_at_k(candidates, sources, k=3) == 1.0


def test_recall_at_k_partial_with_multiple_sources():
    candidates = [candidate("docA", 1), candidate("docC", 9)]
    sources = [source("docA", 1), source("docB", 2)]

    assert metrics.recall_at_k(candidates, sources, k=2) == 0.5


def test_recall_at_k_none_when_no_ground_truth():
    candidates = [candidate("docA", 1)]

    assert metrics.recall_at_k(candidates, [], k=3) is None


def test_precision_at_k_basic():
    candidates = [candidate("docA", 1), candidate("docB", 5), candidate("docA", 1)]
    sources = [source("docA", 1)]

    assert metrics.precision_at_k(candidates, sources, k=3) == 2 / 3


def test_precision_at_k_none_when_no_candidates():
    assert metrics.precision_at_k([], [source("docA", 1)], k=3) is None


def test_hit_rate_at_k_hit_and_miss():
    candidates = [candidate("docB", 5), candidate("docA", 1)]
    sources = [source("docA", 1)]

    assert metrics.hit_rate_at_k(candidates, sources, k=1) == 0.0
    assert metrics.hit_rate_at_k(candidates, sources, k=2) == 1.0


def test_reciprocal_rank_first_relevant_position():
    candidates = [candidate("docB", 5), candidate("docC", 9), candidate("docA", 1)]
    sources = [source("docA", 1)]

    assert metrics.reciprocal_rank(candidates, sources) == 1 / 3


def test_reciprocal_rank_zero_when_never_found():
    candidates = [candidate("docB", 5)]
    sources = [source("docA", 1)]

    assert metrics.reciprocal_rank(candidates, sources) == 0.0


def test_page_wildcard_when_source_has_no_page_number():
    candidates = [candidate("docA", page_number=42)]
    sources = [source("docA", page_number=None)]

    assert metrics.recall_at_k(candidates, sources, k=1) == 1.0


def test_average_metric_ignores_none_values():
    assert metrics.average_metric([1.0, None, 0.0, None]) == 0.5


def test_average_metric_all_none_returns_none():
    assert metrics.average_metric([None, None]) is None


def test_evaluate_retrieval_shape():
    candidates = [candidate("docA", 1)]
    sources = [source("docA", 1)]

    result = metrics.evaluate_retrieval(candidates, sources, k_values=[1, 3])

    assert set(result.keys()) == {
        "mrr", "recall_at_1", "precision_at_1", "hit_rate_at_1",
        "recall_at_3", "precision_at_3", "hit_rate_at_3",
    }
    assert result["mrr"] == 1.0


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_is_abstention_detects_phrase():
    assert metrics.is_abstention("The documents do not contain this information.")


def test_is_abstention_false_for_normal_answer():
    assert not metrics.is_abstention("Evo 2 was trained on 9.3 trillion DNA letters.")


def test_is_abstention_true_for_empty_answer():
    assert metrics.is_abstention(None)
    assert metrics.is_abstention("")


def test_score_abstention_all_four_outcomes():
    assert metrics.score_abstention(True, "I don't have enough information.") == "correct_abstention"
    assert metrics.score_abstention(True, "The answer is 42.") == "missed_abstention"
    assert metrics.score_abstention(False, "Sorry, not available.") == "unexpected_abstention"
    assert metrics.score_abstention(False, "The answer is 42.") == "correct_answer"


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------

def test_score_correctness_all_facts_found():
    assert metrics.score_correctness("The score was 0.95 AUROC.", ["0.95", "AUROC"]) == "correct"


def test_score_correctness_no_facts_found():
    assert metrics.score_correctness("I'm not sure.", ["0.95", "AUROC"]) == "incorrect"


def test_score_correctness_some_facts_found():
    assert metrics.score_correctness("The score was 0.95.", ["0.95", "AUROC"]) == "partial"


def test_score_correctness_no_key_facts_defined_is_unscored():
    assert metrics.score_correctness("Anything.", []) == "unscored"


def test_score_correctness_case_insensitive():
    assert metrics.score_correctness("the score was auroc 0.95", ["0.95", "AUROC"]) == "correct"


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def test_faithfulness_grounded_when_facts_in_evidence():
    answer = "Evo 2 scored 0.95 AUROC on BRCA1."
    evidence = ["Evo 2 achieved 0.95 AUROC on BRCA1 mutation classification."]

    assert metrics.score_faithfulness(answer, evidence, ["0.95", "BRCA1"]) == "grounded"


def test_faithfulness_unsupported_when_fact_missing_from_evidence():
    answer = "Evo 2 scored 0.99 AUROC on BRCA1."
    evidence = ["Some unrelated evidence text."]

    # "BRCA1" appears in the answer and nowhere in the evidence -> unsupported
    assert metrics.score_faithfulness(answer, evidence, ["BRCA1"]) == "unsupported"


def test_faithfulness_unscored_when_no_facts_in_answer():
    answer = "I don't know."
    evidence = ["Evo 2 achieved 0.95 AUROC on BRCA1."]

    assert metrics.score_faithfulness(answer, evidence, ["0.95", "BRCA1"]) == "unscored"


# ---------------------------------------------------------------------------
# Citation correctness
# ---------------------------------------------------------------------------

def test_citations_correct_when_match_found():
    actual_sources = [{"document_id": "docA", "page_number": 1}]
    expected = [source("docA", 1)]

    assert metrics.score_citations(actual_sources, expected) == "correct"


def test_citations_incorrect_when_no_match():
    actual_sources = [{"document_id": "docB", "page_number": 9}]
    expected = [source("docA", 1)]

    assert metrics.score_citations(actual_sources, expected) == "incorrect"


def test_citations_no_citations_when_empty():
    assert metrics.score_citations([], [source("docA", 1)]) == "no_citations"


def test_citations_not_applicable_when_no_expected_sources():
    assert metrics.score_citations([{"document_id": "docA", "page_number": 1}], []) == "not_applicable"
