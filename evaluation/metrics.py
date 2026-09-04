"""Deterministic, reproducible evaluation metrics for ResearchDesk's RAG pipeline.

No LLM judge is used anywhere in this module. Retrieval metrics compare the
RAG system's actual retrieved candidates against the `supporting_sources`
ground truth in the evaluation dataset. Answer metrics compare the actual
generated answer against a curated `key_facts` list per question (substring
presence, not exact-phrase match against the full reference answer).

Relevance policy (retrieval metrics):
    A retrieved chunk is considered relevant to a question if its
    `document_id` matches a supporting source's `document_id`, AND (if the
    supporting source specifies a `page_number`) its `page_number` also
    matches. Chunk-level exact match is not required, because reranking and
    chunking boundaries can validly return a different, overlapping chunk on
    the correct page. Questions with multiple supporting sources are
    satisfied by matching ANY of them per retrieved item; recall counts how
    many of the distinct expected (document_id, page_number) pairs were
    covered by the retrieved set.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _source_key(document_id, page_number):
    return (document_id, page_number)


def _expected_keys(supporting_sources: list[dict]) -> set:
    """Distinct (document_id, page_number) pairs that count as relevant.

    If a supporting source has no page_number, any chunk from that
    document_id is considered relevant (page_number=None acts as a wildcard
    for that document).
    """
    return {
        _source_key(source["document_id"], source.get("page_number"))
        for source in supporting_sources
    }


def _is_relevant(candidate: dict, expected_keys: set) -> bool:
    document_id = candidate.get("document_id")
    page_number = candidate.get("page_number")

    for expected_document_id, expected_page in expected_keys:
        if document_id != expected_document_id:
            continue
        if expected_page is None or page_number == expected_page:
            return True

    return False


def relevant_ranks(candidates: list[dict], supporting_sources: list[dict]) -> list[int]:
    """1-indexed ranks (within `candidates`, in order) of relevant items."""
    expected_keys = _expected_keys(supporting_sources)

    return [
        rank
        for rank, candidate in enumerate(candidates, start=1)
        if _is_relevant(candidate, expected_keys)
    ]


def recall_at_k(candidates: list[dict], supporting_sources: list[dict], k: int) -> float | None:
    """Fraction of distinct expected (document, page) sources found in the top K."""
    expected_keys = _expected_keys(supporting_sources)

    if not expected_keys:
        return None

    top_k = candidates[:k]
    covered = set()

    for expected_document_id, expected_page in expected_keys:
        for candidate in top_k:
            if candidate.get("document_id") != expected_document_id:
                continue
            if expected_page is None or candidate.get("page_number") == expected_page:
                covered.add((expected_document_id, expected_page))
                break

    return len(covered) / len(expected_keys)


def precision_at_k(candidates: list[dict], supporting_sources: list[dict], k: int) -> float | None:
    """Fraction of the top K retrieved candidates that are relevant."""
    top_k = candidates[:k]

    if not top_k:
        return None

    expected_keys = _expected_keys(supporting_sources)
    relevant_count = sum(1 for candidate in top_k if _is_relevant(candidate, expected_keys))

    return relevant_count / len(top_k)


def hit_rate_at_k(candidates: list[dict], supporting_sources: list[dict], k: int) -> float | None:
    """1.0 if at least one relevant item appears in the top K, else 0.0."""
    expected_keys = _expected_keys(supporting_sources)

    if not expected_keys:
        return None

    top_k = candidates[:k]
    return 1.0 if any(_is_relevant(candidate, expected_keys) for candidate in top_k) else 0.0


def reciprocal_rank(candidates: list[dict], supporting_sources: list[dict]) -> float | None:
    """1/rank of the first relevant item across the full candidate list, or 0.0 if none found."""
    expected_keys = _expected_keys(supporting_sources)

    if not expected_keys:
        return None

    for rank, candidate in enumerate(candidates, start=1):
        if _is_relevant(candidate, expected_keys):
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(candidates: list[dict], supporting_sources: list[dict], k_values: list[int]) -> dict:
    """Compute Recall/Precision/HitRate at each K, plus MRR, for one question.

    Returns None-valued metrics (not zeros) when there is no ground truth to
    evaluate against (e.g. unanswerable questions with no supporting_sources)
    so callers can distinguish "no evidence expected" from "system missed it".
    """
    result = {"mrr": reciprocal_rank(candidates, supporting_sources)}

    for k in k_values:
        result[f"recall_at_{k}"] = recall_at_k(candidates, supporting_sources, k)
        result[f"precision_at_{k}"] = precision_at_k(candidates, supporting_sources, k)
        result[f"hit_rate_at_{k}"] = hit_rate_at_k(candidates, supporting_sources, k)

    return result


def average_metric(per_question_values: list[float | None]) -> float | None:
    """Mean of non-None values, or None if every value was unavailable."""
    values = [value for value in per_question_values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

ABSTENTION_PHRASES = [
    "does not contain",
    "doesn't contain",
    "not contain",
    "not mentioned",
    "no information",
    "couldn't find",
    "could not find",
    "cannot answer",
    "can't answer",
    "not provided",
    "not available",
    "not enough information",
    "don't have enough information",
    "do not have enough information",
]


def is_abstention(answer: str | None) -> bool:
    """Heuristic phrase match for whether an answer is an abstention.

    An empty/None answer (e.g. because retrieval returned nothing, or
    generation failed) also counts as an abstention: the system produced no
    claim, which is the safe outcome we want on unanswerable questions.
    """
    if not answer:
        return True

    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def score_abstention(should_abstain: bool, actual_answer: str | None) -> str:
    """Returns 'correct_abstention', 'missed_abstention', 'unexpected_abstention', or 'correct_answer'."""
    actual_abstained = is_abstention(actual_answer)

    if should_abstain:
        return "correct_abstention" if actual_abstained else "missed_abstention"

    return "unexpected_abstention" if actual_abstained else "correct_answer"


# ---------------------------------------------------------------------------
# Answer correctness (key-fact presence, not exact-phrase matching)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower()


def key_facts_found(actual_answer: str | None, key_facts: list[str]) -> list[str]:
    """Which of a question's key_facts appear (case-insensitively) in the answer."""
    if not actual_answer or not key_facts:
        return []

    normalized_answer = _normalize(actual_answer)
    return [fact for fact in key_facts if _normalize(fact) in normalized_answer]


def score_correctness(actual_answer: str | None, key_facts: list[str]) -> str:
    """Deterministic correctness label from key-fact coverage.

    Methodology: each answerable question has a curated list of short,
    source-grounded "key facts" (numbers, names, terms) that must appear in
    a correct answer. This avoids requiring a verbatim match against the
    full reference answer (the earlier evaluation's flaw) while remaining
    fully deterministic and inspectable, per-question, without an LLM judge.

    - no key_facts defined -> "unscored" (methodology gap, not a system failure)
    - 0 facts found -> "incorrect"
    - all facts found -> "correct"
    - some but not all -> "partial"
    """
    if not key_facts:
        return "unscored"

    found = key_facts_found(actual_answer, key_facts)

    if len(found) == 0:
        return "incorrect"
    if len(found) == len(key_facts):
        return "correct"
    return "partial"


# ---------------------------------------------------------------------------
# Faithfulness / groundedness
# ---------------------------------------------------------------------------

def score_faithfulness(actual_answer: str | None, evidence_chunks: list[str], key_facts: list[str]) -> str:
    """Whether the key facts present in the answer are actually backed by the
    retrieved evidence text used for this specific run (not the reference answer).

    This checks groundedness against what the system actually retrieved this
    run, so it can catch a case where the model states a fact that happens to
    be true but wasn't in the evidence it was given.

    - no key_facts or no answer -> "unscored"
    - answer contains no key facts at all -> "unscored" (nothing to check groundedness of)
    - every key fact found in the answer is also present in the evidence -> "grounded"
    - at least one key fact in the answer is missing from the evidence -> "unsupported"
    """
    found_in_answer = key_facts_found(actual_answer, key_facts)

    if not found_in_answer:
        return "unscored"

    evidence_text = _normalize(" ".join(evidence_chunks))
    ungrounded = [fact for fact in found_in_answer if _normalize(fact) not in evidence_text]

    return "unsupported" if ungrounded else "grounded"


# ---------------------------------------------------------------------------
# Citation correctness
# ---------------------------------------------------------------------------

def score_citations(actual_sources: list[dict], supporting_sources: list[dict]) -> str:
    """Whether the RAG system's returned `sources` (real chunk metadata, not
    LLM-invented text) include at least one source matching the dataset's
    expected supporting sources.

    - no supporting_sources defined (e.g. unanswerable question) -> "not_applicable"
    - system returned no sources -> "no_citations"
    - at least one returned source matches an expected source -> "correct"
    - sources were returned but none match -> "incorrect"
    """
    if not supporting_sources:
        return "not_applicable"

    if not actual_sources:
        return "no_citations"

    expected_keys = _expected_keys(supporting_sources)

    for source in actual_sources:
        if _is_relevant(
            {"document_id": source.get("document_id"), "page_number": source.get("page_number")},
            expected_keys,
        ):
            return "correct"

    return "incorrect"
