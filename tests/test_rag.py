"""Unit tests for app/rag.py's retrieval/generation flag plumbing.

All collaborators (embedder, store, generator, reranker) are mocked, so
these tests never load a real Sentence Transformers model, never touch
ChromaDB, and never make a live Gemini call.
"""

import json
import logging
from unittest.mock import Mock

from app.models.generator import GenerationUnavailableError
from app.rag import (
    RAG,
    RECENT_HISTORY_SIZE,
    RELEVANT_HISTORY_MAX_MESSAGES,
    extract_citations,
    select_relevant_history,
)

RESEARCHDESK_LOGGER = "researchdesk"


def _log_payloads(caplog):
    """Parse only this module's structured JSON log records, ignoring the
    plain-text logger.warning() calls that also run alongside them (caplog
    captures all propagating loggers, not just the one named in at_level)."""
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == RESEARCHDESK_LOGGER
    ]


def make_rag():
    """A RAG instance with mocked collaborators, bypassing __init__ so no
    real model/API client is constructed."""
    rag = RAG.__new__(RAG)
    rag.embedder = Mock()
    rag.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    rag.store = Mock()
    rag.generator = Mock()
    rag.reranker = Mock()
    return rag


def metadata(document_id, chunk_id, page_number=None, filename="doc.pdf"):
    return {
        "document_id": document_id,
        "filename": filename,
        "chunk_id": chunk_id,
        "page_number": page_number,
    }


def search_result(items):
    """items: list of (document_text, metadata_dict, distance)."""
    return {
        "documents": [[item[0] for item in items]],
        "metadatas": [[item[1] for item in items]],
        "distances": [[item[2] for item in items]],
    }


# ---------------------------------------------------------------------------
# RAG.retrieve() -- enable_reranking
# ---------------------------------------------------------------------------

def test_retrieve_reranking_enabled_uses_cross_encoder_scores():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.5),
        ("chunk-b", metadata("docB", 0), 0.1),
        ("chunk-c", metadata("docC", 0), 0.3),
    ])

    # Reranker deliberately inverts the distance order: the worst-distance
    # chunk gets the best rerank score, so ordering by rerank_score (not
    # distance) is the only way this assertion can pass.
    rag.reranker.rerank.return_value = [
        ("chunk-a", 0.9),
        ("chunk-c", 0.5),
        ("chunk-b", 0.1),
    ]

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=True,
    )

    rag.reranker.rerank.assert_called_once()

    final_evidence = result["final_evidence"]
    assert [item["document"] for item in final_evidence] == ["chunk-a", "chunk-c", "chunk-b"]
    assert [item["rerank_score"] for item in final_evidence] == [0.9, 0.5, 0.1]


def test_retrieve_reranking_disabled_uses_vector_distance_ordering():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.5),
        ("chunk-b", metadata("docB", 0), 0.1),
        ("chunk-c", metadata("docC", 0), 0.3),
    ])

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=False,
    )

    rag.reranker.rerank.assert_not_called()

    final_evidence = result["final_evidence"]
    # Ascending by distance: chunk-b (0.1), chunk-c (0.3), chunk-a (0.5)
    assert [item["document"] for item in final_evidence] == ["chunk-b", "chunk-c", "chunk-a"]
    assert all(item["rerank_score"] is None for item in final_evidence)


def test_retrieve_reranking_disabled_respects_top_k():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.9),
        ("chunk-b", metadata("docB", 0), 0.2),
        ("chunk-c", metadata("docC", 0), 0.4),
    ])

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=2,
        enable_reranking=False,
    )

    assert [item["document"] for item in result["final_evidence"]] == ["chunk-b", "chunk-c"]


# ---------------------------------------------------------------------------
# RAG.retrieve() -- malformed Chroma metadata (missing document_id/chunk_id)
#
# Regression coverage for the KeyError('document_id') that crashed the
# basic_vector_retrieval Phase 5 run: two orphaned chunks ("test_1"/"test_2",
# inserted directly into data/chroma by a manual smoke-test script) carry
# metadata like {"source": "test", "page": 1} with no document_id or
# chunk_id at all. Retrieval must skip chunks like this, not crash.
# ---------------------------------------------------------------------------

def test_retrieve_skips_chunk_missing_document_id():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("good chunk", metadata("docA", 0), 0.2),
        # Same shape as the real orphaned test_1/test_2 chunks found in
        # data/chroma: no document_id, no chunk_id, no filename.
        ("orphan chunk", {"source": "test", "page": 1}, 0.1),
    ])

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=False,
    )

    documents = [item["document"] for item in result["final_evidence"]]
    assert documents == ["good chunk"]


def test_retrieve_skips_chunk_missing_chunk_id():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("good chunk", metadata("docA", 0), 0.2),
        ("orphan chunk", {"document_id": "docB", "filename": "doc.pdf"}, 0.1),
    ])

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=False,
    )

    documents = [item["document"] for item in result["final_evidence"]]
    assert documents == ["good chunk"]


def test_retrieve_all_malformed_chunks_returns_empty_without_crashing():
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("orphan-1", {"source": "test", "page": 1}, 0.1),
        ("orphan-2", {"source": "test", "page": 2}, 0.2),
    ])

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=False,
    )

    assert result == {"candidates": [], "final_evidence": []}


def test_retrieve_logs_warning_for_malformed_chunk(caplog):
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("orphan chunk", {"source": "test", "page": 1}, 0.1),
    ])

    with caplog.at_level(logging.WARNING, logger="app.rag"):
        rag.retrieve(
            retrieval_question="q",
            search_queries=["q"],
            top_k=3,
            enable_reranking=False,
        )

    assert "malformed metadata" in caplog.text


def test_retrieve_reranking_enabled_also_skips_malformed_chunks():
    """The malformed-metadata guard runs before reranking, so it must hold
    regardless of enable_reranking."""
    rag = make_rag()

    rag.store.search.return_value = search_result([
        ("good chunk", metadata("docA", 0), 0.2),
        ("orphan chunk", {"source": "test", "page": 1}, 0.1),
    ])
    rag.reranker.rerank.return_value = [("good chunk", 0.8)]

    result = rag.retrieve(
        retrieval_question="q",
        search_queries=["q"],
        top_k=3,
        enable_reranking=True,
    )

    rag.reranker.rerank.assert_called_once_with(query="q", documents=["good chunk"])
    documents = [item["document"] for item in result["final_evidence"]]
    assert documents == ["good chunk"]


# ---------------------------------------------------------------------------
# RAG.answer() -- enable_query_rewrite / enable_multi_query / enable_reranking
# ---------------------------------------------------------------------------

def test_answer_default_flags_match_previous_behavior():
    rag = make_rag()
    rag.generator.rewrite_query.return_value = "rewritten question"
    rag.generator.generate_queries.return_value = ["q1", "q2"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    rag.answer(question="follow up", conversation_history=history, trace_id="fixed-trace-id")

    rag.generator.rewrite_query.assert_called_once_with(
        question="follow up", conversation_history=history[-3:], trace_id="fixed-trace-id"
    )
    rag.generator.generate_queries.assert_called_once_with(
        question="rewritten question", num_queries=3, trace_id="fixed-trace-id"
    )
    rag.retrieve.assert_called_once()
    assert rag.retrieve.call_args.kwargs["enable_reranking"] is True
    assert rag.retrieve.call_args.kwargs["search_queries"] == ["q1", "q2"]


def test_answer_query_rewrite_disabled_even_with_history():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    rag.answer(
        question="follow up",
        conversation_history=history,
        enable_query_rewrite=False,
        trace_id="fixed-trace-id",
    )

    rag.generator.rewrite_query.assert_not_called()
    rag.generator.generate_queries.assert_called_once_with(
        question="follow up", num_queries=3, trace_id="fixed-trace-id"
    )


def test_answer_multi_query_disabled_uses_single_query():
    rag = make_rag()
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    rag.answer(question="a question", conversation_history=[], enable_multi_query=False)

    rag.generator.generate_queries.assert_not_called()
    assert rag.retrieve.call_args.kwargs["search_queries"] == ["a question"]


def test_answer_passes_enable_reranking_through_to_retrieve():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    rag.answer(question="a question", conversation_history=[], enable_reranking=False)

    assert rag.retrieve.call_args.kwargs["enable_reranking"] is False


def test_answer_generation_disabled_skips_generator_generate_call():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.reranker.rerank.return_value = [("chunk text", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk text", metadata("docA", 0, page_number=3, filename="doc.pdf"), 0.2),
    ])

    result = rag.answer(
        question="a question",
        conversation_history=[],
        enable_answer_generation=False,
    )

    rag.generator.generate.assert_not_called()
    assert result["answer"] is None
    assert result["error"] is None


def test_answer_generation_disabled_still_populates_retrieval_and_sources():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.reranker.rerank.return_value = [("chunk text", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk text", metadata("docA", 0, page_number=3, filename="doc.pdf"), 0.2),
    ])

    result = rag.answer(
        question="a question",
        conversation_history=[],
        enable_answer_generation=False,
    )

    assert result["sources"] == [
        {
            "citation_id": 1,
            "document_id": "docA",
            "filename": "doc.pdf",
            "page_number": 3,
            "chunk_id": 0,
            "rerank_score": 0.9,
        }
    ]
    assert len(result["retrieval"]["final_evidence"]) == 1
    assert result["retrieval"]["final_evidence"][0]["document"] == "chunk text"
    assert len(result["retrieval"]["candidates"]) == 1
    assert result["retrieval"]["search_queries"] == ["q"]


def test_answer_generation_enabled_by_default_calls_generator():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.generator.generate.return_value = "a generated answer"
    rag.reranker.rerank.return_value = [("chunk text", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk text", metadata("docA", 0), 0.2),
    ])

    result = rag.answer(question="a question", conversation_history=[])

    rag.generator.generate.assert_called_once()
    assert result["answer"] == "a generated answer"
    assert result["error"] is None


def test_answer_all_stages_disabled_uses_original_question_as_only_search_query():
    rag = make_rag()
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    rag.answer(
        question="a question",
        conversation_history=history,
        enable_query_rewrite=False,
        enable_multi_query=False,
        enable_reranking=False,
    )

    rag.generator.rewrite_query.assert_not_called()
    rag.generator.generate_queries.assert_not_called()
    assert rag.retrieve.call_args.kwargs["search_queries"] == ["a question"]
    assert rag.retrieve.call_args.kwargs["enable_reranking"] is False


# ---------------------------------------------------------------------------
# Conversation memory -- recent / relevant-older / summarization
# ---------------------------------------------------------------------------

def _paris_ml_eiffel_history():
    """6-message history: an older Paris exchange, an older unrelated ML
    exchange, then a recent Eiffel Tower exchange. Deliberately built so a
    "Paris" follow-up question should pull in the Paris exchange (shares
    vocabulary) despite it falling outside the last-3 window, while leaving
    the unrelated ML exchange behind."""
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI."},
        {"role": "user", "content": "How tall is the Eiffel Tower?"},
        {"role": "assistant", "content": "The Eiffel Tower is 330 meters tall."},
    ]


def test_prepare_context_preserves_recent_conversation_when_history_is_short():
    rag = make_rag()
    history = [
        {"role": "user", "content": "What is Evo 2?"},
        {"role": "assistant", "content": "Evo 2 is a genomic model."},
    ]

    prepared = rag._prepare_conversation_context(history, "What is it used for?")

    assert prepared == history


def test_prepare_context_selects_relevant_older_message():
    rag = make_rag()
    history = _paris_ml_eiffel_history()

    prepared = rag._prepare_conversation_context(history, "Tell me more about Paris")

    # Shares "Paris" with the question, so it should be pulled forward even
    # though it's outside the plain last-3 window.
    assert history[1] in prepared
    # Mentions France but never says "Paris" -- no overlap, correctly left out.
    assert history[0] not in prepared


def test_prepare_context_excludes_irrelevant_older_message():
    rag = make_rag()
    history = _paris_ml_eiffel_history()

    prepared = rag._prepare_conversation_context(history, "Tell me more about Paris")

    # "What is machine learning?" is in the older portion and shares no
    # vocabulary with the Paris question -- must not be blindly carried
    # forward just because it's part of the conversation.
    assert history[2] not in prepared


def test_prepare_context_history_remains_bounded():
    rag = make_rag()
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message number {i} about topic {i}.",
        }
        for i in range(10)
    ]

    prepared = rag._prepare_conversation_context(history, "What is Evo 2 used for?")

    assert len(prepared) <= RECENT_HISTORY_SIZE + RELEVANT_HISTORY_MAX_MESSAGES
    assert len(prepared) < len(history)


def test_summarization_not_triggered_below_threshold():
    rag = make_rag()
    # 13 messages -> older = 10, exactly at the threshold, not exceeding it.
    history = [{"role": "user", "content": f"Question {i}"} for i in range(13)]

    rag._prepare_conversation_context(history, "Question 12")

    rag.generator.summarize_conversation.assert_not_called()


def test_summarization_triggered_above_threshold():
    rag = make_rag()
    rag.generator.summarize_conversation.return_value = "They discussed several earlier topics."
    # 14 messages -> older = 11, exceeding the threshold of 10.
    history = [{"role": "user", "content": f"Question {i}"} for i in range(14)]

    prepared = rag._prepare_conversation_context(history, "Question 13")

    rag.generator.summarize_conversation.assert_called_once()
    assert prepared[0]["role"] == "system"
    assert "They discussed several earlier topics." in prepared[0]["content"]
    assert len(prepared) == 1 + RECENT_HISTORY_SIZE


def test_answer_rewrite_receives_prepared_context_not_just_recent_slice():
    rag = make_rag()
    rag.generator.rewrite_query.return_value = "standalone question"
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    history = _paris_ml_eiffel_history()

    rag.answer(question="Tell me more about Paris", conversation_history=history)

    actual_context = rag.generator.rewrite_query.call_args.kwargs["conversation_history"]

    # Proves answer() passes the *prepared* context (recent + relevant
    # older), not the naive history[-3:] slice: the relevant older Paris
    # message must be present even though it falls outside the last 3.
    assert history[1] in actual_context
    assert actual_context != history[-3:]


def test_select_relevant_history_returns_empty_when_no_overlap():
    older = [{"role": "user", "content": "What is machine learning?"}]

    result = select_relevant_history("Tell me about Paris", older)

    assert result == []


def test_select_relevant_history_caps_at_max_messages():
    older = [
        {"role": "user", "content": "Tell me about Paris landmarks"},
        {"role": "assistant", "content": "Paris has many landmarks"},
        {"role": "user", "content": "Paris is also known for its food"},
    ]

    result = select_relevant_history("What about Paris?", older, max_messages=1)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# Citations -- extract_citations()
# ---------------------------------------------------------------------------

def _sample_sources():
    return [
        {
            "citation_id": 1,
            "document_id": "docA",
            "filename": "a.pdf",
            "page_number": 1,
            "chunk_id": 0,
            "rerank_score": 0.9,
        },
        {
            "citation_id": 2,
            "document_id": "docB",
            "filename": "b.pdf",
            "page_number": 4,
            "chunk_id": 3,
            "rerank_score": 0.8,
        },
    ]


def test_extract_citations_valid_single_citation():
    sources = _sample_sources()

    result = extract_citations("The system uses X [1].", sources)

    assert result["valid"] == [1]
    assert result["invalid"] == []
    assert result["citation_map"][1] == sources[0]


def test_extract_citations_supports_multiple_citations():
    sources = _sample_sources()

    result = extract_citations("This claim relies on two sources [1][2].", sources)

    assert result["valid"] == [1, 2]
    assert result["citation_map"][1] == sources[0]
    assert result["citation_map"][2] == sources[1]


def test_extract_citations_detects_invalid_out_of_range_citation():
    sources = _sample_sources()

    result = extract_citations("According to the evidence [5], this is true.", sources)

    assert result["valid"] == []
    assert result["invalid"] == [5]
    assert 5 not in result["citation_map"]


def test_extract_citations_separates_valid_and_invalid_in_same_answer():
    sources = _sample_sources()

    result = extract_citations(
        "Fact A is supported [1], but fact B cites a bad source [9].", sources
    )

    assert result["valid"] == [1]
    assert result["invalid"] == [9]
    assert result["citation_map"] == {1: sources[0]}


def test_extract_citations_no_markers_returns_empty():
    sources = _sample_sources()

    result = extract_citations("No citations here at all.", sources)

    assert result == {"valid": [], "invalid": [], "citation_map": {}}


def test_extract_citations_none_answer_returns_empty():
    sources = _sample_sources()

    result = extract_citations(None, sources)

    assert result == {"valid": [], "invalid": [], "citation_map": {}}


# ---------------------------------------------------------------------------
# Citations -- RAG.answer() integration
# ---------------------------------------------------------------------------

def test_answer_evidence_receives_stable_sequential_citation_ids():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.generator.generate.return_value = "An answer with no citations."
    rag.reranker.rerank.return_value = [("chunk-a", 0.9), ("chunk-b", 0.5)]

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.2),
        ("chunk-b", metadata("docB", 1), 0.3),
    ])

    result = rag.answer(question="a question", conversation_history=[])

    citation_ids = [source["citation_id"] for source in result["sources"]]
    assert citation_ids == [1, 2]


def test_answer_citation_map_points_to_real_evidence_metadata():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.generator.generate.return_value = "The system supports this claim [1]."
    rag.reranker.rerank.return_value = [("chunk-a", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 7, page_number=5, filename="real.pdf"), 0.2),
    ])

    result = rag.answer(question="a question", conversation_history=[])

    assert result["citations"]["valid"] == [1]
    cited_source = result["citations"]["citation_map"][1]
    assert cited_source["filename"] == "real.pdf"
    assert cited_source["page_number"] == 5
    assert cited_source["chunk_id"] == 7
    assert cited_source["document_id"] == "docA"


def test_answer_detects_invalid_citation_from_generated_text():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    # Only one piece of evidence is ever supplied, but the model cites [3].
    rag.generator.generate.return_value = "This is supported by evidence [3]."
    rag.reranker.rerank.return_value = [("chunk-a", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.2),
    ])

    result = rag.answer(question="a question", conversation_history=[])

    assert result["citations"]["valid"] == []
    assert result["citations"]["invalid"] == [3]
    assert 3 not in result["citations"]["citation_map"]


def test_citation_metadata_is_never_taken_from_generated_text():
    """Citation metadata must come entirely from the retrieved chunk's own
    metadata -- never by trusting/parsing anything the LLM said. Here the
    model's answer text names a filename/page that don't correspond to any
    real evidence; the citation map for the marker it used must still
    resolve strictly to the real evidence Python already built, not to
    whatever the model claimed inline.
    """
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.generator.generate.return_value = (
        "According to fake-document.pdf, page 999 [1], the answer is X."
    )
    rag.reranker.rerank.return_value = [("chunk-a", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0, page_number=1, filename="real.pdf"), 0.2),
    ])

    result = rag.answer(question="a question", conversation_history=[])

    cited_source = result["citations"]["citation_map"][1]
    assert cited_source["filename"] == "real.pdf"
    assert cited_source["page_number"] == 1
    assert "fake-document.pdf" not in str(cited_source.values())


# ---------------------------------------------------------------------------
# Phase 8 -- trace IDs
# ---------------------------------------------------------------------------

def test_answer_generates_a_trace_id_when_not_provided():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    result = rag.answer(question="a question", conversation_history=[])

    assert result["trace_id"]
    assert len(result["trace_id"]) == 32


def test_answer_uses_provided_trace_id_instead_of_generating_a_new_one():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    result = rag.answer(
        question="a question", conversation_history=[], trace_id="custom-trace-id"
    )

    assert result["trace_id"] == "custom-trace-id"


def test_answer_propagates_trace_id_to_retrieve():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})

    rag.answer(question="a question", conversation_history=[], trace_id="propagated-id")

    assert rag.retrieve.call_args.kwargs["trace_id"] == "propagated-id"


def test_answer_propagates_trace_id_to_generator_calls():
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.generator.generate.return_value = "an answer"
    rag.reranker.rerank.return_value = [("chunk text", 0.9)]

    rag.store.search.return_value = search_result([
        ("chunk text", metadata("docA", 0), 0.2),
    ])

    rag.answer(question="a question", conversation_history=[], trace_id="gen-trace-id")

    assert rag.generator.generate_queries.call_args.kwargs["trace_id"] == "gen-trace-id"
    assert rag.generator.generate.call_args.kwargs["trace_id"] == "gen-trace-id"


def test_retrieve_generates_a_trace_id_when_called_standalone(caplog):
    rag = make_rag()
    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.2),
    ])
    rag.reranker.rerank.return_value = [("chunk-a", 0.9)]

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        rag.retrieve(retrieval_question="q", search_queries=["q"])

    trace_ids = {payload["trace_id"] for payload in _log_payloads(caplog)}
    assert len(trace_ids) == 1
    assert next(iter(trace_ids))


# ---------------------------------------------------------------------------
# Phase 8 -- structured logging and failure handling
# ---------------------------------------------------------------------------

def test_retrieve_logs_structured_events_for_each_stage(caplog):
    rag = make_rag()
    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.2),
    ])
    rag.reranker.rerank.return_value = [("chunk-a", 0.9)]

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        rag.retrieve(retrieval_question="q", search_queries=["q"], trace_id="stage-trace")

    payloads = _log_payloads(caplog)
    events = {payload["event"] for payload in payloads}

    assert "retrieval_started" in events
    assert "retrieval_candidates_selected" in events
    assert "reranking_completed" in events
    assert "final_evidence_selected" in events
    assert all(payload["trace_id"] == "stage-trace" for payload in payloads)


def test_retrieve_vector_search_failure_is_logged_and_degrades_gracefully(caplog):
    rag = make_rag()
    rag.store.search.side_effect = RuntimeError("chroma is down")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        result = rag.retrieve(retrieval_question="q", search_queries=["q"], trace_id="failure-trace")

    # Doesn't crash -- degrades to the existing "no candidates" empty result.
    assert result == {"candidates": [], "final_evidence": []}

    events = [p for p in _log_payloads(caplog) if p["event"] == "vector_search_failed"]
    assert len(events) == 1
    assert events[0]["trace_id"] == "failure-trace"
    assert "chroma is down" in events[0]["error"]


def test_retrieve_reranking_failure_is_logged_and_falls_back_to_distance_sort(caplog):
    rag = make_rag()
    rag.store.search.return_value = search_result([
        ("chunk-a", metadata("docA", 0), 0.5),
        ("chunk-b", metadata("docB", 0), 0.1),
    ])
    rag.reranker.rerank.side_effect = RuntimeError("reranker model failed to load")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        result = rag.retrieve(
            retrieval_question="q", search_queries=["q"], trace_id="rerank-fail-trace"
        )

    # Same fallback behavior as enable_reranking=False: ascending by distance.
    documents = [item["document"] for item in result["final_evidence"]]
    assert documents == ["chunk-b", "chunk-a"]
    assert all(item["rerank_score"] is None for item in result["final_evidence"])

    events = [p for p in _log_payloads(caplog) if p["event"] == "reranking_failed"]
    assert len(events) == 1
    assert "reranker model failed to load" in events[0]["error"]


def test_retrieve_empty_result_is_logged_as_warning(caplog):
    rag = make_rag()
    rag.store.search.return_value = search_result([])

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        rag.retrieve(retrieval_question="q", search_queries=["q"], trace_id="empty-trace")

    events = [p for p in _log_payloads(caplog) if p["event"] == "retrieval_empty"]
    assert len(events) == 1
    assert events[0]["level"] == "WARNING"


def test_answer_generation_failure_is_logged(caplog):
    rag = make_rag()
    rag.generator.generate_queries.return_value = ["q"]
    rag.reranker.rerank.return_value = [("chunk text", 0.9)]
    rag.store.search.return_value = search_result([
        ("chunk text", metadata("docA", 0), 0.2),
    ])
    rag.generator.generate.side_effect = GenerationUnavailableError("Gemini is down")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        result = rag.answer(
            question="a question", conversation_history=[], trace_id="gen-fail-trace"
        )

    assert result["answer"] is None
    assert result["error"]["message"] == "Gemini is down"

    events = [p for p in _log_payloads(caplog) if p["event"] == "answer_generation_failed"]
    assert len(events) == 1
    assert events[0]["trace_id"] == "gen-fail-trace"


def test_citation_validation_logs_invalid_citations_as_warning(caplog):
    sources = _sample_sources()

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        extract_citations("This cites a bad source [9].", sources, trace_id="citation-trace")

    events = [p for p in _log_payloads(caplog) if p["event"] == "citation_validation_completed"]
    assert len(events) == 1
    assert events[0]["level"] == "WARNING"
    assert events[0]["invalid_citations"] == [9]


def test_citation_validation_logs_info_when_all_valid(caplog):
    sources = _sample_sources()

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        extract_citations("This cites a good source [1].", sources, trace_id="citation-ok-trace")

    events = [p for p in _log_payloads(caplog) if p["event"] == "citation_validation_completed"]
    assert len(events) == 1
    assert events[0]["level"] == "INFO"
