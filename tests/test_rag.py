"""Unit tests for app/rag.py's retrieval/generation flag plumbing.

All collaborators (embedder, store, generator, reranker) are mocked, so
these tests never load a real Sentence Transformers model, never touch
ChromaDB, and never make a live Gemini call.
"""

import logging
from unittest.mock import Mock

from app.rag import RAG


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

    rag.answer(question="follow up", conversation_history=history)

    rag.generator.rewrite_query.assert_called_once_with(
        question="follow up", conversation_history=history[-3:]
    )
    rag.generator.generate_queries.assert_called_once_with(
        question="rewritten question", num_queries=3
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

    rag.answer(question="follow up", conversation_history=history, enable_query_rewrite=False)

    rag.generator.rewrite_query.assert_not_called()
    rag.generator.generate_queries.assert_called_once_with(question="follow up", num_queries=3)


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
