"""Unit tests for the Phase 9 deterministic retrieval tool layer.

`RAG` is mocked exactly like tests/test_rag.py's make_rag(): no real
embedding model, no real ChromaDB, no real Gemini call. These tests check
that RetrievalTools delegates to the right existing collaborator with the
right arguments and shapes its return value correctly -- not retrieval
quality, which is already covered by tests/test_rag.py.
"""

from unittest.mock import Mock

from app.agents.tools import RetrievalTools
from app.rag import RAG


def make_rag():
    rag = RAG.__new__(RAG)
    rag.embedder = Mock()
    rag.store = Mock()
    rag.generator = Mock()
    rag.reranker = Mock()
    return rag


def make_tools():
    rag = make_rag()
    rag.retrieve = Mock(return_value={"candidates": [], "final_evidence": []})
    return RetrievalTools(rag), rag


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------

def test_search_documents_calls_rag_retrieve_with_single_query_no_document_filter():
    tools, rag = make_tools()

    tools.search_documents("what is Evo 2?", top_k=5, enable_reranking=False)

    rag.retrieve.assert_called_once_with(
        retrieval_question="what is Evo 2?",
        search_queries=["what is Evo 2?"],
        top_k=5,
        document_id=None,
        enable_reranking=False,
        trace_id=None,
    )


def test_search_documents_returns_final_evidence_list():
    tools, rag = make_tools()
    evidence = [{"document_id": "docA", "chunk_id": 0}]
    rag.retrieve.return_value = {"candidates": [], "final_evidence": evidence}

    result = tools.search_documents("a question")

    assert result == evidence


def test_search_documents_defaults_top_k_and_reranking():
    tools, rag = make_tools()

    tools.search_documents("a question")

    assert rag.retrieve.call_args.kwargs["top_k"] == 3
    assert rag.retrieve.call_args.kwargs["enable_reranking"] is True


def test_search_documents_propagates_trace_id():
    tools, rag = make_tools()

    tools.search_documents("a question", trace_id="trace-123")

    assert rag.retrieve.call_args.kwargs["trace_id"] == "trace-123"


# ---------------------------------------------------------------------------
# search_specific_document
# ---------------------------------------------------------------------------

def test_search_specific_document_passes_document_id_filter():
    tools, rag = make_tools()

    tools.search_specific_document("docA", "a question", top_k=2)

    rag.retrieve.assert_called_once_with(
        retrieval_question="a question",
        search_queries=["a question"],
        top_k=2,
        document_id="docA",
        enable_reranking=True,
        trace_id=None,
    )


def test_search_specific_document_returns_final_evidence_list():
    tools, rag = make_tools()
    evidence = [{"document_id": "docA", "chunk_id": 1}]
    rag.retrieve.return_value = {"candidates": [], "final_evidence": evidence}

    result = tools.search_specific_document("docA", "a question")

    assert result == evidence


# ---------------------------------------------------------------------------
# list_available_documents
# ---------------------------------------------------------------------------

def test_list_available_documents_wraps_store_list_documents():
    tools, rag = make_tools()
    documents = [{"document_id": "docA", "filename": "a.pdf"}]
    rag.store.list_documents.return_value = documents

    result = tools.list_available_documents()

    rag.store.list_documents.assert_called_once_with()
    assert result == documents


# ---------------------------------------------------------------------------
# get_document_metadata
# ---------------------------------------------------------------------------

def test_get_document_metadata_returns_matching_document():
    tools, rag = make_tools()
    rag.store.list_documents.return_value = [
        {"document_id": "docA", "filename": "a.pdf"},
        {"document_id": "docB", "filename": "b.pdf"},
    ]

    result = tools.get_document_metadata("docB")

    assert result == {"document_id": "docB", "filename": "b.pdf"}


def test_get_document_metadata_returns_none_when_not_found():
    tools, rag = make_tools()
    rag.store.list_documents.return_value = [
        {"document_id": "docA", "filename": "a.pdf"},
    ]

    result = tools.get_document_metadata("does-not-exist")

    assert result is None


# ---------------------------------------------------------------------------
# get_document_page
# ---------------------------------------------------------------------------

def test_get_document_page_wraps_store_get_document_chunks():
    tools, rag = make_tools()
    chunks = [{"document_id": "docA", "page_number": 3, "chunk_id": 4}]
    rag.store.get_document_chunks.return_value = chunks

    result = tools.get_document_page("docA", 3)

    rag.store.get_document_chunks.assert_called_once_with("docA", page_number=3)
    assert result == chunks
