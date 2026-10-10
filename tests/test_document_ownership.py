"""Regression tests for per-user document ownership enforcement.

Covers two fixes:
1. RAG/Agent retrieval must be restricted to the authenticated user's own
   documents at the ChromaDB `where`-filter layer (app.rag.RAG.retrieve's
   `owner_id` parameter), not by trusting a caller-supplied document_id or
   filtering results after the fact.
2. DocumentRepository.upsert_indexed must transfer ownership to the new
   uploader when identical content is re-ingested after the previous
   owner deleted it, without weakening duplicate detection.

Follows existing conventions: a real, tmp_path-isolated VectorStore +
deterministic fake embedder (tests/test_pipeline.py's style) for anything
that needs real ChromaDB filtering behavior, `RAG.__new__(RAG)` with mocked
collaborators (tests/test_rag.py's `make_rag()` style) to avoid loading a
real embedding/reranker model or calling Gemini, and the in-memory SQLite
`db_session` fixture (tests/conftest.py) for repository-level checks. No
real Postgres, no real ChromaDB data, no external LLM calls anywhere in
this file.
"""

from unittest.mock import Mock

import pytest

from app.agents.tools import RetrievalTools
from app.ingestion.pipeline import DuplicateDocumentError, ingest_file
from app.ingestion.vector_store import VectorStore
from app.rag import RAG
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import UserRepository


class MarkerEmbedder:
    """Deterministic embedder: text containing a known marker word maps to
    a fixed vector, so ChromaDB's nearest-neighbor search is fully
    predictable regardless of real semantic meaning. A query embedding
    identical to one document's vector guarantees that document would be
    the best (or only) match if the owner filter were not applied."""

    VECTORS = {
        "alpha": [1.0, 0.0, 0.0],
        "bravo": [0.0, 1.0, 0.0],
    }

    def embed(self, texts):
        vectors = []
        for text in texts:
            for marker, vector in self.VECTORS.items():
                if marker in text:
                    vectors.append(vector)
                    break
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def make_rag_with_real_store(tmp_path) -> RAG:
    """A RAG instance with a real, tmp_path-isolated VectorStore + the
    deterministic MarkerEmbedder, but mocked generator/reranker -- bypasses
    __init__ so no real Sentence Transformers model or Gemini client is
    constructed (same technique as tests/test_rag.py's make_rag())."""
    rag = RAG.__new__(RAG)
    rag.store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    rag.embedder = MarkerEmbedder()
    rag.generator = Mock()
    rag.reranker = Mock()
    return rag


def seed_two_users_documents(tmp_path, store, embedder):
    """Ingests one document for "user-a" (about "alpha") and one for
    "user-b" (about "bravo") into the same shared ChromaDB collection."""
    file_a = tmp_path / "a.txt"
    file_a.write_text("This document discusses the alpha project roadmap in detail.", encoding="utf-8")
    result_a = ingest_file(str(file_a), "a.txt", store, embedder, owner_id="user-a")

    file_b = tmp_path / "b.txt"
    file_b.write_text("This document discusses the bravo project roadmap in detail.", encoding="utf-8")
    result_b = ingest_file(str(file_b), "b.txt", store, embedder, owner_id="user-b")

    return result_a["document_id"], result_b["document_id"]


# ---------------------------------------------------------------------------
# RAG retrieval isolation
# ---------------------------------------------------------------------------

def test_rag_retrieve_excludes_other_users_documents(tmp_path):
    rag = make_rag_with_real_store(tmp_path)
    doc_a_id, doc_b_id = seed_two_users_documents(tmp_path, rag.store, rag.embedder)

    # Ask the question that would best match user B's document, as user A.
    result = rag.retrieve(
        retrieval_question="Tell me about the bravo project",
        search_queries=["bravo project"],
        top_k=3,
        owner_id="user-a",
        enable_reranking=False,
    )

    returned_document_ids = {item["document_id"] for item in result["final_evidence"]}
    assert doc_b_id not in returned_document_ids
    assert returned_document_ids <= {doc_a_id}


def test_rag_retrieve_returns_nothing_for_user_with_no_documents(tmp_path):
    rag = make_rag_with_real_store(tmp_path)
    seed_two_users_documents(tmp_path, rag.store, rag.embedder)

    result = rag.retrieve(
        retrieval_question="Tell me about the bravo project",
        search_queries=["bravo project"],
        top_k=3,
        owner_id="user-c",  # has uploaded nothing
        enable_reranking=False,
    )

    assert result["final_evidence"] == []
    assert result["candidates"] == []


def test_rag_answer_citations_never_reference_other_users_documents(tmp_path):
    rag = make_rag_with_real_store(tmp_path)
    doc_a_id, doc_b_id = seed_two_users_documents(tmp_path, rag.store, rag.embedder)
    rag.generator.generate.return_value = "Here is what I found [1]."

    result = rag.answer(
        question="Tell me about the bravo project",
        conversation_history=[],
        owner_id="user-a",
        enable_multi_query=False,
        enable_reranking=False,
    )

    cited_document_ids = {source["document_id"] for source in result["sources"]}
    assert doc_b_id not in cited_document_ids
    assert cited_document_ids <= {doc_a_id}


# ---------------------------------------------------------------------------
# Agent retrieval isolation (RetrievalTools -- the agent's only access path)
# ---------------------------------------------------------------------------

def test_agent_tools_search_documents_excludes_other_users_documents(tmp_path):
    rag = make_rag_with_real_store(tmp_path)
    doc_a_id, doc_b_id = seed_two_users_documents(tmp_path, rag.store, rag.embedder)
    tools = RetrievalTools(rag)

    evidence = tools.search_documents(
        "bravo project", top_k=3, owner_id="user-a", enable_reranking=False
    )

    returned_document_ids = {item["document_id"] for item in evidence}
    assert doc_b_id not in returned_document_ids
    assert returned_document_ids <= {doc_a_id}


def test_agent_tools_search_documents_returns_nothing_for_user_with_no_documents(tmp_path):
    rag = make_rag_with_real_store(tmp_path)
    seed_two_users_documents(tmp_path, rag.store, rag.embedder)
    tools = RetrievalTools(rag)

    evidence = tools.search_documents(
        "bravo project", top_k=3, owner_id="user-c", enable_reranking=False
    )

    assert evidence == []


# ---------------------------------------------------------------------------
# Identical-document ownership (delete by one user, re-upload by another)
# ---------------------------------------------------------------------------

def test_reupload_after_deletion_transfers_ownership_to_new_uploader(db_session):
    users = UserRepository(db_session)
    documents = DocumentRepository(db_session)
    user_a = users.create(email="a@example.com", password_hash="hashed")
    user_b = users.create(email="b@example.com", password_hash="hashed")

    document = documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_a.id, filename="a.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )
    documents.mark_deleted(document)  # user A deletes it

    documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_b.id, filename="b.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )  # user B uploads byte-identical content

    reloaded = documents.get("shared-hash")
    assert reloaded.owner_id == user_b.id
    assert reloaded.status == "indexed"


def test_reupload_after_deletion_removes_access_for_original_owner(db_session):
    users = UserRepository(db_session)
    documents = DocumentRepository(db_session)
    user_a = users.create(email="a@example.com", password_hash="hashed")
    user_b = users.create(email="b@example.com", password_hash="hashed")

    document = documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_a.id, filename="a.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )
    documents.mark_deleted(document)
    documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_b.id, filename="b.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )

    assert documents.list_for_owner(user_a.id) == []
    assert [d.document_id for d in documents.list_for_owner(user_b.id)] == ["shared-hash"]


def test_upsert_indexed_does_not_reassign_owner_for_a_still_active_document(db_session):
    """Defensive check: if this method were ever reached for a row that is
    still "indexed" (not deleted) and owned by someone else -- which
    shouldn't happen via DocumentService.ingest(), since ingest_file's
    duplicate check blocks it first -- ownership must NOT be silently
    transferred."""
    users = UserRepository(db_session)
    documents = DocumentRepository(db_session)
    user_a = users.create(email="a@example.com", password_hash="hashed")
    user_b = users.create(email="b@example.com", password_hash="hashed")

    documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_a.id, filename="a.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )  # still "indexed" -- never deleted

    documents.upsert_indexed(
        document_id="shared-hash", owner_id=user_b.id, filename="b.pdf",
        file_type="PDF", page_count=1, chunk_count=1,
    )

    assert documents.get("shared-hash").owner_id == user_a.id


# ---------------------------------------------------------------------------
# Duplicate detection and metadata tagging continue to work
# ---------------------------------------------------------------------------

def test_ingest_tags_chunks_with_owner_id(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    embedder = MarkerEmbedder()
    file_path = tmp_path / "a.txt"
    file_path.write_text("This document discusses the alpha project roadmap in detail.", encoding="utf-8")

    result = ingest_file(str(file_path), "a.txt", store, embedder, owner_id="user-a")

    chunks = store.get_document_chunks(result["document_id"])
    assert chunks
    assert all(chunk["owner_id"] == "user-a" for chunk in chunks)


def test_duplicate_detection_still_blocks_reingestion_regardless_of_owner_id(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    embedder = MarkerEmbedder()
    file_path = tmp_path / "a.txt"
    file_path.write_text("This document discusses the alpha project roadmap in detail.", encoding="utf-8")

    ingest_file(str(file_path), "a.txt", store, embedder, owner_id="user-a")

    with pytest.raises(DuplicateDocumentError):
        # Same byte-identical content, different (claimed) owner, while the
        # first copy is still actively indexed -- must still be rejected.
        ingest_file(str(file_path), "a.txt", store, embedder, owner_id="user-b")
