"""DocumentService tests. The real ingestion pipeline (embedding, Chroma)
is replaced with a fake `ingest_file` (same pattern as
tests/test_pipeline.py's FakeEmbedder) so these tests exercise only
DocumentService's own logic: ownership, Postgres bookkeeping via
DocumentRepository, and saving uploaded bytes under a throwaway directory
rather than the real data/documents/.
"""

from unittest.mock import Mock

import pytest

from app.ingestion.pipeline import DuplicateDocumentError
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import UserRepository
from app.services import document_service as document_service_module
from app.services.document_service import DocumentAccessDeniedError, DocumentService


def make_fake_ingest_file(result):
    def fake_ingest_file(file_path, filename, store, embedder, owner_id=None, trace_id=None):
        return result

    return fake_ingest_file


def make_document_service(db_session, tmp_path, monkeypatch, ingest_result=None):
    monkeypatch.setattr(document_service_module, "DOCUMENTS_DIR", tmp_path)

    default_result = ingest_result or {
        "document_id": "doc1",
        "filename": "a.pdf",
        "file_type": "pdf",
        "page_count": 3,
        "chunk_count": 5,
    }
    monkeypatch.setattr(
        document_service_module, "ingest_file", make_fake_ingest_file(default_result)
    )

    rag = Mock()
    service = DocumentService(rag=rag, document_repository=DocumentRepository(db_session))
    user = UserRepository(db_session).create(email="user@example.com", password_hash="hashed")
    return service, user, rag


def test_ingest_persists_document_with_owner(db_session, tmp_path, monkeypatch):
    service, user, _ = make_document_service(db_session, tmp_path, monkeypatch)

    result = service.ingest(owner_id=user.id, file_bytes=b"pdf-bytes", filename="a.pdf")

    assert result["document_id"] == "doc1"
    listing = service.list_for_owner(user.id)
    assert len(listing) == 1
    assert listing[0]["filename"] == "a.pdf"


def test_ingest_saves_uploaded_bytes_under_throwaway_dir_not_real_data_dir(db_session, tmp_path, monkeypatch):
    service, user, _ = make_document_service(db_session, tmp_path, monkeypatch)

    service.ingest(owner_id=user.id, file_bytes=b"pdf-bytes", filename="a.pdf")

    saved_files = list(tmp_path.glob("*a.pdf"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"pdf-bytes"


def test_ingest_propagates_duplicate_error_and_cleans_up_saved_file(db_session, tmp_path, monkeypatch):
    def failing_ingest(file_path, filename, store, embedder, owner_id=None, trace_id=None):
        raise DuplicateDocumentError("doc1", filename, 5)

    monkeypatch.setattr(document_service_module, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(document_service_module, "ingest_file", failing_ingest)

    rag = Mock()
    service = DocumentService(rag=rag, document_repository=DocumentRepository(db_session))
    user = UserRepository(db_session).create(email="user@example.com", password_hash="hashed")

    with pytest.raises(DuplicateDocumentError):
        service.ingest(owner_id=user.id, file_bytes=b"pdf-bytes", filename="a.pdf")

    assert list(tmp_path.glob("*a.pdf")) == []
    assert service.list_for_owner(user.id) == []


def test_delete_removes_from_store_and_marks_deleted(db_session, tmp_path, monkeypatch):
    service, user, rag = make_document_service(db_session, tmp_path, monkeypatch)
    service.ingest(owner_id=user.id, file_bytes=b"pdf-bytes", filename="a.pdf")

    service.delete(owner_id=user.id, document_id="doc1")

    rag.store.delete_document.assert_called_once_with("doc1")
    assert service.list_for_owner(user.id) == []


def test_delete_by_non_owner_raises_access_denied(db_session, tmp_path, monkeypatch):
    service, user, rag = make_document_service(db_session, tmp_path, monkeypatch)
    service.ingest(owner_id=user.id, file_bytes=b"pdf-bytes", filename="a.pdf")
    other = UserRepository(db_session).create(email="other@example.com", password_hash="hashed")

    with pytest.raises(DocumentAccessDeniedError):
        service.delete(owner_id=other.id, document_id="doc1")

    rag.store.delete_document.assert_not_called()
