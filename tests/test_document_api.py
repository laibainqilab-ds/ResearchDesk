"""HTTP integration tests for the document endpoints
(POST/GET /documents, DELETE /documents/{id}).

Isolation mirrors two existing patterns rather than inventing a new one:
- tests/test_api.py's `get_db` -> in-memory SQLite `db_session` override and
  `get_rag` -> Mock() override, with TestClient deliberately NOT used as a
  context manager (so the app's lifespan hook, which would call
  Base.metadata.create_all against the *real* Postgres engine, never fires).
- tests/test_document_service.py's monkeypatching of
  app.services.document_service.ingest_file and .DOCUMENTS_DIR, so the real
  ingestion pipeline, real Chroma, and the real data/documents/ directory
  are never touched even though a real multipart UploadFile request is sent
  and real router/service/repository code runs.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_rag
from app.ingestion.parsers import UnsupportedFileTypeError
from app.ingestion.pipeline import DocumentParsingError, DuplicateDocumentError, EmptyDocumentError
from app.main import app
from app.services import document_service as document_service_module

DEFAULT_INGEST_RESULT = {
    "document_id": "doc1",
    "filename": "a.pdf",
    "file_type": "PDF",
    "page_count": 3,
    "chunk_count": 5,
}


def _fake_ingest_file(file_path, filename, store, embedder, trace_id=None):
    return dict(DEFAULT_INGEST_RESULT, filename=filename)


def _raising_ingest_file(error):
    def fake(file_path, filename, store, embedder, trace_id=None):
        raise error

    return fake


@pytest.fixture()
def client(db_session, tmp_path, monkeypatch):
    def override_get_db():
        yield db_session

    fake_rag = Mock()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rag] = lambda: fake_rag

    monkeypatch.setattr(document_service_module, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(document_service_module, "ingest_file", _fake_ingest_file)

    test_client = TestClient(app)
    test_client.fake_rag = fake_rag  # exposed for assertions on store.delete_document
    yield test_client

    app.dependency_overrides.clear()


def signup_and_get_headers(client, email="user@example.com", password="password123"):
    response = client.post("/auth/signup", json={"email": email, "password": password})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(client, headers, filename="a.pdf", content=b"pdf-bytes"):
    return client.post("/documents", files={"file": (filename, content)}, headers=headers)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def test_upload_document_requires_authentication(client):
    response = upload(client, headers={})

    assert response.status_code == 401


def test_upload_document_success_persists_and_returns_metadata(client):
    headers = signup_and_get_headers(client)

    response = upload(client, headers)

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc1"
    assert body["filename"] == "a.pdf"
    assert body["file_type"] == "PDF"
    assert body["page_count"] == 3
    assert body["chunk_count"] == 5
    assert "created_at" in body

    listing = client.get("/documents", headers=headers).json()
    assert len(listing) == 1
    assert listing[0]["document_id"] == "doc1"


def test_upload_document_unsupported_file_type_returns_415(client, monkeypatch):
    headers = signup_and_get_headers(client)
    monkeypatch.setattr(
        document_service_module,
        "ingest_file",
        _raising_ingest_file(UnsupportedFileTypeError("Unsupported file type '.zip'.")),
    )

    response = upload(client, headers, filename="a.zip")

    assert response.status_code == 415


def test_upload_document_duplicate_returns_409(client, monkeypatch):
    headers = signup_and_get_headers(client)
    monkeypatch.setattr(
        document_service_module,
        "ingest_file",
        _raising_ingest_file(DuplicateDocumentError("doc1", "a.pdf", 5)),
    )

    response = upload(client, headers)

    assert response.status_code == 409


def test_upload_document_empty_returns_422(client, monkeypatch):
    headers = signup_and_get_headers(client)
    monkeypatch.setattr(
        document_service_module,
        "ingest_file",
        _raising_ingest_file(EmptyDocumentError("'a.pdf' contains no usable text after cleaning.")),
    )

    response = upload(client, headers)

    assert response.status_code == 422


def test_upload_document_parsing_error_returns_422(client, monkeypatch):
    headers = signup_and_get_headers(client)
    monkeypatch.setattr(
        document_service_module,
        "ingest_file",
        _raising_ingest_file(DocumentParsingError("Failed to parse 'a.pdf': boom")),
    )

    response = upload(client, headers)

    assert response.status_code == 422


def test_upload_document_saves_to_throwaway_dir_not_real_data_dir(client, tmp_path):
    headers = signup_and_get_headers(client)

    upload(client, headers)

    saved_files = list(tmp_path.glob("*a.pdf"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"pdf-bytes"


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def test_list_documents_requires_authentication(client):
    response = client.get("/documents")

    assert response.status_code == 401


def test_list_documents_empty_for_new_user(client):
    headers = signup_and_get_headers(client)

    response = client.get("/documents", headers=headers)

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_only_own_documents(client):
    headers_a = signup_and_get_headers(client, "a@example.com")
    headers_b = signup_and_get_headers(client, "b@example.com")

    upload(client, headers_a)

    assert len(client.get("/documents", headers=headers_a).json()) == 1
    assert client.get("/documents", headers=headers_b).json() == []


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_document_requires_authentication(client):
    response = client.delete("/documents/doc1")

    assert response.status_code == 401


def test_delete_document_success_returns_204_and_removes_from_listing(client):
    headers = signup_and_get_headers(client)
    upload(client, headers)

    response = client.delete("/documents/doc1", headers=headers)

    assert response.status_code == 204
    assert client.get("/documents", headers=headers).json() == []
    client.fake_rag.store.delete_document.assert_called_once_with("doc1")


def test_delete_document_by_non_owner_returns_404(client):
    headers_a = signup_and_get_headers(client, "a@example.com")
    headers_b = signup_and_get_headers(client, "b@example.com")
    upload(client, headers_a)

    response = client.delete("/documents/doc1", headers=headers_b)

    assert response.status_code == 404


def test_delete_document_unknown_id_returns_404(client):
    headers = signup_and_get_headers(client)

    response = client.delete("/documents/does-not-exist", headers=headers)

    assert response.status_code == 404
