import json
import logging

import pytest

from app.ingestion.parsers import UnsupportedFileTypeError
from app.ingestion.pipeline import (
    DocumentParsingError,
    DuplicateDocumentError,
    EmptyDocumentError,
    ingest_file,
)
from app.ingestion.vector_store import VectorStore

RESEARCHDESK_LOGGER = "researchdesk"


def _log_payloads(caplog):
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == RESEARCHDESK_LOGGER
    ]


class FakeEmbedder:
    """Deterministic stand-in for the real Sentence Transformers embedder so
    pipeline tests don't depend on downloading/loading the actual model."""

    def embed(self, texts):
        return [[float(len(text) % 7), 0.0, 1.0] for text in texts]


class FailingEmbedder:
    """Raises during embed() to exercise the ingestion embedding-failure path."""

    def embed(self, texts):
        raise RuntimeError("embedding backend unavailable")


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma"))


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _make_pdf(file_path, pages_text):
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    for text in pages_text:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(str(file_path))
    document.close()


def test_ingest_txt_file(tmp_path, store, embedder):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a test document about ResearchDesk.", encoding="utf-8")

    result = ingest_file(str(file_path), "notes.txt", store, embedder)

    assert result["file_type"] == "TXT"
    assert result["chunk_count"] >= 1
    assert result["page_count"] is None
    assert store.count() == result["chunk_count"]


def test_ingest_markdown_file(tmp_path, store, embedder):
    file_path = tmp_path / "guide.md"
    file_path.write_text("# Guide\n\nUseful content here.", encoding="utf-8")

    result = ingest_file(str(file_path), "guide.md", store, embedder)

    assert result["file_type"] == "MARKDOWN"
    assert result["chunk_count"] >= 1
    assert result["page_count"] is None


def test_ingest_empty_txt_is_rejected(tmp_path, store, embedder):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\n  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        ingest_file(str(file_path), "empty.txt", store, embedder)

    assert store.count() == 0


def test_ingest_unsupported_extension_is_rejected(tmp_path, store, embedder):
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"not a real archive")

    with pytest.raises(UnsupportedFileTypeError):
        ingest_file(str(file_path), "archive.zip", store, embedder)

    assert store.count() == 0


def test_ingest_malformed_pdf_raises_parsing_error(tmp_path, store, embedder):
    pytest.importorskip("pymupdf")

    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"%PDF-1.4 this is not a real pdf body")

    with pytest.raises(DocumentParsingError):
        ingest_file(str(file_path), "broken.pdf", store, embedder)

    assert store.count() == 0


def test_duplicate_content_is_rejected(tmp_path, store, embedder):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Duplicate content check.", encoding="utf-8")

    first = ingest_file(str(file_path), "doc.txt", store, embedder)

    with pytest.raises(DuplicateDocumentError):
        ingest_file(str(file_path), "doc.txt", store, embedder)

    assert store.count() == first["chunk_count"]


def test_same_content_different_filename_is_still_duplicate(tmp_path, store, embedder):
    content = "Identical bytes, different filename."
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text(content, encoding="utf-8")
    file_b.write_text(content, encoding="utf-8")

    ingest_file(str(file_a), "a.txt", store, embedder)

    with pytest.raises(DuplicateDocumentError):
        ingest_file(str(file_b), "b.txt", store, embedder)


def test_multiple_distinct_documents_coexist(tmp_path, store, embedder):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("Document A content.", encoding="utf-8")
    file_b.write_text("Document B content, which is different.", encoding="utf-8")

    result_a = ingest_file(str(file_a), "a.txt", store, embedder)
    result_b = ingest_file(str(file_b), "b.txt", store, embedder)

    document_ids = {document["document_id"] for document in store.list_documents()}

    assert result_a["document_id"] != result_b["document_id"]
    assert {result_a["document_id"], result_b["document_id"]} == document_ids


def test_delete_one_document_preserves_others(tmp_path, store, embedder):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("Document A content.", encoding="utf-8")
    file_b.write_text("Document B content, which is different.", encoding="utf-8")

    result_a = ingest_file(str(file_a), "a.txt", store, embedder)
    result_b = ingest_file(str(file_b), "b.txt", store, embedder)

    store.delete_document(result_a["document_id"])

    remaining_ids = {document["document_id"] for document in store.list_documents()}

    assert result_a["document_id"] not in remaining_ids
    assert result_b["document_id"] in remaining_ids
    assert store.count() == result_b["chunk_count"]


def test_pdf_page_metadata_is_preserved(tmp_path, store, embedder):
    file_path = tmp_path / "sample.pdf"
    _make_pdf(file_path, ["PDF page content for metadata test."])

    result = ingest_file(str(file_path), "sample.pdf", store, embedder)

    assert result["file_type"] == "PDF"
    assert result["page_count"] == 1

    records = store.collection.get(
        where={"document_id": result["document_id"]},
        include=["metadatas"],
    )

    metadata = records["metadatas"][0]
    assert metadata["page_number"] == 1
    assert metadata["document_id"] == result["document_id"]
    assert metadata["filename"] == "sample.pdf"
    assert metadata["file_type"] == "PDF"
    assert metadata["chunk_id"] == 0
    assert metadata["source"] == str(file_path)


def test_txt_metadata_has_no_page_number_key(tmp_path, store, embedder):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("TXT content for metadata test.", encoding="utf-8")

    result = ingest_file(str(file_path), "notes.txt", store, embedder)

    records = store.collection.get(
        where={"document_id": result["document_id"]},
        include=["metadatas"],
    )

    metadata = records["metadatas"][0]
    assert "page_number" not in metadata
    assert metadata["file_type"] == "TXT"
    assert metadata["source"] == str(file_path)


# ---------------------------------------------------------------------------
# Phase 8 -- structured logging for ingestion failure paths
#
# Exception-raising behavior for these failure paths is already covered
# above; these tests only check that each also produces a structured log
# event with useful diagnostics, not the exception behavior itself.
# ---------------------------------------------------------------------------

def test_duplicate_document_rejection_is_logged(tmp_path, store, embedder, caplog):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Duplicate content check.", encoding="utf-8")

    ingest_file(str(file_path), "doc.txt", store, embedder)

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        with pytest.raises(DuplicateDocumentError):
            ingest_file(str(file_path), "doc.txt", store, embedder)

    events = [p for p in _log_payloads(caplog) if p["event"] == "ingestion_duplicate_rejected"]
    assert len(events) == 1
    assert events[0]["filename"] == "doc.txt"


def test_unsupported_file_type_is_logged(tmp_path, store, embedder, caplog):
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"not a real archive")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        with pytest.raises(UnsupportedFileTypeError):
            ingest_file(str(file_path), "archive.zip", store, embedder)

    events = [p for p in _log_payloads(caplog) if p["event"] == "ingestion_unsupported_file_type"]
    assert len(events) == 1


def test_empty_document_is_logged(tmp_path, store, embedder, caplog):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\n  ", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        with pytest.raises(EmptyDocumentError):
            ingest_file(str(file_path), "empty.txt", store, embedder)

    events = [p for p in _log_payloads(caplog) if p["event"] == "ingestion_empty_document"]
    assert len(events) == 1


def test_embedding_failure_is_logged_and_still_rolled_back(tmp_path, store, caplog):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Content that will fail to embed.", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        with pytest.raises(DocumentParsingError):
            ingest_file(str(file_path), "doc.txt", store, FailingEmbedder())

    # Existing rollback behavior preserved -- nothing left behind on failure.
    assert store.count() == 0

    events = [
        p for p in _log_payloads(caplog) if p["event"] == "ingestion_embedding_or_storage_failed"
    ]
    assert len(events) == 1
    assert "embedding backend unavailable" in events[0]["error"]


def test_successful_ingestion_is_logged(tmp_path, store, embedder, caplog):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Some content for logging test.", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=RESEARCHDESK_LOGGER):
        result = ingest_file(str(file_path), "notes.txt", store, embedder)

    events = [p for p in _log_payloads(caplog) if p["event"] == "ingestion_completed"]
    assert len(events) == 1
    assert events[0]["document_id"] == result["document_id"]
    assert events[0]["chunk_count"] == result["chunk_count"]
