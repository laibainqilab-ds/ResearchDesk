import hashlib
import logging
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.parsers import UnsupportedFileTypeError, parse_document
from app.ingestion.text_cleaner import clean_text
from app.observability import log_event, new_trace_id

logger = logging.getLogger(__name__)


class EmptyDocumentError(Exception):
    """Raised when a document contains no usable text after cleaning."""


class DocumentParsingError(Exception):
    """Raised when a document cannot be parsed, embedded, or stored."""


class DuplicateDocumentError(Exception):
    """Raised when a document with identical content is already indexed."""

    def __init__(self, document_id: str, filename: str, chunk_count: int):
        super().__init__(
            f"'{filename}' already exists in the index "
            f"(document_id={document_id[:16]}..., {chunk_count} chunks)."
        )
        self.document_id = document_id
        self.filename = filename
        self.chunk_count = chunk_count


def compute_document_id(file_bytes: bytes) -> str:
    """Content-based document ID so re-uploading identical bytes is detectable
    regardless of filename."""
    return hashlib.sha256(file_bytes).hexdigest()


def ingest_file(file_path: str, filename: str, store, embedder, trace_id: str | None = None) -> dict:
    """Parse, chunk, embed, and store a single file.

    `store` must provide count_document_chunks, add_documents, delete_document.
    `embedder` must provide embed(texts) -> list[list[float]].

    Raises UnsupportedFileTypeError, EmptyDocumentError, DuplicateDocumentError,
    or DocumentParsingError. On success returns a summary dict with
    document_id, filename, file_type, page_count, and chunk_count.
    """
    trace_id = trace_id or new_trace_id()

    log_event(trace_id, "ingestion_started", filename=filename)

    path = Path(file_path)
    file_bytes = path.read_bytes()
    document_id = compute_document_id(file_bytes)

    existing_chunk_count = store.count_document_chunks(document_id)

    if existing_chunk_count > 0:
        log_event(
            trace_id,
            "ingestion_duplicate_rejected",
            level=logging.INFO,
            filename=filename,
            document_id=document_id,
            existing_chunk_count=existing_chunk_count,
        )
        raise DuplicateDocumentError(document_id, filename, existing_chunk_count)

    try:
        pages, file_type = parse_document(file_path, filename)
    except UnsupportedFileTypeError as error:
        log_event(
            trace_id,
            "ingestion_unsupported_file_type",
            level=logging.WARNING,
            filename=filename,
            error=str(error),
        )
        raise
    except Exception as error:
        log_event(
            trace_id,
            "ingestion_parsing_failed",
            level=logging.WARNING,
            filename=filename,
            error=str(error),
        )
        raise DocumentParsingError(f"Failed to parse '{filename}': {error}") from error

    cleaned_pages = [
        {
            "page_number": page["page_number"],
            "text": clean_text(page["text"]),
        }
        for page in pages
    ]

    if not any(page["text"] for page in cleaned_pages):
        log_event(
            trace_id,
            "ingestion_empty_document",
            level=logging.WARNING,
            filename=filename,
            page_count=len(cleaned_pages),
        )
        raise EmptyDocumentError(f"'{filename}' contains no usable text after cleaning.")

    chunks = chunk_pages(
        pages=cleaned_pages,
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        source=str(path),
    )

    ids = [f"{chunk.document_id}_chunk_{chunk.chunk_id}" for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = [_build_metadata(chunk) for chunk in chunks]

    try:
        embeddings = embedder.embed(texts)
        store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as error:
        logger.warning(
            "Ingestion failed for '%s' after chunking, rolling back partial data: %s",
            filename,
            error,
        )
        log_event(
            trace_id,
            "ingestion_embedding_or_storage_failed",
            level=logging.WARNING,
            filename=filename,
            document_id=document_id,
            chunk_count=len(chunks),
            error=str(error),
        )
        store.delete_document(document_id)
        raise DocumentParsingError(
            f"Failed to embed or store '{filename}': {error}"
        ) from error

    page_numbers = [
        page["page_number"] for page in cleaned_pages if page["page_number"] is not None
    ]

    log_event(
        trace_id,
        "ingestion_completed",
        filename=filename,
        document_id=document_id,
        file_type=file_type,
        chunk_count=len(chunks),
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "page_count": max(page_numbers) if page_numbers else None,
        "chunk_count": len(chunks),
    }


def _build_metadata(chunk) -> dict:
    metadata = {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "file_type": chunk.file_type,
        "chunk_id": chunk.chunk_id,
    }

    if chunk.source is not None:
        metadata["source"] = chunk.source

    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number

    return metadata
