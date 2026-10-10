"""Document Service: owns document lifecycle (ingest, list, delete) and
ownership. Chunks/embeddings stay in ChromaDB via RAG.store /
app.ingestion.pipeline; ownership and lifecycle bookkeeping live in
PostgreSQL via DocumentRepository -- this service is the only place that
talks to both.
"""

from pathlib import Path

from app.ingestion.pipeline import (
    DocumentParsingError,
    DuplicateDocumentError,
    EmptyDocumentError,
    compute_document_id,
    ingest_file,
)
from app.ingestion.parsers import UnsupportedFileTypeError
from app.rag import RAG
from app.repositories.document_repository import DocumentRepository

DOCUMENTS_DIR = Path("data/documents")


class DocumentAccessDeniedError(Exception):
    """Raised when a user tries to delete a document they don't own."""


class DocumentService:
    def __init__(self, rag: RAG, document_repository: DocumentRepository):
        self.rag = rag
        self.documents = document_repository

    def ingest(self, owner_id: str, file_bytes: bytes, filename: str) -> dict:
        """Save the uploaded file, ingest it into Chroma via the existing
        pipeline, and persist ownership/lifecycle bookkeeping in Postgres.

        Raises UnsupportedFileTypeError, EmptyDocumentError,
        DuplicateDocumentError, or DocumentParsingError -- same contract as
        app.ingestion.pipeline.ingest_file.
        """
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        document_id = compute_document_id(file_bytes)
        save_path = DOCUMENTS_DIR / f"{document_id[:16]}_{filename}"

        try:
            save_path.write_bytes(file_bytes)

            result = ingest_file(
                file_path=str(save_path),
                filename=filename,
                store=self.rag.store,
                embedder=self.rag.embedder,
                owner_id=owner_id,
            )
        except (UnsupportedFileTypeError, DuplicateDocumentError, EmptyDocumentError, DocumentParsingError):
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            raise

        document = self.documents.upsert_indexed(
            document_id=result["document_id"],
            owner_id=owner_id,
            filename=result["filename"],
            file_type=result["file_type"],
            page_count=result["page_count"],
            chunk_count=result["chunk_count"],
        )

        return {**result, "created_at": document.created_at}

    def list_for_owner(self, owner_id: str) -> list[dict]:
        return [
            {
                "document_id": document.document_id,
                "filename": document.filename,
                "file_type": document.file_type,
                "page_count": document.page_count,
                "chunk_count": document.chunk_count,
                "created_at": document.created_at,
            }
            for document in self.documents.list_for_owner(owner_id)
        ]

    def delete(self, owner_id: str, document_id: str) -> None:
        document = self.documents.get(document_id)

        if document is None or document.owner_id != owner_id or document.status != "indexed":
            raise DocumentAccessDeniedError(document_id)

        self.rag.store.delete_document(document_id)
        self.documents.mark_deleted(document)
