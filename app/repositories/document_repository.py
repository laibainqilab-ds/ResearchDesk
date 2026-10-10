from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, document_id: str) -> Document | None:
        return self.db.get(Document, document_id)

    def list_for_owner(self, owner_id: str) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.owner_id == owner_id, Document.status == "indexed")
            .order_by(Document.created_at.desc())
        )
        return list(self.db.execute(statement).scalars().all())

    def upsert_indexed(
        self,
        document_id: str,
        owner_id: str,
        filename: str,
        file_type: str | None,
        page_count: int | None,
        chunk_count: int,
    ) -> Document:
        document = self.get(document_id)

        if document is None:
            document = Document(
                document_id=document_id,
                owner_id=owner_id,
                filename=filename,
                file_type=file_type,
                page_count=page_count,
                chunk_count=chunk_count,
                status="indexed",
            )
            self.db.add(document)
        else:
            document.filename = filename
            document.file_type = file_type
            document.page_count = page_count
            document.chunk_count = chunk_count
            document.status = "indexed"

        self.db.commit()
        self.db.refresh(document)
        return document

    def mark_deleted(self, document: Document) -> None:
        document.status = "deleted"
        self.db.commit()
