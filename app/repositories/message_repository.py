from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message, MessageSource


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_chat(self, chat_id: str) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )
        return list(self.db.execute(statement).scalars().all())

    def get(self, message_id: str) -> Message | None:
        return self.db.get(Message, message_id)

    def add_user_message(self, chat_id: str, content: str) -> Message:
        message = Message(chat_id=chat_id, role="user", content=content)
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def add_assistant_message(
        self,
        chat_id: str,
        content: str,
        mode: str,
        is_error: bool = False,
        trace_id: str | None = None,
        sources: list[dict] | None = None,
    ) -> Message:
        message = Message(
            chat_id=chat_id,
            role="assistant",
            mode=mode,
            content=content,
            is_error=is_error,
            trace_id=trace_id,
        )
        self.db.add(message)
        self.db.flush()  # assign message.id before attaching sources

        for source in sources or []:
            self.db.add(
                MessageSource(
                    message_id=message.id,
                    citation_id=source.get("citation_id", 0),
                    document_id=source.get("document_id"),
                    filename=source.get("filename"),
                    page_number=source.get("page_number"),
                    chunk_id=source.get("chunk_id"),
                    rerank_score=source.get("rerank_score"),
                )
            )

        self.db.commit()
        self.db.refresh(message)
        return message
