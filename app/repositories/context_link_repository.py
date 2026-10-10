from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatContextLink


class ContextLinkRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: str, source_chat_id: str, target_chat_id: str) -> ChatContextLink:
        link = ChatContextLink(
            user_id=user_id,
            source_chat_id=source_chat_id,
            target_chat_id=target_chat_id,
        )
        self.db.add(link)
        self.db.commit()
        self.db.refresh(link)
        return link

    def list_for_source_chat(self, source_chat_id: str) -> list[ChatContextLink]:
        statement = select(ChatContextLink).where(ChatContextLink.source_chat_id == source_chat_id)
        return list(self.db.execute(statement).scalars().all())

    def get(self, link_id: str) -> ChatContextLink | None:
        return self.db.get(ChatContextLink, link_id)

    def delete(self, link: ChatContextLink) -> None:
        self.db.delete(link)
        self.db.commit()
