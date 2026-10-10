from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chat, _utcnow


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: str, title: str = "New chat") -> Chat:
        chat = Chat(user_id=user_id, title=title)
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def list_for_user(self, user_id: str) -> list[Chat]:
        statement = (
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
        )
        return list(self.db.execute(statement).scalars().all())

    def get(self, chat_id: str) -> Chat | None:
        return self.db.get(Chat, chat_id)

    def get_for_user(self, chat_id: str, user_id: str) -> Chat | None:
        chat = self.get(chat_id)
        if chat is None or chat.user_id != user_id:
            return None
        return chat

    def touch(self, chat: Chat) -> None:
        """Mark the chat updated (bump updated_at) -- called after a new
        message is appended so chat lists sort by recent activity."""
        chat.updated_at = _utcnow()
        self.db.commit()

    def delete(self, chat: Chat) -> None:
        self.db.delete(chat)
        self.db.commit()
