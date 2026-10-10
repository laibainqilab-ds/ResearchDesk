"""SQLAlchemy ORM models for ResearchDesk's persistent application data.

Document chunks, embeddings, and vector metadata stay in ChromaDB
(app.ingestion.vector_store.VectorStore) -- these tables only hold the
application-level records that need relational queries, ownership checks,
and durability: users, chats, messages, citations, document bookkeeping,
compact run records, and cross-chat context links.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    chats: Mapped[list["Chat"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New chat")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="chats")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # "rag" | "agent"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False, index=True)

    chat: Mapped["Chat"] = relationship(back_populates="messages")
    sources: Mapped[list["MessageSource"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="MessageSource.citation_id"
    )
    run: Mapped["Run | None"] = relationship(back_populates="message", uselist=False, cascade="all, delete-orphan")


class MessageSource(Base):
    """One citation/evidence source attached to an assistant message."""

    __tablename__ = "message_sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), nullable=False, index=True)
    citation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    message: Mapped["Message"] = relationship(back_populates="sources")


class Document(Base):
    """Bookkeeping for an ingested document: ownership and lifecycle status.

    The content itself (chunks/embeddings) lives in ChromaDB, keyed by the
    same document_id (sha256 of file bytes) computed by
    app.ingestion.pipeline.compute_document_id.
    """

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="indexed")  # "indexed" | "deleted"
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    owner: Mapped["User"] = relationship(back_populates="documents")


class Run(Base):
    """Compact, persisted record of one RAG or Agent run -- enough for the
    Retrieval Inspector / Agent Trace views, never the full live LangGraph
    state."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), nullable=False, index=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), nullable=False, unique=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)  # "rag" | "agent"
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # RAG mode: retrieval inspector payload (original/rewritten question,
    # search queries, candidates, final evidence).
    retrieval: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Agent mode: route taken, retry count, validation outcome, and the
    # compact AgentTraceEntry list for the Agent Trace view.
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    agent_trace: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    message: Mapped["Message"] = relationship(back_populates="run")


class ChatContextLink(Base):
    """An explicit, user-created link letting one chat pull relevant
    context from another chat the same user owns."""

    __tablename__ = "chat_context_links"
    __table_args__ = (UniqueConstraint("source_chat_id", "target_chat_id", name="uq_context_link_pair"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    source_chat_id: Mapped[str] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_chat_id: Mapped[str] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
