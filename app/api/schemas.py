from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatCreateRequest(BaseModel):
    title: str = "New chat"


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    citation_id: int
    document_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    chunk_id: int | None = None
    rerank_score: float | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    mode: str | None = None
    content: str
    is_error: bool
    trace_id: str | None = None
    created_at: datetime
    sources: list[MessageSourceOut] = Field(default_factory=list)


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    mode: Literal["rag", "agent"] = "rag"


class PostMessageResponse(BaseModel):
    chat_id: str
    message_id: str
    role: str
    mode: str
    content: str
    is_error: bool
    trace_id: str | None = None
    sources: list[dict] = Field(default_factory=list)


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    file_type: str | None = None
    page_count: int | None = None
    chunk_count: int
    created_at: datetime


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chat_id: str
    message_id: str
    mode: str
    trace_id: str | None = None
    retrieval: dict | None = None
    route: str | None = None
    retry_count: int | None = None
    is_valid: bool | None = None
    agent_trace: list | None = None
    created_at: datetime


class ContextLinkCreateRequest(BaseModel):
    target_chat_id: str


class ContextLinkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_chat_id: str
    target_chat_id: str
    created_at: datetime
