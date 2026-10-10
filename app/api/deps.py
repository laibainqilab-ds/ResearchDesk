"""Shared FastAPI dependencies: DB session, current authenticated user, and
a process-wide RAG singleton (mirrors the Streamlit app's
@st.cache_resource load_rag() -- the embedder/reranker/generator/vector
store are expensive to construct and safe to share across requests).
"""

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.rag import RAG
from app.repositories.chat_repository import ChatRepository
from app.repositories.context_link_repository import ContextLinkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.run_repository import RunRepository
from app.repositories.user_repository import UserRepository
from app.security import InvalidTokenError, decode_access_token
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.rag_service import RagService

_bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_rag() -> RAG:
    return RAG()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = decode_access_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = UserRepository(db).get_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_chat_service(db: Session = Depends(get_db), rag: RAG = Depends(get_rag)) -> ChatService:
    return ChatService(
        chat_repository=ChatRepository(db),
        message_repository=MessageRepository(db),
        run_repository=RunRepository(db),
        context_link_repository=ContextLinkRepository(db),
        rag_service=RagService(rag),
        agent_service=AgentService(rag),
    )


def get_document_service(db: Session = Depends(get_db), rag: RAG = Depends(get_rag)) -> DocumentService:
    return DocumentService(rag=rag, document_repository=DocumentRepository(db))
