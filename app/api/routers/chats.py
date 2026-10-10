from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_chat_service, get_current_user
from app.api.schemas import (
    ChatCreateRequest,
    ChatOut,
    ContextLinkCreateRequest,
    ContextLinkOut,
    MessageOut,
    PostMessageRequest,
    PostMessageResponse,
)
from app.db.models import User
from app.services.chat_service import ChatNotFoundError, ChatService

router = APIRouter(prefix="/chats", tags=["chats"])


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
def create_chat(
    payload: ChatCreateRequest,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatOut:
    chat = chat_service.create_chat(user_id=user.id, title=payload.title)
    return ChatOut.model_validate(chat)


@router.get("", response_model=list[ChatOut])
def list_chats(
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[ChatOut]:
    return [ChatOut.model_validate(chat) for chat in chat_service.list_chats(user.id)]


@router.get("/{chat_id}", response_model=ChatOut)
def get_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatOut:
    try:
        chat = chat_service.get_chat(chat_id, user.id)
    except ChatNotFoundError:
        raise _not_found()
    return ChatOut.model_validate(chat)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    try:
        chat_service.delete_chat(chat_id, user.id)
    except ChatNotFoundError:
        raise _not_found()


@router.get("/{chat_id}/messages", response_model=list[MessageOut])
def list_messages(
    chat_id: str,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[MessageOut]:
    try:
        messages = chat_service.list_messages(chat_id, user.id)
    except ChatNotFoundError:
        raise _not_found()
    return [MessageOut.model_validate(message) for message in messages]


@router.post("/{chat_id}/messages", response_model=PostMessageResponse, status_code=status.HTTP_201_CREATED)
def post_message(
    chat_id: str,
    payload: PostMessageRequest,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> PostMessageResponse:
    try:
        result = chat_service.post_message(
            chat_id=chat_id,
            user_id=user.id,
            content=payload.content,
            mode=payload.mode,
        )
    except ChatNotFoundError:
        raise _not_found()
    return PostMessageResponse(**result)


@router.post("/{chat_id}/context-links", response_model=ContextLinkOut, status_code=status.HTTP_201_CREATED)
def create_context_link(
    chat_id: str,
    payload: ContextLinkCreateRequest,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ContextLinkOut:
    try:
        link = chat_service.create_context_link(user.id, chat_id, payload.target_chat_id)
    except ChatNotFoundError:
        raise _not_found()
    return ContextLinkOut.model_validate(link)
