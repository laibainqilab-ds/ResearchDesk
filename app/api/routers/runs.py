from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_chat_service, get_current_user
from app.api.schemas import RunOut
from app.db.models import User
from app.services.chat_service import ChatNotFoundError, ChatService

router = APIRouter(tags=["runs"])


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")


@router.get("/chats/{chat_id}/runs", response_model=list[RunOut])
def list_runs(
    chat_id: str,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[RunOut]:
    try:
        runs = chat_service.list_runs(chat_id, user.id)
    except ChatNotFoundError:
        raise _not_found()
    return [RunOut.model_validate(run) for run in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(
    run_id: str,
    user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> RunOut:
    try:
        run = chat_service.get_run(run_id, user.id)
    except ChatNotFoundError:
        raise _not_found()
    return RunOut.model_validate(run)
