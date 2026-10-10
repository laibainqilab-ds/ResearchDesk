from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Run


class RunRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        chat_id: str,
        message_id: str,
        mode: str,
        trace_id: str | None = None,
        retrieval: dict | None = None,
        route: str | None = None,
        retry_count: int | None = None,
        is_valid: bool | None = None,
        agent_trace: list | None = None,
    ) -> Run:
        run = Run(
            chat_id=chat_id,
            message_id=message_id,
            mode=mode,
            trace_id=trace_id,
            retrieval=retrieval,
            route=route,
            retry_count=retry_count,
            is_valid=is_valid,
            agent_trace=agent_trace,
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def get(self, run_id: str) -> Run | None:
        return self.db.get(Run, run_id)

    def get_for_message(self, message_id: str) -> Run | None:
        statement = select(Run).where(Run.message_id == message_id)
        return self.db.execute(statement).scalar_one_or_none()

    def list_for_chat(self, chat_id: str) -> list[Run]:
        statement = select(Run).where(Run.chat_id == chat_id).order_by(Run.created_at.asc())
        return list(self.db.execute(statement).scalars().all())
