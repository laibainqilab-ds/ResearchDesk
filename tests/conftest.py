import sys
from pathlib import Path

# Ensure the repo root is importable as `app.*` regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db import models  # noqa: F401 -- ensure all tables are registered on Base.metadata


@pytest.fixture()
def db_session():
    """An isolated in-memory SQLite database, schema-equivalent to the real
    Postgres `researchdesk` database, for repository/service tests that
    need a real ORM session without touching a live Postgres server."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)

    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
