import logging

import psycopg
from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_rag
from app.config import DATABASE_URL
from app.rag import RAG

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# A health check must fail fast, not hang. The shared, pooled SQLAlchemy
# engine (app/db/session.py) has no timeout configured -- and is left that
# way here deliberately, since adding one there would change behavior for
# every normal request, not just this check. Instead this uses a separate,
# short-lived, direct connection with its own bounded timeout: connect_timeout
# bounds a stalled TCP handshake (the dominant real-world "unreachable or
# stalled" case), and statement_timeout additionally bounds the query itself
# in case a connection succeeds but the server doesn't respond. A fresh
# connection -- not the shared pool -- is also the semantically correct
# check here: it proves a *new* connection can be established right now,
# not that a previously-opened pooled one merely still exists.
DATABASE_HEALTH_CHECK_TIMEOUT_SECONDS = 3


def _check_database() -> bool:
    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=DATABASE_HEALTH_CHECK_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={DATABASE_HEALTH_CHECK_TIMEOUT_SECONDS * 1000}",
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
        return True
    except Exception as error:
        logger.warning("Health check: database dependency unavailable: %s", error)
        return False


def _check_vector_store(rag: RAG) -> bool:
    try:
        rag.store.client.heartbeat()
        return True
    except Exception as error:
        logger.warning("Health check: vector store dependency unavailable: %s", error)
        return False


@router.get("/health")
def health(response: Response, rag: RAG = Depends(get_rag)) -> dict:
    """Liveness + dependency check, per the architecture doc's API table
    ("is the backend and its dependencies up"). Checks PostgreSQL and
    ChromaDB only -- never Gemini (no LLM call belongs in a health check)
    and never mutates any data. Exposes only up/down per dependency, no
    connection strings or raw exception text.
    """
    database_ok = _check_database()
    vector_store_ok = _check_vector_store(rag)
    healthy = database_ok and vector_store_ok

    response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if healthy else "unhealthy",
        "dependencies": {
            "database": "ok" if database_ok else "unhealthy",
            "vector_store": "ok" if vector_store_ok else "unhealthy",
        },
    }
