"""API-level tests through FastAPI's TestClient. `get_db` is overridden to
the in-memory SQLite db_session fixture (so no live Postgres server is
required to run the suite) and `get_rag` is overridden to a Mock RAG (so no
real Gemini/Chroma call happens) -- the real app wiring (routers, deps,
services, repositories) all run as in production.
"""

from unittest.mock import Mock

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_rag
from app.api.routers import health as health_module
from app.main import app


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    fake_rag = Mock()
    fake_rag.answer.return_value = {
        "answer": "Answer [1].",
        "sources": [{"citation_id": 1, "document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_id": 0, "rerank_score": 0.9}],
        "citations": {"valid": [1], "invalid": [], "citation_map": {}},
        "retrieval": {"original_question": "q", "rewritten_question": "q", "search_queries": ["q"], "candidates": [], "final_evidence": []},
        "error": None,
    }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rag] = lambda: fake_rag

    # Deliberately not used as a context manager: entering it would fire the
    # app's startup event (Base.metadata.create_all against the *real*
    # Postgres engine), making this test suite depend on a live DB server.
    # The dependency overrides above are enough to exercise every route.
    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.clear()


def signup_and_get_headers(client, email="user@example.com", password="password123"):
    response = client.post("/auth/signup", json={"email": email, "password": password})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, *args, **kwargs):
        return None


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return _FakeCursor()


def _fake_db_connect_success(*args, **kwargs):
    """A fake psycopg.connect() that never touches the network -- health's
    database check now bypasses the shared, SQLite-overridden SQLAlchemy
    session entirely (see app/api/routers/health.py), so every health test
    must mock at this boundary instead, or it would silently depend on a
    real, live PostgreSQL server to pass."""
    return _FakeConnection()


def test_health(client, monkeypatch):
    monkeypatch.setattr(health_module.psycopg, "connect", _fake_db_connect_success)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {"database": "ok", "vector_store": "ok"},
    }


def test_health_database_check_requests_a_bounded_connect_and_statement_timeout(client, monkeypatch):
    """Regression test for the hang fix's actual mechanism: a fresh
    connection must be requested with an explicit connect_timeout (bounds
    a stalled TCP handshake -- the dominant real-world "unreachable or
    stalled" case) and statement_timeout (bounds the query itself in case
    a connection succeeds but the server never responds). psycopg/libpq
    enforce the real timing; a mocked unit test can't reproduce an actual
    multi-second stall without making the suite slow and flaky, so this
    verifies the call is correctly configured to request a bound, which is
    what the fix actually changed."""
    captured = {}

    def fake_connect(conninfo, **kwargs):
        captured.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(health_module.psycopg, "connect", fake_connect)

    client.get("/health")

    assert captured["connect_timeout"] == health_module.DATABASE_HEALTH_CHECK_TIMEOUT_SECONDS
    assert captured["connect_timeout"] <= 5, "a health check must stay fast"
    assert (
        f"statement_timeout={health_module.DATABASE_HEALTH_CHECK_TIMEOUT_SECONDS * 1000}"
        in captured["options"]
    )


def test_health_reports_unhealthy_with_503_when_database_is_down(client, monkeypatch):
    def fake_connect(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(health_module.psycopg, "connect", fake_connect)

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["dependencies"]["database"] == "unhealthy"
    assert body["dependencies"]["vector_store"] == "ok"
    # No connection string, stack trace, or raw exception text leaked.
    assert "connection refused" not in response.text
    assert "OperationalError" not in response.text


def test_health_reports_unhealthy_with_503_when_database_connection_times_out(client, monkeypatch):
    """The realistic failure shape once connect_timeout actually expires --
    a stalled, not merely refused, connection -- distinct from the clean
    "connection refused" case above. Both are handled by the same generic
    except clause in _check_database, so this also confirms that catch is
    broad enough for the real exception psycopg/libpq raise here."""
    def fake_connect(*args, **kwargs):
        raise psycopg.OperationalError("connection timeout expired")

    monkeypatch.setattr(health_module.psycopg, "connect", fake_connect)

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["dependencies"]["database"] == "unhealthy"
    assert "timeout expired" not in response.text


def test_health_reports_unhealthy_with_503_when_vector_store_is_down(client, monkeypatch):
    monkeypatch.setattr(health_module.psycopg, "connect", _fake_db_connect_success)

    class _BrokenRag:
        class store:
            class client:
                @staticmethod
                def heartbeat():
                    raise RuntimeError("chroma unavailable")

    app.dependency_overrides[get_rag] = lambda: _BrokenRag()

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["dependencies"]["database"] == "ok"
    assert body["dependencies"]["vector_store"] == "unhealthy"
    assert "chroma unavailable" not in response.text


def test_signup_then_login(client):
    response = client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"})
    assert response.status_code == 201

    response = client.post("/auth/login", json={"email": "a@example.com", "password": "password123"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_signup_duplicate_email_rejected(client):
    client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"})

    response = client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"})

    assert response.status_code == 409


def test_login_wrong_password_rejected(client):
    client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"})

    response = client.post("/auth/login", json={"email": "a@example.com", "password": "wrong"})

    assert response.status_code == 401


def test_chat_endpoints_require_authentication(client):
    response = client.get("/chats")

    assert response.status_code == 401


def test_create_list_and_delete_chat(client):
    headers = signup_and_get_headers(client)

    response = client.post("/chats", json={"title": "My chat"}, headers=headers)
    assert response.status_code == 201
    chat_id = response.json()["id"]

    response = client.get("/chats", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = client.delete(f"/chats/{chat_id}", headers=headers)
    assert response.status_code == 204

    response = client.get(f"/chats/{chat_id}", headers=headers)
    assert response.status_code == 404


def test_post_rag_message_persists_and_returns_answer(client):
    headers = signup_and_get_headers(client)
    chat_id = client.post("/chats", json={"title": "c"}, headers=headers).json()["id"]

    response = client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "What is the answer?", "mode": "rag"},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["content"] == "Answer [1]."
    assert body["mode"] == "rag"
    assert len(body["sources"]) == 1

    history = client.get(f"/chats/{chat_id}/messages", headers=headers).json()
    assert [message["role"] for message in history] == ["user", "assistant"]

    runs = client.get(f"/chats/{chat_id}/runs", headers=headers).json()
    assert len(runs) == 1
    assert runs[0]["mode"] == "rag"


def test_get_chat_returns_full_message_history_in_order(client):
    headers = signup_and_get_headers(client)
    chat_id = client.post("/chats", json={"title": "c"}, headers=headers).json()["id"]

    client.post(
        f"/chats/{chat_id}/messages", json={"content": "first question", "mode": "rag"}, headers=headers
    )
    client.post(
        f"/chats/{chat_id}/messages", json={"content": "second question", "mode": "rag"}, headers=headers
    )

    response = client.get(f"/chats/{chat_id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == chat_id
    assert "title" in body and "created_at" in body and "updated_at" in body
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    contents = [m["content"] for m in body["messages"]]
    assert contents[0] == "first question"
    assert contents[2] == "second question"
    assert body["messages"][1]["sources"][0]["document_id"] == "doc1"


def test_get_chat_returns_empty_messages_for_a_new_chat(client):
    headers = signup_and_get_headers(client)
    chat_id = client.post("/chats", json={"title": "empty"}, headers=headers).json()["id"]

    response = client.get(f"/chats/{chat_id}", headers=headers)

    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_get_chat_other_user_cannot_access(client):
    headers1 = signup_and_get_headers(client, "owner@example.com")
    headers2 = signup_and_get_headers(client, "intruder@example.com")
    chat_id = client.post("/chats", json={"title": "private"}, headers=headers1).json()["id"]
    client.post(
        f"/chats/{chat_id}/messages", json={"content": "secret", "mode": "rag"}, headers=headers1
    )

    response = client.get(f"/chats/{chat_id}", headers=headers2)

    assert response.status_code == 404


def test_get_chat_does_not_duplicate_messages_when_called_repeatedly(client):
    headers = signup_and_get_headers(client)
    chat_id = client.post("/chats", json={"title": "c"}, headers=headers).json()["id"]
    client.post(
        f"/chats/{chat_id}/messages", json={"content": "q", "mode": "rag"}, headers=headers
    )

    first = client.get(f"/chats/{chat_id}", headers=headers).json()
    second = client.get(f"/chats/{chat_id}", headers=headers).json()

    assert len(first["messages"]) == 2
    assert len(second["messages"]) == 2
    assert [m["id"] for m in first["messages"]] == [m["id"] for m in second["messages"]]


def test_other_user_cannot_access_chat(client):
    headers1 = signup_and_get_headers(client, "user1@example.com")
    headers2 = signup_and_get_headers(client, "user2@example.com")
    chat_id = client.post("/chats", json={"title": "c"}, headers=headers1).json()["id"]

    response = client.get(f"/chats/{chat_id}", headers=headers2)
    assert response.status_code == 404

    response = client.post(
        f"/chats/{chat_id}/messages", json={"content": "q", "mode": "rag"}, headers=headers2
    )
    assert response.status_code == 404


def test_context_link_requires_ownership_of_target_chat(client):
    headers1 = signup_and_get_headers(client, "user1@example.com")
    headers2 = signup_and_get_headers(client, "user2@example.com")
    my_chat = client.post("/chats", json={"title": "mine"}, headers=headers1).json()["id"]
    their_chat = client.post("/chats", json={"title": "theirs"}, headers=headers2).json()["id"]

    response = client.post(
        f"/chats/{my_chat}/context-links",
        json={"target_chat_id": their_chat},
        headers=headers1,
    )

    assert response.status_code == 404
