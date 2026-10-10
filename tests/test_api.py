"""API-level tests through FastAPI's TestClient. `get_db` is overridden to
the in-memory SQLite db_session fixture (so no live Postgres server is
required to run the suite) and `get_rag` is overridden to a Mock RAG (so no
real Gemini/Chroma call happens) -- the real app wiring (routers, deps,
services, repositories) all run as in production.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_rag
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


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
