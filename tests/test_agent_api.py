"""HTTP integration tests for the Agent-mode chat path
(POST /chats/{chat_id}/messages with mode="agent", GET /runs/{run_id}).

Isolation mirrors tests/test_api.py's pattern (`get_db` -> in-memory SQLite
`db_session` override, `get_rag` -> a single shared Mock() override,
TestClient NOT used as a context manager so the app's lifespan hook never
touches the real Postgres engine) plus tests/test_agents_graph.py's proven
mocking boundary for the LangGraph workflow: only `rag.generator.generate`
(the Gemini call) and `rag.retrieve` (the Chroma call) are mocked -- the
real compiled LangGraph graph and the real router/retrieval/answer/
validation node functions all run for real, through the real FastAPI
route, ChatService, and AgentService.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_rag
from app.main import app

ROUTE_DOCUMENT_QA = '{"route": "document_qa", "reasoning": "simple lookup"}'
VALID_OK = (
    '{"claims_supported": true, "addresses_question": true, '
    '"citation_correct": true, "unsupported_claims": [], "reasoning": "ok"}'
)
INVALID_UNSUPPORTED = (
    '{"claims_supported": false, "addresses_question": true, '
    '"citation_correct": true, "unsupported_claims": ["x"], "reasoning": "bad"}'
)


def make_evidence(document_id="docA", chunk_id=0):
    return {
        "document_id": document_id,
        "filename": "a.pdf",
        "page_number": 1,
        "chunk_id": chunk_id,
        "rerank_score": 0.9,
        "document": "X is true.",
    }


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    fake_rag = Mock()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rag] = lambda: fake_rag

    test_client = TestClient(app)  # not a context manager -- avoids firing lifespan against real Postgres
    test_client.fake_rag = fake_rag
    yield test_client

    app.dependency_overrides.clear()


def signup_and_get_headers(client, email="user@example.com", password="password123"):
    response = client.post("/auth/signup", json={"email": email, "password": password})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_chat(client, headers, title="agent test chat"):
    return client.post("/chats", json={"title": title}, headers=headers).json()["id"]


def post_agent_message(client, chat_id, headers, content="What is X?"):
    return client.post(
        f"/chats/{chat_id}/messages",
        json={"content": content, "mode": "agent"},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Required test 1: successful Agent request and persisted run
# ---------------------------------------------------------------------------

def test_post_agent_message_happy_path_persists_run_and_response(client):
    headers = signup_and_get_headers(client)
    chat_id = create_chat(client, headers)

    client.fake_rag.generator.generate.side_effect = [ROUTE_DOCUMENT_QA, "The answer is X [1].", VALID_OK]
    client.fake_rag.retrieve.return_value = {"candidates": [], "final_evidence": [make_evidence()]}

    response = post_agent_message(client, chat_id, headers)

    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "agent"
    assert body["is_error"] is False
    assert body["content"] == "The answer is X [1]."
    assert len(body["sources"]) == 1

    runs = client.get(f"/chats/{chat_id}/runs", headers=headers).json()
    assert len(runs) == 1
    run = runs[0]
    assert run["mode"] == "agent"
    assert run["route"] == "document_qa"
    assert run["retry_count"] == 0
    assert run["is_valid"] is True
    assert run["retrieval"] is None  # RAG-only field must stay empty for agent-mode runs
    assert [entry["stage"] for entry in run["agent_trace"]] == [
        "router",
        "retrieval",
        "answer",
        "validation",
    ]

    detail = client.get(f"/runs/{run['id']}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["id"] == run["id"]
    assert detail.json()["route"] == "document_qa"


# ---------------------------------------------------------------------------
# Required test 2: retry/validation failure with persisted metadata
# ---------------------------------------------------------------------------

def test_post_agent_message_retry_exhausted_still_finalizes_invalid_answer(client):
    headers = signup_and_get_headers(client)
    chat_id = create_chat(client, headers)

    client.fake_rag.generator.generate.side_effect = [
        ROUTE_DOCUMENT_QA,
        "Answer attempt 1 [1].",
        INVALID_UNSUPPORTED,
        "Answer attempt 2 [1].",
        INVALID_UNSUPPORTED,
    ]
    client.fake_rag.retrieve.return_value = {"candidates": [], "final_evidence": [make_evidence()]}

    response = post_agent_message(client, chat_id, headers)

    assert response.status_code == 201
    body = response.json()
    assert body["is_error"] is False  # a non-null (if invalid) answer is not treated as a request error
    assert body["content"] == "Answer attempt 2 [1]."

    run = client.get(f"/chats/{chat_id}/runs", headers=headers).json()[0]
    assert run["retry_count"] == 1
    assert run["is_valid"] is False
    assert [entry["stage"] for entry in run["agent_trace"]] == [
        "router",
        "retrieval",
        "answer",
        "validation",
        "retrieval",
        "answer",
        "validation",
    ]


# ---------------------------------------------------------------------------
# Required test 3: cross-user run ownership
# ---------------------------------------------------------------------------

def test_get_run_by_non_owner_returns_404(client):
    headers_a = signup_and_get_headers(client, "a@example.com")
    headers_b = signup_and_get_headers(client, "b@example.com")
    chat_id = create_chat(client, headers_a)

    client.fake_rag.generator.generate.side_effect = [ROUTE_DOCUMENT_QA, "An answer [1].", VALID_OK]
    client.fake_rag.retrieve.return_value = {"candidates": [], "final_evidence": [make_evidence()]}

    post_agent_message(client, chat_id, headers_a)
    run_id = client.get(f"/chats/{chat_id}/runs", headers=headers_a).json()[0]["id"]

    response = client.get(f"/runs/{run_id}", headers=headers_b)

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Optional test: no-evidence abstention (included -- clean, no scope expansion)
# ---------------------------------------------------------------------------

def test_post_agent_message_no_evidence_abstains_without_extra_llm_calls(client):
    headers = signup_and_get_headers(client)
    chat_id = create_chat(client, headers)

    client.fake_rag.generator.generate.side_effect = [ROUTE_DOCUMENT_QA]
    client.fake_rag.retrieve.return_value = {"candidates": [], "final_evidence": []}

    response = post_agent_message(client, chat_id, headers)

    assert response.status_code == 201
    body = response.json()
    assert body["is_error"] is False
    assert "couldn't find enough information" in body["content"]

    run = client.get(f"/chats/{chat_id}/runs", headers=headers).json()[0]
    assert run["retry_count"] == 0
    assert run["is_valid"] is True
    assert client.fake_rag.generator.generate.call_count == 1
