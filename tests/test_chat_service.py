"""ChatService orchestration tests: real repositories against the in-memory
SQLite db_session fixture, but fake RagService/AgentService collaborators
(no real Gemini/Chroma/LangGraph calls) -- mirrors the existing
tests/test_agents_tools.py style of mocking only the LLM/retrieval
boundary, not the persistence layer under test.
"""

import pytest

from app.agents.state import AgentState, AgentTraceEntry, RouteType, ValidationResult
from app.repositories.chat_repository import ChatRepository
from app.repositories.context_link_repository import ContextLinkRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.run_repository import RunRepository
from app.repositories.user_repository import UserRepository
from app.services.chat_service import ChatNotFoundError, ChatService


class FakeRagService:
    def __init__(self, response=None):
        self.response = response or {
            "answer": "The answer is 42 [1].",
            "sources": [{"citation_id": 1, "document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_id": 0, "rerank_score": 0.8}],
            "citations": {"valid": [1], "invalid": [], "citation_map": {}},
            "retrieval": {"original_question": "q", "rewritten_question": "q", "search_queries": ["q"], "candidates": [], "final_evidence": []},
            "error": None,
        }
        self.calls = []

    def answer(self, question, owner_id, conversation_history=None, document_id=None, trace_id=None):
        self.calls.append({
            "question": question,
            "owner_id": owner_id,
            "conversation_history": conversation_history,
        })
        return self.response


class FakeAgentService:
    def __init__(self, agent_state=None):
        self.agent_state = agent_state or AgentState(
            user_query="q",
            selected_route=RouteType.DOCUMENT_QA,
            retrieved_evidence=[],
            final_answer="The agent's final answer.",
            validation_result=ValidationResult(
                is_valid=True, claims_supported=True, citation_correct=True, addresses_question=True,
            ),
            agent_trace=[AgentTraceEntry(stage="router", status="ok")],
        )
        self.calls = []

    def run(self, user_query, owner_id, conversation_context=None, trace_id=None):
        self.calls.append({
            "user_query": user_query,
            "owner_id": owner_id,
            "conversation_context": conversation_context,
        })
        return self.agent_state


def make_chat_service(db_session, rag_service=None, agent_service=None):
    service = ChatService(
        chat_repository=ChatRepository(db_session),
        message_repository=MessageRepository(db_session),
        run_repository=RunRepository(db_session),
        context_link_repository=ContextLinkRepository(db_session),
        rag_service=rag_service or FakeRagService(),
        agent_service=agent_service or FakeAgentService(),
    )
    user = UserRepository(db_session).create(email="user@example.com", password_hash="hashed")
    return service, user


# ---------------------------------------------------------------------------
# Chat lifecycle + ownership
# ---------------------------------------------------------------------------

def test_get_chat_raises_not_found_for_non_owner(db_session):
    service, user = make_chat_service(db_session)
    other = UserRepository(db_session).create(email="other@example.com", password_hash="hashed")
    chat = service.create_chat(user.id)

    with pytest.raises(ChatNotFoundError):
        service.get_chat(chat.id, other.id)


def test_delete_chat_removes_it_for_owner(db_session):
    service, user = make_chat_service(db_session)
    chat = service.create_chat(user.id)

    service.delete_chat(chat.id, user.id)

    with pytest.raises(ChatNotFoundError):
        service.get_chat(chat.id, user.id)


# ---------------------------------------------------------------------------
# post_message -- rag mode
# ---------------------------------------------------------------------------

def test_post_message_rag_persists_user_and_assistant_messages(db_session):
    rag_service = FakeRagService()
    service, user = make_chat_service(db_session, rag_service=rag_service)
    chat = service.create_chat(user.id)

    result = service.post_message(chat.id, user.id, "What is the answer?", mode="rag")

    assert result["role"] == "assistant"
    assert result["mode"] == "rag"
    assert result["content"] == "The answer is 42 [1]."
    assert result["is_error"] is False
    assert len(result["sources"]) == 1

    messages = service.list_messages(chat.id, user.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "What is the answer?"


def test_post_message_rag_records_a_run_with_retrieval_payload(db_session):
    service, user = make_chat_service(db_session)
    chat = service.create_chat(user.id)

    service.post_message(chat.id, user.id, "q", mode="rag")

    runs = service.list_runs(chat.id, user.id)
    assert len(runs) == 1
    assert runs[0].mode == "rag"
    assert runs[0].retrieval["original_question"] == "q"


def test_post_message_rag_generation_error_marks_message_as_error(db_session):
    rag_service = FakeRagService(response={
        "answer": None,
        "sources": [],
        "citations": {"valid": [], "invalid": [], "citation_map": {}},
        "retrieval": {},
        "error": {"message": "Gemini unavailable"},
    })
    service, user = make_chat_service(db_session, rag_service=rag_service)
    chat = service.create_chat(user.id)

    result = service.post_message(chat.id, user.id, "q", mode="rag")

    assert result["is_error"] is True
    assert "Gemini unavailable" in result["content"]


def test_post_message_passes_prior_messages_as_conversation_history(db_session):
    rag_service = FakeRagService()
    service, user = make_chat_service(db_session, rag_service=rag_service)
    chat = service.create_chat(user.id)

    service.post_message(chat.id, user.id, "first question", mode="rag")
    service.post_message(chat.id, user.id, "second question", mode="rag")

    second_call_history = rag_service.calls[1]["conversation_history"]
    assert {"role": "user", "content": "first question"} in second_call_history
    assert {"role": "assistant", "content": "The answer is 42 [1]."} in second_call_history


def test_post_message_raises_not_found_for_non_owner(db_session):
    service, user = make_chat_service(db_session)
    other = UserRepository(db_session).create(email="other@example.com", password_hash="hashed")
    chat = service.create_chat(user.id)

    with pytest.raises(ChatNotFoundError):
        service.post_message(chat.id, other.id, "q", mode="rag")


# ---------------------------------------------------------------------------
# post_message -- agent mode
# ---------------------------------------------------------------------------

def test_post_message_agent_persists_final_answer_and_agent_trace(db_session):
    agent_service = FakeAgentService()
    service, user = make_chat_service(db_session, agent_service=agent_service)
    chat = service.create_chat(user.id)

    result = service.post_message(chat.id, user.id, "q", mode="agent")

    assert result["content"] == "The agent's final answer."
    assert result["is_error"] is False

    runs = service.list_runs(chat.id, user.id)
    assert runs[0].mode == "agent"
    assert runs[0].route == "document_qa"
    assert runs[0].is_valid is True
    assert runs[0].agent_trace == [{"stage": "router", "status": "ok", "duration_seconds": None, "decision": None, "tools_used": []}]


def test_post_message_agent_no_answer_marks_message_as_error(db_session):
    agent_state = AgentState(user_query="q", final_answer=None)
    service, user = make_chat_service(db_session, agent_service=FakeAgentService(agent_state))
    chat = service.create_chat(user.id)

    result = service.post_message(chat.id, user.id, "q", mode="agent")

    assert result["is_error"] is True
    assert result["content"] == "(no answer produced)"


# ---------------------------------------------------------------------------
# Cross-chat context links
# ---------------------------------------------------------------------------

def test_create_context_link_requires_ownership_of_both_chats(db_session):
    service, user = make_chat_service(db_session)
    other = UserRepository(db_session).create(email="other@example.com", password_hash="hashed")
    mine = service.create_chat(user.id)
    theirs = service.create_chat(other.id)

    with pytest.raises(ChatNotFoundError):
        service.create_context_link(user.id, mine.id, theirs.id)


def test_post_message_pulls_relevant_context_from_linked_chat(db_session):
    rag_service = FakeRagService()
    service, user = make_chat_service(db_session, rag_service=rag_service)
    source_chat = service.create_chat(user.id)
    target_chat = service.create_chat(user.id)

    # Seed the linked chat with a message that overlaps the new question's
    # keywords, so select_relevant_history actually picks it up.
    service.post_message(target_chat.id, user.id, "quantum entanglement experiment results", mode="rag")
    service.create_context_link(user.id, source_chat.id, target_chat.id)

    service.post_message(source_chat.id, user.id, "quantum entanglement experiment", mode="rag")

    last_call_history = rag_service.calls[-1]["conversation_history"]
    assert any("quantum entanglement" in message["content"] for message in last_call_history)
