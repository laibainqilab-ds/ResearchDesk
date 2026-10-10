"""Chat Service: owns chat/message lifecycle and orchestrates RAG/Agent
Service calls for POST /chats/{chat_id}/messages.

All PostgreSQL access goes through the repositories passed in -- this
service never touches a database session directly, and RagService /
AgentService never see one at all.
"""

import logging

from app.agents.trace_view import summarize_workflow_run
from app.db.models import Chat, Message, Run
from app.observability import new_trace_id
from app.rag import select_relevant_history
from app.repositories.chat_repository import ChatRepository
from app.repositories.context_link_repository import ContextLinkRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.run_repository import RunRepository
from app.services.agent_service import AgentService
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

MAX_CROSS_CHAT_CONTEXT_MESSAGES = 2


class ChatNotFoundError(Exception):
    """Raised when a chat doesn't exist, or exists but isn't owned by the
    requesting user -- both cases return 404, so chat existence is never
    leaked to a non-owner."""


class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        message_repository: MessageRepository,
        run_repository: RunRepository,
        context_link_repository: ContextLinkRepository,
        rag_service: RagService,
        agent_service: AgentService,
    ):
        self.chats = chat_repository
        self.messages = message_repository
        self.runs = run_repository
        self.context_links = context_link_repository
        self.rag_service = rag_service
        self.agent_service = agent_service

    # -- Chat lifecycle ----------------------------------------------------

    def create_chat(self, user_id: str, title: str = "New chat") -> Chat:
        return self.chats.create(user_id=user_id, title=title)

    def list_chats(self, user_id: str) -> list[Chat]:
        return self.chats.list_for_user(user_id)

    def get_chat(self, chat_id: str, user_id: str) -> Chat:
        chat = self.chats.get_for_user(chat_id, user_id)
        if chat is None:
            raise ChatNotFoundError(chat_id)
        return chat

    def delete_chat(self, chat_id: str, user_id: str) -> None:
        chat = self.get_chat(chat_id, user_id)
        self.chats.delete(chat)

    def list_messages(self, chat_id: str, user_id: str) -> list[Message]:
        self.get_chat(chat_id, user_id)  # ownership check
        return self.messages.list_for_chat(chat_id)

    # -- Runs ----------------------------------------------------------------

    def list_runs(self, chat_id: str, user_id: str) -> list[Run]:
        self.get_chat(chat_id, user_id)  # ownership check
        return self.runs.list_for_chat(chat_id)

    def get_run(self, run_id: str, user_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise ChatNotFoundError(run_id)
        self.get_chat(run.chat_id, user_id)  # ownership check
        return run

    # -- Cross-chat context links -------------------------------------------

    def create_context_link(self, user_id: str, source_chat_id: str, target_chat_id: str):
        # Ownership of both chats is required -- get_chat raises
        # ChatNotFoundError (and therefore a 404, not a 403) for a chat the
        # user doesn't own, so link creation can't be used to probe for the
        # existence of other users' chats.
        self.get_chat(source_chat_id, user_id)
        self.get_chat(target_chat_id, user_id)
        return self.context_links.create(user_id, source_chat_id, target_chat_id)

    def _gather_cross_chat_context(self, chat_id: str, question: str) -> list[dict]:
        """Pull relevant messages from every chat linked as context for
        `chat_id`, scored against `question` with the same deterministic
        keyword-overlap selection RAG already uses for same-chat history."""
        links = self.context_links.list_for_source_chat(chat_id)

        if not links:
            return []

        context: list[dict] = []

        for link in links:
            linked_messages = self.messages.list_for_chat(link.target_chat_id)
            as_history = [
                {"role": message.role, "content": message.content}
                for message in linked_messages
            ]
            relevant = select_relevant_history(
                question, as_history, max_messages=MAX_CROSS_CHAT_CONTEXT_MESSAGES
            )
            context.extend(relevant)

        return context

    # -- Messages ------------------------------------------------------------

    def post_message(self, chat_id: str, user_id: str, content: str, mode: str) -> dict:
        """Persist the user message, run RAG or Agent mode, persist the
        assistant message + sources + a compact Run record, and return a
        response payload for the API layer."""
        chat = self.get_chat(chat_id, user_id)

        trace_id = new_trace_id()

        prior_messages = self.messages.list_for_chat(chat_id)
        conversation_history = [
            {"role": message.role, "content": message.content} for message in prior_messages
        ]
        cross_chat_context = self._gather_cross_chat_context(chat_id, content)

        self.messages.add_user_message(chat_id, content)

        if mode == "agent":
            result = self._run_agent(content, conversation_history + cross_chat_context, trace_id)
        else:
            result = self._run_rag(content, conversation_history + cross_chat_context, trace_id)

        assistant_message = self.messages.add_assistant_message(
            chat_id=chat_id,
            content=result["content"],
            mode=mode,
            is_error=result["is_error"],
            trace_id=trace_id,
            sources=result["sources"],
        )

        self.runs.create(
            chat_id=chat_id,
            message_id=assistant_message.id,
            mode=mode,
            trace_id=trace_id,
            retrieval=result.get("retrieval"),
            route=result.get("route"),
            retry_count=result.get("retry_count"),
            is_valid=result.get("is_valid"),
            agent_trace=result.get("agent_trace"),
        )

        self.chats.touch(chat)

        return {
            "chat_id": chat_id,
            "message_id": assistant_message.id,
            "role": "assistant",
            "mode": mode,
            "content": result["content"],
            "is_error": result["is_error"],
            "trace_id": trace_id,
            "sources": result["sources"],
        }

    def _run_rag(self, question: str, conversation_history: list[dict], trace_id: str) -> dict:
        result = self.rag_service.answer(
            question=question,
            conversation_history=conversation_history,
            trace_id=trace_id,
        )

        error = result.get("error")

        if error is not None:
            content = (
                "Retrieval completed successfully, but answer generation "
                f"failed: {error['message']}"
            )
            is_error = True
        else:
            content = result["answer"]
            is_error = False

        return {
            "content": content,
            "is_error": is_error,
            "sources": result["sources"],
            "retrieval": result.get("retrieval"),
        }

    def _run_agent(self, question: str, conversation_context: list[dict], trace_id: str) -> dict:
        agent_state = self.agent_service.run(
            user_query=question,
            conversation_context=conversation_context,
            trace_id=trace_id,
        )
        summary = summarize_workflow_run(agent_state)

        content = agent_state.final_answer or "(no answer produced)"
        is_error = agent_state.final_answer is None

        sources = [
            {
                "citation_id": index,
                "document_id": evidence.get("document_id"),
                "filename": evidence.get("filename"),
                "page_number": evidence.get("page_number"),
                "chunk_id": evidence.get("chunk_id"),
                "rerank_score": evidence.get("rerank_score"),
            }
            for index, evidence in enumerate(agent_state.retrieved_evidence, start=1)
        ]

        return {
            "content": content,
            "is_error": is_error,
            "sources": sources,
            "route": summary["route"],
            "retry_count": summary["retry_count"],
            "is_valid": summary["is_valid"],
            "agent_trace": [entry.model_dump() for entry in agent_state.agent_trace],
        }
