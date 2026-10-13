"""Chat Service: owns chat/message lifecycle and orchestrates RAG/Agent
Service calls for POST /chats/{chat_id}/messages.

All PostgreSQL access goes through the repositories passed in -- this
service never touches a database session directly, and RagService /
AgentService never see one at all.
"""

import logging

from app.agents.trace_view import summarize_workflow_run
from app.db.models import Chat, Message, Run, _utcnow
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

# The ChatCreateRequest/api_client.py default -- every chat the UI creates
# arrives here with exactly this title, since nothing currently lets a user
# set one at creation time. Used as a sentinel so create_chat can replace it
# with something distinguishable, while still respecting a genuinely custom
# title (anything else) from a caller that does set one.
DEFAULT_CHAT_TITLE = "New chat"

MAX_GENERATED_TITLE_LENGTH = 60


def _title_from_message(content: str) -> str | None:
    """A concise, deterministic chat title from the user's first message --
    no LLM call: just the message itself, collapsed and truncated. Returns
    None for whitespace-only content, so callers can leave the existing
    title alone rather than overwrite it with an empty string."""
    collapsed = " ".join(content.split())

    if not collapsed:
        return None

    if len(collapsed) <= MAX_GENERATED_TITLE_LENGTH:
        return collapsed

    return collapsed[: MAX_GENERATED_TITLE_LENGTH - 1].rstrip() + "…"


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

    def create_chat(self, user_id: str, title: str = DEFAULT_CHAT_TITLE) -> Chat:
        """A fresh chat with no messages yet.

        An unmodified default title is replaced with a timestamp-stamped
        placeholder (e.g. "Untitled chat — Oct 14, 03:34:07 PM") so that
        creating several empty chats doesn't leave them all showing the
        same indistinguishable name in the sidebar -- distinguishable from
        the moment they're persisted, not just at display time. A caller
        that explicitly passes something other than the default is assumed
        to want that exact title and is never overridden.
        """
        if title == DEFAULT_CHAT_TITLE:
            title = f"Untitled chat — {_utcnow():%b %d, %I:%M:%S %p}"

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
        """Run RAG or Agent mode, then -- once an answer comes back -- persist
        both the user message and the assistant message together, plus
        sources and a compact Run record, and return a response payload for
        the API layer.

        Per the architecture doc (section 6, "Asking a question"): history
        is pulled from Postgres first, handed to RAG/Agent Service, and
        "once it gets an answer back it saves both the user's message and
        the assistant's reply" -- not the user's message immediately,
        before generation runs. A generation failure that's caught
        gracefully (GenerationUnavailableError, surfaced as
        result["is_error"]) still reaches this point normally, so both
        messages and the run are still saved together. An unexpected,
        uncaught exception during the RAG/Agent call -- not the documented
        failure path -- means neither message is persisted; the doc does
        not describe that case, and this is the accepted, documented
        trade-off rather than an oversight.
        """
        chat = self.get_chat(chat_id, user_id)

        trace_id = new_trace_id()

        prior_messages = self.messages.list_for_chat(chat_id)
        is_first_message = not prior_messages
        conversation_history = [
            {"role": message.role, "content": message.content} for message in prior_messages
        ]
        cross_chat_context = self._gather_cross_chat_context(chat_id, content)

        if mode == "agent":
            result = self._run_agent(content, conversation_history + cross_chat_context, trace_id, user_id)
        else:
            result = self._run_rag(content, conversation_history + cross_chat_context, trace_id, user_id)

        self.messages.add_user_message(chat_id, content)

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

        if is_first_message:
            # Based only on the user's question, independent of whether
            # generation succeeded -- the question itself is already
            # meaningful content, and a failed answer shouldn't leave the
            # chat stuck on its placeholder title. Nothing else can have
            # set a title before this point (no rename feature exists), so
            # this never overwrites anything a user explicitly chose.
            generated_title = _title_from_message(content)

            if generated_title:
                self.chats.update_title(chat, generated_title)

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

    def _run_rag(self, question: str, conversation_history: list[dict], trace_id: str, owner_id: str) -> dict:
        result = self.rag_service.answer(
            question=question,
            owner_id=owner_id,
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

    def _run_agent(self, question: str, conversation_context: list[dict], trace_id: str, owner_id: str) -> dict:
        agent_state = self.agent_service.run(
            user_query=question,
            owner_id=owner_id,
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
