"""Thin service wrapper around the existing app.rag.RAG pipeline.

No PostgreSQL access here -- conversation history and cross-chat context
are assembled by the caller (ChatService, via its repositories) and passed
in as plain data; this service only ever talks to RAG/Chroma/Gemini.
"""

from app.rag import RAG


class RagService:
    def __init__(self, rag: RAG):
        self.rag = rag

    def answer(
        self,
        question: str,
        owner_id: str,
        conversation_history: list[dict] | None = None,
        document_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict:
        """`owner_id` is required (no default) -- this is the only
        production call path into RAG.retrieve()/answer(), so it is the
        actual enforcement point restricting retrieval (and therefore the
        answer and its citations) to documents the current user owns."""
        return self.rag.answer(
            question=question,
            conversation_history=conversation_history or [],
            document_id=document_id,
            owner_id=owner_id,
            trace_id=trace_id,
        )
