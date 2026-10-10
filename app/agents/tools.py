"""Deterministic retrieval/storage tool interfaces for the Phase 9 agent workflow.

`RetrievalTools` is the only seam through which agents (Router/Research/
Answer, built in later steps) are allowed to fetch evidence or document
information. It wraps the existing Phase 1-8 pipeline -- `RAG.retrieve()`
and `VectorStore` -- rather than duplicating retrieval logic, and none of
its methods make an LLM call: query rewriting and multi-query generation
stay with `RAG.answer()` / the Generator, since those require a model call
and this layer is intentionally LLM-free.

Every method returns plain dicts/lists using the same evidence shape
`RAG.retrieve()` already produces, so no new schema is introduced for data
that already has one.
"""

from app.rag import RAG


class RetrievalTools:
    def __init__(self, rag: RAG):
        self.rag = rag

    def search_documents(
        self,
        query: str,
        top_k: int = 3,
        owner_id: str | None = None,
        enable_reranking: bool = True,
        trace_id: str | None = None,
    ) -> list[dict]:
        """Semantic search across all indexed documents. Returns the final,
        already-deduplicated and (optionally) reranked evidence list.

        `owner_id`, when given, restricts results to documents owned by that
        user -- see RAG.retrieve()'s `owner_id` filter. Optional only so
        existing unit tests of this tool (which have no user concept) keep
        working; every agent-workflow caller (app.agents.graph,
        app.agents.research) always threads through the current
        AgentState.owner_id."""
        retrieval = self.rag.retrieve(
            retrieval_question=query,
            search_queries=[query],
            top_k=top_k,
            document_id=None,
            owner_id=owner_id,
            enable_reranking=enable_reranking,
            trace_id=trace_id,
        )
        return retrieval["final_evidence"]

    def search_specific_document(
        self,
        document_id: str,
        query: str,
        top_k: int = 3,
        owner_id: str | None = None,
        enable_reranking: bool = True,
        trace_id: str | None = None,
    ) -> list[dict]:
        """Semantic search scoped to a single document via metadata filtering."""
        retrieval = self.rag.retrieve(
            retrieval_question=query,
            search_queries=[query],
            top_k=top_k,
            document_id=document_id,
            owner_id=owner_id,
            enable_reranking=enable_reranking,
            trace_id=trace_id,
        )
        return retrieval["final_evidence"]

    def list_available_documents(self, owner_id: str | None = None) -> list[dict]:
        """Documents currently indexed, with chunk/page counts -- restricted
        to `owner_id` when given."""
        return self.rag.store.list_documents(owner_id=owner_id)

    def get_document_metadata(self, document_id: str, owner_id: str | None = None) -> dict | None:
        """Metadata for one document owned by `owner_id`, or None if it
        isn't indexed (or isn't owned by that user)."""
        for document in self.rag.store.list_documents(owner_id=owner_id):
            if document["document_id"] == document_id:
                return document
        return None

    def get_document_page(
        self, document_id: str, page_number: int, owner_id: str | None = None
    ) -> list[dict]:
        """Exact lookup of the chunks belonging to one page of one document
        (not a similarity search), restricted to `owner_id` when given."""
        return self.rag.store.get_document_chunks(
            document_id, page_number=page_number, owner_id=owner_id
        )
