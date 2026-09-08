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
        enable_reranking: bool = True,
        trace_id: str | None = None,
    ) -> list[dict]:
        """Semantic search across all indexed documents. Returns the final,
        already-deduplicated and (optionally) reranked evidence list."""
        retrieval = self.rag.retrieve(
            retrieval_question=query,
            search_queries=[query],
            top_k=top_k,
            document_id=None,
            enable_reranking=enable_reranking,
            trace_id=trace_id,
        )
        return retrieval["final_evidence"]

    def search_specific_document(
        self,
        document_id: str,
        query: str,
        top_k: int = 3,
        enable_reranking: bool = True,
        trace_id: str | None = None,
    ) -> list[dict]:
        """Semantic search scoped to a single document via metadata filtering."""
        retrieval = self.rag.retrieve(
            retrieval_question=query,
            search_queries=[query],
            top_k=top_k,
            document_id=document_id,
            enable_reranking=enable_reranking,
            trace_id=trace_id,
        )
        return retrieval["final_evidence"]

    def list_available_documents(self) -> list[dict]:
        """All documents currently indexed, with chunk/page counts."""
        return self.rag.store.list_documents()

    def get_document_metadata(self, document_id: str) -> dict | None:
        """Metadata for one document, or None if it isn't indexed."""
        for document in self.rag.store.list_documents():
            if document["document_id"] == document_id:
                return document
        return None

    def get_document_page(self, document_id: str, page_number: int) -> list[dict]:
        """Exact lookup of the chunks belonging to one page of one document
        (not a similarity search)."""
        return self.rag.store.get_document_chunks(document_id, page_number=page_number)
