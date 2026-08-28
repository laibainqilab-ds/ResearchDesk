import logging

from app.ingestion.embedder import Embedder
from app.ingestion.reranker import Reranker
from app.ingestion.vector_store import VectorStore
from app.models.generator import Generator, GenerationUnavailableError

logger = logging.getLogger(__name__)


class RAG:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.generator = Generator()
        self.reranker = Reranker()

    def retrieve(
        self,
        retrieval_question: str,
        search_queries: list[str],
        top_k: int = 3,
        document_id: str | None = None,
    ) -> dict:
        where = None

        if document_id:
            where = {"document_id": document_id}

        candidate_k = max(top_k * 3, 10)

        all_results = []

        for search_query in search_queries:
            query_embedding = self.embedder.embed([search_query])[0]

            results = self.store.search(
                query_embedding=query_embedding,
                n_results=candidate_k,
                where=where,
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            ):
                all_results.append(
                    {
                        "document": document,
                        "metadata": metadata,
                        "distance": distance,
                        "search_query": search_query,
                    }
                )

        unique_results = {}

        for result in all_results:
            metadata = result["metadata"]

            chunk_key = (
                metadata["document_id"],
                metadata["chunk_id"],
            )

            if (
                chunk_key not in unique_results
                or result["distance"]
                < unique_results[chunk_key]["distance"]
            ):
                unique_results[chunk_key] = result

        candidates = list(unique_results.values())

        if not candidates:
            return {
                "candidates": [],
                "final_evidence": [],
            }

        candidate_documents = [
            candidate["document"]
            for candidate in candidates
        ]

        reranked_documents = self.reranker.rerank(
            query=retrieval_question,
            documents=candidate_documents,
        )

        reranked_lookup = {
            document: score
            for document, score in reranked_documents
        }

        for candidate in candidates:
            candidate["rerank_score"] = reranked_lookup[
                candidate["document"]
            ]

        ranked_candidates = sorted(
            candidates,
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        selected_results = ranked_candidates[:top_k]

        inspector_candidates = []

        for result in ranked_candidates:
            metadata = result["metadata"]

            inspector_candidates.append(
                {
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata["chunk_id"],
                    "search_query": result["search_query"],
                    "retrieval_distance": result["distance"],
                    "rerank_score": result["rerank_score"],
                    "document": result["document"],
                }
            )

        final_evidence = [
            {
                "document_id": result["metadata"]["document_id"],
                "filename": result["metadata"]["filename"],
                "page_number": result["metadata"].get("page_number"),
                "chunk_id": result["metadata"]["chunk_id"],
                "rerank_score": result["rerank_score"],
                "document": result["document"],
            }
            for result in selected_results
        ]

        return {
            "candidates": inspector_candidates,
            "final_evidence": final_evidence,
        }

    def answer(
        self,
        question: str,
        top_k: int = 3,
        conversation_history: list[dict] | None = None,
        document_id: str | None = None,
    ) -> dict:
        conversation_history = conversation_history or []

        recent_history = conversation_history[-3:]

        if recent_history:
            try:
                retrieval_question = self.generator.rewrite_query(
                    question=question,
                    conversation_history=recent_history,
                )
            except GenerationUnavailableError as error:
                logger.warning("Query rewriting unavailable, using original question: %s", error)
                retrieval_question = question
        else:
            retrieval_question = question

        try:
            search_queries = self.generator.generate_queries(
                question=retrieval_question,
                num_queries=3,
            )
        except GenerationUnavailableError as error:
            logger.warning("Multi-query generation unavailable, using single query: %s", error)
            search_queries = [retrieval_question]

        retrieval = self.retrieve(
            retrieval_question=retrieval_question,
            search_queries=search_queries,
            top_k=top_k,
            document_id=document_id,
        )

        selected_results = retrieval["final_evidence"]

        if not selected_results:
            return {
                "answer": (
                    "I couldn't find enough information in the "
                    "provided documents to answer this question."
                ),
                "sources": [],
                "retrieval": {
                    "original_question": question,
                    "rewritten_question": retrieval_question,
                    "search_queries": search_queries,
                    "candidates": [],
                    "final_evidence": [],
                },
                "error": None,
            }

        context_parts = []
        sources = []

        for result in selected_results:
            document = result["document"]

            source = result["filename"]
            page = result["page_number"]
            chunk_id = result["chunk_id"]

            if page is not None:
                source_info = f"{source}, page {page}"
            else:
                source_info = source

            context_parts.append(
                f"[Source: {source_info}]\n{document}"
            )

            sources.append(
                {
                    "document_id": result["document_id"],
                    "filename": source,
                    "page_number": page,
                    "chunk_id": chunk_id,
                    "rerank_score": result["rerank_score"],
                }
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Give a complete answer in 2 sentences.

Answer:
"""

        try:
            answer = self.generator.generate(prompt)
            error = None
        except GenerationUnavailableError as generation_error:
            logger.warning("Answer generation unavailable: %s", generation_error)
            answer = None
            error = {
                "quota_exceeded": generation_error.quota_exceeded,
                "message": str(generation_error),
            }

        return {
            "answer": answer,
            "sources": sources,
            "retrieval": {
                "original_question": question,
                "rewritten_question": retrieval_question,
                "search_queries": search_queries,
                "candidates": retrieval["candidates"],
                "final_evidence": retrieval["final_evidence"],
            },
            "error": error,
        }
