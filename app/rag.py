from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore
from app.models.generator import Generator


class RAG:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.generator = Generator()

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
            retrieval_question = self.generator.rewrite_query(
                question=question,
                conversation_history=recent_history,
            )
        else:
            retrieval_question = question

        search_queries = self.generator.generate_queries(
            question=retrieval_question,
            num_queries=3,
        )

        where = None

        if document_id:
            where = {"document_id": document_id}

        all_results = []

        for search_query in search_queries:
            query_embedding = self.embedder.embed([search_query])[0]

            results = self.store.search(
                query_embedding=query_embedding,
                n_results=top_k,
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
                or result["distance"] < unique_results[chunk_key]["distance"]
            ):
                unique_results[chunk_key] = result

        ranked_results = sorted(
            unique_results.values(),
            key=lambda result: result["distance"],
        )

        selected_results = ranked_results[:top_k]

        if not selected_results:
            return {
                "answer": (
                    "I couldn't find enough information in the "
                    "provided documents to answer this question."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for result in selected_results:
            document = result["document"]
            metadata = result["metadata"]

            source = metadata["filename"]
            page = metadata.get("page_number")
            chunk_id = metadata["chunk_id"]

            if page is not None:
                source_info = f"{source}, page {page}"
            else:
                source_info = source

            context_parts.append(
                f"[Source: {source_info}]\n{document}"
            )

            sources.append(
                {
                    "document_id": metadata["document_id"],
                    "filename": source,
                    "page_number": page,
                    "chunk_id": chunk_id,
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

        answer = self.generator.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
        }

