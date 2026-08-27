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
    ) -> dict:
        conversation_history = conversation_history or []

        recent_history = conversation_history[-3:]

        if recent_history:
            history_text = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in recent_history
            )

            retrieval_question = (
                f"Previous conversation:\n"
                f"{history_text}\n\n"
                f"Current question:\n"
                f"{question}"
            )
        else:
            retrieval_question = question

        query_embedding = self.embedder.embed([retrieval_question])[0]

        results = self.store.search(
            query_embedding=query_embedding,
            n_results=top_k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not distances or distances[0] > 0.9:
            return {
                "answer": (
                    "I couldn't find enough information in the "
                    "provided documents to answer this question."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for document, metadata in zip(documents, metadatas):
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