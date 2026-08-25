from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore
from app.models.generator import Generator


class RAG:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.generator = Generator()

    def answer(self, question: str) -> str:
        query_embedding = self.embedder.embed([question])[0]

        results = self.store.search(
            query_embedding=query_embedding,
            n_results=3,
        )

        # Only use the most relevant chunk for the small generator model
        context = results["documents"][0][0]

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Give a complete answer in 2 sentences.

Answer:
"""

        return self.generator.generate(prompt)