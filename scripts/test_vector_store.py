from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore


embedder = Embedder()
store = VectorStore()

texts = [
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are commonly used in deep learning.",
]

embeddings = embedder.embed(texts)

store.add_documents(
    ids=["test_1", "test_2"],
    documents=texts,
    embeddings=embeddings,
    metadatas=[
        {"source": "test", "page": 1},
        {"source": "test", "page": 2},
    ],
)

print(f"Documents stored: {store.count()}")