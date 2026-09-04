import tempfile

from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore

# Uses a throwaway Chroma directory, not data/chroma -- this is a manual
# smoke test with fake data and must never write into the real,
# production-persisted collection the live app and evaluation runs read from.
with tempfile.TemporaryDirectory() as temp_dir:
    embedder = Embedder()
    store = VectorStore(persist_directory=temp_dir)

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