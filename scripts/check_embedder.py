from app.ingestion.embedder import Embedder


embedder = Embedder()

texts = [
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are commonly used in deep learning.",
]

embeddings = embedder.embed(texts)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"First 5 values: {embeddings[0][:5]}")