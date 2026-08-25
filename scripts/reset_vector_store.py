from app.ingestion.vector_store import VectorStore


store = VectorStore()

print(f"Before reset: {store.count()}")

store.reset()

print(f"After reset: {store.count()}")