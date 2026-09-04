from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore


embedder = Embedder()
store = VectorStore()

query = "What is Evo 2 used for?"

query_embedding = embedder.embed([query])[0]

results = store.search(
    query_embedding=query_embedding,
    n_results=3,
)

print(f"\nQuery: {query}\n")

for i, document in enumerate(results["documents"][0], start=1):
    metadata = results["metadatas"][0][i - 1]

    print(f"Result {i}:")
    print(f"Page: {metadata['page_number']}")
    print(f"Text: {document[:500]}")
    print()