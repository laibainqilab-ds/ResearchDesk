from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


documents = [
    "The cat is sleeping on the sofa.",
    "Python is a programming language.",
    "Machine learning allows computers to learn patterns from data.",
    "The dog is playing in the garden.",
    "Neural networks are commonly used in deep learning.",
]

query = "Which animal is sleeping?"


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])


similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


results = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)


print("Query:")
print(query)

print("\nResults:")

top_k = 3

for document, score in results[:top_k]:
    print(f"{score:.4f} - {document}")