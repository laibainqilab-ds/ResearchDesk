chunks = [
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are commonly used in deep learning.",
    "Python is a popular programming language.",
    "The company was founded in 2015.",
    "The company operates in several countries.",
]

print(f"Number of chunks: {len(chunks)}")

total_words = sum(len(chunk.split()) for chunk in chunks)

print(f"Total words: {total_words}")

top_k = 2

selected_chunks = chunks[:top_k]

selected_words = sum(
    len(chunk.split())
    for chunk in selected_chunks
)

print(f"\nChunks sent to LLM: {top_k}")
print(f"Words sent to LLM: {selected_words}")