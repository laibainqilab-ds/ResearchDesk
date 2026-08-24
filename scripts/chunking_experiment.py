text = """
Artificial intelligence is a field of computer science.
Machine learning is a subset of artificial intelligence.
It allows computers to learn patterns from data.
Deep learning is a type of machine learning.
It uses neural networks with many layers.
These models can process complex information.
"""

words = text.split()

chunk_size = 20
overlap = 0

chunks = []

start = 0

while start < len(words):
    end = start + chunk_size
    chunk = words[start:end]
    chunks.append(" ".join(chunk))

    start += chunk_size - overlap


print(f"Total words: {len(words)}")
print(f"Chunk size: {chunk_size}")
print(f"Overlap: {overlap}")
print(f"Number of chunks: {len(chunks)}")

print("\nChunks:\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print()