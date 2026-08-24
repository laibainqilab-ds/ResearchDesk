chunk = {
    "text": "Machine learning allows computers to learn patterns from data.",
    "document_id": "doc_001",
    "filename": "AI_Introduction.pdf",
    "file_type": "PDF",
    "page_number": 12,
    "chunk_id": 37,
}
print("Chunk text:")
print(chunk["text"])

print("\nMetadata:")
print(f"Document ID: {chunk['document_id']}")
print(f"Filename: {chunk['filename']}")
print(f"File type: {chunk['file_type']}")
print(f"Page: {chunk['page_number']}")
print(f"Chunk ID: {chunk['chunk_id']}")