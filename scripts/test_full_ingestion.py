from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.text_cleaner import clean_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import Embedder
from app.ingestion.vector_store import VectorStore


pdf_path = "data/documents/SCRIPT FOR DISCOVERY.pdf"

# 1. Parse PDF
pages = parse_pdf(pdf_path)

# 2. Clean text
cleaned_pages = [
    {
        "page_number": page["page_number"],
        "text": clean_text(page["text"]),
    }
    for page in pages
]

# 3. Create chunks
chunks = chunk_pages(
    pages=cleaned_pages,
    document_id="doc_001",
    filename="SCRIPT FOR DISCOVERY.pdf",
)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

# 4. Generate embeddings
embedder = Embedder()
texts = [chunk.text for chunk in chunks]
embeddings = embedder.embed(texts)

# 5. Store in ChromaDB
store = VectorStore()

ids = [
    f"{chunk.document_id}_chunk_{chunk.chunk_id}"
    for chunk in chunks
]

metadatas = [
    {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "file_type": chunk.file_type,
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.page_number,
    }
    for chunk in chunks
]

store.add_documents(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Documents stored in ChromaDB: {store.count()}")