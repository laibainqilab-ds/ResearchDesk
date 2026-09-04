from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.text_cleaner import clean_text
from app.ingestion.chunker import chunk_pages


pdf_path = "data/documents/SCRIPT FOR DISCOVERY.pdf"

pages = parse_pdf(pdf_path)

cleaned_pages = [
    {
        "page_number": page["page_number"],
        "text": clean_text(page["text"]),
    }
    for page in pages
]

chunks = chunk_pages(
    pages=cleaned_pages,
    document_id="doc_001",
    filename="SCRIPT FOR DISCOVERY.pdf",
)

print(f"Number of pages: {len(pages)}")
print(f"Number of chunks: {len(chunks)}")

for chunk in chunks[:5]:
    print(f"\nChunk {chunk.chunk_id}:")
    print(f"Page: {chunk.page_number}")
    print(f"Text: {chunk.text[:300]}")