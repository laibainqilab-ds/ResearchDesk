from app.models.document import DocumentChunk


def chunk_pages(
    pages: list[dict],
    document_id: str,
    filename: str,
    file_type: str = "PDF",
    chunk_size: int = 500,
    overlap: int = 50,
    source: str | None = None,
) -> list[DocumentChunk]:
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        words = text.split()

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    filename=filename,
                    file_type=file_type,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page["page_number"],
                    source=source,
                )
            )

            chunk_id += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks