from pydantic import BaseModel, Field
from typing import Optional


class DocumentChunk(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_id: int
    text: str
    page_number: Optional[int] = None