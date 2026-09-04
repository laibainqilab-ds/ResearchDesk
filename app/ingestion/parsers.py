from pathlib import Path

from app.ingestion.pdf_parser import parse_pdf


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension has no registered parser."""


def file_type_for(filename: str) -> str:
    """Map a filename's extension to a ResearchDesk file_type label.

    Raises UnsupportedFileTypeError for anything outside PDF/TXT/Markdown.
    """
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return "PDF"
    if extension == ".txt":
        return "TXT"
    if extension == ".md":
        return "MARKDOWN"

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{extension or filename}'. "
        f"Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def parse_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    return [{"page_number": None, "text": text}]


def parse_md(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    return [{"page_number": None, "text": text}]


def parse_document(file_path: str, filename: str) -> tuple[list[dict], str]:
    """Parse a document into pages using the parser selected by extension.

    Returns (pages, file_type). Each page is {"page_number": int | None, "text": str}.
    """
    file_type = file_type_for(filename)

    if file_type == "PDF":
        pages = parse_pdf(file_path)
    elif file_type == "TXT":
        pages = parse_txt(file_path)
    else:
        pages = parse_md(file_path)

    return pages, file_type
