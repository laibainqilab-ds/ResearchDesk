import pytest

from app.ingestion.parsers import UnsupportedFileTypeError, file_type_for, parse_document


def test_file_type_for_pdf():
    assert file_type_for("report.pdf") == "PDF"


def test_file_type_for_txt_is_case_insensitive():
    assert file_type_for("notes.TXT") == "TXT"


def test_file_type_for_markdown():
    assert file_type_for("README.md") == "MARKDOWN"


def test_file_type_for_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        file_type_for("archive.zip")


def test_file_type_for_no_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        file_type_for("no_extension")


def test_parse_txt_returns_single_page_without_page_number(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world.\nSecond line.", encoding="utf-8")

    pages, file_type = parse_document(str(file_path), "sample.txt")

    assert file_type == "TXT"
    assert len(pages) == 1
    assert pages[0]["page_number"] is None
    assert "Hello world" in pages[0]["text"]


def test_parse_markdown_returns_single_page_without_page_number(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Heading\n\nSome **bold** content.", encoding="utf-8")

    pages, file_type = parse_document(str(file_path), "sample.md")

    assert file_type == "MARKDOWN"
    assert len(pages) == 1
    assert pages[0]["page_number"] is None
    assert "Heading" in pages[0]["text"]


def test_parse_document_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"not a real archive")

    with pytest.raises(UnsupportedFileTypeError):
        parse_document(str(file_path), "archive.zip")


def _make_two_page_pdf(file_path):
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    page1 = document.new_page()
    page1.insert_text((72, 72), "First page content")
    page2 = document.new_page()
    page2.insert_text((72, 72), "Second page content")
    document.save(str(file_path))
    document.close()


def test_parse_pdf_preserves_page_numbers(tmp_path):
    file_path = tmp_path / "sample.pdf"
    _make_two_page_pdf(file_path)

    pages, file_type = parse_document(str(file_path), "sample.pdf")

    assert file_type == "PDF"
    assert [page["page_number"] for page in pages] == [1, 2]
    assert "First page" in pages[0]["text"]
    assert "Second page" in pages[1]["text"]
