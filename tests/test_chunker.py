import pytest

from app.ingestion.chunker import chunk_pages


def _words_text(count: int) -> str:
    return " ".join(f"w{i}" for i in range(1, count + 1))


def test_single_page_fewer_words_than_chunk_size_produces_one_chunk():
    pages = [{"page_number": 1, "text": "only three words"}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt")

    assert len(chunks) == 1
    assert chunks[0].text == "only three words"
    assert chunks[0].chunk_id == 0
    assert chunks[0].page_number == 1


def test_chunk_boundaries_and_overlap_match_expected_slices():
    # 12 words, chunk_size=5, overlap=2 -> windows [0:5] [3:8] [6:11] [9:14],
    # stepping by (chunk_size - overlap) = 3 words each time, last window
    # clipped by slicing once start+chunk_size reaches/exceeds the word count.
    pages = [{"page_number": 1, "text": _words_text(12)}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=2)

    assert [c.text for c in chunks] == [
        "w1 w2 w3 w4 w5",
        "w4 w5 w6 w7 w8",
        "w7 w8 w9 w10 w11",
        "w10 w11 w12",
    ]
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3]


def test_chunk_exactly_matching_chunk_size_produces_single_chunk():
    pages = [{"page_number": 1, "text": _words_text(5)}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "w1 w2 w3 w4 w5"


def test_zero_overlap_produces_non_overlapping_chunks():
    pages = [{"page_number": 1, "text": _words_text(10)}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=0)

    assert [c.text for c in chunks] == [
        "w1 w2 w3 w4 w5",
        "w6 w7 w8 w9 w10",
    ]


def test_empty_page_text_produces_no_chunks_for_that_page():
    pages = [{"page_number": 1, "text": ""}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt")

    assert chunks == []


def test_empty_pages_list_produces_no_chunks():
    assert chunk_pages(pages=[], document_id="doc1", filename="f.txt") == []


def test_overlap_equal_to_chunk_size_raises_value_error():
    pages = [{"page_number": 1, "text": _words_text(12)}]

    with pytest.raises(ValueError, match="Overlap must be smaller than chunk size."):
        chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=5)


def test_overlap_greater_than_chunk_size_raises_value_error():
    pages = [{"page_number": 1, "text": _words_text(12)}]

    with pytest.raises(ValueError, match="Overlap must be smaller than chunk size."):
        chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=6)


def test_chunk_id_is_continuous_across_multiple_pages():
    pages = [
        {"page_number": 1, "text": "a b c"},
        {"page_number": 2, "text": "d e f"},
    ]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt", chunk_size=5, overlap=1)

    assert [(c.chunk_id, c.page_number, c.text) for c in chunks] == [
        (0, 1, "a b c"),
        (1, 2, "d e f"),
    ]


def test_metadata_propagates_to_every_chunk():
    pages = [{"page_number": None, "text": "x y z"}]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc2",
        filename="f2.pdf",
        file_type="PDF",
        source="/some/path",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc2"
    assert chunk.filename == "f2.pdf"
    assert chunk.file_type == "PDF"
    assert chunk.page_number is None
    assert chunk.source == "/some/path"


def test_file_type_and_source_default_when_not_provided():
    pages = [{"page_number": 1, "text": "x y z"}]

    chunks = chunk_pages(pages=pages, document_id="doc1", filename="f.txt")

    assert chunks[0].file_type == "PDF"
    assert chunks[0].source is None
