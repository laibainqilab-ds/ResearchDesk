from app.ingestion.vector_store import VectorStore


def make_store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma"))


def test_count_document_chunks(tmp_path):
    store = make_store(tmp_path)

    store.add_documents(
        ids=["doc1_chunk_0", "doc1_chunk_1"],
        documents=["chunk one text", "chunk two text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"document_id": "doc1", "filename": "a.txt", "file_type": "TXT", "chunk_id": 0},
            {"document_id": "doc1", "filename": "a.txt", "file_type": "TXT", "chunk_id": 1},
        ],
    )

    store.add_documents(
        ids=["doc2_chunk_0"],
        documents=["other document text"],
        embeddings=[[0.5, 0.6]],
        metadatas=[
            {"document_id": "doc2", "filename": "b.txt", "file_type": "TXT", "chunk_id": 0},
        ],
    )

    assert store.count_document_chunks("doc1") == 2
    assert store.count_document_chunks("doc2") == 1
    assert store.count_document_chunks("doc3") == 0


def test_delete_document_removes_only_that_document(tmp_path):
    store = make_store(tmp_path)

    store.add_documents(
        ids=["doc1_chunk_0", "doc1_chunk_1"],
        documents=["chunk one text", "chunk two text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"document_id": "doc1", "filename": "a.txt", "file_type": "TXT", "chunk_id": 0},
            {"document_id": "doc1", "filename": "a.txt", "file_type": "TXT", "chunk_id": 1},
        ],
    )

    store.add_documents(
        ids=["doc2_chunk_0"],
        documents=["other document text"],
        embeddings=[[0.5, 0.6]],
        metadatas=[
            {"document_id": "doc2", "filename": "b.txt", "file_type": "TXT", "chunk_id": 0},
        ],
    )

    store.delete_document("doc1")

    assert store.count_document_chunks("doc1") == 0
    assert store.count() == 1

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc2"


def test_list_documents_reports_file_type_and_page_count(tmp_path):
    store = make_store(tmp_path)

    store.add_documents(
        ids=["doc1_chunk_0", "doc1_chunk_1"],
        documents=["page one text", "page two text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": 0,
                "page_number": 1,
            },
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": 1,
                "page_number": 3,
            },
        ],
    )

    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0]["file_type"] == "PDF"
    assert documents[0]["page_count"] == 3
    assert documents[0]["chunk_count"] == 2


def test_list_documents_page_count_none_when_no_page_numbers(tmp_path):
    store = make_store(tmp_path)

    store.add_documents(
        ids=["doc1_chunk_0"],
        documents=["txt chunk text"],
        embeddings=[[0.1, 0.2]],
        metadatas=[
            {"document_id": "doc1", "filename": "a.txt", "file_type": "TXT", "chunk_id": 0},
        ],
    )

    documents = store.list_documents()

    assert documents[0]["page_count"] is None


def test_list_documents_empty_when_store_empty(tmp_path):
    store = make_store(tmp_path)

    assert store.list_documents() == []


def _add_two_page_document(store):
    store.add_documents(
        ids=["doc1_chunk_0", "doc1_chunk_1", "doc1_chunk_2"],
        documents=["page one text", "page two text A", "page two text B"],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        metadatas=[
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": 0,
                "page_number": 1,
            },
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": 1,
                "page_number": 2,
            },
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": 2,
                "page_number": 2,
            },
        ],
    )

    store.add_documents(
        ids=["doc2_chunk_0"],
        documents=["other document, page one"],
        embeddings=[[0.7, 0.8]],
        metadatas=[
            {
                "document_id": "doc2",
                "filename": "b.pdf",
                "file_type": "PDF",
                "chunk_id": 0,
                "page_number": 1,
            },
        ],
    )


def test_get_document_chunks_returns_all_chunks_for_document(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    chunks = store.get_document_chunks("doc1")

    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk["document_id"] == "doc1" for chunk in chunks)


def test_get_document_chunks_filters_by_page_number(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    chunks = store.get_document_chunks("doc1", page_number=2)

    assert [chunk["chunk_id"] for chunk in chunks] == [1, 2]
    assert all(chunk["page_number"] == 2 for chunk in chunks)


def test_get_document_chunks_does_not_cross_documents(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    chunks = store.get_document_chunks("doc2", page_number=1)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "doc2"


def test_get_document_chunks_includes_text(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    chunks = store.get_document_chunks("doc1", page_number=1)

    assert chunks[0]["document"] == "page one text"


def test_get_document_chunks_returns_empty_for_unknown_document(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    assert store.get_document_chunks("doc-missing") == []


def test_get_document_chunks_returns_empty_for_page_with_no_chunks(tmp_path):
    store = make_store(tmp_path)
    _add_two_page_document(store)

    assert store.get_document_chunks("doc1", page_number=99) == []
