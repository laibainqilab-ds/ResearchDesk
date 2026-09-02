import chromadb


class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="researchdesk"
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[dict]:
        """Return distinct documents currently indexed, derived from chunk metadata."""
        if self.count() == 0:
            return []

        records = self.collection.get(include=["metadatas"])

        documents: dict[str, dict] = {}

        for metadata in records["metadatas"]:
            document_id = metadata.get("document_id")

            if document_id is None:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("filename"),
                    "file_type": metadata.get("file_type"),
                    "chunk_count": 0,
                    "page_count": None,
                }

            documents[document_id]["chunk_count"] += 1

            page_number = metadata.get("page_number")

            if page_number is not None:
                current_max = documents[document_id]["page_count"] or 0
                documents[document_id]["page_count"] = max(current_max, page_number)

        return list(documents.values())

    def count_document_chunks(self, document_id: str) -> int:
        """Number of chunks currently stored for a given document_id.

        Used to detect whether a document (by content hash) is already indexed
        before ingesting it again.
        """
        records = self.collection.get(where={"document_id": document_id})
        return len(records["ids"])

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a single document, leaving others intact."""
        self.collection.delete(where={"document_id": document_id})

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        where: dict | None = None,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
    )

    def reset(self) -> None:
        self.client.delete_collection("researchdesk")
        self.collection = self.client.get_or_create_collection(
            name="researchdesk"
        )