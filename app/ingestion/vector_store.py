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

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def reset(self) -> None:
        self.client.delete_collection("researchdesk")
        self.collection = self.client.get_or_create_collection(
            name="researchdesk"
        )