from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()