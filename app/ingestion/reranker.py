from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[tuple[str, float]]:
        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (document, float(score))
            for document, score in ranked
        ]