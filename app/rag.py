import logging
import re

from app.ingestion.embedder import Embedder
from app.ingestion.reranker import Reranker
from app.ingestion.vector_store import VectorStore
from app.models.generator import Generator, GenerationUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation memory
#
# The conversation context passed to query rewriting is always bounded:
# the most recent RECENT_HISTORY_SIZE messages, plus either a handful of
# deterministically-selected older messages (cheap, no LLM call) or, once
# the older portion grows past SUMMARIZATION_THRESHOLD_MESSAGES, a single
# summarized message (one LLM call) in their place. It is never the raw,
# unbounded conversation_history list.
# ---------------------------------------------------------------------------

RECENT_HISTORY_SIZE = 3
RELEVANT_HISTORY_MAX_MESSAGES = 2
RELEVANCE_MIN_SCORE = 0.15
SUMMARIZATION_THRESHOLD_MESSAGES = 10

_WORD_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "about", "as", "by", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "do", "does", "did", "i",
    "you", "he", "she", "we", "they", "them", "his", "her", "their",
}


def _content_words(text: str) -> set[str]:
    words = _WORD_PATTERN.findall((text or "").lower())
    return {word for word in words if word not in _STOPWORDS}


def select_relevant_history(
    question: str,
    older_messages: list[dict],
    max_messages: int = RELEVANT_HISTORY_MAX_MESSAGES,
    min_score: float = RELEVANCE_MIN_SCORE,
) -> list[dict]:
    """Deterministic keyword-overlap relevance selection over older
    conversation turns -- no LLM call.

    Scores each older message by Jaccard word overlap with the current
    question, keeps only messages at or above `min_score`, and returns at
    most `max_messages` of them in their original chronological order (so
    the resulting context still reads as a coherent partial conversation,
    not a relevance-sorted jumble).
    """
    question_words = _content_words(question)

    if not question_words or not older_messages:
        return []

    scored_indices = []

    for index, message in enumerate(older_messages):
        message_words = _content_words(message.get("content", ""))

        if not message_words:
            continue

        overlap = question_words & message_words
        union = question_words | message_words
        score = len(overlap) / len(union) if union else 0.0

        if score >= min_score:
            scored_indices.append((score, index))

    scored_indices.sort(key=lambda item: item[0], reverse=True)
    selected_indices = {index for _, index in scored_indices[:max_messages]}

    return [
        message
        for index, message in enumerate(older_messages)
        if index in selected_indices
    ]


# ---------------------------------------------------------------------------
# Citations
#
# Evidence sent to the LLM is numbered [1], [2], ... matching sources[i]
# 1:1. The model is instructed to cite using only those markers, but its
# citations are never trusted at face value -- extract_citations() parses
# them back out of the generated text and checks each one against the real
# evidence list; anything out of range is reported as invalid, not silently
# resolved into a citation.
# ---------------------------------------------------------------------------

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(answer_text: str | None, sources: list[dict]) -> dict:
    if not answer_text:
        return {"valid": [], "invalid": [], "citation_map": {}}

    found_numbers = sorted({int(match) for match in CITATION_PATTERN.findall(answer_text)})

    valid = []
    invalid = []
    citation_map = {}

    for number in found_numbers:
        if 1 <= number <= len(sources):
            valid.append(number)
            citation_map[number] = sources[number - 1]
        else:
            invalid.append(number)

    return {"valid": valid, "invalid": invalid, "citation_map": citation_map}


class RAG:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.generator = Generator()
        self.reranker = Reranker()

    def retrieve(
        self,
        retrieval_question: str,
        search_queries: list[str],
        top_k: int = 3,
        document_id: str | None = None,
        enable_reranking: bool = True,
    ) -> dict:
        where = None

        if document_id:
            where = {"document_id": document_id}

        candidate_k = max(top_k * 3, 10)

        all_results = []

        for search_query in search_queries:
            query_embedding = self.embedder.embed([search_query])[0]

            results = self.store.search(
                query_embedding=query_embedding,
                n_results=candidate_k,
                where=where,
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            ):
                all_results.append(
                    {
                        "document": document,
                        "metadata": metadata,
                        "distance": distance,
                        "search_query": search_query,
                    }
                )

        unique_results = {}

        for result in all_results:
            metadata = result["metadata"]

            candidate_document_id = metadata.get("document_id")
            candidate_chunk_id = metadata.get("chunk_id")

            if candidate_document_id is None or candidate_chunk_id is None:
                logger.warning(
                    "Skipping retrieved chunk with malformed metadata "
                    "(missing document_id and/or chunk_id): %r",
                    metadata,
                )
                continue

            chunk_key = (candidate_document_id, candidate_chunk_id)

            if (
                chunk_key not in unique_results
                or result["distance"]
                < unique_results[chunk_key]["distance"]
            ):
                unique_results[chunk_key] = result

        candidates = list(unique_results.values())

        if not candidates:
            return {
                "candidates": [],
                "final_evidence": [],
            }

        if enable_reranking:
            candidate_documents = [
                candidate["document"]
                for candidate in candidates
            ]

            reranked_documents = self.reranker.rerank(
                query=retrieval_question,
                documents=candidate_documents,
            )

            reranked_lookup = {
                document: score
                for document, score in reranked_documents
            }

            for candidate in candidates:
                candidate["rerank_score"] = reranked_lookup[
                    candidate["document"]
                ]

            ranked_candidates = sorted(
                candidates,
                key=lambda result: result["rerank_score"],
                reverse=True,
            )
        else:
            for candidate in candidates:
                candidate["rerank_score"] = None

            ranked_candidates = sorted(
                candidates,
                key=lambda result: result["distance"],
            )

        selected_results = ranked_candidates[:top_k]

        inspector_candidates = []

        for result in ranked_candidates:
            metadata = result["metadata"]

            inspector_candidates.append(
                {
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_id"),
                    "search_query": result["search_query"],
                    "retrieval_distance": result["distance"],
                    "rerank_score": result["rerank_score"],
                    "document": result["document"],
                }
            )

        final_evidence = [
            {
                "document_id": result["metadata"].get("document_id"),
                "filename": result["metadata"].get("filename"),
                "page_number": result["metadata"].get("page_number"),
                "chunk_id": result["metadata"].get("chunk_id"),
                "rerank_score": result["rerank_score"],
                "document": result["document"],
            }
            for result in selected_results
        ]

        return {
            "candidates": inspector_candidates,
            "final_evidence": final_evidence,
        }

    def _summarize_older_history(self, older_messages: list[dict]) -> str | None:
        try:
            return self.generator.summarize_conversation(older_messages)
        except GenerationUnavailableError as error:
            logger.warning(
                "Conversation summarization unavailable, falling back to "
                "relevance selection: %s",
                error,
            )
            return None

    def _prepare_conversation_context(
        self,
        conversation_history: list[dict],
        question: str,
    ) -> list[dict]:
        """Bounded context for query rewriting: recent-N messages are always
        included; older messages are either relevance-filtered (small
        history) or condensed into one summary message via a single Gemini
        call (large history) -- never passed through unbounded."""
        if not conversation_history:
            return []

        recent = conversation_history[-RECENT_HISTORY_SIZE:]
        older = conversation_history[:-RECENT_HISTORY_SIZE]

        if not older:
            return recent

        if len(older) > SUMMARIZATION_THRESHOLD_MESSAGES:
            summary = self._summarize_older_history(older)

            if summary:
                return [
                    {
                        "role": "system",
                        "content": f"Summary of earlier conversation: {summary}",
                    }
                ] + recent
            # Summarization unavailable -- fall through to relevance
            # selection rather than silently dropping older context.

        relevant = select_relevant_history(question, older)
        return relevant + recent

    def answer(
        self,
        question: str,
        top_k: int = 3,
        conversation_history: list[dict] | None = None,
        document_id: str | None = None,
        enable_query_rewrite: bool = True,
        enable_multi_query: bool = True,
        enable_reranking: bool = True,
        enable_answer_generation: bool = True,
    ) -> dict:
        conversation_history = conversation_history or []

        # Conversation-context preparation (and its possible summarization
        # Gemini call) only ever runs when rewriting is actually enabled --
        # experiments/callers with enable_query_rewrite=False must see zero
        # additional Gemini calls from conversation memory, unchanged from
        # before this existed.
        if enable_query_rewrite:
            prepared_history = self._prepare_conversation_context(conversation_history, question)
        else:
            prepared_history = []

        if enable_query_rewrite and prepared_history:
            try:
                retrieval_question = self.generator.rewrite_query(
                    question=question,
                    conversation_history=prepared_history,
                )
            except GenerationUnavailableError as error:
                logger.warning("Query rewriting unavailable, using original question: %s", error)
                retrieval_question = question
        else:
            retrieval_question = question

        if enable_multi_query:
            try:
                search_queries = self.generator.generate_queries(
                    question=retrieval_question,
                    num_queries=3,
                )
            except GenerationUnavailableError as error:
                logger.warning("Multi-query generation unavailable, using single query: %s", error)
                search_queries = [retrieval_question]
        else:
            search_queries = [retrieval_question]

        retrieval = self.retrieve(
            retrieval_question=retrieval_question,
            search_queries=search_queries,
            top_k=top_k,
            document_id=document_id,
            enable_reranking=enable_reranking,
        )

        selected_results = retrieval["final_evidence"]

        if not selected_results:
            return {
                "answer": (
                    "I couldn't find enough information in the "
                    "provided documents to answer this question."
                ),
                "sources": [],
                "citations": {"valid": [], "invalid": [], "citation_map": {}},
                "retrieval": {
                    "original_question": question,
                    "rewritten_question": retrieval_question,
                    "search_queries": search_queries,
                    "prepared_history": prepared_history,
                    "candidates": [],
                    "final_evidence": [],
                },
                "error": None,
            }

        context_parts = []
        sources = []

        for citation_id, result in enumerate(selected_results, start=1):
            document = result["document"]

            source = result["filename"]
            page = result["page_number"]
            chunk_id = result["chunk_id"]

            if page is not None:
                location = f"{source} — page {page} — chunk_{chunk_id}"
            else:
                location = f"{source} — chunk_{chunk_id}"

            context_parts.append(
                f"[{citation_id}] {location}\n{document}"
            )

            sources.append(
                {
                    "citation_id": citation_id,
                    "document_id": result["document_id"],
                    "filename": source,
                    "page_number": page,
                    "chunk_id": chunk_id,
                    "rerank_score": result["rerank_score"],
                }
            )

        if not enable_answer_generation:
            return {
                "answer": None,
                "sources": sources,
                "citations": {"valid": [], "invalid": [], "citation_map": {}},
                "retrieval": {
                    "original_question": question,
                    "rewritten_question": retrieval_question,
                    "search_queries": search_queries,
                    "prepared_history": prepared_history,
                    "candidates": retrieval["candidates"],
                    "final_evidence": retrieval["final_evidence"],
                },
                "error": None,
            }

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using only the evidence below.

Evidence:
{context}

Question:
{question}

Instructions:
- Answer only using the evidence above. Do not use outside knowledge.
- Cite every factual claim using the evidence numbers in brackets, e.g. [1], [2].
- If a claim depends on more than one piece of evidence, cite all of them, e.g. [1][2].
- Never invent, guess, or reformat a filename, page number, or source detail -- the bracketed evidence numbers are the only citations you may use.
- If the evidence does not contain enough information to answer, say so directly instead of guessing.

Answer:
"""

        try:
            answer = self.generator.generate(prompt)
            error = None
        except GenerationUnavailableError as generation_error:
            logger.warning("Answer generation unavailable: %s", generation_error)
            answer = None
            error = {
                "message": str(generation_error),
            }

        citations = extract_citations(answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "retrieval": {
                "original_question": question,
                "rewritten_question": retrieval_question,
                "search_queries": search_queries,
                "prepared_history": prepared_history,
                "candidates": retrieval["candidates"],
                "final_evidence": retrieval["final_evidence"],
            },
            "error": error,
        }
