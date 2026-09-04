import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.observability import log_event


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

REQUEST_TIMEOUT_SECONDS = 180


class GenerationUnavailableError(Exception):
    """Raised when the Gemini backend cannot fulfill a generation request.

    Wraps connection, timeout, API, and unusable-response failures so callers
    can degrade gracefully instead of crashing. The original exception is kept
    on `.original` for logging/debugging.
    """

    def __init__(self, message: str, *, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class Generator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise GenerationUnavailableError(
                "GEMINI_API_KEY is not set. Add it to your .env file before "
                "starting ResearchDesk."
            )

        self.model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
            ),
        )

    def generate(self, prompt: str, trace_id: str | None = None) -> str:
        start = time.perf_counter()

        log_event(
            trace_id,
            "gemini_call_started",
            model=self.model_name,
            prompt_length=len(prompt),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except genai_errors.APIError as error:
            logger.warning(
                "Gemini generation request failed with code %s: %s",
                error.code,
                error,
            )
            log_event(
                trace_id,
                "gemini_call_failed",
                level=logging.WARNING,
                model=self.model_name,
                error_code=error.code,
                error=str(error),
                latency_seconds=time.perf_counter() - start,
            )
            raise GenerationUnavailableError(
                f"The answer-generation service returned an error "
                f"(code {error.code}) and could not generate an answer.",
                original=error,
            ) from error
        except Exception as error:
            # Covers connection failures, timeouts, and other transport-level
            # errors from the underlying HTTP client that aren't raised as
            # genai_errors.APIError.
            logger.warning("Gemini generation request failed: %s", error)
            log_event(
                trace_id,
                "gemini_call_failed",
                level=logging.WARNING,
                model=self.model_name,
                error=str(error),
                latency_seconds=time.perf_counter() - start,
            )
            raise GenerationUnavailableError(
                "The answer-generation service is temporarily unreachable. "
                "Please check the connection and try again.",
                original=error,
            ) from error

        text = getattr(response, "text", None)

        if not isinstance(text, str) or not text.strip():
            logger.warning("Gemini response contained no usable text: %r", response)
            log_event(
                trace_id,
                "gemini_call_failed",
                level=logging.WARNING,
                model=self.model_name,
                error="empty_or_malformed_response",
                latency_seconds=time.perf_counter() - start,
            )
            raise GenerationUnavailableError(
                "The answer-generation service returned an unexpected response format.",
                original=None,
            )

        log_event(
            trace_id,
            "gemini_call_succeeded",
            model=self.model_name,
            response_length=len(text),
            latency_seconds=time.perf_counter() - start,
        )

        return text.strip()

    def rewrite_query(
        self,
        question: str,
        conversation_history: list[dict],
        trace_id: str | None = None,
    ) -> str:
        if not conversation_history:
            return question

        # conversation_history is expected to already be bounded by the
        # caller (RAG._prepare_conversation_context: recent messages plus
        # either a few relevance-selected older ones or a single summary
        # message) -- it is used as given, not re-truncated here.
        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history
        )

        log_event(trace_id, "query_rewrite_requested", history_length=len(conversation_history))

        prompt = f"""
Rewrite the user's current question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- this
- that
- the previous one
- the following year

The rewritten query must preserve the user's intended meaning.

Do not answer the question.
Return only the rewritten search query.

Conversation history:
{history_text}

Current question:
{question}

Standalone search query:
"""

        return self.generate(prompt, trace_id=trace_id).strip()

    def generate_queries(
        self,
        question: str,
        num_queries: int = 3,
        trace_id: str | None = None,
    ) -> list[str]:
        log_event(trace_id, "multi_query_requested", requested_count=num_queries)

        prompt = f"""
Generate {num_queries} different search queries for the user's question.

The queries should:
- preserve the original meaning
- use different wording or perspectives
- improve the chance of finding relevant information
- be suitable for semantic search over document chunks

Do not answer the question.
Return exactly {num_queries} search queries, one per line.
Do not number them.
Do not add explanations.

Question:
{question}

Search queries:
"""

        response = self.generate(prompt, trace_id=trace_id).strip()

        queries = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        queries = queries[:num_queries]

        log_event(trace_id, "multi_query_generated", generated_count=len(queries))

        return queries

    def summarize_conversation(self, messages: list[dict], trace_id: str | None = None) -> str:
        if not messages:
            return ""

        conversation_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )

        log_event(trace_id, "conversation_summarization_requested", message_count=len(messages))

        prompt = f"""
Summarize the following earlier conversation in 2-3 sentences, preserving
any facts, names, numbers, or topics that later questions might refer back
to.

Do not answer any question.
Return only the summary.

Conversation:
{conversation_text}

Summary:
"""

        return self.generate(prompt, trace_id=trace_id).strip()
