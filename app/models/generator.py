import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

logger = logging.getLogger(__name__)


class GenerationUnavailableError(Exception):
    """Raised when the Gemini API cannot fulfill a generation request.

    Wraps quota/rate-limit and other API-level failures so callers can
    degrade gracefully instead of crashing. The original exception is kept
    on `.original` for logging/debugging.
    """

    def __init__(self, message: str, *, quota_exceeded: bool, original: Exception):
        super().__init__(message)
        self.quota_exceeded = quota_exceeded
        self.original = original


def _is_quota_error(error: errors.APIError) -> bool:
    status = (getattr(error, "status", None) or "").upper()

    return error.code == 429 or status == "RESOURCE_EXHAUSTED"


class Generator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to the .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except errors.ClientError as error:
            quota_exceeded = _is_quota_error(error)

            logger.warning(
                "Gemini generation request failed (quota_exceeded=%s): %s",
                quota_exceeded,
                error,
            )

            if quota_exceeded:
                message = (
                    "The answer-generation service has reached its usage "
                    "quota. Please try again later."
                )
            else:
                message = (
                    "The answer-generation service rejected this request "
                    "and could not generate an answer."
                )

            raise GenerationUnavailableError(
                message,
                quota_exceeded=quota_exceeded,
                original=error,
            ) from error
        except errors.ServerError as error:
            logger.warning("Gemini generation request failed: %s", error)

            raise GenerationUnavailableError(
                (
                    "The answer-generation service is temporarily "
                    "unavailable. Please try again shortly."
                ),
                quota_exceeded=False,
                original=error,
            ) from error

        return response.text

    def rewrite_query(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> str:
        if not conversation_history:
            return question

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history[-3:]
        )

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

        return self.generate(prompt).strip()

    def generate_queries(
        self,
        question: str,
        num_queries: int = 3,
    ) -> list[str]:
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

        response = self.generate(prompt).strip()

        queries = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        return queries[:num_queries]

