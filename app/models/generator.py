import json
import logging
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "https://wish-excited-organizational-difficulties.trycloudflare.com"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"

REQUEST_TIMEOUT_SECONDS = 180


class GenerationUnavailableError(Exception):
    """Raised when the remote Ollama/Qwen server cannot fulfill a generation request.

    Wraps connection, timeout, HTTP, and malformed-response failures so callers
    can degrade gracefully instead of crashing. The original exception is kept
    on `.original` for logging/debugging.
    """

    def __init__(self, message: str, *, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class Generator:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as error:
            logger.warning(
                "Ollama generation request failed with HTTP %s: %s",
                error.code,
                error,
            )
            raise GenerationUnavailableError(
                f"The answer-generation service returned an error "
                f"(HTTP {error.code}) and could not generate an answer.",
                original=error,
            ) from error
        except OSError as error:
            # Covers URLError (DNS/connection failures), TimeoutError, and
            # other socket-level errors when the remote server is unreachable.
            logger.warning("Ollama generation request failed: %s", error)
            raise GenerationUnavailableError(
                "The answer-generation service is temporarily unreachable. "
                "Please check the connection and try again.",
                original=error,
            ) from error

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as error:
            logger.warning("Ollama returned malformed JSON: %s", raw_body[:200])
            raise GenerationUnavailableError(
                "The answer-generation service returned an unreadable response.",
                original=error,
            ) from error

        if not isinstance(body, dict) or not isinstance(body.get("response"), str):
            logger.warning("Ollama response missing 'response' field: %s", body)
            raise GenerationUnavailableError(
                "The answer-generation service returned an unexpected response format.",
                original=None,
            )

        return body["response"].strip()

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
