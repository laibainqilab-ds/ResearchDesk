import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


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
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

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

