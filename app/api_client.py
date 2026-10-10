"""Thin HTTP client the Streamlit frontend uses to talk to the FastAPI
backend. Streamlit itself makes no direct RAG/LangGraph/database calls
anymore -- every stateful operation goes through these functions.
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=120)


def _handle(response: httpx.Response):
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ApiError(response.status_code, detail)
    if response.status_code == 204:
        return None
    return response.json()


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def health() -> dict:
    with _client() as client:
        return _handle(client.get("/health"))


def signup(email: str, password: str) -> str:
    with _client() as client:
        data = _handle(client.post("/auth/signup", json={"email": email, "password": password}))
    return data["access_token"]


def login(email: str, password: str) -> str:
    with _client() as client:
        data = _handle(client.post("/auth/login", json={"email": email, "password": password}))
    return data["access_token"]


def list_chats(token: str) -> list[dict]:
    with _client() as client:
        return _handle(client.get("/chats", headers=_auth_headers(token)))


def create_chat(token: str, title: str = "New chat") -> dict:
    with _client() as client:
        return _handle(client.post("/chats", json={"title": title}, headers=_auth_headers(token)))


def delete_chat(token: str, chat_id: str) -> None:
    with _client() as client:
        _handle(client.delete(f"/chats/{chat_id}", headers=_auth_headers(token)))


def list_messages(token: str, chat_id: str) -> list[dict]:
    with _client() as client:
        return _handle(client.get(f"/chats/{chat_id}/messages", headers=_auth_headers(token)))


def post_message(token: str, chat_id: str, content: str, mode: str) -> dict:
    with _client() as client:
        return _handle(
            client.post(
                f"/chats/{chat_id}/messages",
                json={"content": content, "mode": mode},
                headers=_auth_headers(token),
            )
        )


def list_runs(token: str, chat_id: str) -> list[dict]:
    with _client() as client:
        return _handle(client.get(f"/chats/{chat_id}/runs", headers=_auth_headers(token)))


def create_context_link(token: str, chat_id: str, target_chat_id: str) -> dict:
    with _client() as client:
        return _handle(
            client.post(
                f"/chats/{chat_id}/context-links",
                json={"target_chat_id": target_chat_id},
                headers=_auth_headers(token),
            )
        )


def list_documents(token: str) -> list[dict]:
    with _client() as client:
        return _handle(client.get("/documents", headers=_auth_headers(token)))


def upload_document(token: str, filename: str, file_bytes: bytes) -> dict:
    with _client() as client:
        return _handle(
            client.post(
                "/documents",
                files={"file": (filename, file_bytes)},
                headers=_auth_headers(token),
            )
        )


def delete_document(token: str, document_id: str) -> None:
    with _client() as client:
        _handle(client.delete(f"/documents/{document_id}", headers=_auth_headers(token)))
