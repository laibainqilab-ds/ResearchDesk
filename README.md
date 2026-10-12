# ResearchDesk

Document research assistant: upload PDFs/TXT/Markdown, then ask grounded,
cited questions over them via either a standard RAG pipeline or a
multi-agent (LangGraph) workflow.

Architecture: Streamlit (frontend) → FastAPI (backend) → services
(Chat / Document / RAG / Agent) → PostgreSQL / ChromaDB / Gemini / LangGraph.
Streamlit and FastAPI run as two separate processes and talk over HTTP —
Streamlit never accesses PostgreSQL, ChromaDB, Gemini, or LangGraph directly.

## Prerequisites

- Python 3.11+ (this repo has been exercised under 3.11 and 3.14).
- A reachable PostgreSQL server. Nothing in this repo provisions or starts
  one — create the database/user yourself before first run.
- A [Gemini API key](https://ai.google.dev/) for answer generation.
- Internet access on first run only, to download the embedding
  (`BAAI/bge-small-en-v1.5`) and reranking (`BAAI/bge-reranker-base`) models
  from Hugging Face. They're cached locally after that. Setting `HF_TOKEN`
  is optional and only raises the Hugging Face Hub rate limit.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the repo root (start from `.env.example`) with:

   | Variable | Required | Notes |
   |---|---|---|
   | `DATABASE_URL` | Yes | e.g. `postgresql://user:password@127.0.0.1:5432/researchdesk`. The backend (and anything importing `app.main`, including the API test client) fails fast at import time if this is unset (`app/db/session.py`). |
   | `GEMINI_API_KEY` | Yes | Used by `app/models/generator.py` for RAG/agent answer generation. |
   | `GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash` if unset. |
   | `JWT_SECRET` | No | If unset, one is generated on first run and appended to `.env` automatically (`app/config.py`), so sessions survive restarts. Set it explicitly only if you need the same value across multiple environments/processes. |
   | `API_BASE_URL` | No | Used only by the Streamlit frontend (`app/api_client.py`) to reach the backend. Defaults to `http://127.0.0.1:8000`. |

3. Create the PostgreSQL database and user referenced by `DATABASE_URL`
   (no setup script exists in this repo for this step).

## Database

There is no migration tool (e.g. Alembic). Tables are created automatically
from the SQLAlchemy models (`app/db/models.py`) the first time the FastAPI
backend starts, via `Base.metadata.create_all()` in the app lifespan
(`app/main.py`). This only ever adds missing tables — it never drops or
alters existing ones, so it's safe to restart against an existing database.

ChromaDB is embedded (no separate service to run) and persists to
`data/chroma/`, created automatically on first use.

## Running

Run both processes from the repo root — paths like `data/chroma`,
`data/documents`, and `evaluation/` are relative to it.

**Backend (FastAPI):**
```
uvicorn app.main:app --reload --port 8000
```

**Frontend (Streamlit), in a second terminal:**
```
streamlit run app/streamlit_app.py
```

Check the backend is up with `GET http://127.0.0.1:8000/health` →
`{"status": "ok"}`. This is a liveness check only — it does not verify
PostgreSQL, ChromaDB, or Gemini connectivity (`app/api/routers/health.py`).

## Tests

```
pytest
```

397 tests. The suite is self-contained — `tests/conftest.py` provisions an
isolated in-memory SQLite database per test and uses fake embedding/generation
components, so no live PostgreSQL, ChromaDB, or Gemini access is required to
run it. A `.env` with a valid `DATABASE_URL` must still exist, since
importing `app.main` (used by the API test client) imports `app.db.session`,
which raises immediately if `DATABASE_URL` is unset.

---
Last verified against commit `9eb475b` (2026-10-11).
