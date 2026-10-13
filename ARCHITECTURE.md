# ResearchDesk — Architecture

This document has two purposes, kept explicitly separate:

1. **Intended architecture** — a summary of `ResearchDesk_Architecture_v2 (3).pdf`
   ("ResearchDesk — Proposed Future Architecture", header: *"Draft for review
   before implementation"*), the approved design source of truth for the
   FastAPI/PostgreSQL/service-layer split.
2. **Actual implementation** — what the current code in this repository (on
   branch `fix/architecture-compliance-gaps`) actually does, verified by
   reading the source directly, not inferred from the PDF.

Every row below is independently verified against code. Where the PDF
describes something and the code doesn't (yet) do it, that is stated
explicitly rather than assumed.

## 1. Why this change (PDF §1–2)

The PDF's stated motivation: the pre-existing system was a single Streamlit
process running the UI, RAG pipeline, and LangGraph agent workflow together,
with conversation state living only in Streamlit's session state (lost on
session end). The proposed fix is not to rewrite RAG or the agent — both
"already work and have been evaluated" — but to insert a FastAPI backend
and a PostgreSQL database between Streamlit and the existing logic.

**Verified in code:** this split exists today. `app/main.py` is a FastAPI
app; `app/streamlit_app.py` talks to it over HTTP via `app/api_client.py`
(an `httpx` client), not via direct in-process calls. `app/db/session.py`
configures a SQLAlchemy engine against `DATABASE_URL` (PostgreSQL in
production, in-memory SQLite in tests per `tests/conftest.py`).

## 2. Target vs. actual data flow (PDF §2)

PDF target:

```
Streamlit (frontend) -> FastAPI -> services (Chat / Document / RAG / Agent)
                      -> PostgreSQL / ChromaDB / Gemini / LangGraph
```

**Verified in code:** this is the actual call shape. `app/streamlit_app.py`
calls functions in `app/api_client.py`, which make HTTP requests to routers
in `app/api/routers/`. Routers depend on services in `app/services/`
(`ChatService`, `DocumentService`, `RagService`, `AgentService`, via
`app/api/deps.py`). Services call either repositories (`app/repositories/`,
Postgres-only) or the unchanged RAG/ingestion/agent modules directly.

## 3. Components and responsibilities (PDF §3)

| Component | PDF responsibility | Verified in code |
|---|---|---|
| FastAPI | "Just the door" — authenticates, routes to a service, no business logic | `app/api/routers/*.py` are thin: they call `Depends(get_current_user)` / `Depends(get_*_service)` and delegate; no business logic found in the routers themselves |
| Chat Service | Owns chat lifecycle, message load/save, RAG-vs-agent routing | `app/services/chat_service.py::ChatService` — `create_chat`, `get_chat`, `post_message` (routes on `mode`), `_gather_cross_chat_context` |
| Document Service | Owns uploads — dedup, parse, chunk, embed | `app/services/document_service.py::DocumentService.ingest` — hashes via `compute_document_id`, calls `app.ingestion.pipeline.ingest_file` |
| RAG Service / Agent Service | "Thin wrappers" around unchanged `rag.py` / LangGraph logic; **must not** access PostgreSQL directly | `app/services/rag_service.py` and `app/services/agent_service.py` — grepped for any `db`/`session`/`repository` import: **none found**. Confirms the PDF's explicit separation rule holds in code today. |
| Repositories | PostgreSQL access only | `app/repositories/` has one file per entity (`chat_repository.py`, `document_repository.py`, `message_repository.py`, `run_repository.py`, `user_repository.py`, `context_link_repository.py`) |
| ChromaDB / BGE / Gemini | Unchanged | `app/ingestion/vector_store.py`, `app/ingestion/embedder.py`, `app/ingestion/reranker.py`, `app/models/generator.py` — still called directly by `RagService`/`AgentService`/`DocumentService`, not through a repository |

## 4. API (PDF §4)

The PDF specifies 12 endpoints. All 12 exist in the current code, plus a
few additions the PDF doesn't mention (listed separately — these are not
compliance gaps, just extra surface area):

| PDF endpoint | Verified in code |
|---|---|
| `POST /auth/login` | `app/api/routers/auth.py` |
| `POST /chats` | `app/api/routers/chats.py` |
| `GET /chats` | `app/api/routers/chats.py` |
| `GET /chats/{chat_id}` | `app/api/routers/chats.py` — returns `ChatDetailOut` (chat + full `messages` list), matching PDF §6 "loads a chat's full history" |
| `POST /chats/{chat_id}/messages` | `app/api/routers/chats.py` — single endpoint, `mode: "rag" \| "agent"` in the body, matching PDF §4's explicit design choice of one endpoint over two |
| `DELETE /chats/{chat_id}` | `app/api/routers/chats.py` |
| `POST /chats/{chat_id}/context-links` | `app/api/routers/chats.py` |
| `POST /documents` | `app/api/routers/documents.py` |
| `GET /documents` | `app/api/routers/documents.py` |
| `DELETE /documents/{document_id}` | `app/api/routers/documents.py` |
| `GET /runs/{run_id}` | `app/api/routers/runs.py` |
| `GET /health` | `app/api/routers/health.py` |

Not in the PDF table, present in code: `POST /auth/signup`, `GET
/chats/{chat_id}/messages`, `GET /chats/{chat_id}/runs`. These are
reasonable extensions of the same resource model, not contradictions of it.

## 5. State management (PDF §5)

The PDF distinguishes four kinds of state. Verified against code:

- **UI state** (current chat tab, unsent input) — lives only in
  `st.session_state` in `app/streamlit_app.py`. Confirmed nothing here is
  persisted to Postgres.
- **Persistent state** (users, chats, messages, documents, citations, runs)
  — `app/db/models.py`: `User`, `Chat`, `Message`, `Document`, `Run`,
  `ChatContextLink`, plus `MessageSource` (the PDF calls this entity
  "Citation"; the code's table name is `message_sources` — same role, a
  naming difference only).
- **Request state** (rewritten query, candidate lists before/after
  reranking) — verified not persisted: `RAG.retrieve()`/`RAG.answer()` in
  `app/rag.py` return these as in-memory dict values to the caller; nothing
  in `RagService` writes them to a repository directly — only `ChatService`
  decides what subset becomes the persisted `Run.retrieval` JSON blob.
- **Agent execution state** (live LangGraph `AgentState`) — verified not
  persisted: `app/agents/state.py::AgentState` is constructed and consumed
  within one `run_agent_workflow()` call in `app/agents/graph.py`; what
  `ChatService` persists afterward is the compact `Run` record (`route`,
  `retry_count`, `is_valid`, `agent_trace` JSON) per PDF §5/§9, not the full
  state object.

Identifiers: `chat_id`, `message_id`, `run_id`, `trace_id` — all four exist
as modeled fields in `app/db/models.py` (`trace_id` on both `Message` and
`Run`), matching the PDF's description of their relationship.

## 6. Main flows (PDF §6)

| PDF flow | Verified in code |
|---|---|
| New chat | `POST /chats` -> `ChatService.create_chat` -> `ChatRepository` insert -> `chat_id` returned |
| Asking a question | `ChatService.post_message`: loads history, calls `RagService`/`AgentService` depending on `mode`, **then** persists the user and assistant messages after generation completes, using separate database commits (see below) |
| Resuming a chat | `GET /chats/{chat_id}` -> `ChatDetailOut.messages`, ordered by `created_at` (`Chat.messages` relationship has `order_by="Message.created_at"` in `app/db/models.py`) |
| Uploading a document | `DocumentService.ingest`: hash via `compute_document_id`, existing parse/clean/chunk/embed pipeline, chunks to ChromaDB, bookkeeping row to Postgres via `DocumentRepository.upsert_indexed` |
| Agent request | Same endpoint, `mode="agent"` -> `ChatService._run_agent` -> `AgentService.run` -> `run_agent_workflow` (router -> retrieval/research -> answer -> validation -> retry <= 1 -> finalize) |
| Cross-chat context | `ChatService.create_context_link` calls `get_chat` for **both** chat ids before creating the link (ownership check); `_gather_cross_chat_context` reuses `select_relevant_history` — the same keyword-relevance function already used for same-chat history, not a separate summarizer, matching PDF §6's explicit statement |

### Message persistence ordering — this branch's fix

The PDF states (§6, "Asking a question"): history is loaded, handed to the
service, and "once it gets an answer back it saves both the user's message
and the assistant's reply" — i.e. not the user's message immediately,
before generation runs.

Prior to this branch, `ChatService.post_message` persisted the user message
*before* calling RAG/Agent. The current diff on this branch
(`app/services/chat_service.py`) moves `self.messages.add_user_message(...)`
to *after* the RAG/Agent call returns, so message persistence now happens
only once generation completes — matching the PDF. This is covered by
`tests/test_chat_service.py`:
`test_post_message_does_not_persist_user_message_before_generation_completes`,
`test_post_message_agent_does_not_persist_user_message_before_generation_completes`,
and `test_post_message_persists_both_messages_together_even_on_generation_failure`
— all passing (see test run in `PROJECT_REPORT.md`).

### `GET /chats/{chat_id}` returning full history — this branch's fix

Prior to this branch, `GET /chats/{chat_id}` returned `ChatOut` (no
messages); the Streamlit frontend called a separate `GET
/chats/{chat_id}/messages` to populate history. The current diff adds
`ChatDetailOut` (schema) and changes the router to return it, and
`app/streamlit_app.py::refresh_messages` now reads `chat["messages"]` from
the single `GET /chats/{chat_id}` call — matching the PDF's "resuming a
chat" flow description literally (one call pulls the chat plus its full
history).

### `GET /health` dependency checks — this branch's fix

The PDF lists `GET /health` as "is the backend and its dependencies up."
Prior to this branch, the endpoint returned a static `{"status": "ok"}`
with no dependency check. The current diff (`app/api/routers/health.py`)
adds a bounded-timeout PostgreSQL check (fresh `psycopg` connection,
`connect_timeout` + `statement_timeout`, not the shared pooled engine) and
a ChromaDB heartbeat check, returning 503 with a per-dependency
up/down breakdown when either is unavailable. It never calls Gemini (no LLM
call belongs in a liveness check). Covered by 5 new tests in
`tests/test_api.py` (`test_health*`), all passing.

**Not verified/claimed:** these three fixes are confirmed by passing unit
tests only (see `PROJECT_REPORT.md` for the exact test run). No live
end-to-end check against a running PostgreSQL + ChromaDB stack was
performed for these three specific changes in this audit — unlike the
cross-chat-context-linking fix, which does have a live-verified addendum in
`docs/QA_E2E_REPORT.md` §8. Do not read "tests pass" as "verified live
against a real deployment."

## 7. Data model (PDF §7)

| PDF entity | Verified in code |
|---|---|
| User | `app/db/models.py::User` |
| Chat | `app/db/models.py::Chat` |
| Message | `app/db/models.py::Message` (links to `Run` via `message.run`) |
| Document | `app/db/models.py::Document` (metadata only; chunks/embeddings in ChromaDB, as the PDF specifies) |
| Citation | `app/db/models.py::MessageSource` (same role as PDF's "Citation" — table name differs) |
| Run | `app/db/models.py::Run` (RAG mode: `retrieval` JSON; agent mode: `route`/`retry_count`/`is_valid`/`agent_trace`) |
| ChatContextLink | `app/db/models.py::ChatContextLink`, with a `UniqueConstraint` on `(source_chat_id, target_chat_id)` |

## 8–9. RAG and Agent — "unchanged" (PDF §8–9)

The PDF states the RAG pipeline (rewriting, multi-query, retrieval,
reranking, generation, citations) and the LangGraph agent workflow (router,
retrieval/research, answer, validation, retry <= 1, finalize) are not being
rewritten, only re-wired to be called from a service layer.

**Verified in code:** `app/rag.py` and `app/agents/graph.py` contain the
full pipeline/workflow logic directly; `RagService`/`AgentService` are thin
pass-throughs (see §3 table). `MAX_VALIDATION_RETRIES = 1` in
`app/agents/validation.py`, enforced in `app/agents/graph.py`, matches
"retry <= 1" exactly.

**Not independently re-verified in this pass:** whether the RAG/agent
*internal logic itself* is byte-for-byte identical to some pre-FastAPI
version is not something this audit checked line-by-line — only that it is
the only thing `RagService`/`AgentService` call, and that it is not
duplicated or forked elsewhere in the codebase.

## 10. Deployment (PDF §10)

PDF target: `Browser -> Streamlit -> HTTPS -> FastAPI -> PostgreSQL /
ChromaDB / Gemini`, as two separate processes, no queues/workers/
microservices, LangGraph and RAG Service running inside the FastAPI process.

**Verified in code:** `README.md` documents running `uvicorn
app.main:app` and `streamlit run app/streamlit_app.py` as two separate
processes. `app/api_client.py` talks to `API_BASE_URL` (default
`http://127.0.0.1:8000`) over plain HTTP in local/dev configuration.

**Not verified — explicitly not claimed:** this repository and this audit
provide **no evidence of an HTTPS deployment**. `API_BASE_URL` defaults to
plain `http://`, there is no TLS termination configured anywhere in this
repo, and no deployment environment was inspected. The PDF's "HTTPS" arrow
in its deployment diagram is the *target* for a real deployment, not
something implemented or verified here. Likewise, no queue/worker/
microservice claims are made or need to be — the code matches the PDF's
"none of that" description by simple absence (no such code exists), which
was confirmed by inspecting `app/main.py` and the process list in
`README.md`'s run instructions.

## 11. Suggested project layout (PDF §11) — known deviation

PDF suggestion:

```
backend/app/   — api/, services/, repositories/, models/, rag/ (unchanged),
                 agents/ (unchanged), ingestion/ (unchanged), main.py
frontend/streamlit_app/ — pages/, api_client.py
```

**Actual repository layout (verified):** flat, single `app/` package at
the repo root containing `api/`, `services/`, `repositories/`, `db/`,
`models/`, `rag.py`, `agents/`, `ingestion/`, `main.py`, `api_client.py`,
and `streamlit_app.py` together — there is no `backend/` or `frontend/`
top-level split.

This is a **known, explicit deviation** from the PDF's suggested layout.
The PDF itself frames §11 as a "suggested project layout," not a hard
requirement ("rag/, agents/, and ingestion/ are basically just today's code
moved into this structure — not rewritten" — i.e. the suggestion is about
directory structure, not behavior). Every functional/behavioral element of
§1–10 is implemented regardless of this directory-structure difference.
This deviation has not been resolved as part of this change; doing so would
be a repository restructuring, which was explicitly out of scope for this
pass.

## 12. Summary — what's new vs. unchanged (PDF §12)

PDF's claim: new = FastAPI, service layer, repository/database layer,
Postgres, persistent chats/documents, run history, optional cross-chat
linking. Unchanged = Streamlit frontend, ChromaDB, BGE models, Gemini, RAG
pipeline, LangGraph agent workflow.

**Verified in code:** matches. All "new" items exist as described in §1–7
above. All "unchanged" items are called directly by the service layer
without modification to their internal logic, per §3 and §8–9.
