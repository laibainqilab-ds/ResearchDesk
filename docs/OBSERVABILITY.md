# ResearchDesk — Observability (Phase 8)

This document describes ResearchDesk's request tracing and structured
logging: where trace IDs are created, how they propagate, where logs are
stored, what's logged, how failures are represented, and how to use a
trace ID to diagnose a request. It's deliberately simple: Python's standard
`logging` module, one local JSON-lines file, no distributed tracing, no
external services, no new database.

## 1. Trace IDs

A trace ID is a plain UUID4 hex string (`app/observability.py::new_trace_id()`),
32 lowercase hex characters, no dashes. Nothing more elaborate than that —
this project deliberately does not use distributed-tracing infrastructure
(no OpenTelemetry, no span trees, no trace-context headers).

**Where they're created**: at the top of `RAG.answer()` and `RAG.retrieve()`
in `app/rag.py` — `trace_id = trace_id or new_trace_id()`. If a caller
already has a trace ID (e.g. `answer()` calling `retrieve()` for the same
request), it's passed through instead of generating a new one. If
`retrieve()` is called standalone (e.g. the Retrieval Inspector's manual
test), it generates its own. `ingest_file()` in `app/ingestion/pipeline.py`
follows the same pattern for document ingestion, which is a separate kind
of operation from a RAG query.

**How they propagate**: as an explicit `trace_id` keyword argument, threaded
through every stage of one request:

```
RAG.answer(trace_id)
 ├─ RAG._prepare_conversation_context(trace_id)
 │   └─ RAG._summarize_older_history(trace_id) -> Generator.summarize_conversation(trace_id)
 ├─ Generator.rewrite_query(trace_id)          -> Generator.generate(trace_id)
 ├─ Generator.generate_queries(trace_id)       -> Generator.generate(trace_id)
 ├─ RAG.retrieve(trace_id)
 │   ├─ vector search / embedding (per search query)
 │   └─ Reranker.rerank (via RAG.retrieve)
 ├─ Generator.generate(trace_id)               -- final answer generation
 └─ extract_citations(trace_id)                -- citation validation
```

Every one of those stages logs at least one structured event tagged with
the same `trace_id`, so every log line belonging to one request can be
found with a single `grep <trace_id> logs/researchdesk.jsonl`.

`RAG.answer()`'s return dict includes `"trace_id"` at the top level, so
callers (Streamlit, `evaluation/run_evaluation.py`, tests) always have it
without needing to generate their own.

**Streamlit integration**: the Chat page stores each assistant message's
`trace_id` alongside it in `st.session_state.messages`, and displays it as
a small caption under the message (and prominently in the error message
text when generation fails) — so if a user reports "this answer looked
wrong" or "I got an error", the trace ID needed to find the exact log
entries is already visible on screen. The Retrieval Inspector's manual test
does the same for its own trace ID.

## 2. Structured logging

Logging is Python's standard `logging` module (`app/observability.py`) —
no new abstraction was introduced. `log_event(trace_id, event, level=INFO,
**fields)` builds a dict (`timestamp`, `trace_id`, `event`, `level`, plus
whatever `fields` the caller passes) and logs it as one JSON string via a
dedicated `"researchdesk"` logger. `configure_logging()` attaches a local
`FileHandler` to that logger; it's idempotent, so it's safe to call from
`RAG.__init__()` on every process start without stacking duplicate handlers
or duplicating log lines.

**Where logs are stored**: `logs/researchdesk.jsonl`, one JSON object per
line, relative to wherever the app is run from (same convention as
`data/chroma`, `data/documents`). The `logs/` directory is gitignored — logs
are a local runtime artifact, not part of the repo.

**What's logged** (event names, all in `app/rag.py`, `app/models/generator.py`,
and `app/ingestion/pipeline.py`):

| Stage | Events |
|---|---|
| Request lifecycle | `request_started`, `request_completed` (with `outcome`: `answered` / `no_evidence` / `retrieval_only` / `generation_failed`, and `total_latency_seconds`) |
| Query rewriting | `query_rewrite_requested`, `query_rewrite_failed` |
| Multi-query generation | `multi_query_requested`, `multi_query_generated`, `multi_query_failed` |
| Conversation summarization | `conversation_summarization_requested`, `conversation_summarization_failed` |
| Retrieval | `retrieval_started`, `vector_search_failed`, `malformed_chunk_skipped`, `retrieval_empty`, `retrieval_candidates_selected` (chunk IDs + distances), `final_evidence_selected` (chunk IDs) |
| Reranking | `reranking_completed` (scores), `reranking_failed` |
| Gemini calls (underlying every stage above that talks to the model) | `gemini_call_started`, `gemini_call_succeeded` (with `latency_seconds`), `gemini_call_failed` (with error code/message) |
| Answer generation | `answer_generation_failed` |
| Citation validation | `citation_validation_completed` (valid/invalid citation numbers; WARNING level if any are invalid) |
| Ingestion | `ingestion_started`, `ingestion_duplicate_rejected`, `ingestion_unsupported_file_type`, `ingestion_parsing_failed`, `ingestion_empty_document`, `ingestion_embedding_or_storage_failed`, `ingestion_completed` |

**What's never logged**: API keys, credentials, or `.env` contents. Gemini
calls log the model name, prompt/response *lengths*, and latency — never
the full prompt or response text (which would include full retrieved
document content, unnecessarily verbose and unnecessary to log in full for
diagnosis). Chunk identifiers, distances, and scores are logged; the
underlying chunk *text* is not.

## 3. How failures are represented

Every failure case audited for Phase 8 either already had, or now has, both
(a) a specific exception type or graceful in-band fallback, and (b) a
structured log event with enough detail to diagnose it without needing to
reproduce it:

- **Ingestion** (`app/ingestion/pipeline.py`): unsupported file type, empty
  document, duplicate document, and parsing/embedding/storage failures all
  raise their existing specific exception (`UnsupportedFileTypeError`,
  `EmptyDocumentError`, `DuplicateDocumentError`, `DocumentParsingError` —
  unchanged) and now also log a matching event before doing so. Embedding/
  storage failure still rolls back any partial data, unchanged.
- **Retrieval** (`app/rag.py::RAG.retrieve()`): a vector-store/embedding
  failure for one search query is logged and that query is skipped rather
  than crashing the whole request (new — previously this would raise
  uncaught). If every candidate search fails, or the corpus genuinely has no
  match, retrieval degrades to the existing "no evidence" behavior, now
  logged as `retrieval_empty`.
- **Reranking**: a cross-encoder failure is logged and falls back to the
  same distance-sort behavior already used when reranking is disabled (new
  — previously this would raise uncaught).
- **Gemini/API failures** (`app/models/generator.py`): unchanged behavior —
  every Gemini call still raises `GenerationUnavailableError` on failure,
  caught by `RAG.answer()`'s existing per-stage fallbacks (skip rewriting,
  skip multi-query, or surface a clean generation-failed message to the
  user) — now each of those fallback paths also logs a structured event.
- **Invalid/malformed citations**: not a failure that raises — `RAG.answer()`
  never trusted the model's citation markers ([1], [2], ...) at face value;
  `extract_citations()` already separated valid from invalid ones. Now that
  check also logs a `citation_validation_completed` event (WARNING level
  when any citation is invalid), so a malformed citation is visible in logs
  even though it doesn't fail the request.

**User-facing vs. logged detail**: what the user sees stays the same as
before Phase 8 — a clean, honest message (e.g. "the answer-generation
service is temporarily unreachable", or "I couldn't find enough information
in the provided documents") — now with the trace ID appended when relevant.
The detailed diagnostic (exception message, which stage, latency, chunk
IDs) goes to the log file, not the UI.

## 4. Diagnosing a request from a trace ID

1. Get the trace ID — from the Streamlit Chat page (shown under each
   assistant message, and in the error text if generation failed), the
   Retrieval Inspector, or `RAG.answer()`'s return value / a raw evaluation
   result's `trace_id` field.
2. `grep "<trace_id>" logs/researchdesk.jsonl` (or, in PowerShell,
   `Select-String -Path logs/researchdesk.jsonl -Pattern "<trace_id>"`) —
   returns every event for that one request, in order.
3. Read the sequence of `event` values to see exactly which stages ran and
   in what order (e.g. `request_started` → `retrieval_started` →
   `retrieval_candidates_selected` → `reranking_completed` →
   `final_evidence_selected` → `gemini_call_started` → `gemini_call_failed`
   → `answer_generation_failed` → `request_completed`) — the last few events
   before the trail stops (or the `outcome` field on `request_completed`)
   show exactly what happened and where it diverged from the happy path.
4. Each event's extra fields (chunk IDs, distances, scores, error messages,
   latency) give the diagnostic detail without needing to reproduce the
   request.

## 5. Deliberately not done

Per the Phase 8 scope, none of the following were introduced: FastAPI,
LangSmith, cloud/external logging or monitoring, distributed tracing, a
database for logs, or any change to retrieval strategy or the evaluation
framework. Trace IDs are request-scoped UUIDs, not spans; there is no
cross-service correlation because ResearchDesk is a single local process.
