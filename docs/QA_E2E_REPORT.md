# ResearchDesk — End-to-End QA Report: Per-User Retrieval Isolation

This document records the results of a real, end-to-end QA verification run
against the live ResearchDesk application (FastAPI, PostgreSQL, ChromaDB
1.5.9, Gemini) following the merge of PR #21 (FastAPI backend) and PR #23
(per-user document retrieval isolation). It was performed manually via the
real running stack, not the automated pytest suite — the automated suite
(397 passed at the time of this run) uses an isolated SQLite database and
fake embedding/generation components, so this run exists specifically to
verify the real integrations and real cross-user isolation behavior.

## 1. Scope

- Real FastAPI backend at `http://127.0.0.1:8000`, real PostgreSQL database,
  real ChromaDB collection (`researchdesk`), real BGE embedder/reranker,
  real Gemini generation.
- Two QA accounts, reused from two pre-existing (test-origin) user records
  rather than newly created, per explicit approval.
- One disposable, synthetic `.txt` test document, uploaded, queried, and
  deleted during the run.
- Covered: authentication, document upload and ownership, RAG retrieval and
  citations, RAG and Agent chat, cross-user isolation across every access
  path the current API exposes, and deletion.
- Not covered: concurrent/simultaneous uploads, large or multi-page
  documents, cross-chat context linking, and the LangGraph "research"
  (multi-subquestion) route.

## 2. Results

**40 of 40 executed checks passed. 0 failed. 2 reported N/A** (features with
no corresponding endpoint in the current API — not counted as pass or fail).

| Area | Result |
|---|---|
| Authentication | **PASS** — login succeeded for both accounts with distinct tokens; unauthenticated access to a protected endpoint was rejected (`401`); sessions were not shared between accounts. |
| Logout / token revocation | **N/A** — no such endpoint exists in the application (JWTs are stateless with a fixed expiry; there is no revocation mechanism). |
| Document upload and ownership | **PASS** — upload succeeded; the resulting PostgreSQL `documents` row and ChromaDB chunk were both correctly tagged to the uploading account; re-uploading identical content while still indexed was correctly rejected (`409`). |
| RAG retrieval and citations | **PASS** — the query returned an answer containing the test document's planted synthetic fact, with citations referencing only the test document, never either of the two pre-existing legacy documents. |
| Agent chat | **PASS** — produced a persisted run with expected route/trace data, retrievable only by its owner. |
| Deletion | **PASS** — deletion removed the ChromaDB chunk entirely (verified by a filtered, read-only lookup) and soft-deleted the PostgreSQL row (`status=deleted`), matching the application's existing, intended delete semantics. |

## 3. Cross-user isolation findings

Every access path exposed by the current API was tested with two accounts
and passed in both directions where applicable:

| Access path | Result |
|---|---|
| Document listing | Account B's list excludes Account A's document; Account A's list includes it |
| Retrieval / search | Account B's query for the same question returns no trace of Account A's planted fact |
| Citation / source lookup | Account B's citations never reference Account A's document |
| Chat history | Account B → `404` on Account A's chat and its messages |
| Run details | Account B → `404` on Account A's run |
| Reverse direction | Account A → `404` on Account B's chat |
| Document deletion | Account B → `404` attempting to delete Account A's document |
| Agent-tool exact lookup (`get_document_metadata`, `get_document_page`) | Verified **in-process only** (no HTTP route exposes these) — found for the owner, `None`/empty for the other account |
| Direct single-document HTTP lookup | **N/A** — no such endpoint exists in the API |

No owner-filtering or authorization logic was altered or weakened to make
any of the above pass; all checks ran against the unmodified, already-merged
code.

## 4. Before / after data comparison

| Table / store | Before | After |
|---|---|---|
| `users` | 6 | 6 *(2 are the reused QA accounts; unchanged by this run)* |
| `chats` | 4 | 4 |
| `messages` | 8 | 8 |
| `message_sources` | 12 | 12 |
| `runs` | 4 | 4 |
| `chat_context_links` | 0 | 0 |
| `documents` | 0 | **1** *(the one approved soft-deleted test record — see §5)* |
| ChromaDB total chunks | 21 | 21 |
| Legacy chunk count — `doc_001` | 16 | 16 *(untouched)* |
| Legacy chunk count — FYDP document | 5 | 5 *(untouched)* |

All pre-existing user, chat, and run IDs were independently re-verified
present and unmodified after the run completed. No email addresses,
passwords, password hashes, JWTs, or API keys were recorded in this report
or printed during the run.

## 5. Persistent QA artifacts

Two things remain from this QA run, both pre-approved in advance:

- **Two reused QA user accounts.** They were not newly created for this
  test; two pre-existing (test-origin) accounts were reused by explicit
  approval. They remain in the `users` table — the application has no
  account-deletion endpoint of any kind, so there is no supported way to
  remove them.
- **One soft-deleted `documents` row.** Its ChromaDB content was fully
  removed by the deletion step; the PostgreSQL row itself persists with
  `status='deleted'`, which is the application's existing, intended
  soft-delete behavior, not a leftover from incomplete cleanup.

Everything else created during the run (test chats, messages, citations,
and run records) was fully removed via the application's own supported
`DELETE` endpoints and confirmed back at baseline (§4).

## 6. Confirmed bugs, design limitations, and untested areas

**Confirmed bugs:** none found during this run.

**Design limitations** (existing, intentional or pre-existing gaps — not
defects introduced by this run or by PR #23):
- No account-deletion endpoint exists anywhere in the application.
- No logout or token-revocation mechanism exists (JWTs are stateless).
- No direct single-document HTTP lookup endpoint exists.
- Three `RetrievalTools` methods (`list_available_documents`,
  `get_document_metadata`, `get_document_page`) exist and are correctly
  owner-filtered, but are not wired to any HTTP route or reachable from the
  live LangGraph agent graph — they were verified in-process only.

**Untested in this run** (out of scope, not attempted):
- Concurrent or simultaneous document uploads.
- Large or multi-page documents.
- Cross-chat context linking.
- The LangGraph "research" (multi-subquestion) route.

## 7. Readiness assessment

This run confirms that the per-user document retrieval isolation introduced
in PR #23 **works correctly end-to-end against the real application stack**
(real PostgreSQL, real ChromaDB, real Gemini) — no cross-user leakage of
document content, filenames, citations, chat history, or run data was
observed across any access path tested, in either direction, between two
distinct accounts.

**This is not a full production-readiness sign-off.** It covers one upload,
one document, two accounts, and a deliberately narrow set of interactions.
Before treating the application as broadly production-ready, the design
limitations in §6 (particularly the absence of account deletion) and the
untested areas should be addressed or explicitly accepted as known gaps.
