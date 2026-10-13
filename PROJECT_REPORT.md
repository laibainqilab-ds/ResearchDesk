# ResearchDesk — Project Report

This report summarizes verified, existing project work only: experiment
results, QA evidence, test results, and architecture implementation status.
No metric, test outcome, or requirement below is invented — each section
states exactly where its numbers come from, and sections 2 and 6 explicitly
flag what could not be verified or is out of scope for the available
evidence.

## 1. Architecture implementation

Full comparison against the approved architecture PDF
(`ResearchDesk_Architecture_v2 (3).pdf`) is in `ARCHITECTURE.md`, section by
section. Summary:

- **Implemented and verified by reading the code:** the FastAPI backend /
  Streamlit frontend split; the Chat / Document / RAG / Agent service
  layer; the repository layer as the only path to PostgreSQL; all 12 API
  endpoints the PDF specifies; the full data model (`User`, `Chat`,
  `Message`, `Document`, `Run`, `ChatContextLink`, plus `MessageSource` for
  the PDF's "Citation"); the cross-chat-context ownership check and
  relevance-matching reuse; the RAG pipeline and LangGraph agent workflow
  (including the `retry <= 1` rule) called unchanged through thin service
  wrappers.
- **This branch's specific fixes**, each backed by new passing unit tests
  (see §5): `GET /chats/{chat_id}` now returns the chat's full message
  history in one call (`ChatDetailOut`); `ChatService.post_message` now
  persists the user message and assistant response after generation
  completes, using separate database commits — not the user message
  immediately before generation runs; `GET /health` now
  checks PostgreSQL and ChromaDB with bounded timeouts and returns 503 on
  failure, instead of an unconditional static `{"status": "ok"}`.
- **Known, explicit deviation:** the PDF's suggested `backend/app/` +
  `frontend/streamlit_app/` directory split (§11 of the PDF) was not done —
  the repository keeps a single flat `app/` package. The PDF frames this as
  a suggestion about directory structure, not a behavioral requirement, and
  every functional element of the design is implemented regardless.
  Restructuring the repository was explicitly out of scope for this change.
- **Not verified — explicitly not claimed:** HTTPS deployment, since this
  repo and this audit have no deployment environment to check (`API_BASE_URL`
  defaults to plain `http://` in local/dev config); live end-to-end
  behavior of the three fixes above against a real running PostgreSQL +
  ChromaDB stack (only unit-test evidence exists for them — see §5 and the
  distinction in `docs/QA_E2E_REPORT.md` vs. its §8 addendum, which *is*
  live-verified but only for cross-chat context linking).

## 2. Chunk-size experiment (250 / 500 / 750)

**Script:** `scripts/chunk_size_experiment.py`. **Result availability
verified:** raw per-question results, per-config summaries, and a
`comparison_summary.json` exist and were re-read directly for this report.
Per the script's own design, these outputs are written outside the
repository (a scratch directory, never under `data/chroma` or `evaluation/`)
and are not tracked by git — they are evaluation run artifacts, not
checked-in evaluation data. This report quotes them directly rather than
duplicating them into a tracked file, since doing so was explicitly out of
scope (the script forbids writing into `evaluation/`).

All three configurations ran against the same 60-question dataset (50
answerable + 10 unanswerable, same as `docs/EVALUATION.md`'s dataset), same
two real source PDFs, same BGE embedder/reranker, same `top_k=5` retrieval
call shape — only `chunk_size`/`overlap` varied. No errors occurred in any
of the three runs (the script aborts hard on any precondition failure, and
none were raised).

| Config | chunk_size | overlap | chunk_count | Recall@3 | Precision@3 | Hit Rate@3 | Recall@5 | Precision@5 | Hit Rate@5 | MRR | Avg latency (answerable, s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `smaller_250_50` | 250 | 50 | 35 | 0.800 | 0.447 | 0.900 | 0.887 | 0.344 | 0.960 | 0.846 | 5.58 |
| `baseline_500_50` | 500 | 50 | 21 | 0.857 | 0.400 | 0.920 | 0.893 | 0.264 | 0.940 | 0.842 | 10.27 |
| `larger_750_50` | 750 | 50 | 15 | 0.857 | 0.400 | 0.940 | 0.900 | 0.256 | 0.960 | 0.862 | 13.11 |

(Figures rounded to 3 decimals for readability; source values are the exact
floats in each config's `*_summary.json` / `comparison_summary.json`, e.g.
`baseline_500_50`'s `mean_recall_at_3 = 0.8566666666666667`.)

**What this data supports:** Recall@3/Hit Rate@3/MRR are all highest at
750, lowest at 250, with 500 in between for Recall@3/Hit Rate@3 but not MRR.
Precision@3 is essentially flat between 500 and 750 (0.400 both) and highest
at 250 (0.447) — fewer, more targeted chunks per retrieved item at a
smaller chunk size. Latency scales with chunk size in this run (5.58s ->
10.27s -> 13.11s average, answerable questions only), consistent with
larger chunks meaning more total chunks the embedder processes per rebuild
and larger reranker inputs. All three configs retrieved exactly 5
candidates on average for unanswerable questions (`unanswerable_mean_
candidates_retrieved: 5.0` each) — i.e. retrieval always returns candidates
regardless of chunk size; abstention correctness itself was not measured
here (no generation step was run, same retrieval-only scope as the Phase 5
evaluation below).

**What this data does not support:** a single-run-per-config result (no
repeat trials), so small differences (e.g. Precision@3 at 500 vs. 750 being
identically 0.400) carry no variance estimate. This report does not assert
a "best" configuration from this data alone — that judgment is left to
whoever reviews these numbers against the trade-offs that matter for the
deployment (retrieval quality vs. chunk count vs. latency).

## 3. System comparison / Phase 5 retrieval evaluation

Full write-up: `docs/EVALUATION.md` (12 sections, already complete and
verified in a prior audit pass — read in full, internally consistent,
sourced from `evaluation/rag_report_*.json` files that exist on disk).
Not duplicated here in full; cross-referencing instead per the instruction
to avoid restating unsupported or redundant metrics.

Summary of its final finding (quoted, not re-derived): across five isolated
retrieval configurations (basic vector retrieval; +query rewriting;
+multi-query; +multi-query+reranking; the full `final_pipeline`) run on the
same 60-question dataset, **`multi_query_plus_reranking` is identified as
the best practical retrieval configuration** — it captures nearly all of
`final_pipeline`'s retrieval-quality gain (MRR 0.863 vs. 0.875) at roughly
60% of its latency, with zero incremental Gemini cost from query rewriting.
Reranking produced the single largest quality jump of any component
(Recall@3 0.597 -> 0.857 when added to multi-query); query rewriting alone
showed no measurable aggregate effect on this dataset, attributed in
`docs/EVALUATION.md` to there being only 3 follow-up-category questions in
the 60-question set.

**Not usable as evidence, and not cited above:**
`evaluation/rag_results_multi_agent_quota_exhausted.json` — a separate,
later, Phase 9 multi-agent evaluation run that only completed 13 of 60
questions before hitting Gemini quota exhaustion. It is not a complete
result set, is not part of the Phase 5 comparison, and is excluded from
this report rather than partially summarized.

## 4. QA evidence

Full report: `docs/QA_E2E_REPORT.md`. Verified by reading in full in a
prior audit pass. Summary:

- **§1–7 (original audit):** manual end-to-end pass against one document
  and two accounts over a deliberately narrow set of interactions, with
  explicitly logged untested areas (concurrent uploads, large/multi-page
  documents, the LangGraph "research" multi-subquestion route) and a
  documented design limitation (no account-deletion endpoint).
- **§8 (addendum, 2026-10-12):** cross-chat context linking, verified live
  against the real stack (real PostgreSQL, real Gemini) in a separate
  session — a planted fact in one chat was correctly surfaced into a
  pronoun-resolved rewritten question and a correctly cited answer in a
  linked chat, confirming linked content reaches query rewriting and
  generation, not just the context-gathering step in isolation.

**Not covered by live QA, and not claimed as such:** this branch's three
architecture-compliance fixes (health-check dependency checks, message
persistence ordering, `GET /chats/{chat_id}` returning full history) have
unit-test evidence only (§5) — no live-stack addendum like §8 exists for
them. The LangGraph "research" route remains untested against the live
stack, per `docs/QA_E2E_REPORT.md` §6/§8.

## 5. Test suite

Full suite run on branch `fix/architecture-compliance-gaps`, 2026-10-13:

```
pytest -q
449 passed, 5 warnings in 31.37s
```

This includes 23 new tests added in this change:
`tests/test_text_cleaner.py` (12 tests covering whitespace collapsing
across space/tab/newline/CR/`\v`/`\f`, leading/trailing strip, empty and
whitespace-only input, and the verified fact that non-whitespace control
bytes like `\x00` are *not* removed by `clean_text`'s regex) and
`tests/test_chunker.py` (11 tests covering single- and multi-chunk word
boundaries, overlap-stepped windows, zero overlap, an exact chunk-size-sized
page, empty page text, an empty pages list, `overlap >= chunk_size` raising
`ValueError`, `chunk_id` continuity across multiple pages, and metadata
(`document_id`/`filename`/`file_type`/`page_number`/`source`) propagation).
Both functions had no prior pytest coverage — only manual, non-asserting
scripts (`scripts/check_text_cleaner.py`, `scripts/check_chunker.py`)
existed before this change.

The 5 warnings are pre-existing library deprecation/security warnings
(`asyncio.iscoroutinefunction`, a Pydantic `_UnionGenericAlias` deprecation,
a FastAPI `HTTP_422_UNPROCESSABLE_ENTITY` rename notice, and a short-JWT-key
`InsecureKeyLengthWarning` from a test fixture's 12-byte test secret) — none
introduced by this change, and none are test failures.

## 6. Completed, limited, and unverified — summary

**Completed and verified in this pass:**
- Unit tests for `clean_text` and `chunk_pages` (§5).
- `ARCHITECTURE.md` written against the approved PDF, with explicit
  intended-vs-actual distinctions throughout.
- `PROJECT_REPORT.md` (this file), sourced only from data confirmed
  accessible on disk.
- README.md test-count correction (397 -> 449, the only two occurrences in
  the repo, both referring to this same verified run — no other historical
  test-count references exist elsewhere in the repo).
- Full test suite passing (449/449).

**Limitations (inherent to the underlying evidence, not gaps introduced
here):**
- Chunk-size experiment: single run per configuration, no variance
  estimate; results live outside the repo by the script's own design.
- Phase 5 evaluation: see `docs/EVALUATION.md` §11 (small corpus, only 3
  follow-up questions, single run per experiment, retrieval-only scope).
- QA: narrow manual coverage (one document, two accounts); several areas
  explicitly logged as untested rather than silently assumed fine.

**Unverified / out of scope for this pass — do not treat as confirmed:**
- Live end-to-end behavior of this branch's three architecture-compliance
  fixes against a real PostgreSQL + ChromaDB deployment (unit-test evidence
  only).
- HTTPS deployment (no deployment environment exists to check).
- The LangGraph "research" multi-subquestion route against the live stack.
- Whether `ARCHITECTURE.md` / `PROJECT_REPORT.md` need to satisfy any
  additional required sections beyond what the approved PDF and the
  existing evaluation/QA documents support — no rubric or deliverables
  template beyond the architecture PDF was available for this pass.
