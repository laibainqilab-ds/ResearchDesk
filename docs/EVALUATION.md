# ResearchDesk — Phase 5 Evaluation

This document reports the results of the Phase 5 retrieval evaluation: five
retrieval configurations, run against the same 60-question dataset and the
same two-document corpus, compared on deterministic retrieval and citation
metrics. All numbers below are read directly from the generated report files
in `evaluation/` (`rag_report_basic_vector_retrieval_retrieval_only.json`,
`rag_report_vector_plus_query_rewriting_retrieval_only.json`,
`rag_report_multi_query_retrieval_retrieval_only.json`,
`rag_report_multi_query_plus_reranking_retrieval_only.json`, and
`rag_report_retrieval_only.json` for `final_pipeline`) — none are estimated
or invented.

## 1. Evaluation Objective

The project's stated principle is: *do not add a component until you have
evidence it helps.* Phase 4 added three retrieval components — query
rewriting, multi-query generation, and reranking — all wired to be always on,
with no way to measure any of them in isolation. The purpose of this
evaluation is to close that gap: isolate each component, measure it against
a fixed dataset and a fixed corpus, and determine which combination of
components is actually worth its cost before treating any of them as
load-bearing.

Every experiment below was run in **retrieval-only mode**
(`enable_answer_generation=False`) — the full retrieval pipeline (query
rewriting where applicable, multi-query generation where applicable,
vector search, reranking where applicable, and source/citation construction)
runs exactly as it would in production, but the final Gemini answer-generation
call is skipped. This makes retrieval and citation-correctness metrics
measurable at a fraction of the Gemini call cost, without touching scoring
logic or retrieval behavior.

## 2. Dataset Composition

`evaluation/rag_evaluation.json` — **60 questions**, **50 answerable**, **10
unanswerable**.

| Category | Count |
|---|---:|
| direct | 19 |
| multi_chunk | 10 |
| unanswerable | 10 |
| multi_hop | 8 |
| multi_document | 5 |
| ambiguous | 5 |
| follow_up | 3 |
| **Total** | **60** |

The corpus indexed at evaluation time contains 2 real documents: *SCRIPT FOR
DISCOVERY.pdf* and *1-FYDP-Proposal-Form_0 (1)-1.pdf*.

## 3. Retrieval Metrics

Computed by `evaluation/metrics.py` over the 50 answerable questions that
have `supporting_sources` (unanswerable questions have no retrieval ground
truth, so they're excluded from these metrics — not scored as failures).

A retrieved chunk counts as *relevant* to a question if its `document_id`
matches one of the question's expected supporting sources, and — where the
expected source specifies a page — its `page_number` also matches. Exact
chunk-level matches aren't required, since chunking/reranking can validly
surface a different, overlapping chunk from the correct page.

- **Recall@K** — of the distinct expected `(document, page)` sources for a
  question, what fraction were found somewhere in the top K retrieved
  candidates?
- **Precision@K** — of the top K retrieved candidates, what fraction were
  actually relevant?
- **Hit Rate@K** — did *at least one* relevant candidate appear in the top K?
  (binary per question, then averaged)
- **MRR** (Mean Reciprocal Rank) — 1 / (rank of the first relevant candidate),
  averaged across questions. Rewards putting the right answer near the top,
  not just somewhere in the list.

K=3 and K=5 are both reported, computed from a single retrieval call per
question (`top_k=5` at evaluation time, wider than Chat's production
`top_k=3`, so both cuts can be derived from one pass).

## 4. The Five Experiments

Each experiment is a fixed combination of three independent flags on
`RAG.answer()` — `enable_query_rewrite`, `enable_multi_query`,
`enable_reranking` — defined in `evaluation/experiments.py`.

| # | Experiment | Query rewriting | Multi-query | Reranking |
|---|---|:---:|:---:|:---:|
| 1 | `basic_vector_retrieval` | off | off | off |
| 2 | `vector_plus_query_rewriting` | **on** | off | off |
| 3 | `multi_query_retrieval` | off | **on** | off |
| 4 | `multi_query_plus_reranking` | off | **on** | **on** |
| 5 | `final_pipeline` | **on** | **on** | **on** |

Experiment 1 is the control: a single embedded query, ranked by raw vector
distance, nothing else. Every other experiment adds exactly one or more
components on top of that control so its effect can be isolated.

Query rewriting only ever activates when conversation history is non-empty —
on this dataset, that's exactly 3 of the 60 questions (the `follow_up`
category). Multi-query generation and reranking apply to all 60 questions
whenever enabled.

## 5. Results

| Metric | #1 basic | #2 +rewrite | #3 multi-query | #4 multi-query+rerank | #5 final_pipeline |
|---|---:|---:|---:|---:|---:|
| Recall@3 | 0.613 | 0.613 | 0.597 | **0.857** | **0.857** |
| Precision@3 | 0.287 | 0.287 | 0.280 | **0.400** | **0.400** |
| Hit Rate@3 | 0.760 | 0.760 | 0.740 | **0.920** | **0.920** |
| Recall@5 | 0.763 | 0.763 | 0.763 | **0.910** | **0.910** |
| Precision@5 | 0.240 | 0.240 | 0.232 | 0.272 | 0.272 |
| Hit Rate@5 | 0.860 | 0.860 | 0.880 | **0.960** | **0.960** |
| MRR | 0.637 | 0.637 | 0.628 | 0.863 | **0.875** |
| Citation correctness | 43/50 | 43/50 | 44/50 | **48/50** | **48/50** |
| Avg latency | 0.10s | 0.57s | 13.19s | 24.78s | 40.77s |
| Median latency | 0.10s | 0.09s | 8.35s | 21.10s | 24.28s |
| Max latency | 0.23s | 14.79s | 58.36s | 69.31s | 158.42s |
| Model | gemini-3.5-flash-lite (all five) | | | | |

(Retrieval numbers rounded to 3 decimals for readability; exact values are
in the underlying report JSON files.)

## 6. Latency / Quality Trade-offs

Latency increases monotonically as components are added, but retrieval
quality does **not** — it takes a real step down with multi-query alone
(#3), then its biggest step up with reranking (#4), and only a marginal
further step with rewriting on top (#5):

- **#1 → #2** (+rewrite): +0.47s average latency, **zero** change in any
  retrieval metric.
- **#1 → #3** (+multi-query): +13.09s average latency, and Recall@3/
  Precision@3/Hit Rate@3/MRR all **decreased** slightly relative to the
  single-query baseline.
- **#3 → #4** (+reranking): +11.59s average latency, and Recall@3 jumped
  +0.260, Hit Rate@3 +0.180, MRR +0.235 — by far the largest quality jump of
  any single component in this evaluation, for local CPU cost only (the
  cross-encoder reranker makes no additional Gemini calls).
- **#4 → #5** (+rewrite on top of multi-query+reranking): +16.0s average
  latency (a 64% increase), for a **+0.012** MRR change and **no** change in
  any other retrieval metric or in citation correctness.

## 7. Per-Experiment Analysis

**#1 basic_vector_retrieval (control).** Single query, raw vector-distance
ranking. Recall@3 0.613, MRR 0.637. This is the floor every other
configuration is measured against.

**#2 vector_plus_query_rewriting.** Retrieval metrics are **identical** to
#1 across every single metric (Recall@3/5, Precision@3/5, Hit Rate@3/5, MRR,
even citation correctness at 43/50). This is expected given the dataset:
rewriting only fires for the 3 `follow_up` questions, and 3-in-60 isn't
enough to move any aggregate metric measurably. Median latency is
essentially unchanged (0.09s vs 0.10s) — most questions are completely
unaffected — while average latency rises (0.10s → 0.57s) purely because of
the 3 questions that do incur a Gemini rewrite call. This does **not** mean
query rewriting is universally useless — it means it showed no measurable
aggregate benefit *on this dataset*, which has too few follow-up questions
to evaluate it properly (see Limitations).

**#3 multi_query_retrieval.** Somewhat counterintuitively, adding multi-query
*alone* (no reranking) made Recall@3, Precision@3, Hit Rate@3, and MRR all
**worse** than the single-query baseline, while Hit Rate@5 improved slightly
(0.860 → 0.880). Merging candidates retrieved from 3 differently-phrased
queries produces a more diverse candidate pool, but ranking that pool by raw
vector distance alone doesn't reliably surface the most relevant items near
the top — it can dilute top-3 precision even as it very slightly widens
top-5 coverage. Average latency rose sharply (13.19s) since this is the
first experiment making a real Gemini call (multi-query generation) on
every one of the 60 questions.

**#4 multi_query_plus_reranking.** This is where multi-query's value
actually shows up. Adding the local BGE cross-encoder reranker on top of
the same multi-query candidate pool used in #3 turns the previous
regression into the largest gain in the whole evaluation: Recall@3 0.597 →
0.857, Hit Rate@3 0.740 → 0.920, MRR 0.628 → 0.863. In other words,
multi-query's larger, more diverse candidate pool is only useful once
something relevance-aware is sorting it — raw distance can't exploit it, a
cross-encoder can. Citation correctness also reaches its best result here,
48/50. The additional latency from #3 to #4 is primarily the local CPU
reranking work; reranking itself makes no additional Gemini calls.

**#5 final_pipeline.** Same retrieval configuration as #4, plus query
rewriting. Every retrieval metric is either identical to #4 or marginally
higher — Recall@3/Precision@3/Hit Rate@3/Recall@5/Precision@5/Hit Rate@5 are
all exactly the same as #4, and MRR ticks up from 0.863 to 0.875 (+0.012),
attributable to the 3 follow-up questions being rewritten before retrieval.
Citation correctness is unchanged at 48/50. Average latency, however, rises
substantially — 24.78s → 40.77s (+64%) — and the maximum single-question
latency more than doubles (69.31s → 158.42s), driven by a small number of
slow calls having an outsized effect on the mean.

## 8. Final Finding

**Experiment #4 (`multi_query_plus_reranking`) is the best practical
retrieval configuration on this dataset.** It captures nearly all of the
achievable retrieval-quality gain — MRR 0.863 vs. #5's 0.875, a difference
small enough to plausibly be within run-to-run noise on a 50-question
sample — at roughly **60% of Experiment #5's latency**, with zero
incremental Gemini cost from query rewriting.

Experiment #5 had essentially the same retrieval metrics as #4 (identical
Recall@3/5, Precision@3/5, and Hit Rate@3/5; citation correctness unchanged
at 48/50), with only a small MRR increase (0.863 → 0.875), but substantially
higher average latency (24.78s → 40.77s). On this dataset, that trade is not
worth it: query rewriting's marginal MRR benefit does not justify a 64%
increase in average latency and adding 3 more Gemini calls per run — but this
conclusion is specific to a dataset with only 3 follow-up questions, not a
general claim that rewriting doesn't help (see Limitations).

**This is a retrieval-quality finding only.** All five experiments were run
with `enable_answer_generation=False`, so nothing here says anything about
which configuration produces the best *final answers* — that requires a
separate comparison with generation enabled. Retrieval quality and answer
quality are different questions; a better retriever does not automatically
guarantee a better answer, and #4 has not been shown to produce the best
final answers, only the best-value retrieval evidence, of the five
configurations tested here.

## 9. Why Answer Correctness, Faithfulness, and Abstention Are N/A

All five experiments used `enable_answer_generation=False`, so
`actual_answer` is `None` for every one of the 300 question-runs (60
questions × 5 experiments). `evaluation/report.py` explicitly detects this
(`retrieval_only: true` in each results file) and reports:

- `correctness_counts`: `"N/A — retrieval-only run"`
- `faithfulness_counts`: `"N/A — retrieval-only run"`
- `abstention`: `"N/A — retrieval-only run"`

instead of scoring a `None` answer, which would otherwise silently produce
misleading numbers — e.g. every question would count as a "correct
abstention" (a `None` answer trivially satisfies the abstention heuristic)
or as "incorrect" (a `None` answer contains none of the expected key facts),
neither of which reflects anything real about system quality.

## 10. Citation Correctness

Citation correctness is the one answer-quality-adjacent metric still
meaningful in retrieval-only mode, because `score_citations()` only needs
the system's returned source metadata (`actual_sources` — built entirely
from retrieved chunk metadata) against the dataset's expected supporting
sources; it never needs the generated answer text.

Results: **43/50 (#1, #2), 44/50 (#3), 48/50 (#4 and #5)**. Citation
correctness tracks retrieval quality directly — the configurations with
better Recall/Hit Rate also cite the right source more often, which is the
expected relationship, since citations are built from the same top-K
evidence that the retrieval metrics score.

## 11. Limitations

- **Small corpus.** Only 2 real documents (21 chunks total) are indexed.
  Results — especially for the `multi_document` and `multi_hop` categories —
  reflect this specific small corpus and may not generalize to a larger,
  more varied document collection.
- **Only 3 follow-up questions.** Query rewriting is only ever exercised by
  the 3 `follow_up`-category questions out of 60. That's too few to draw any
  conclusion about rewriting's value in general — only that it made no
  measurable difference *on this particular dataset's follow-up questions*.
  A dataset with substantially more multi-turn conversation could plausibly
  show a different result.
- **Single run per experiment.** Each configuration was run once, not
  repeated across multiple trials, so there's no variance estimate — some of
  the small differences reported here (e.g. #4 vs #5's MRR gap) could
  narrow or widen on a repeat run.
- **Retrieval-only scope.** These results say nothing about final answer
  quality, faithfulness, or abstention behavior — that requires a separate
  evaluation with answer generation enabled.
- **Citation correctness is a presence check, not full precision.** It
  confirms whether at least one returned source matches an expected one, not
  whether every returned citation is individually correct.
- **Deterministic scoring only.** No LLM-judge was used anywhere in this
  evaluation; all metrics are computed by fixed, inspectable code in
  `evaluation/metrics.py`.

## 12. Conclusion

Across five isolated retrieval configurations measured on the same dataset
and corpus, the evidence shows: query rewriting alone produced no
measurable retrieval benefit on this dataset (too few follow-up questions to
exercise it); multi-query alone was actually slightly worse than the
single-query baseline on top-3 metrics, because a wider, more diverse
candidate pool without relevance-aware ranking doesn't reliably surface the
best items near the top; reranking is what actually makes multi-query's
wider candidate pool pay off, delivering the largest quality jump in the
whole evaluation for local-only cost; and adding rewriting on top of
multi-query+reranking bought a small further MRR gain at a disproportionate
latency cost, on this dataset. The practical recommendation from this
evidence is **multi-query + reranking**, not the full four-component
pipeline, as the retrieval configuration for further work — with the
explicit caveat that this is a retrieval-quality conclusion from a small
corpus and a dataset with limited follow-up coverage, not a final-answer-
quality conclusion and not a claim that any component is universally
without value.
