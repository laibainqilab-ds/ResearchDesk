import json
from pathlib import Path

import streamlit as st

from app.ingestion.parsers import UnsupportedFileTypeError, file_type_for
from app.ingestion.pipeline import (
    DocumentParsingError,
    DuplicateDocumentError,
    EmptyDocumentError,
    compute_document_id,
    ingest_file,
)
from app.models.generator import GenerationUnavailableError
from app.rag import RAG


DOCUMENTS_DIR = Path("data/documents")


st.set_page_config(
    page_title="ResearchDesk",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Presentation (CSS) — layout/typography/status only, no behavior changes
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }

        .rd-brand {
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.1rem;
        }

        .rd-tagline {
            font-size: 0.85rem;
            opacity: 0.65;
            margin-bottom: 1.25rem;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
        }

        .rd-status-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }

        .rd-dot {
            width: 8px;
            height: 8px;
            min-width: 8px;
            border-radius: 50%;
        }

        .rd-dot-ok { background-color: #2e7d32; }
        .rd-dot-warn { background-color: #b45309; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status(label: str, ok: bool, text: str) -> None:
    dot_class = "rd-dot-ok" if ok else "rd-dot-warn"
    st.markdown(
        f'<div class="rd-status-row">'
        f'<span class="rd-dot {dot_class}"></span>'
        f'<span><strong>{label}:</strong> {text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


inject_css()


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ResearchDesk models...")
def load_rag() -> RAG:
    return RAG()


def get_rag() -> tuple[RAG | None, str | None]:
    try:
        return load_rag(), None
    except ValueError as error:
        return None, str(error)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_score(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


def location_label(item: dict) -> str:
    filename = item.get("filename") or "Unknown file"
    page_number = item.get("page_number")
    chunk_id = item.get("chunk_id", "unknown")

    if page_number is not None:
        return f"{filename} · page {page_number} · chunk {chunk_id}"
    return f"{filename} · chunk {chunk_id}"


def render_sources(sources: list[dict]) -> None:
    for source in sources:
        with st.container(border=True):
            st.markdown(f"**{source.get('filename') or 'Unknown file'}**")

            page_number = source.get("page_number")
            page_label = f"page {page_number}" if page_number is not None else "page n/a"

            st.caption(
                f"{page_label} · chunk {source.get('chunk_id', 'unknown')} "
                f"· relevance {format_score(source.get('rerank_score'))}"
            )


def render_generation_error_message(error: dict) -> str:
    """Compose a clean, honest user-facing message from the backend's structured error."""
    return f"Retrieval completed successfully, but answer generation failed: {error['message']}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "rag" not in st.session_state or "rag_error" not in st.session_state:
    st.session_state.rag, st.session_state.rag_error = get_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None

if "last_retrieval_source" not in st.session_state:
    st.session_state.last_retrieval_source = None

if "last_generation_error" not in st.session_state:
    st.session_state.last_generation_error = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="rd-brand">ResearchDesk</div>', unsafe_allow_html=True)
    st.markdown('<div class="rd-tagline">Document Research Assistant</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Chat", "Documents", "Retrieval Inspector", "Evaluation"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**System status**")

    if st.session_state.rag is None:
        render_status("Retrieval", False, "Unavailable")
    else:
        indexed_chunks = st.session_state.rag.store.count()

        if indexed_chunks > 0:
            render_status("Retrieval", True, "Ready")
        else:
            render_status("Retrieval", False, "No documents indexed")

    if st.session_state.rag_error:
        render_status("Generation", False, f"Unavailable — {st.session_state.rag_error}")
    elif st.session_state.last_generation_error:
        render_status("Generation", False, "Last attempt failed")
    else:
        render_status("Generation", True, "Configured")

    if st.session_state.rag is not None:
        st.metric("Indexed chunks", st.session_state.rag.store.count())


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

if page == "Chat":
    st.header("Chat")
    st.caption("Ask questions about your indexed documents and get cited, grounded answers.")

    if st.session_state.rag_error:
        st.warning(
            "ResearchDesk could not fully initialize, so chat is disabled. "
            f"{st.session_state.rag_error}"
        )
    elif not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### Ask ResearchDesk")
            st.write(
                "ResearchDesk answers your questions using only the documents "
                "that have been indexed into its vector store, and every "
                "answer is grounded in cited source passages below it."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("is_error"):
                st.warning(message["content"])
            else:
                st.write(message["content"])

            sources = message.get("sources")

            if message["role"] == "assistant" and sources:
                with st.expander(f"Sources ({len(sources)})"):
                    render_sources(sources)

    question = st.chat_input(
        "Ask a question about your documents",
        disabled=bool(st.session_state.rag_error),
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation_history = st.session_state.messages[:-1]

        with st.spinner("Searching documents..."):
            result = st.session_state.rag.answer(
                question=question,
                conversation_history=conversation_history,
            )

        st.session_state.last_retrieval = result.get("retrieval")
        st.session_state.last_retrieval_source = "Chat"

        error = result.get("error")
        st.session_state.last_generation_error = error

        if error is not None:
            content = render_generation_error_message(error)
            is_error = True
        else:
            content = result["answer"]
            is_error = False

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "sources": result["sources"],
                "is_error": is_error,
            }
        )

        st.rerun()


# ---------------------------------------------------------------------------
# Documents page
# ---------------------------------------------------------------------------

elif page == "Documents":
    st.header("Documents")
    st.caption("Upload PDF, TXT, or Markdown documents and manage what's indexed.")

    if st.session_state.rag is None:
        st.warning("ResearchDesk could not initialize the vector store.")
    else:
        st.subheader("Upload documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Ingest uploaded files", type="primary"):
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                save_path = None

                with st.status(
                    f"Processing {uploaded_file.name}...", expanded=True
                ) as status:
                    try:
                        file_type_for(uploaded_file.name)
                        st.write("Validating file type ✓")

                        file_bytes = uploaded_file.getvalue()
                        document_id = compute_document_id(file_bytes)
                        save_path = DOCUMENTS_DIR / f"{document_id[:16]}_{uploaded_file.name}"
                        save_path.write_bytes(file_bytes)

                        st.write("Parsing and chunking...")

                        result = ingest_file(
                            file_path=str(save_path),
                            filename=uploaded_file.name,
                            store=st.session_state.rag.store,
                            embedder=st.session_state.rag.embedder,
                        )

                        st.write("Embedding and storing ✓")

                        pages_note = (
                            f", {result['page_count']} pages"
                            if result["page_count"]
                            else ""
                        )
                        status.update(
                            label=(
                                f"{uploaded_file.name} — indexed "
                                f"({result['chunk_count']} chunks{pages_note})"
                            ),
                            state="complete",
                        )
                    except UnsupportedFileTypeError as error:
                        status.update(
                            label=f"{uploaded_file.name} — unsupported file type",
                            state="error",
                        )
                        st.error(str(error))
                    except DuplicateDocumentError as error:
                        if save_path is not None:
                            save_path.unlink(missing_ok=True)
                        status.update(
                            label=f"{uploaded_file.name} — already indexed",
                            state="complete",
                        )
                        st.info(str(error))
                    except EmptyDocumentError as error:
                        if save_path is not None:
                            save_path.unlink(missing_ok=True)
                        status.update(
                            label=f"{uploaded_file.name} — empty document",
                            state="error",
                        )
                        st.error(str(error))
                    except DocumentParsingError as error:
                        if save_path is not None:
                            save_path.unlink(missing_ok=True)
                        status.update(
                            label=f"{uploaded_file.name} — failed to process",
                            state="error",
                        )
                        st.error(str(error))

            st.rerun()

        st.divider()
        st.subheader("Indexed documents")

        documents = st.session_state.rag.store.list_documents()

        if not documents:
            with st.container(border=True):
                st.markdown("#### No documents indexed yet")
                st.write(
                    "Upload a PDF, TXT, or Markdown file above to get started."
                )
        else:
            column1, column2 = st.columns(2)
            column1.metric("Documents", len(documents))
            column2.metric("Total chunks", sum(d["chunk_count"] for d in documents))

            st.divider()

            for document in documents:
                with st.container(border=True):
                    info_col, action_col = st.columns([4, 1])

                    with info_col:
                        st.markdown(f"**{document.get('filename') or 'Unknown file'}**")

                        page_note = (
                            f"{document['page_count']} pages · "
                            if document.get("page_count")
                            else ""
                        )
                        st.caption(
                            f"{document.get('file_type') or 'UNKNOWN'} · "
                            f"{page_note}{document['chunk_count']} chunks"
                        )
                        st.caption(f"Document ID: {document['document_id'][:16]}...")

                    with action_col:
                        if st.button(
                            "Delete",
                            key=f"delete_{document['document_id']}",
                        ):
                            st.session_state.rag.store.delete_document(
                                document["document_id"]
                            )
                            st.rerun()


# ---------------------------------------------------------------------------
# Retrieval Inspector page
# ---------------------------------------------------------------------------

elif page == "Retrieval Inspector":
    st.header("Retrieval Inspector")
    st.caption(
        "Question → Query rewriting → Search queries → Retrieval candidates "
        "→ Deduplication → Reranking → Final evidence"
    )

    st.divider()

    with st.expander("Run a retrieval-only test (no final answer generation)"):
        st.write(
            "This runs real query generation and retrieval for your "
            "question, without generating a final written answer."
        )

        inspector_question = st.text_input(
            "Question",
            placeholder="Example: How does Evo 2 help with genetic research?",
        )

        if st.button("Run Retrieval Inspector", type="primary"):
            if not inspector_question.strip():
                st.warning("Enter a question first.")
            elif st.session_state.rag is None:
                st.warning("ResearchDesk could not initialize the retrieval pipeline.")
            else:
                with st.spinner("Generating search queries..."):
                    try:
                        search_queries = st.session_state.rag.generator.generate_queries(
                            question=inspector_question,
                            num_queries=3,
                        )
                    except GenerationUnavailableError:
                        search_queries = []

                if not search_queries:
                    search_queries = [inspector_question]

                with st.spinner("Running retrieval pipeline..."):
                    retrieval = st.session_state.rag.retrieve(
                        retrieval_question=inspector_question,
                        search_queries=search_queries,
                        top_k=3,
                    )

                st.session_state.last_retrieval = {
                    "original_question": inspector_question,
                    "rewritten_question": inspector_question,
                    "search_queries": search_queries,
                    "candidates": retrieval["candidates"],
                    "final_evidence": retrieval["final_evidence"],
                }
                st.session_state.last_retrieval_source = "Manual test"

                st.rerun()

    retrieval = st.session_state.last_retrieval

    if not retrieval:
        st.info(
            "Ask a question in Chat, or run the retrieval-only test above, "
            "to inspect the retrieval pipeline."
        )
    else:
        st.divider()
        st.caption(f"Showing retrieval from: {st.session_state.last_retrieval_source}")

        question_col, rewritten_col = st.columns(2)

        with question_col:
            st.markdown("**Original question**")
            st.write(retrieval["original_question"])

        with rewritten_col:
            st.markdown("**Rewritten question**")
            st.write(retrieval["rewritten_question"])

        st.subheader("Generated search queries")

        for index, query in enumerate(retrieval["search_queries"], start=1):
            st.write(f"{index}. {query}")

        candidates = retrieval["candidates"]
        final_evidence = retrieval["final_evidence"]

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Search queries", len(retrieval["search_queries"]))
        metric2.metric("Unique candidates", len(candidates))
        metric3.metric("Final evidence", len(final_evidence))

        st.divider()

        st.subheader("Retrieved and reranked candidates")
        st.caption("Ranked by BGE reranking score, after deduplication by document and chunk.")

        if not candidates:
            st.warning("No candidates were retrieved.")
        else:
            for index, candidate in enumerate(candidates, start=1):
                with st.expander(f"Rank {index} — {location_label(candidate)}"):
                    st.caption(f"Search query: {candidate.get('search_query', 'n/a')}")

                    distance_col, score_col = st.columns(2)
                    distance_col.metric(
                        "Retrieval distance",
                        format_score(candidate.get("retrieval_distance")),
                    )
                    score_col.metric(
                        "Reranking score",
                        format_score(candidate.get("rerank_score")),
                    )

                    st.text_area(
                        "Chunk text",
                        candidate.get("document", ""),
                        height=180,
                        key=f"candidate_{index}",
                    )

        st.divider()

        st.subheader("Final evidence passed to generator")
        st.caption("The exact top-K chunks used to generate the answer.")

        if not final_evidence:
            st.warning("No final evidence was selected.")
        else:
            for index, evidence in enumerate(final_evidence, start=1):
                with st.expander(
                    f"Evidence {index} — {location_label(evidence)}",
                    expanded=True,
                ):
                    st.caption(
                        f"Reranking score: {format_score(evidence.get('rerank_score'))}"
                    )

                    st.text_area(
                        "Chunk text",
                        evidence.get("document", ""),
                        height=150,
                        key=f"evidence_{index}",
                    )


# ---------------------------------------------------------------------------
# Evaluation page
# ---------------------------------------------------------------------------

elif page == "Evaluation":
    st.header("Evaluation")
    st.caption("Results from the most recently generated Phase 5 evaluation report.")

    report_path = Path("evaluation/rag_report.json")

    if not report_path.exists():
        with st.container(border=True):
            st.markdown("#### No evaluation report yet")
            st.write(
                "This page displays the results of the last evaluation run. "
                "No report has been generated yet. Run the following from the "
                "project root, then reload this page:"
            )
            st.code(
                "python -m evaluation.run_evaluation\n"
                "python -m evaluation.report",
                language="powershell",
            )
    else:
        eval_report = json.loads(report_path.read_text(encoding="utf-8"))

        st.caption(f"Experiment: {eval_report.get('experiment', 'n/a')}")

        st.subheader("Dataset")
        dataset_info = eval_report["dataset"]

        column1, column2, column3 = st.columns(3)
        column1.metric("Total questions", dataset_info["total_questions"])
        column2.metric("Answerable", dataset_info["answerable_questions"])
        column3.metric("Unanswerable", dataset_info["unanswerable_questions"])

        st.write("**By category:**", dataset_info["by_category"])
        st.write("**Documents represented:**", ", ".join(dataset_info["documents_represented"]) or "none")

        st.divider()
        st.subheader("Retrieval metrics")
        st.caption(
            f"Computed over {eval_report['retrieval']['questions_evaluated']} answerable "
            "questions with known supporting sources."
        )

        retrieval_metrics = {
            key: value for key, value in eval_report["retrieval"].items()
            if key != "questions_evaluated"
        }
        metric_columns = st.columns(len(retrieval_metrics) or 1)

        for column, (metric_name, value) in zip(metric_columns, retrieval_metrics.items()):
            column.metric(metric_name, format_score(value))

        st.divider()
        st.subheader("Answer metrics")

        answer_col, faithfulness_col, citation_col = st.columns(3)

        with answer_col:
            st.markdown("**Correctness**")
            st.write(eval_report["answers"]["correctness_counts"])

        with faithfulness_col:
            st.markdown("**Faithfulness**")
            st.write(eval_report["answers"]["faithfulness_counts"])

        with citation_col:
            st.markdown("**Citations**")
            st.write(eval_report["answers"]["citation_counts"])

        st.divider()
        st.subheader("Abstention")
        abstention_info = eval_report["abstention"]

        abstain_col1, abstain_col2, abstain_col3, abstain_col4 = st.columns(4)
        abstain_col1.metric("Correct abstentions", abstention_info["correct_abstentions"])
        abstain_col2.metric("Missed abstentions", abstention_info["missed_abstentions"])
        abstain_col3.metric("Unexpected abstentions", abstention_info["unexpected_abstentions"])
        abstention_accuracy = abstention_info["abstention_accuracy"]
        abstain_col4.metric(
            "Abstention accuracy",
            f"{abstention_accuracy:.0%}" if abstention_accuracy is not None else "N/A",
        )

        st.divider()
        st.subheader("Performance")
        performance_info = eval_report["performance"]

        perf_col1, perf_col2, perf_col3 = st.columns(3)
        avg_latency = performance_info["average_latency_seconds"]
        median_latency = performance_info["median_latency_seconds"]
        perf_col1.metric("Avg latency", f"{avg_latency:.2f}s" if avg_latency is not None else "N/A")
        perf_col2.metric("Median latency", f"{median_latency:.2f}s" if median_latency is not None else "N/A")
        perf_col3.metric("Model(s)", ", ".join(performance_info["model_names_used"]) or "n/a")

        st.caption(
            f"Token usage: {performance_info['token_usage']}. "
            f"{performance_info['unavailable_metrics_reason']}"
        )

        st.divider()
        st.subheader(f"Failed questions ({len(eval_report['failures'])})")

        if not eval_report["failures"]:
            st.success("No failures recorded in the last run.")
        else:
            for failure in eval_report["failures"]:
                with st.expander(f"{failure['id']} [{failure['category']}] — {failure['question']}"):
                    st.write("Reasons:", ", ".join(failure["reasons"]))
