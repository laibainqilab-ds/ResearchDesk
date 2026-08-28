import streamlit as st

from app.models.generator import GenerationUnavailableError
from app.rag import RAG


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
    st.caption("Documents currently indexed in the vector store.")

    if st.session_state.rag is None:
        st.warning("ResearchDesk could not initialize the vector store.")
    else:
        documents = st.session_state.rag.store.list_documents()

        if not documents:
            with st.container(border=True):
                st.markdown("#### No documents indexed yet")
                st.write(
                    "Documents will appear here once they have been ingested "
                    "into the vector store."
                )
        else:
            column1, column2 = st.columns(2)
            column1.metric("Documents", len(documents))
            column2.metric("Total chunks", sum(d["chunk_count"] for d in documents))

            st.divider()

            grid_columns = st.columns(3)

            for index, document in enumerate(documents):
                with grid_columns[index % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{document.get('filename') or 'Unknown file'}**")
                        st.caption(f"Document ID: {document['document_id']}")
                        st.caption(f"{document['chunk_count']} chunks indexed")

    st.divider()
    st.caption("Document upload and management will be added in a later phase.")


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
    st.caption("Automated quality measurement for retrieval and answers.")

    with st.container(border=True):
        st.markdown("#### Coming in a future phase")
        st.write(
            "Evaluation is not implemented yet. A future phase will add "
            "automated measurement of retrieval and answer quality. No "
            "evaluation metrics are available in this version."
        )
