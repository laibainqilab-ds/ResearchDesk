import streamlit as st

from app.rag import RAG


st.set_page_config(
    page_title="ResearchDesk",
    page_icon="🔬",
    layout="wide",
)


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 ResearchDesk")
    st.caption("Document Research Assistant")

    st.divider()

    page = st.radio(
        "Navigate",
        ["Chat", "Documents", "Retrieval Inspector", "Evaluation"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**Status**")

    if st.session_state.rag_error:
        st.error("Answer generation unavailable")
        st.caption(st.session_state.rag_error)
    else:
        st.success("Models loaded")

    if st.session_state.rag is not None:
        indexed_chunks = st.session_state.rag.store.count()
        st.metric("Indexed chunks", indexed_chunks)


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
        st.info("Ask a question about your documents to get started.")

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

        if error is not None:
            content = f"Retrieval completed successfully. {error['message']}"
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
            st.info("No documents are indexed yet.")
        else:
            column1, column2 = st.columns(2)
            column1.metric("Documents", len(documents))
            column2.metric("Total chunks", sum(d["chunk_count"] for d in documents))

            st.divider()

            st.dataframe(
                documents,
                column_config={
                    "document_id": "Document ID",
                    "filename": "Filename",
                    "chunk_count": "Chunks",
                },
                hide_index=True,
                use_container_width=True,
            )

    st.info("Document upload and management will be added in a later phase.")


# ---------------------------------------------------------------------------
# Retrieval Inspector page
# ---------------------------------------------------------------------------

elif page == "Retrieval Inspector":
    st.header("Retrieval Inspector")
    st.caption(
        "Question → Query rewriting → Multi-query search → "
        "Candidate retrieval → Reranking → Final evidence"
    )

    st.divider()

    with st.expander("Run a retrieval-only test (no answer generation)"):
        st.write(
            "This bypasses query rewriting and query generation so you can "
            "exercise the retrieval pipeline without using Gemini quota."
        )

        inspector_question = st.text_input(
            "Question",
            placeholder="Example: How does Evo 2 help with genetic research?",
        )

        default_queries = [
            "Evo 2 applications in genomic research",
            "Evo 2 DNA sequence analysis and genetic research",
            "Evo 2 genetic variant prediction and biological applications",
        ]

        query_1 = st.text_input("Search query 1", value=default_queries[0])
        query_2 = st.text_input("Search query 2", value=default_queries[1])
        query_3 = st.text_input("Search query 3", value=default_queries[2])

        if st.button("Run Retrieval Inspector"):
            if not inspector_question.strip():
                st.warning("Enter a question first.")
            elif st.session_state.rag is None:
                st.warning("ResearchDesk could not initialize the retrieval pipeline.")
            else:
                search_queries = [
                    query.strip()
                    for query in [query_1, query_2, query_3]
                    if query.strip()
                ]

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

        st.subheader("Final evidence passed to the LLM")

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
    st.info(
        "Evaluation is not implemented yet. Phase 5 will add automated "
        "measurement of retrieval and answer quality."
    )
