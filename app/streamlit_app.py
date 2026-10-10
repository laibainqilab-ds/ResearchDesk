import streamlit as st

from app import api_client
from app.api_client import ApiError
from evaluation import report as evaluation_report
from pathlib import Path


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

if "token" not in st.session_state:
    st.session_state.token = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auth_error" not in st.session_state:
    st.session_state.auth_error = None


def is_authenticated() -> bool:
    return st.session_state.token is not None


def logout() -> None:
    st.session_state.token = None
    st.session_state.user_email = None
    st.session_state.current_chat_id = None
    st.session_state.messages = []


def refresh_messages() -> None:
    if st.session_state.current_chat_id is None:
        st.session_state.messages = []
        return
    st.session_state.messages = api_client.list_messages(
        st.session_state.token, st.session_state.current_chat_id
    )


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

if not is_authenticated():
    st.markdown('<div class="rd-brand">ResearchDesk</div>', unsafe_allow_html=True)
    st.markdown('<div class="rd-tagline">Document Research Assistant</div>', unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")

        if submitted:
            try:
                token = api_client.login(email, password)
                st.session_state.token = token
                st.session_state.user_email = email
                st.session_state.auth_error = None
                st.rerun()
            except ApiError as error:
                st.session_state.auth_error = error.detail
            except Exception as error:
                st.session_state.auth_error = f"Could not reach the ResearchDesk API: {error}"

    with signup_tab:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input(
                "Password (min 8 characters)", type="password", key="signup_password"
            )
            submitted_signup = st.form_submit_button("Create account", type="primary")

        if submitted_signup:
            try:
                token = api_client.signup(new_email, new_password)
                st.session_state.token = token
                st.session_state.user_email = new_email
                st.session_state.auth_error = None
                st.rerun()
            except ApiError as error:
                st.session_state.auth_error = error.detail
            except Exception as error:
                st.session_state.auth_error = f"Could not reach the ResearchDesk API: {error}"

    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)

    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

try:
    chats = api_client.list_chats(st.session_state.token)
    api_unavailable = None
except Exception as error:
    chats = []
    api_unavailable = str(error)

with st.sidebar:
    st.markdown('<div class="rd-brand">ResearchDesk</div>', unsafe_allow_html=True)
    st.markdown('<div class="rd-tagline">Document Research Assistant</div>', unsafe_allow_html=True)

    st.caption(f"Signed in as {st.session_state.user_email}")
    if st.button("Log out"):
        logout()
        st.rerun()

    page = st.radio(
        "Navigate",
        ["Chat", "Documents", "Retrieval Inspector", "Agent Trace", "Evaluation"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Chats**")

    if api_unavailable:
        render_status("API", False, f"Unavailable — {api_unavailable}")
    else:
        if st.button("+ New chat"):
            chat = api_client.create_chat(st.session_state.token)
            st.session_state.current_chat_id = chat["id"]
            refresh_messages()
            st.rerun()

        for chat in chats:
            is_current = chat["id"] == st.session_state.current_chat_id
            label = ("➡ " if is_current else "") + chat["title"]

            chat_col, delete_col = st.columns([5, 1])
            with chat_col:
                if st.button(label, key=f"select_chat_{chat['id']}", use_container_width=True):
                    st.session_state.current_chat_id = chat["id"]
                    refresh_messages()
                    st.rerun()
            with delete_col:
                if st.button("🗑", key=f"delete_chat_{chat['id']}"):
                    api_client.delete_chat(st.session_state.token, chat["id"])
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                    st.rerun()


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

if page == "Chat":
    st.header("Chat")
    st.caption("Ask questions about your indexed documents and get cited, grounded answers.")

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    elif st.session_state.current_chat_id is None:
        with st.container(border=True):
            st.markdown("#### Ask ResearchDesk")
            st.write(
                "Start a new chat from the sidebar, or select an existing one, then ask "
                "questions about your indexed documents. Every answer is grounded in cited "
                "source passages below it."
            )
    else:
        mode = st.radio(
            "Mode",
            ["rag", "agent"],
            format_func=lambda value: "Standard RAG" if value == "rag" else "Multi-agent workflow",
            horizontal=True,
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

                trace_id = message.get("trace_id")

                if message["role"] == "assistant" and trace_id:
                    st.caption(f"trace ID: {trace_id} · mode: {message.get('mode', 'n/a')}")

        question = st.chat_input("Ask a question about your documents")

        if question:
            with st.spinner("Thinking..."):
                try:
                    api_client.post_message(
                        st.session_state.token, st.session_state.current_chat_id, question, mode
                    )
                except ApiError as error:
                    st.error(f"Request failed: {error.detail}")
                else:
                    refresh_messages()
            st.rerun()

        with st.expander("Link another chat as context"):
            other_chats = [chat for chat in chats if chat["id"] != st.session_state.current_chat_id]

            if not other_chats:
                st.caption("No other chats available to link.")
            else:
                target = st.selectbox(
                    "Pull relevant context from:",
                    options=other_chats,
                    format_func=lambda chat: chat["title"],
                )

                if st.button("Link chat"):
                    try:
                        api_client.create_context_link(
                            st.session_state.token, st.session_state.current_chat_id, target["id"]
                        )
                        st.success(f"Linked '{target['title']}' as context for this chat.")
                    except ApiError as error:
                        st.error(f"Could not link chat: {error.detail}")


# ---------------------------------------------------------------------------
# Documents page
# ---------------------------------------------------------------------------

elif page == "Documents":
    st.header("Documents")
    st.caption("Upload PDF, TXT, or Markdown documents and manage what's indexed.")

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    else:
        st.subheader("Upload documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Ingest uploaded files", type="primary"):
            for uploaded_file in uploaded_files:
                with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                    try:
                        result = api_client.upload_document(
                            st.session_state.token, uploaded_file.name, uploaded_file.getvalue()
                        )
                        pages_note = f", {result['page_count']} pages" if result.get("page_count") else ""
                        status.update(
                            label=(
                                f"{uploaded_file.name} — indexed "
                                f"({result['chunk_count']} chunks{pages_note})"
                            ),
                            state="complete",
                        )
                    except ApiError as error:
                        if error.status_code == 409:
                            status.update(label=f"{uploaded_file.name} — already indexed", state="complete")
                            st.info(error.detail)
                        elif error.status_code == 415:
                            status.update(label=f"{uploaded_file.name} — unsupported file type", state="error")
                            st.error(error.detail)
                        else:
                            status.update(label=f"{uploaded_file.name} — failed to process", state="error")
                            st.error(error.detail)

            st.rerun()

        st.divider()
        st.subheader("Indexed documents")

        documents = api_client.list_documents(st.session_state.token)

        if not documents:
            with st.container(border=True):
                st.markdown("#### No documents indexed yet")
                st.write("Upload a PDF, TXT, or Markdown file above to get started.")
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

                        page_note = f"{document['page_count']} pages · " if document.get("page_count") else ""
                        st.caption(
                            f"{document.get('file_type') or 'UNKNOWN'} · "
                            f"{page_note}{document['chunk_count']} chunks"
                        )
                        st.caption(f"Document ID: {document['document_id'][:16]}...")

                    with action_col:
                        if st.button("Delete", key=f"delete_doc_{document['document_id']}"):
                            api_client.delete_document(st.session_state.token, document["document_id"])
                            st.rerun()


# ---------------------------------------------------------------------------
# Retrieval Inspector page
# ---------------------------------------------------------------------------

elif page == "Retrieval Inspector":
    st.header("Retrieval Inspector")
    st.caption(
        "Question → Query rewriting → Search queries → Retrieval candidates "
        "→ Deduplication → Reranking → Final evidence, read from the persisted "
        "run record of a Standard RAG chat message."
    )

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    elif st.session_state.current_chat_id is None:
        st.info("Select a chat from the sidebar and ask a Standard RAG question to inspect it here.")
    else:
        runs = [
            run for run in api_client.list_runs(st.session_state.token, st.session_state.current_chat_id)
            if run["mode"] == "rag"
        ]

        if not runs:
            st.info("No Standard RAG runs yet in this chat. Ask a question in Chat mode 'Standard RAG'.")
        else:
            selected_run = st.selectbox(
                "Run",
                options=list(reversed(runs)),
                format_func=lambda run: f"{run['created_at']} · trace {run['trace_id']}",
            )

            retrieval = selected_run["retrieval"] or {}
            st.caption(f"trace ID: {selected_run['trace_id']}")

            question_col, rewritten_col = st.columns(2)

            with question_col:
                st.markdown("**Original question**")
                st.write(retrieval.get("original_question", "n/a"))

            with rewritten_col:
                st.markdown("**Rewritten question**")
                st.write(retrieval.get("rewritten_question", "n/a"))

            search_queries = retrieval.get("search_queries", [])
            st.subheader("Generated search queries")
            for index, query in enumerate(search_queries, start=1):
                st.write(f"{index}. {query}")

            candidates = retrieval.get("candidates", [])
            final_evidence = retrieval.get("final_evidence", [])

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Search queries", len(search_queries))
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
                        distance_col.metric("Retrieval distance", format_score(candidate.get("retrieval_distance")))
                        score_col.metric("Reranking score", format_score(candidate.get("rerank_score")))

                        st.text_area(
                            "Chunk text", candidate.get("document", ""), height=180, key=f"candidate_{index}"
                        )

            st.divider()
            st.subheader("Final evidence passed to generator")

            if not final_evidence:
                st.warning("No final evidence was selected.")
            else:
                for index, evidence in enumerate(final_evidence, start=1):
                    with st.expander(f"Evidence {index} — {location_label(evidence)}", expanded=True):
                        st.caption(f"Reranking score: {format_score(evidence.get('rerank_score'))}")
                        st.text_area(
                            "Chunk text", evidence.get("document", ""), height=150, key=f"evidence_{index}"
                        )


# ---------------------------------------------------------------------------
# Agent Trace page
# ---------------------------------------------------------------------------

elif page == "Agent Trace":
    st.header("Agent Trace")
    st.caption(
        "Shows the persisted run record of a Multi-agent workflow chat message "
        "(Router -> Retrieval/Research -> Answer -> Validation)."
    )

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    elif st.session_state.current_chat_id is None:
        st.info("Select a chat from the sidebar and ask a question in 'Multi-agent workflow' mode.")
    else:
        runs = [
            run for run in api_client.list_runs(st.session_state.token, st.session_state.current_chat_id)
            if run["mode"] == "agent"
        ]

        if not runs:
            st.info("No agent workflow runs yet in this chat. Ask a question in Chat mode 'Multi-agent workflow'.")
        else:
            selected_run = st.selectbox(
                "Run",
                options=list(reversed(runs)),
                format_func=lambda run: f"{run['created_at']} · trace {run['trace_id']}",
            )

            message_for_run = next(
                (m for m in st.session_state.messages if m.get("trace_id") == selected_run["trace_id"]),
                None,
            )

            st.caption(f"trace ID: {selected_run['trace_id']}")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Selected route", selected_run.get("route") or "n/a")
            metric2.metric("Retries used", selected_run.get("retry_count", "n/a"))

            is_valid = selected_run.get("is_valid")
            if is_valid is None:
                validation_label = "n/a"
            elif is_valid:
                validation_label = "Valid"
            else:
                validation_label = "Invalid"
            metric3.metric("Validation", validation_label)

            st.subheader("Final answer")
            st.write(message_for_run["content"] if message_for_run else "(message not found)")

            st.divider()
            st.subheader("Agent trace")
            trace_rows = [
                {
                    "stage": entry.get("stage"),
                    "status": entry.get("status"),
                    "duration_seconds": entry.get("duration_seconds"),
                    "decision": entry.get("decision"),
                    "tools_used": ", ".join(entry.get("tools_used") or []) or "none",
                }
                for entry in (selected_run.get("agent_trace") or [])
            ]
            st.dataframe(trace_rows, use_container_width=True, hide_index=True)

            if message_for_run and message_for_run.get("sources"):
                st.divider()
                st.subheader(f"Retrieved evidence ({len(message_for_run['sources'])})")

                for index, evidence in enumerate(message_for_run["sources"], start=1):
                    with st.expander(f"Evidence {index} — {location_label(evidence)}"):
                        st.caption(f"Reranking score: {format_score(evidence.get('rerank_score'))}")


# ---------------------------------------------------------------------------
# Evaluation page
# ---------------------------------------------------------------------------

elif page == "Evaluation":
    st.header("Evaluation")
    st.caption("Phase 5 retrieval evaluation, read from the generated report files in evaluation/.")

    dataset_file = Path("evaluation/rag_evaluation.json")

    st.subheader("Dataset overview")

    if not dataset_file.exists():
        st.info("evaluation/rag_evaluation.json not found -- dataset overview unavailable.")
    else:
        dataset_by_id = evaluation_report.load_dataset()
        dataset_info = evaluation_report.build_dataset_summary(dataset_by_id)

        column1, column2, column3, column4 = st.columns(4)
        column1.metric("Total questions", dataset_info["total_questions"])
        column2.metric("Answerable", dataset_info["answerable_questions"])
        column3.metric("Unanswerable", dataset_info["unanswerable_questions"])
        column4.metric("Documents represented", len(dataset_info["documents_represented"]))

        st.write("**By category:**", dataset_info["by_category"])
        st.write("**Documents:**", ", ".join(dataset_info["documents_represented"]) or "none")

    st.divider()
    st.subheader("Evaluation report")

    available_reports = evaluation_report.discover_reports()
    loaded_reports = {
        path: report
        for path in available_reports
        if (report := evaluation_report.load_report_safely(path)) is not None
    }

    if not available_reports:
        with st.container(border=True):
            st.markdown("#### No evaluation report yet")
            st.write(
                "No report has been generated yet. Run the following from the "
                "project root, then reload this page:"
            )
            st.code(
                "python -m evaluation.run_evaluation\n"
                "python -m evaluation.report",
                language="powershell",
            )
    elif not loaded_reports:
        st.error(
            f"Found {len(available_reports)} report file(s) in evaluation/, but none "
            "could be parsed as valid JSON."
        )
    else:
        default_path = Path("evaluation/rag_report.json")
        report_paths = list(loaded_reports.keys())
        default_index = report_paths.index(default_path) if default_path in report_paths else 0

        selected_path = st.selectbox(
            "Select a report",
            options=report_paths,
            index=default_index,
            format_func=lambda path: evaluation_report.report_label(path, loaded_reports[path]),
        )

        eval_report = loaded_reports[selected_path]

        st.caption(f"Source: {selected_path.name}")

        st.subheader("Retrieval metrics")

        retrieval_info = eval_report.get("retrieval", {})
        st.caption(
            f"Computed over {retrieval_info.get('questions_evaluated', 'n/a')} answerable "
            "questions with known supporting sources."
        )

        retrieval_metrics = {
            key: value for key, value in retrieval_info.items()
            if key != "questions_evaluated"
        }
        metric_columns = st.columns(len(retrieval_metrics) or 1)

        for column, (metric_name, value) in zip(metric_columns, retrieval_metrics.items()):
            column.metric(metric_name, format_score(value))

        st.divider()
        st.subheader("Answer quality")
        st.caption(
            "Correctness and faithfulness require a real generated answer, so they are "
            "only available for full-generation reports. Citation correctness only needs "
            "retrieved source metadata, so it remains available even for retrieval-only reports."
        )

        answer_col, faithfulness_col, citation_col = st.columns(3)

        with answer_col:
            st.markdown("**Correctness**")
            if evaluation_report.has_dict_metric(eval_report, "answers", "correctness_counts"):
                st.write(eval_report["answers"]["correctness_counts"])
            else:
                st.info("Not available (retrieval-only report).")

        with faithfulness_col:
            st.markdown("**Faithfulness**")
            if evaluation_report.has_dict_metric(eval_report, "answers", "faithfulness_counts"):
                st.write(eval_report["answers"]["faithfulness_counts"])
            else:
                st.info("Not available (retrieval-only report).")

        with citation_col:
            st.markdown("**Citations**")
            if evaluation_report.has_dict_metric(eval_report, "answers", "citation_counts"):
                st.write(eval_report["answers"]["citation_counts"])
            else:
                st.info("Not available.")

        st.divider()
        st.subheader("Abstention")

        if evaluation_report.has_dict_metric(eval_report, "abstention"):
            abstention_info = eval_report["abstention"]

            abstain_col1, abstain_col2, abstain_col3, abstain_col4, abstain_col5 = st.columns(5)
            abstain_col1.metric("Correct abstentions", abstention_info["correct_abstentions"])
            abstain_col2.metric("Expected abstentions", abstention_info["expected_abstentions"])
            abstain_col3.metric("Missed abstentions", abstention_info["missed_abstentions"])
            abstain_col4.metric("Unexpected abstentions", abstention_info["unexpected_abstentions"])
            abstention_accuracy = abstention_info["abstention_accuracy"]
            abstain_col5.metric(
                "Abstention accuracy",
                f"{abstention_accuracy:.0%}" if abstention_accuracy is not None else "N/A",
            )
        else:
            st.info(
                "Abstention metrics are not available for this report -- it was run in "
                "retrieval-only mode, so there are no generated answers to check for abstention."
            )

        st.divider()
        st.subheader("Performance")

        performance_info = eval_report.get("performance", {})

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        avg_latency = performance_info.get("average_latency_seconds")
        median_latency = performance_info.get("median_latency_seconds")
        min_latency = performance_info.get("min_latency_seconds")
        max_latency = performance_info.get("max_latency_seconds")
        perf_col1.metric("Avg latency", f"{avg_latency:.2f}s" if avg_latency is not None else "N/A")
        perf_col2.metric("Median latency", f"{median_latency:.2f}s" if median_latency is not None else "N/A")
        perf_col3.metric("Min latency", f"{min_latency:.2f}s" if min_latency is not None else "N/A")
        perf_col4.metric("Max latency", f"{max_latency:.2f}s" if max_latency is not None else "N/A")

        st.caption(f"Model(s): {', '.join(performance_info.get('model_names_used', [])) or 'n/a'}")
        st.caption(evaluation_report.token_usage_message(performance_info))

        st.divider()

        failures = eval_report.get("failures", [])
        st.subheader(f"Failed questions ({len(failures)})")

        if not failures:
            st.success("No failures recorded in this report.")
        else:
            for failure in failures:
                with st.expander(f"{failure['id']} [{failure['category']}] — {failure['question']}"):
                    st.write("Reasons:", ", ".join(failure["reasons"]))

    st.divider()
    st.subheader("Experiment comparison — the five required Phase 5 configurations")
    st.caption(
        "All five experiments below were run in retrieval-only mode (no answer "
        "generation), read directly from their existing report files. This compares "
        "retrieval quality and latency only -- it does not indicate final answer quality."
    )

    comparison_rows = evaluation_report.build_experiment_comparison_rows()

    comparison_table = [
        {
            "Experiment": row["experiment"],
            "Mode": "retrieval-only" if row["retrieval_only"] else "full generation",
            "Recall@5": format_score(row["recall_at_5"]) if row["available"] else "not run yet",
            "Precision@5": format_score(row["precision_at_5"]) if row["available"] else "not run yet",
            "Hit Rate@5": format_score(row["hit_rate_at_5"]) if row["available"] else "not run yet",
            "MRR": format_score(row["mrr"]) if row["available"] else "not run yet",
            "Avg latency": (
                f"{row['average_latency_seconds']:.2f}s"
                if row["available"] and row["average_latency_seconds"] is not None
                else "not run yet"
            ),
        }
        for row in comparison_rows
    ]

    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    comparison_summary = evaluation_report.summarize_best_retrieval_configuration(comparison_rows)

    if comparison_summary:
        st.info(comparison_summary)

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        "- Only total per-question latency is available; internal timing for the "
        "retrieval step vs. the generation step is not separately exposed.\n"
        "- Token usage is currently unavailable -- the generator only returns "
        "response text, not usage metadata.\n"
        "- The five comparison experiments above were run retrieval-only. They "
        "measure retrieval quality only and cannot be used to judge final answer "
        "quality, faithfulness, or abstention behavior -- that requires a separate "
        "run with answer generation enabled."
    )
