from pathlib import Path

import streamlit as st

from app import api_client
from app.api_client import ApiError
from app.chat_ui_helpers import agent_progress_outcome, find_run_by_trace_id
from app.ui_helpers import (
    answer_markdown,
    auth_brand_html,
    bar_html,
    card_html,
    chips_html,
    display_chat_title,
    document_meta_html,
    document_title_html,
    esc,
    failure_row_html,
    file_badge_html,
    format_date,
    format_run_time,
    kpis_html,
    kv_rows_html,
    message_meta_html,
    pipeline_html,
    result_row_html,
    retrieval_metric_label,
    run_summary_html,
    score_badge_html,
    section_label_html,
    section_title_html,
    short_id,
    sidebar_brand_html,
    source_chips_html,
    stepper_html,
    tiles_html,
    tools_details_html,
    topbar_title_html,
    user_bubble_html,
    user_chip_html,
    validation_html,
)
from app.ui_styles import AUTH_CSS, CHAT_CSS, GLOBAL_CSS
from evaluation import report as evaluation_report

AGENT_REQUEST_TIMEOUT_SECONDS = 300

# Neutral Material-icon avatar for answers (Streamlit's default avatars are
# filled with the theme's red/orange, which would clash with the accent).
CHAT_AVATAR = ":material/auto_awesome:"

NAV_GROUPS = {
    "Workspace": [("Chat", ":material/chat:"), ("Documents", ":material/description:")],
    "Inspect": [
        ("Retrieval Inspector", ":material/search:"),
        ("Agent Trace", ":material/account_tree:"),
        ("Evaluation", ":material/bar_chart:"),
    ],
}
PAGE_NAMES = [name for items in NAV_GROUPS.values() for name, _ in items]


st.set_page_config(
    page_title="ResearchDesk",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Presentation (CSS lives in app/ui_styles.py; colours/fonts in .streamlit/config.toml)
# ---------------------------------------------------------------------------

def inject_css(extra: str = "") -> None:
    st.markdown(f"<style>{GLOBAL_CSS}{extra}</style>", unsafe_allow_html=True)


inject_css()


def render_status(label: str, ok: bool, text: str) -> None:
    dot_class = "rd-dot-ok" if ok else "rd-dot-warn"
    st.markdown(
        f'<div class="rd-status-row">'
        f'<span class="rd-dot {dot_class}"></span>'
        f'<span><strong>{esc(label)}:</strong> {esc(text)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def topbar(title: str, subtitle: str = "", weights: tuple = (5, 4)):
    """The page title on the left. Returns the right-hand column, where each
    page puts its own controls (run picker, mode switch, ...)."""
    with st.container(key="topbar"):
        left, right = st.columns(list(weights), vertical_alignment="center")
        left.markdown(topbar_title_html(title, subtitle), unsafe_allow_html=True)

    return right


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_score(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


def badge_score(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def place_label(item: dict) -> str:
    page_number = item.get("page_number")
    chunk_id = item.get("chunk_id", "unknown")

    if page_number is not None:
        return f"page {page_number} · chunk {chunk_id}"
    return f"chunk {chunk_id}"


def format_seconds(value: object) -> str:
    return f"{value:.2f}s" if isinstance(value, (int, float)) else "N/A"


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

if st.session_state.get("page") not in PAGE_NAMES:
    st.session_state.page = "Chat"


def is_authenticated() -> bool:
    return st.session_state.token is not None


def logout() -> None:
    st.session_state.token = None
    st.session_state.user_email = None
    st.session_state.current_chat_id = None
    st.session_state.messages = []
    st.session_state.page = "Chat"


def refresh_messages() -> None:
    if st.session_state.current_chat_id is None:
        st.session_state.messages = []
        return
    chat = api_client.get_chat(st.session_state.token, st.session_state.current_chat_id)
    st.session_state.messages = chat["messages"]


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

if not is_authenticated():
    st.markdown(f"<style>{AUTH_CSS}</style>", unsafe_allow_html=True)
    st.markdown(auth_brand_html(), unsafe_allow_html=True)

    with st.container(key="auth_card"):
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

    st.markdown(
        '<div class="rd-auth-note">Your documents and chats are private to your account.</div>',
        unsafe_allow_html=True,
    )
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


def nav_button(name: str, icon: str) -> None:
    slug = name.lower().replace(" ", "_")
    active = st.session_state.page == name
    key = f"nav_active_{slug}" if active else f"nav_{slug}"

    if st.button(name, key=key, icon=icon, type="tertiary", use_container_width=True) and not active:
        st.session_state.page = name
        st.rerun()


def chat_row(chat: dict) -> None:
    is_current = chat["id"] == st.session_state.current_chat_id
    title = display_chat_title(chat)
    key = f"chat_active_{chat['id']}" if is_current else f"chat_{chat['id']}"

    def select() -> None:
        st.session_state.current_chat_id = chat["id"]
        refresh_messages()
        st.rerun()

    if not is_current:
        if st.button(title, key=key, type="tertiary", use_container_width=True):
            select()
        return

    title_col, delete_col = st.columns([6, 1], vertical_alignment="center")

    with title_col:
        if st.button(title, key=key, type="tertiary", use_container_width=True):
            select()

    with delete_col:
        with st.popover("", icon=":material/delete:", help="Delete chat"):
            st.caption("Delete this chat and its messages?")

            if st.button("Delete chat", key=f"delete_chat_{chat['id']}", type="primary"):
                api_client.delete_chat(st.session_state.token, chat["id"])
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()


with st.sidebar:
    st.markdown(sidebar_brand_html(), unsafe_allow_html=True)

    if api_unavailable:
        render_status("API", False, f"Unavailable — {api_unavailable}")
    elif st.button("New chat", key="new_chat", type="primary", icon=":material/add:", use_container_width=True):
        chat = api_client.create_chat(st.session_state.token)
        st.session_state.current_chat_id = chat["id"]
        st.session_state.page = "Chat"
        refresh_messages()
        st.rerun()

    for group, items in NAV_GROUPS.items():
        st.markdown(section_label_html(group), unsafe_allow_html=True)

        for name, icon in items:
            nav_button(name, icon)

    if not api_unavailable:
        st.markdown(section_label_html("Recent chats"), unsafe_allow_html=True)

        for chat in chats:
            chat_row(chat)

    with st.container(key="user_chip"):
        info_col, out_col = st.columns([5, 1], vertical_alignment="center")
        info_col.markdown(user_chip_html(st.session_state.user_email), unsafe_allow_html=True)

        if out_col.button("", key="logout", icon=":material/logout:", help="Log out"):
            logout()
            st.rerun()


page = st.session_state.page


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

if page == "Chat":
    inject_css(CHAT_CSS)

    current_chat = next((chat for chat in chats if chat["id"] == st.session_state.current_chat_id), None)
    chat_title = ""

    if current_chat:
        chat_title = display_chat_title(current_chat)

    right = topbar("Chat", chat_title, (4, 5))

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    elif st.session_state.current_chat_id is None:
        st.markdown(
            card_html(
                "Ask ResearchDesk",
                "<p>Start a new chat from the sidebar, or select an existing one, then ask "
                "questions about your indexed documents. Every answer is grounded in cited "
                "source passages below it.</p>",
            ),
            unsafe_allow_html=True,
        )
    else:
        link_col, mode_col = right.columns([5, 6], vertical_alignment="center")

        with link_col:
            with st.popover("Link chat as context", icon=":material/link:", use_container_width=True):
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

        with mode_col:
            picked = st.segmented_control(
                "Answer mode",
                ["rag", "agent"],
                format_func=lambda value: "Standard RAG" if value == "rag" else "Multi-agent",
                default="rag",
                key="mode_control",
                label_visibility="collapsed",
            )

        # Clicking the selected option again clears a segmented control; keep the last mode then.
        mode = picked or st.session_state.get("last_mode", "rag")
        st.session_state.last_mode = mode

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(user_bubble_html(message["content"]), unsafe_allow_html=True)
                continue

            with st.chat_message("assistant", avatar=CHAT_AVATAR):
                if message.get("is_error"):
                    st.warning(message["content"])
                else:
                    st.markdown(answer_markdown(message["content"]), unsafe_allow_html=True)

                chips = source_chips_html(message.get("sources"))

                if chips:
                    st.markdown(chips, unsafe_allow_html=True)

                trace_id = message.get("trace_id")

                if trace_id:
                    st.markdown(message_meta_html(message.get("mode"), trace_id), unsafe_allow_html=True)

        question = st.chat_input("Ask a question about your documents")

        if question and mode == "agent":
            with st.status("Running the multi-agent workflow...", expanded=True) as status:
                status.write(
                    "This routes your question, gathers evidence (possibly across "
                    "multiple sub-questions for complex queries), drafts an answer, "
                    "and validates it before replying. This can take longer than "
                    "Standard RAG."
                )
                result = None

                try:
                    result = api_client.post_message(
                        st.session_state.token,
                        st.session_state.current_chat_id,
                        question,
                        mode,
                        timeout_seconds=AGENT_REQUEST_TIMEOUT_SECONDS,
                    )
                except ApiError as error:
                    status.update(label="Request failed.", state="error")
                    st.error(f"Request failed: {error.detail}")
                except Exception:
                    status.update(label="Request timed out or the connection failed.", state="error")
                    st.error("The agent request did not finish in time. Please try again.")
                else:
                    run = None
                    try:
                        runs = api_client.list_runs(
                            st.session_state.token, st.session_state.current_chat_id
                        )
                        run = find_run_by_trace_id(runs, result.get("trace_id"))
                    except Exception:
                        run = None  # Non-fatal -- the message itself already succeeded.

                    state, label = agent_progress_outcome(result, run)
                    status.update(label=label, state=state)

                    if run and run.get("agent_trace"):
                        for entry in run["agent_trace"]:
                            status.write(f"- **{entry.get('stage', '?')}**: {entry.get('status', '?')}")

                    refresh_messages()
            st.rerun()
        elif question:
            with st.spinner("Thinking..."):
                try:
                    api_client.post_message(
                        st.session_state.token, st.session_state.current_chat_id, question, mode
                    )
                except ApiError as error:
                    st.error(f"Request failed: {error.detail}")
                except Exception:
                    st.error("The request did not finish in time. Please try again.")
                else:
                    refresh_messages()
            st.rerun()


# ---------------------------------------------------------------------------
# Documents page
# ---------------------------------------------------------------------------

elif page == "Documents":
    documents = [] if api_unavailable else api_client.list_documents(st.session_state.token)
    total_chunks = sum(document["chunk_count"] for document in documents)
    topbar("Documents", f"{len(documents)} documents · {total_chunks} chunks" if documents else "")

    if api_unavailable:
        st.warning(f"ResearchDesk API is unavailable: {api_unavailable}")
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
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

        if not documents:
            with st.container(border=True):
                st.markdown("#### No documents indexed yet")
                st.write("Upload a PDF, TXT, or Markdown file above to get started.")
        else:
            query = st.text_input(
                "Search documents",
                placeholder="Search documents…",
                icon=":material/search:",
                label_visibility="collapsed",
            )
            needle = query.strip().lower()
            shown = [document for document in documents if needle in (document.get("filename") or "").lower()]

            if not shown:
                st.caption(f"No documents match '{query.strip()}'.")

            for document in shown:
                with st.container(border=True, key=f"doc_{document['document_id']}"):
                    badge_col, info_col, action_col = st.columns([1, 8, 3], vertical_alignment="center")

                    badge_col.markdown(file_badge_html(document.get("file_type")), unsafe_allow_html=True)
                    info_col.markdown(
                        document_title_html(document.get("filename"))
                        + document_meta_html(
                            document.get("file_type"),
                            document.get("page_count"),
                            document["chunk_count"],
                            document["document_id"],
                        ),
                        unsafe_allow_html=True,
                    )

                    # [3, 2] (not the original [2, 1]): measured via real rendered DOM
                    # that delete_col was only ~62px wide, too narrow for an icon +
                    # "Delete" on one line -- see the matching CSS comment in
                    # app/ui_styles.py for the full measurement.
                    date_col, delete_col = action_col.columns([3, 2], vertical_alignment="center")
                    date_col.markdown(
                        f'<div class="rd-date">{esc(format_date(document.get("created_at")))}</div>',
                        unsafe_allow_html=True,
                    )

                    if delete_col.button(
                        "Delete", key=f"delete_doc_{document['document_id']}", icon=":material/delete:"
                    ):
                        api_client.delete_document(st.session_state.token, document["document_id"])
                        st.rerun()


# ---------------------------------------------------------------------------
# Retrieval Inspector page
# ---------------------------------------------------------------------------

elif page == "Retrieval Inspector":
    right = topbar("Retrieval Inspector", "Standard RAG runs")

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
            selected_run = right.selectbox(
                "Run",
                options=list(reversed(runs)),
                format_func=lambda run: f"{format_run_time(run['created_at'])} · trace {short_id(run['trace_id'])}",
                label_visibility="collapsed",
            )

            retrieval = selected_run["retrieval"] or {}
            search_queries = retrieval.get("search_queries", [])
            candidates = retrieval.get("candidates", [])
            final_evidence = retrieval.get("final_evidence", [])

            st.markdown(
                run_summary_html(
                    retrieval.get("original_question", "n/a"),
                    retrieval.get("rewritten_question", "n/a"),
                    "rag",
                    selected_run["trace_id"],
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                pipeline_html(len(search_queries), len(candidates), len(final_evidence), search_queries),
                unsafe_allow_html=True,
            )

            st.markdown(
                section_title_html("Final evidence passed to the generator", f"{len(final_evidence)} results"),
                unsafe_allow_html=True,
            )

            if not final_evidence:
                st.warning("No final evidence was selected.")
            else:
                st.markdown(
                    "".join(
                        result_row_html(
                            index,
                            evidence.get("filename"),
                            place_label(evidence),
                            [score_badge_html("rerank", badge_score(evidence.get("rerank_score")), True)],
                            passage=evidence.get("document", ""),
                            open_=index == 1,
                        )
                        for index, evidence in enumerate(final_evidence, start=1)
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                section_title_html(
                    "All retrieved candidates", "ranked by reranking score, after deduplication"
                ),
                unsafe_allow_html=True,
            )

            if not candidates:
                st.warning("No candidates were retrieved.")
            else:
                st.markdown(
                    "".join(
                        result_row_html(
                            index,
                            candidate.get("filename"),
                            place_label(candidate),
                            [
                                score_badge_html("distance", badge_score(candidate.get("retrieval_distance"))),
                                score_badge_html("rerank", badge_score(candidate.get("rerank_score")), True),
                            ],
                            passage=candidate.get("document", ""),
                            note=f"Search query: {candidate.get('search_query', 'n/a')}",
                        )
                        for index, candidate in enumerate(candidates, start=1)
                    ),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Agent Trace page
# ---------------------------------------------------------------------------

elif page == "Agent Trace":
    right = topbar("Agent Trace", "Multi-agent runs")

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
            selected_run = right.selectbox(
                "Run",
                options=list(reversed(runs)),
                format_func=lambda run: f"{format_run_time(run['created_at'])} · trace {short_id(run['trace_id'])}",
                label_visibility="collapsed",
            )

            messages = st.session_state.messages
            answer_index = next(
                (i for i, m in enumerate(messages) if m.get("trace_id") == selected_run["trace_id"]), None
            )
            message_for_run = messages[answer_index] if answer_index is not None else None
            question_for_run = None

            if answer_index and messages[answer_index - 1].get("role") == "user":
                question_for_run = messages[answer_index - 1].get("content")

            st.markdown(
                run_summary_html(question_for_run, None, "agent", selected_run["trace_id"]),
                unsafe_allow_html=True,
            )

            retry_count = selected_run.get("retry_count")
            st.markdown(
                tiles_html(
                    [
                        ("Selected route", esc(selected_run.get("route") or "n/a")),
                        ("Retries used", esc("n/a" if retry_count is None else retry_count)),
                        ("Validation", validation_html(selected_run.get("is_valid"))),
                    ]
                ),
                unsafe_allow_html=True,
            )

            trace_entries = selected_run.get("agent_trace") or []
            stepper = stepper_html(trace_entries)

            if stepper:
                st.markdown(stepper + tools_details_html(trace_entries), unsafe_allow_html=True)
            else:
                st.info("No trace entries were recorded for this run.")

            with st.container(border=True):
                st.caption("Final answer")

                if message_for_run:
                    st.markdown(answer_markdown(message_for_run["content"]), unsafe_allow_html=True)
                else:
                    st.write("(message not found)")

            if message_for_run and message_for_run.get("sources"):
                sources = message_for_run["sources"]

                st.markdown(
                    section_title_html("Retrieved evidence", f"{len(sources)} results"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "".join(
                        result_row_html(
                            source.get("citation_id") or index,
                            source.get("filename"),
                            place_label(source),
                            [score_badge_html("rerank", badge_score(source.get("rerank_score")), True)],
                        )
                        for index, source in enumerate(sources, start=1)
                    ),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Evaluation page
# ---------------------------------------------------------------------------

elif page == "Evaluation":
    right = topbar("Evaluation", "Phase 5 retrieval evaluation")

    dataset_file = Path("evaluation/rag_evaluation.json")
    dataset_info = None

    if dataset_file.exists():
        dataset_info = evaluation_report.build_dataset_summary(evaluation_report.load_dataset())

    available_reports = evaluation_report.discover_reports()
    loaded_reports = {
        path: report
        for path in available_reports
        if (report := evaluation_report.load_report_safely(path)) is not None
    }

    eval_report = None
    selected_path = None

    if loaded_reports:
        default_path = Path("evaluation/rag_report.json")
        report_paths = list(loaded_reports.keys())
        default_index = report_paths.index(default_path) if default_path in report_paths else 0

        selected_path = right.selectbox(
            "Report",
            options=report_paths,
            index=default_index,
            format_func=lambda path: evaluation_report.report_label(path, loaded_reports[path]),
            label_visibility="collapsed",
        )
        eval_report = loaded_reports[selected_path]

    if dataset_info is None:
        st.info("evaluation/rag_evaluation.json not found -- dataset overview unavailable.")

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
        retrieval_info = eval_report.get("retrieval", {})
        performance_info = eval_report.get("performance", {})
        has_abstention = evaluation_report.has_dict_metric(eval_report, "abstention")
        abstention_info = eval_report["abstention"] if has_abstention else {}
        abstention_accuracy = abstention_info.get("abstention_accuracy")

        questions_value = str(dataset_info["total_questions"]) if dataset_info else "N/A"
        questions_sub = (
            f"{dataset_info['answerable_questions']} answerable · {dataset_info['unanswerable_questions']} unanswerable"
            if dataset_info
            else ""
        )
        recall = retrieval_info.get("recall_at_5")
        mrr = retrieval_info.get("mrr")

        st.caption(f"Source: {selected_path.name}")
        st.markdown(
            kpis_html(
                [
                    ("Questions", questions_value, questions_sub),
                    ("Recall@5", badge_score(recall) if recall is not None else "N/A", ""),
                    ("MRR", badge_score(mrr) if mrr is not None else "N/A", ""),
                    (
                        "Abstention accuracy",
                        f"{abstention_accuracy:.0%}" if abstention_accuracy is not None else "N/A",
                        "" if has_abstention else "retrieval-only report",
                    ),
                    ("Avg latency", format_seconds(performance_info.get("average_latency_seconds")), ""),
                ]
            ),
            unsafe_allow_html=True,
        )

        # --- answer quality / abstention
        quality_column, abstention_column = st.columns(2)

        bars = []
        for title, metric_key in (
            ("Correctness", "correctness_counts"),
            ("Faithfulness", "faithfulness_counts"),
            ("Citations", "citation_counts"),
        ):
            if evaluation_report.has_dict_metric(eval_report, "answers", metric_key):
                bars.append(bar_html(title, list(eval_report["answers"][metric_key].items())))
            else:
                bars.append(f'<p class="rd-note">{esc(title)}: not available (retrieval-only report).</p>')

        quality_column.markdown(
            card_html(
                "Answer quality",
                "".join(bars),
                "Correctness and faithfulness need a real generated answer, so they exist only for "
                "full-generation reports. Citation correctness only needs retrieved source metadata.",
            ),
            unsafe_allow_html=True,
        )

        if has_abstention:
            abstention_body = kv_rows_html(
                [
                    ("Expected abstentions", abstention_info["expected_abstentions"]),
                    ("Correct abstentions", abstention_info["correct_abstentions"]),
                    ("Missed abstentions", abstention_info["missed_abstentions"]),
                    ("Unexpected abstentions", abstention_info["unexpected_abstentions"]),
                    (
                        "Abstention accuracy",
                        f"{abstention_accuracy:.0%}" if abstention_accuracy is not None else "N/A",
                    ),
                ]
            )
        else:
            abstention_body = (
                '<p class="rd-note">Abstention metrics are not available for this report -- it was run '
                "in retrieval-only mode, so there are no generated answers to check for abstention.</p>"
            )

        abstention_column.markdown(card_html("Abstention", abstention_body), unsafe_allow_html=True)

        # --- retrieval metrics / performance
        retrieval_column, performance_column = st.columns(2)

        retrieval_metrics = [
            (retrieval_metric_label(key), format_score(value))
            for key, value in retrieval_info.items()
            if key != "questions_evaluated"
        ]
        retrieval_column.markdown(
            card_html(
                "Retrieval metrics",
                kv_rows_html(retrieval_metrics),
                f"Computed over {retrieval_info.get('questions_evaluated', 'n/a')} answerable "
                "questions with known supporting sources.",
            ),
            unsafe_allow_html=True,
        )

        models = ", ".join(performance_info.get("model_names_used", [])) or "n/a"
        performance_column.markdown(
            card_html(
                "Performance",
                kv_rows_html(
                    [
                        ("Average latency", format_seconds(performance_info.get("average_latency_seconds"))),
                        ("Median latency", format_seconds(performance_info.get("median_latency_seconds"))),
                        ("Min latency", format_seconds(performance_info.get("min_latency_seconds"))),
                        ("Max latency", format_seconds(performance_info.get("max_latency_seconds"))),
                        ("Model(s)", models),
                    ]
                ),
                evaluation_report.token_usage_message(performance_info),
            ),
            unsafe_allow_html=True,
        )

        # --- dataset
        if dataset_info:
            documents_text = ", ".join(dataset_info["documents_represented"]) or "none"
            st.markdown(
                card_html(
                    "Dataset by category",
                    chips_html(dataset_info["by_category"]) + f'<p class="rd-note">Documents: {esc(documents_text)}</p>',
                ),
                unsafe_allow_html=True,
            )

        # --- failures
        failures = eval_report.get("failures", [])
        st.markdown(
            section_title_html("Failed questions", f"{len(failures)} in this report"),
            unsafe_allow_html=True,
        )

        if not failures:
            st.success("No failures recorded in this report.")
        else:
            st.markdown("".join(failure_row_html(failure) for failure in failures), unsafe_allow_html=True)

    # --- experiment comparison (independent of the selected report)
    st.markdown(
        section_title_html("Experiment comparison", "the five required Phase 5 configurations"),
        unsafe_allow_html=True,
    )
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
