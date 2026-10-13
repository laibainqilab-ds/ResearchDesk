"""Pure, Streamlit-free helpers that build the small HTML pieces used by every
screen (pills, ID chips, cards, steppers, bars), so the whole app presents
information the same way.

Kept separate from app/streamlit_app.py so they're unit-testable (importing
that file executes top-level st.* calls). Everything dynamic -- file names,
questions, model answers, document text -- is HTML-escaped here, because the
results are rendered with unsafe_allow_html=True.
"""

import html
import re
from datetime import datetime

_MODE_LABELS = {"rag": "Standard RAG", "agent": "Multi-agent"}
_CITATION = re.compile(r"\[(\d+)\]")

_LOGO_SVG = (
    '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'
    '<path d="M9 14h6M9 17.5h4"/></svg>'
)
_CHECK_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 4.5 4.5L19 7"/></svg>'
)
_WARN_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4 3 20h18z"/>'
    '<path d="M12 10v4M12 17h.01"/></svg>'
)
_CROSS_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>'
)
_CHEVRON_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'
)

# stage -> (display name, one-line description). The descriptions say what
# each step of the real workflow does (see app/agents/*).
_STAGES = {
    "router": ("Router", "Chooses a route for the question"),
    "retrieval": ("Retrieval", "Searches your indexed documents"),
    "research_decompose": ("Research: split", "Breaks a complex question into sub-questions"),
    "research_retrieve": ("Research: search", "Searches the documents for each sub-question"),
    "research_sufficiency": ("Research: check", "Checks whether the evidence is enough"),
    "answer": ("Answer", "Drafts a cited answer from the evidence"),
    "validation": ("Validation", "Checks citations and support (one retry allowed)"),
}

_RETRIEVAL_METRIC_LABELS = {
    "recall_at_3": "Recall@3",
    "precision_at_3": "Precision@3",
    "hit_rate_at_3": "Hit rate@3",
    "recall_at_5": "Recall@5",
    "precision_at_5": "Precision@5",
    "hit_rate_at_5": "Hit rate@5",
    "mrr": "MRR",
}


# ----------------------------------------------------------------- basics

def esc(value: object) -> str:
    """HTML-escape any value ("" for None) and flatten line breaks to spaces.

    Streamlit renders these snippets through Markdown, which ends an HTML block
    at the first blank line -- so dynamic text must never carry a raw newline.
    """
    text = html.escape("" if value is None else str(value))

    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def esc_pre(value: object) -> str:
    """Escape text for a <pre> block, keeping its line breaks as &#10; so the
    snippet itself stays on one line (see esc)."""
    text = html.escape("" if value is None else str(value))

    return text.replace("\r\n", "&#10;").replace("\n", "&#10;").replace("\r", "&#10;")


def short_id(value: object, head: int = 8, tail: int = 4) -> str:
    """Compact form of a long ID, e.g. "07fb10a9…3308".

    IDs of head + tail characters or fewer are returned unchanged; a
    missing value gives "". The full ID stays available as a tooltip.
    """
    if not value:
        return ""

    text = str(value)

    if len(text) <= head + tail:
        return text

    return f"{text[:head]}…{text[-tail:]}"


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value

    if not isinstance(value, str) or not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_date(value: object) -> str:
    """"Oct 13, 2026" from an ISO timestamp; "" if it can't be read."""
    parsed = _parse_datetime(value)

    return f"{parsed:%b} {parsed.day}, {parsed.year}" if parsed else ""


def format_run_time(value: object) -> str:
    """"Oct 13, 10:00" from an ISO timestamp; the raw text if it can't be read."""
    parsed = _parse_datetime(value)

    if not parsed:
        return "" if value is None else str(value)

    return f"{parsed:%b} {parsed.day}, {parsed:%H:%M}"


def pill_html(text: object) -> str:
    return f'<span class="rd-pill">{esc(text)}</span>'


def id_chip_html(label: str, full_id: object, tooltip_label: str | None = None) -> str:
    """A monospace chip such as "Trace 07fb10a9…3308" with the full ID in
    the tooltip. Returns "" when there is no ID."""
    if not full_id:
        return ""

    tooltip = esc(f"{tooltip_label or label}: {full_id}")

    return f'<span class="rd-trace" title="{tooltip}">{esc(label)} {esc(short_id(full_id))}</span>'


def meta_row_html(parts: list[str]) -> str:
    """One quiet row of pills/chips. Empty parts are skipped; no parts gives ""."""
    inner = "".join(part for part in parts if part)

    return f'<div class="rd-meta">{inner}</div>' if inner else ""


def _count_label(count: object, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")

    return f"{count} {word}"


def display_chat_title(chat: dict) -> str:
    """The chat's title as shown in the sidebar and topbar.

    New chats are persisted with a timestamp-stamped placeholder (see
    ChatService.create_chat) that's already distinguishable as-is. This
    only reformats chats saved before that existed, which still carry the
    literal "New chat" ChatCreateRequest default -- those fall back to
    their (already-persisted) created_at so multiple old empty chats don't
    all display identically, without needing a data migration.
    """
    if chat["title"] == "New chat":
        return f"Untitled chat — {format_run_time(chat['created_at'])}"
    return chat["title"]


def mode_label(mode: str | None) -> str:
    return _MODE_LABELS.get(mode or "", mode or "n/a")


def message_meta_html(mode: str | None, trace_id: str | None) -> str:
    """Row under an assistant answer: mode pill plus shortened trace ID."""
    return meta_row_html([pill_html(mode_label(mode)), id_chip_html("Trace", trace_id, "Trace ID")])


# ------------------------------------------------------------ brand / chrome

def sidebar_brand_html() -> str:
    return (
        '<div class="rd-brand-row"><div class="rd-logo">' + _LOGO_SVG.format(size=18) + "</div>"
        '<div><div class="rd-brand">ResearchDesk</div>'
        '<div class="rd-tagline">Document research assistant</div></div></div>'
    )


def auth_brand_html() -> str:
    return (
        '<div class="rd-auth-brand"><div class="rd-logo rd-logo-lg">' + _LOGO_SVG.format(size=24) + "</div>"
        '<div class="rd-auth-name">ResearchDesk</div>'
        '<div class="rd-tagline">Document research assistant</div></div>'
    )


def user_chip_html(email: str | None) -> str:
    initial = (email or "?").strip()[:1].upper() or "?"

    return (
        f'<div class="rd-user-chip"><div class="rd-avatar">{esc(initial)}</div>'
        f'<div class="rd-user-text"><div class="rd-k">Signed in as</div>'
        f'<div class="rd-email">{esc(email)}</div></div></div>'
    )


def section_label_html(text: str) -> str:
    return f'<div class="rd-section">{esc(text)}</div>'


def topbar_title_html(title: str, subtitle: str = "") -> str:
    sub = f'<span class="rd-subtitle">{esc(subtitle)}</span>' if subtitle else ""

    return f'<div class="rd-topbar-title"><span class="rd-title">{esc(title)}</span>{sub}</div>'


def section_title_html(title: str, note: str = "") -> str:
    right = f'<span class="rd-note">{esc(note)}</span>' if note else ""

    return f'<div class="rd-section-title"><h3>{esc(title)}</h3>{right}</div>'


# ------------------------------------------------------------------- chat

def user_bubble_html(text: object) -> str:
    escaped = html.escape("" if text is None else str(text))
    escaped = escaped.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")

    return f'<div class="rd-user-row"><div class="rd-user">{escaped}</div></div>'


def answer_markdown(text: str | None) -> str:
    """The model's answer, made safe to render with unsafe_allow_html.

    "<" is escaped so no HTML tag in the model's output (or in a document it
    echoed) can take effect; Markdown formatting is untouched. The [1], [2]
    markers the prompt asks for become small citation chips.
    """
    safe = (text or "").replace("<", "&lt;")

    return _CITATION.sub(lambda match: f'<span class="rd-cite">{match.group(1)}</span>', safe)


def source_chips_html(sources: list[dict] | None) -> str:
    """One chip per cited source: number, file, page. Chunk and relevance
    are in the tooltip."""
    chips = []

    for source in sources or []:
        number = source.get("citation_id")
        name = source.get("filename") or "Unknown file"
        page = source.get("page_number")
        location = f"{name} · p. {page}" if page is not None else name

        tooltip = f"chunk {source.get('chunk_id', 'unknown')}"
        score = source.get("rerank_score")

        if isinstance(score, (int, float)):
            tooltip += f" · relevance {score:.2f}"

        number_html = f"<b>{esc(number)}</b>" if number is not None else ""
        chips.append(f'<span class="rd-source" title="{esc(tooltip)}">{number_html}{esc(location)}</span>')

    return f'<div class="rd-sources">{"".join(chips)}</div>' if chips else ""


# -------------------------------------------------------------- documents

def file_badge_html(file_type: str | None) -> str:
    label = (file_type or "").strip().upper()
    css = {"PDF": "pdf", "TXT": "txt", "MD": "md", "MARKDOWN": "md"}.get(label, "other")

    return f'<div class="rd-filebadge rd-ft-{css}">{esc(label[:4] or "FILE")}</div>'


def document_meta_html(
    file_type: str | None,
    page_count: int | None,
    chunk_count: int,
    document_id: str | None,
) -> str:
    """Row under a document's name: type, pages, chunks and a short ID."""
    parts = [pill_html(file_type or "UNKNOWN")]

    if page_count:
        parts.append(pill_html(_count_label(page_count, "page")))

    parts.append(pill_html(_count_label(chunk_count, "chunk")))
    parts.append(id_chip_html("ID", document_id, "Document ID"))

    return meta_row_html(parts)


def document_title_html(filename: str | None) -> str:
    return f'<div class="rd-doc-name">{esc(filename or "Unknown file")}</div>'


# ---------------------------------------------------- inspector / trace pages

def _field(label: str, value_html: str) -> str:
    return f'<div class="rd-field"><div class="rd-k">{esc(label)}</div><div class="rd-v">{value_html}</div></div>'


def run_summary_html(question: object, rewritten: object | None, mode: str | None, trace_id: str | None) -> str:
    """The card at the top of a run: question, (rewritten question), mode, trace ID."""
    fields = [_field("Question", esc(question) or "n/a")]

    if rewritten is not None:
        fields.append(_field("Rewritten as", esc(rewritten) or "n/a"))

    fields.append(_field("Mode", pill_html(mode_label(mode))))
    fields.append(_field("Trace ID", f'<span class="rd-trace rd-wrap" title="Trace ID">{esc(trace_id) or "n/a"}</span>'))

    return f'<div class="rd-card rd-summary">{"".join(fields)}</div>'


def pipeline_html(n_queries: int, n_candidates: int, n_evidence: int, queries: list[str]) -> str:
    """How the answer was found: the retrieval steps with their counts, then the queries used."""
    arrow = '<span class="rd-arrow">›</span>'
    steps = [
        ("Question", False),
        ("Rewritten", False),
        (_count_label(n_queries, "search query", "search queries"), False),
        (_count_label(n_candidates, "candidate"), False),
        ("Reranked", False),
        (f"{n_evidence} final evidence", True),
    ]
    row = arrow.join(f'<span class="rd-step{" rd-step-strong" if strong else ""}">{esc(text)}</span>' for text, strong in steps)
    chips = "".join(pill_html(query) for query in queries)
    chips_html = f'<div class="rd-meta rd-queries">{chips}</div>' if chips else ""

    return (
        '<div class="rd-card"><div class="rd-k">How this answer was found</div>'
        f'<div class="rd-pipeline">{row}</div>{chips_html}</div>'
    )


def score_badge_html(label: str, value: str, accent: bool = False) -> str:
    css = "rd-badge rd-badge-accent" if accent else "rd-badge"

    return f'<span class="{css}">{esc(label)} {esc(value)}</span>'


def result_row_html(
    rank: int,
    filename: str | None,
    location: str,
    badges: list[str],
    passage: str | None = None,
    note: str | None = None,
    open_: bool = False,
) -> str:
    """One retrieved/cited passage. With a passage it expands (a native
    <details>); without one it is a plain row."""
    head = (
        f'<span class="rd-rank">{esc(rank)}</span>'
        f'<span class="rd-result-name"><span class="rd-fname">{esc(filename or "Unknown file")}</span> '
        f'<span class="rd-loc">{esc(location)}</span></span>'
        f'<span class="rd-badges">{"".join(badges)}</span>'
    )

    if passage is None and note is None:
        return f'<div class="rd-result rd-result-static">{head}</div>'

    body = ""

    if note:
        body += f'<div class="rd-k">{esc(note)}</div>'

    if passage is not None:
        body += f'<div class="rd-k">Passage text</div><pre class="rd-passage">{esc_pre(passage)}</pre>'

    open_attr = " open" if open_ else ""

    return (
        f'<details class="rd-result"{open_attr}><summary>{head}<span class="rd-chevron">{_CHEVRON_SVG}</span></summary>'
        f'<div class="rd-result-body">{body}</div></details>'
    )


def stage_info(stage: object) -> tuple[str, str]:
    """(display name, description) for an agent trace stage."""
    known = _STAGES.get(str(stage))

    return known if known else (str(stage).replace("_", " ").title(), "")


def _status_html(status: object) -> str:
    text = str(status or "")

    if text == "ok":
        return f'<span class="rd-status rd-st-ok" title="Completed">{_CHECK_SVG}</span>'

    if text == "failed":
        return f'<span class="rd-status rd-st-bad" title="Failed">{_CROSS_SVG}<span>failed</span></span>'

    label = {"fallback": "fallback used", "no_evidence": "no evidence"}.get(text, text or "unknown")

    return f'<span class="rd-status rd-st-warn" title="{esc(label)}">{_WARN_SVG}<span>{esc(label)}</span></span>'


def _truncate(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def tools_used(entries: list[dict] | None) -> list[str]:
    """Every tool used across the trace, in first-use order, without duplicates."""
    seen: list[str] = []

    for entry in entries or []:
        for tool in entry.get("tools_used") or []:
            if tool not in seen:
                seen.append(tool)

    return seen


def stepper_html(entries: list[dict] | None) -> str:
    """The agent run as a numbered vertical stepper. "" when there are no entries."""
    if not entries:
        return ""

    steps = []

    for index, entry in enumerate(entries, start=1):
        name, description = stage_info(entry.get("stage"))
        duration = entry.get("duration_seconds")
        decision = entry.get("decision")
        tags = ""

        if decision:
            tags += f'<span class="rd-badge" title="{esc(decision)}">{esc(_truncate(str(decision)))}</span>'

        for tool in entry.get("tools_used") or []:
            tags += f'<span class="rd-badge rd-badge-tool">{esc(tool)}</span>'

        duration_html = f'<span class="rd-duration">{duration:.1f}s</span>' if isinstance(duration, (int, float)) else ""
        description_html = f'<div class="rd-step-desc">{esc(description)}</div>' if description else ""

        steps.append(
            f'<div class="rd-trace-step"><div class="rd-trace-rail"><span class="rd-step-num">{index}</span>'
            f'<span class="rd-rail-line"></span></div>'
            f'<div class="rd-trace-main"><div class="rd-trace-head">'
            f'<div class="rd-trace-title"><div class="rd-step-name">{esc(name)}</div>{description_html}</div>'
            f'<div class="rd-trace-tags">{tags}</div>{duration_html}{_status_html(entry.get("status"))}</div></div></div>'
        )

    return f'<div class="rd-card rd-stepper">{"".join(steps)}</div>'


def tools_details_html(entries: list[dict] | None) -> str:
    tools = tools_used(entries)
    body = "".join(f'<span class="rd-badge rd-badge-tool">{esc(tool)}</span>' for tool in tools)

    if not body:
        body = '<span class="rd-note">No tools were used in this run.</span>'

    return (
        '<details class="rd-result rd-tools"><summary><span class="rd-fname">Tools used</span>'
        f'<span class="rd-chevron">{_CHEVRON_SVG}</span></summary>'
        f'<div class="rd-result-body rd-badges-wrap">{body}</div></details>'
    )


def validation_html(is_valid: bool | None) -> str:
    if is_valid is None:
        return '<span class="rd-v">n/a</span>'

    css, text = ("rd-dot-ok", "Valid") if is_valid else ("rd-dot-bad", "Invalid")

    return f'<span class="rd-validation"><span class="rd-dot {css}"></span>{text}</span>'


def tile_html(label: str, value_html: str) -> str:
    return f'<div class="rd-card rd-tile"><div class="rd-k">{esc(label)}</div><div class="rd-tile-v">{value_html}</div></div>'


def tiles_html(tiles: list[tuple[str, str]]) -> str:
    return f'<div class="rd-tiles">{"".join(tile_html(label, value) for label, value in tiles)}</div>'


# ------------------------------------------------------------- evaluation

def retrieval_metric_label(key: str) -> str:
    return _RETRIEVAL_METRIC_LABELS.get(key, key)


def kpi_tile_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="rd-kpi-sub">{esc(sub)}</div>' if sub else ""

    return f'<div class="rd-card rd-kpi"><div class="rd-k">{esc(label)}</div><div class="rd-kpi-v">{esc(value)}</div>{sub_html}</div>'


def kpis_html(items: list[tuple[str, str, str]]) -> str:
    return f'<div class="rd-kpis">{"".join(kpi_tile_html(*item) for item in items)}</div>'


def card_html(title: str, inner_html: str, note: str = "") -> str:
    note_html = f'<p class="rd-card-note">{esc(note)}</p>' if note else ""

    return f'<div class="rd-card"><h3 class="rd-card-title">{esc(title)}</h3>{note_html}{inner_html}</div>'


def kv_rows_html(pairs: list[tuple[str, object]]) -> str:
    rows = "".join(f'<div class="rd-kv"><span>{esc(key)}</span><strong>{esc(value)}</strong></div>' for key, value in pairs)

    return f'<div class="rd-kvs">{rows}</div>'


def bar_html(title: str, parts: list[tuple[str, int]]) -> str:
    """A proportional bar with a legend. Zero-count parts are left out of the
    bar but kept in the legend, so every number is still visible."""
    total = sum(count for _, count in parts) or 1
    segments = "".join(
        f'<span class="rd-seg rd-seg-{index % 4}" style="width:{100 * count / total:.1f}%" title="{esc(label)} {esc(count)}"></span>'
        for index, (label, count) in enumerate(parts)
        if count
    )
    legend = "".join(
        f'<span class="rd-legend"><span class="rd-swatch rd-seg-{index % 4}"></span>{esc(label)} {esc(count)}</span>'
        for index, (label, count) in enumerate(parts)
    )

    return (
        f'<div class="rd-bar-block"><div class="rd-bar-title">{esc(title)}</div>'
        f'<div class="rd-bar">{segments}</div><div class="rd-legends">{legend}</div></div>'
    )


def chips_html(counts: dict) -> str:
    chips = "".join(f'<span class="rd-chip">{esc(key)}<strong>{esc(value)}</strong></span>' for key, value in counts.items())

    return f'<div class="rd-chips">{chips}</div>'


def failure_row_html(failure: dict) -> str:
    reasons = ", ".join(str(reason) for reason in failure.get("reasons") or []) or "no reason recorded"

    return (
        '<details class="rd-result"><summary>'
        f'<span class="rd-trace">{esc(failure.get("id"))}</span>{pill_html(failure.get("category"))}'
        f'<span class="rd-result-name rd-fail-q">{esc(failure.get("question"))}</span>'
        f'<span class="rd-chevron">{_CHEVRON_SVG}</span></summary>'
        f'<div class="rd-result-body"><div class="rd-k">Reasons</div>{esc(reasons)}</div></details>'
    )
