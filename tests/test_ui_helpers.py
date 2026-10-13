from datetime import datetime

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
    esc_pre,
    failure_row_html,
    file_badge_html,
    format_date,
    format_run_time,
    id_chip_html,
    kpis_html,
    kv_rows_html,
    message_meta_html,
    meta_row_html,
    mode_label,
    pill_html,
    pipeline_html,
    result_row_html,
    retrieval_metric_label,
    run_summary_html,
    score_badge_html,
    short_id,
    sidebar_brand_html,
    source_chips_html,
    stage_info,
    stepper_html,
    tools_details_html,
    tools_used,
    topbar_title_html,
    user_bubble_html,
    user_chip_html,
    validation_html,
)

ATTACK = '<script>alert(1)</script><img src=x onerror=alert(1)>"\''


def assert_inert(markup: str) -> None:
    """None of ATTACK's tags or attribute break-outs survived as live HTML."""
    assert "<script" not in markup
    assert "<img" not in markup
    assert "onerror=alert(1)>" not in markup


# ------------------------------------------------------------------ basics

def test_esc_handles_none_and_escapes():
    assert esc(None) == ""
    assert esc("<b>") == "&lt;b&gt;"
    assert esc(5) == "5"


def test_short_id():
    assert short_id("07fb10a9ce2b48e59e53e28543fe3308") == "07fb10a9…3308"
    assert short_id("abc123") == "abc123"
    assert short_id("a" * 12) == "a" * 12
    assert short_id(None) == ""
    assert short_id("") == ""


def test_format_date_reads_iso_values():
    assert format_date("2026-10-13T10:00:00") == "Oct 13, 2026"
    assert format_date("2026-10-05T10:00:00Z") == "Oct 5, 2026"
    assert format_date("2026-10-05T10:00:00+00:00") == "Oct 5, 2026"
    assert format_date(datetime(2026, 1, 2, 3, 4)) == "Jan 2, 2026"


def test_format_date_is_empty_for_unreadable_values():
    assert format_date(None) == ""
    assert format_date("") == ""
    assert format_date("not a date") == ""
    assert format_date(12345) == ""


def test_format_run_time():
    assert format_run_time("2026-10-13T10:05:00") == "Oct 13, 10:05"
    assert format_run_time(None) == ""
    assert format_run_time("garbage") == "garbage"


def test_display_chat_title_passes_through_a_real_title():
    chat = {"title": "What is Evo 2?", "created_at": "2026-10-14T10:05:00"}
    assert display_chat_title(chat) == "What is Evo 2?"


def test_display_chat_title_formats_legacy_new_chat_rows_using_created_at():
    chat = {"title": "New chat", "created_at": "2026-10-14T10:05:00"}
    assert display_chat_title(chat) == "Untitled chat — Oct 14, 10:05"


def test_display_chat_title_distinguishes_legacy_chats_created_at_different_times():
    first = {"title": "New chat", "created_at": "2026-10-14T10:05:00"}
    second = {"title": "New chat", "created_at": "2026-10-14T10:06:00"}
    assert display_chat_title(first) != display_chat_title(second)


def test_pill_and_chip_escape():
    assert pill_html("<b>x</b>") == '<span class="rd-pill">&lt;b&gt;x&lt;/b&gt;</span>'
    assert_inert(id_chip_html("Trace", ATTACK))


def test_id_chip_shows_short_id_with_full_id_in_tooltip():
    chip = id_chip_html("Trace", "07fb10a9ce2b48e59e53e28543fe3308", "Trace ID")

    assert "Trace 07fb10a9…3308" in chip
    assert 'title="Trace ID: 07fb10a9ce2b48e59e53e28543fe3308"' in chip
    assert id_chip_html("Trace", None) == ""


def test_meta_row_skips_empty_parts():
    assert meta_row_html(["", "<span>a</span>", ""]) == '<div class="rd-meta"><span>a</span></div>'
    assert meta_row_html([]) == ""
    assert meta_row_html(["", ""]) == ""


def test_mode_label_and_message_meta():
    assert mode_label("rag") == "Standard RAG"
    assert mode_label("agent") == "Multi-agent"
    assert mode_label(None) == "n/a"
    assert mode_label("custom") == "custom"

    markup = message_meta_html("rag", "07fb10a9ce2b48e59e53e28543fe3308")
    assert "Standard RAG" in markup and "Trace 07fb10a9…3308" in markup
    assert "Trace" not in message_meta_html("rag", None)
    assert_inert(message_meta_html(ATTACK, ATTACK))


# ------------------------------------------------------------ brand / chrome

def test_brand_and_chrome_pieces_render():
    assert "ResearchDesk" in sidebar_brand_html()
    assert "ResearchDesk" in auth_brand_html()
    assert "Document research assistant" in sidebar_brand_html()


def test_user_chip_uses_initial_and_escapes():
    chip = user_chip_html("laiba@example.com")
    assert ">L<" in chip and "laiba@example.com" in chip
    assert ">?<" in user_chip_html(None)
    assert_inert(user_chip_html(ATTACK))


def test_topbar_and_section_titles_escape():
    assert "Chat" in topbar_title_html("Chat", "my chat")
    assert "rd-subtitle" not in topbar_title_html("Chat")
    assert_inert(topbar_title_html(ATTACK, ATTACK))


# -------------------------------------------------------------------- chat

def test_user_bubble_escapes_and_keeps_line_breaks():
    markup = user_bubble_html("line one\nline two")
    assert "line one<br>line two" in markup
    assert_inert(user_bubble_html(ATTACK))


def test_answer_markdown_turns_markers_into_citation_chips():
    markup = answer_markdown("Evo 2 scores variants [1] and generates DNA [2][3].")

    assert markup.count('class="rd-cite"') == 3
    assert '<span class="rd-cite">1</span>' in markup
    assert "[1]" not in markup


def test_answer_markdown_neutralises_html_but_keeps_markdown():
    markup = answer_markdown("**bold** and - list\n" + ATTACK)

    assert "**bold**" in markup and "- list" in markup
    # Only "<" is escaped (so Markdown such as "> quotes" still works). Without a
    # "<" no tag can open, which is the property that matters.
    assert "<script" not in markup and "<img" not in markup
    assert "&lt;script>" in markup and "&lt;img" in markup


def test_answer_markdown_handles_empty():
    assert answer_markdown(None) == ""
    assert answer_markdown("") == ""


def test_source_chips_show_number_file_page_and_tooltip():
    markup = source_chips_html(
        [{"citation_id": 1, "filename": "paper.pdf", "page_number": 4, "chunk_id": 3, "rerank_score": 0.9123}]
    )

    assert "<b>1</b>paper.pdf · p. 4" in markup
    assert 'title="chunk 3 · relevance 0.91"' in markup


def test_source_chips_handle_missing_fields_and_escape():
    markup = source_chips_html([{"filename": None, "page_number": None}])
    assert "Unknown file" in markup and "<b>" not in markup

    assert source_chips_html([]) == ""
    assert source_chips_html(None) == ""
    assert_inert(source_chips_html([{"citation_id": ATTACK, "filename": ATTACK, "chunk_id": ATTACK}]))


# --------------------------------------------------------------- documents

def test_file_badge_classes():
    assert "rd-ft-pdf" in file_badge_html("pdf")
    assert "rd-ft-txt" in file_badge_html("TXT")
    assert "rd-ft-md" in file_badge_html("md")
    assert "rd-ft-other" in file_badge_html("docx")
    assert ">FILE<" in file_badge_html(None)
    assert_inert(file_badge_html(ATTACK))


def test_document_meta_shows_type_pages_chunks_and_short_id():
    markup = document_meta_html("PDF", 12, 34, "f79b5fe2" + "0" * 48)

    assert ">PDF<" in markup and "12 pages" in markup and "34 chunks" in markup
    assert "ID f79b5fe2…" in markup


def test_document_meta_singular_and_missing_values():
    markup = document_meta_html(None, None, 1, None)
    assert "UNKNOWN" in markup and "1 chunk<" in markup
    assert "page" not in markup and "ID" not in markup
    assert "1 page<" in document_meta_html("PDF", 1, 2, "abc")


def test_document_title_escapes_and_defaults():
    assert "Unknown file" in document_title_html(None)
    assert_inert(document_title_html(ATTACK))


# ------------------------------------------------------ inspector / trace

def test_run_summary_with_and_without_rewritten_question():
    with_rewrite = run_summary_html("what is evo 2", "What is Evo 2?", "rag", "5c91e2b7aaaabbbbccccdddd0000a04d")
    assert "Rewritten as" in with_rewrite and "What is Evo 2?" in with_rewrite
    assert "5c91e2b7aaaabbbbccccdddd0000a04d" in with_rewrite  # full ID stays visible

    assert "Rewritten as" not in run_summary_html("q", None, "agent", "t")
    assert "n/a" in run_summary_html(None, None, None, None)
    assert_inert(run_summary_html(ATTACK, ATTACK, ATTACK, ATTACK))


def test_pipeline_counts_pluralise_and_escape():
    one = pipeline_html(1, 1, 1, ["only query"])
    assert "1 search query<" in one and "1 candidate<" in one

    many = pipeline_html(2, 4, 2, ["a", "b"])
    assert "2 search queries" in many and "4 candidates" in many and "2 final evidence" in many

    assert "rd-queries" not in pipeline_html(0, 0, 0, [])
    assert_inert(pipeline_html(1, 1, 1, [ATTACK]))


def test_result_row_static_vs_expandable():
    static = result_row_html(1, "paper.pdf", "page 4 · chunk 3", [score_badge_html("rerank", "0.91", True)])
    assert "<details" not in static and "rd-result-static" in static

    expandable = result_row_html(2, "paper.pdf", "page 4", [], passage="some passage", note="Search query: evo")
    assert "<details" in expandable and "some passage" in expandable and "Search query: evo" in expandable
    assert " open" not in expandable.split(">")[0]
    assert "<details class=\"rd-result\" open>" in result_row_html(1, "f", "l", [], passage="p", open_=True)


def test_result_row_escapes_everything_dynamic():
    markup = result_row_html(ATTACK, ATTACK, ATTACK, [], passage=ATTACK, note=ATTACK)
    assert_inert(markup)


def test_score_badge_accent_class():
    assert "rd-badge-accent" in score_badge_html("rerank", "0.91", True)
    assert "rd-badge-accent" not in score_badge_html("distance", "0.2")
    assert_inert(score_badge_html(ATTACK, ATTACK))


def test_stage_info_known_and_unknown():
    assert stage_info("router")[0] == "Router"
    assert stage_info("research_sufficiency")[0] == "Research: check"
    assert stage_info("some_new_stage") == ("Some New Stage", "")


def test_stepper_empty_gives_empty_string():
    assert stepper_html([]) == ""
    assert stepper_html(None) == ""


def test_stepper_shows_every_step_with_status_duration_decision_and_tools():
    entries = [
        {"stage": "router", "status": "ok", "duration_seconds": 0.84, "decision": "document_qa", "tools_used": []},
        {"stage": "retrieval", "status": "ok", "duration_seconds": 1.2, "decision": "2 evidence item(s)", "tools_used": ["search_documents"]},
        {"stage": "answer", "status": "failed", "duration_seconds": None, "decision": None, "tools_used": []},
        {"stage": "validation", "status": "fallback", "duration_seconds": 1.0, "decision": "valid", "tools_used": []},
    ]
    markup = stepper_html(entries)

    for name in ("Router", "Retrieval", "Answer", "Validation"):
        assert name in markup
    assert markup.count('class="rd-step-num"') == 4
    assert "0.8s" in markup and "1.2s" in markup
    assert "document_qa" in markup and "search_documents" in markup
    assert "rd-st-ok" in markup and "rd-st-bad" in markup and "rd-st-warn" in markup
    assert "fallback used" in markup


def test_stepper_handles_no_evidence_unknown_status_and_long_decisions():
    markup = stepper_html([
        {"stage": "answer", "status": "no_evidence", "decision": "x" * 300},
        {"stage": "mystery", "status": "weird"},
    ])
    assert "no evidence" in markup and "weird" in markup
    assert "…" in markup and "x" * 100 not in markup.split('title="')[0]


def test_stepper_escapes_everything_dynamic():
    assert_inert(stepper_html([{"stage": ATTACK, "status": ATTACK, "decision": ATTACK, "tools_used": [ATTACK]}]))


def test_tools_used_is_unique_and_ordered():
    entries = [{"tools_used": ["a", "b"]}, {"tools_used": ["b", "c"]}, {}, {"tools_used": None}]
    assert tools_used(entries) == ["a", "b", "c"]
    assert tools_used(None) == []


def test_tools_details_lists_tools_or_says_none():
    assert "search_documents" in tools_details_html([{"tools_used": ["search_documents"]}])
    assert "No tools were used" in tools_details_html([{"tools_used": []}])
    assert_inert(tools_details_html([{"tools_used": [ATTACK]}]))


def test_validation_html_states():
    assert "Valid" in validation_html(True) and "rd-dot-ok" in validation_html(True)
    assert "Invalid" in validation_html(False) and "rd-dot-bad" in validation_html(False)
    assert "n/a" in validation_html(None)


# -------------------------------------------------------------- evaluation

def test_retrieval_metric_labels():
    assert retrieval_metric_label("recall_at_5") == "Recall@5"
    assert retrieval_metric_label("mrr") == "MRR"
    assert retrieval_metric_label("something_else") == "something_else"


def test_kpis_and_cards_escape():
    markup = kpis_html([("Questions", "60", "50 answerable"), ("MRR", "0.863", "")])
    assert "60" in markup and "50 answerable" in markup and markup.count("rd-kpi-sub") == 1
    assert_inert(kpis_html([(ATTACK, ATTACK, ATTACK)]))
    assert_inert(card_html(ATTACK, "<p>ok</p>", ATTACK))
    assert "<p>ok</p>" in card_html("Title", "<p>ok</p>")


def test_kv_rows_escape():
    assert "Correct" in kv_rows_html([("Correct", 4)]) and "<strong>4</strong>" in kv_rows_html([("Correct", 4)])
    assert_inert(kv_rows_html([(ATTACK, ATTACK)]))


def test_bar_html_widths_and_zero_parts():
    markup = bar_html("Correctness", [("correct", 29), ("partial", 15), ("incorrect", 6), ("unscored", 0)])

    assert markup.count('class="rd-seg rd-seg-') == 3          # zero-count part is not drawn ...
    assert "unscored 0" in markup                                # ... but stays in the legend
    assert "width:58.0%" in markup                               # 29 of 50
    assert "rd-bar" in bar_html("Empty", [("a", 0)])             # never divides by zero
    assert_inert(bar_html(ATTACK, [(ATTACK, 1)]))


def test_chips_and_failure_rows_escape():
    assert "direct" in chips_html({"direct": 19}) and "<strong>19</strong>" in chips_html({"direct": 19})
    assert_inert(chips_html({ATTACK: ATTACK}))

    row = failure_row_html({"id": "q014", "category": "direct", "question": "What?", "reasons": ["incorrect_answer"]})
    assert "q014" in row and "incorrect_answer" in row and "What?" in row
    assert "no reason recorded" in failure_row_html({"id": "q1", "category": "c", "question": "q", "reasons": []})
    assert_inert(failure_row_html({"id": ATTACK, "category": ATTACK, "question": ATTACK, "reasons": [ATTACK]}))


def test_quotes_in_tooltips_cannot_break_out_of_attributes():
    chip = id_chip_html("Trace", 'x" onmouseover="alert(1)')
    assert 'title="Trace: x&quot; onmouseover=&quot;alert(1)"' in chip

    source = source_chips_html([{"chunk_id": 'x" onmouseover="alert(1)'}])
    assert "&quot; onmouseover=&quot;" in source
    assert '" onmouseover="' not in source


def test_dynamic_text_never_puts_a_line_break_inside_a_snippet():
    """Markdown ends an HTML block at a blank line, so every snippet must be one line."""
    nasty = "first line\n\nsecond paragraph\r\nthird"

    assert "\n" not in esc(nasty) and "\r" not in esc(nasty)
    assert "\n" not in esc_pre(nasty) and "&#10;" in esc_pre(nasty)
    assert "<br>" in user_bubble_html(nasty) and "\n" not in user_bubble_html(nasty)

    snippets = [
        result_row_html(1, nasty, nasty, [], passage=nasty, note=nasty, open_=True),
        run_summary_html(nasty, nasty, "rag", nasty),
        pipeline_html(1, 1, 1, [nasty]),
        stepper_html([{"stage": "router", "status": "ok", "decision": nasty, "tools_used": [nasty]}]),
        source_chips_html([{"citation_id": 1, "filename": nasty, "page_number": 1, "chunk_id": nasty}]),
        document_meta_html(nasty, 1, 1, nasty), document_title_html(nasty),
        failure_row_html({"id": nasty, "category": nasty, "question": nasty, "reasons": [nasty]}),
        kv_rows_html([(nasty, nasty)]), chips_html({nasty: nasty}), bar_html(nasty, [(nasty, 1)]),
        card_html(nasty, "x", nasty), kpis_html([(nasty, nasty, nasty)]), tools_details_html([{"tools_used": [nasty]}]),
    ]

    for snippet in snippets:
        assert "\n" not in snippet and "\r" not in snippet
