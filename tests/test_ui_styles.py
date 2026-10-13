import re

from app import ui_helpers as h
from app.ui_styles import AUTH_CSS, CHAT_CSS, GLOBAL_CSS

ALL_CSS = GLOBAL_CSS + CHAT_CSS + AUTH_CSS


def _sample_markup() -> str:
    """One call of every HTML builder, with data that triggers its optional parts."""
    entries = [
        {"stage": "router", "status": "ok", "duration_seconds": 0.8, "decision": "document_qa", "tools_used": []},
        {"stage": "retrieval", "status": "ok", "duration_seconds": 1.0, "decision": "2 evidence", "tools_used": ["search_documents"]},
        {"stage": "answer", "status": "failed", "duration_seconds": 1.0, "decision": None, "tools_used": []},
        {"stage": "validation", "status": "fallback", "duration_seconds": 1.0, "decision": "valid", "tools_used": []},
        {"stage": "answer", "status": "no_evidence", "duration_seconds": 1.0, "decision": None, "tools_used": []},
    ]
    sources = [{"citation_id": 1, "filename": "a.pdf", "page_number": 2, "chunk_id": 3, "rerank_score": 0.9}]

    return "".join([
        h.sidebar_brand_html(), h.auth_brand_html(), h.user_chip_html("a@b.co"), h.section_label_html("Chats"),
        h.topbar_title_html("Chat", "sub"), h.section_title_html("Title", "note"),
        h.user_bubble_html("hi"), h.answer_markdown("x [1]"), h.source_chips_html(sources),
        h.message_meta_html("rag", "abcdefghijklmnop"),
        h.file_badge_html("PDF"), h.file_badge_html("TXT"), h.file_badge_html("MD"), h.file_badge_html("x"),
        h.document_meta_html("PDF", 2, 3, "abcdefghijklmnop"), h.document_title_html("a.pdf"),
        h.run_summary_html("q", "rq", "rag", "tid"), h.pipeline_html(2, 3, 1, ["a"]),
        h.result_row_html(1, "a.pdf", "p1", [h.score_badge_html("r", "0.9", True), h.score_badge_html("d", "0.1")]),
        h.result_row_html(1, "a.pdf", "p1", [], passage="text", note="n", open_=True),
        h.stepper_html(entries), h.tools_details_html(entries), h.tools_details_html([]),
        h.validation_html(True), h.validation_html(False), h.validation_html(None),
        h.tiles_html([("a", "b")]), h.kpis_html([("a", "b", "c")]),
        h.card_html("t", "x", "n"), h.kv_rows_html([("a", 1)]),
        h.bar_html("t", [("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1)]), h.chips_html({"a": 1}),
        h.failure_row_html({"id": "q1", "category": "c", "question": "q", "reasons": ["r"]}),
    ])


def _css_classes(css: str) -> set[str]:
    return set(re.findall(r"\.(rd-[a-z0-9-]+)", css))


def test_every_class_the_helpers_emit_is_styled():
    used = set()

    for value in re.findall(r'class="([^"]+)"', _sample_markup()):
        used.update(name for name in value.split() if name.startswith("rd-"))

    missing = sorted(used - _css_classes(ALL_CSS))

    assert not missing, f"classes emitted by ui_helpers but not defined in ui_styles: {missing}"


def test_stylesheets_are_safe_to_embed_in_a_style_tag():
    for css in (GLOBAL_CSS, CHAT_CSS, AUTH_CSS):
        assert "</style" not in css.lower()
        assert css.count("{") == css.count("}")


def test_accent_matches_the_theme_config():
    config = open(".streamlit/config.toml", encoding="utf-8").read()

    assert "--rd-accent: #3DD6B5" in GLOBAL_CSS and 'primaryColor = "#3DD6B5"' in config
    assert "--rd-bg: #0E1116" in GLOBAL_CSS and 'backgroundColor = "#0E1116"' in config
