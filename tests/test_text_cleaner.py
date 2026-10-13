from app.ingestion.text_cleaner import clean_text


def test_collapses_multiple_spaces():
    assert clean_text("Hello    world") == "Hello world"


def test_collapses_newlines():
    assert clean_text("Hello\n\nworld") == "Hello world"


def test_collapses_tabs():
    assert clean_text("Hello\tworld") == "Hello world"


def test_collapses_carriage_returns():
    assert clean_text("Hello\rworld") == "Hello world"


def test_collapses_vertical_tab_and_form_feed():
    # \v and \f are matched by \s, same as the other whitespace variants above.
    assert clean_text("Hello\x0bworld") == "Hello world"
    assert clean_text("Hello\x0cworld") == "Hello world"


def test_collapses_mixed_whitespace_runs():
    assert clean_text("Hello \n\t  world") == "Hello world"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("   Hello world   ") == "Hello world"


def test_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_whitespace_only_string_returns_empty_string():
    assert clean_text("   \n\t  ") == ""


def test_single_word_is_unchanged():
    assert clean_text("word") == "word"


def test_preserves_single_internal_spaces():
    assert clean_text("Hello world") == "Hello world"


def test_does_not_remove_non_whitespace_control_characters():
    # \s only matches whitespace (space/tab/newline/CR/\v/\f); null and other
    # non-whitespace control bytes are not touched by clean_text's regex.
    assert clean_text("Hello\x00world") == "Hello\x00world"
    assert clean_text("Hello\x01\x02world") == "Hello\x01\x02world"
