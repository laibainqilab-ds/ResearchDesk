"""Unit tests for app/observability.py: trace ID generation and structured
JSON logging. Uses caplog targeted at the "researchdesk" logger directly
(via `caplog.at_level(..., logger="researchdesk")`) so these tests work
regardless of whether configure_logging() has run elsewhere in the same
pytest session, and never touch the real logs/ directory.
"""

import json
import logging

import pytest

from app import observability


LOGGER_NAME = observability.LOGGER_NAME


def test_new_trace_id_returns_unique_values():
    first = observability.new_trace_id()
    second = observability.new_trace_id()

    assert first != second


def test_new_trace_id_is_a_plain_uuid4_hex_string():
    trace_id = observability.new_trace_id()

    # Plain UUID4, no distributed-tracing infrastructure: 32 hex characters,
    # no dashes (uuid.uuid4().hex).
    assert len(trace_id) == 32
    assert all(character in "0123456789abcdef" for character in trace_id)


def test_log_event_emits_valid_json_with_trace_id_and_event(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        observability.log_event("trace-123", "unit_test_event", some_field="value")

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)

    assert payload["trace_id"] == "trace-123"
    assert payload["event"] == "unit_test_event"
    assert payload["some_field"] == "value"
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_log_event_respects_warning_level(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        observability.log_event("trace-456", "unit_test_warning", level=logging.WARNING)

    assert caplog.records[0].levelno == logging.WARNING
    payload = json.loads(caplog.records[0].message)
    assert payload["level"] == "WARNING"


def test_log_event_serializes_non_json_native_fields_without_raising(caplog):
    class Unserializable:
        def __str__(self):
            return "custom-repr"

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        observability.log_event("trace-789", "unit_test_object_field", weird=Unserializable())

    payload = json.loads(caplog.records[0].message)
    assert payload["weird"] == "custom-repr"


def test_log_event_works_without_trace_id():
    # Must not raise even if a caller doesn't have a trace_id yet.
    observability.log_event(None, "no_trace_event")


@pytest.fixture
def isolated_observability_state():
    """Save/restore app.observability's module-level logging state so a
    test that calls configure_logging() doesn't leak a file handler or a
    disabled-propagation flag into other tests in the same pytest session.
    """
    original_configured = observability._configured
    original_handlers = list(observability._logger.handlers)
    original_propagate = observability._logger.propagate
    original_level = observability._logger.level

    yield

    for handler in observability._logger.handlers:
        if handler not in original_handlers:
            observability._logger.removeHandler(handler)
            handler.close()

    observability._configured = original_configured
    observability._logger.propagate = original_propagate
    observability._logger.setLevel(original_level)


def test_configure_logging_creates_log_file(tmp_path, isolated_observability_state):
    observability._configured = False
    log_file = tmp_path / "nested" / "researchdesk.jsonl"

    observability.configure_logging(log_file=log_file)
    observability.log_event("trace-file", "file_creation_test")

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "file_creation_test"


def test_configure_logging_is_idempotent(tmp_path, isolated_observability_state):
    observability._configured = False
    log_file = tmp_path / "researchdesk.jsonl"

    observability.configure_logging(log_file=log_file)
    handlers_after_first_call = len(observability._logger.handlers)

    observability.configure_logging(log_file=log_file)
    handlers_after_second_call = len(observability._logger.handlers)

    assert handlers_after_second_call == handlers_after_first_call
