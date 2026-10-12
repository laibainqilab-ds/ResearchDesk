"""Tests for app/api_client.py's timeout configuration.

Only covers the one behavior this change touched: that post_message can
request a longer timeout (for slower agent-mode runs) without changing the
default used everywhere else. No network calls are made -- httpx.Client
exposes its configured timeout without connecting.
"""

import httpx

from app import api_client


def test_default_client_uses_default_timeout():
    with api_client._client() as client:
        assert client.timeout == httpx.Timeout(api_client.DEFAULT_TIMEOUT_SECONDS)


def test_client_honors_an_explicit_timeout_override():
    with api_client._client(timeout=300) as client:
        assert client.timeout == httpx.Timeout(300)


def test_post_message_defaults_to_the_standard_timeout_when_unspecified():
    import inspect

    signature = inspect.signature(api_client.post_message)
    assert signature.parameters["timeout_seconds"].default is None


# ---------------------------------------------------------------------------
# post_message's actual timeout selection (timeout_seconds is None, not
# falsy-ness, decides the fallback). No network call is made -- _client is
# replaced with a fake that just records the timeout it was built with.
# ---------------------------------------------------------------------------

class _FakeResponse:
    status_code = 201

    def json(self):
        return {"ok": True}


class _RecordingClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse()


def _patch_client_recorder(monkeypatch):
    captured = {}

    def fake_client(timeout=api_client.DEFAULT_TIMEOUT_SECONDS):
        captured["timeout"] = timeout
        return _RecordingClient(timeout)

    monkeypatch.setattr(api_client, "_client", fake_client)
    return captured


def test_post_message_uses_default_timeout_when_omitted(monkeypatch):
    captured = _patch_client_recorder(monkeypatch)

    api_client.post_message("token", "chat-1", "hello", "rag")

    assert captured["timeout"] == api_client.DEFAULT_TIMEOUT_SECONDS


def test_post_message_respects_an_explicit_timeout(monkeypatch):
    captured = _patch_client_recorder(monkeypatch)

    api_client.post_message("token", "chat-1", "hello", "agent", timeout_seconds=300)

    assert captured["timeout"] == 300


def test_post_message_respects_a_falsy_but_explicit_timeout(monkeypatch):
    """Regression test for the `or` vs `is None` bug: 0 is falsy but a
    meaningful, explicitly-supplied value -- it must not silently fall back
    to DEFAULT_TIMEOUT_SECONDS. `timeout_seconds or DEFAULT_TIMEOUT_SECONDS`
    would incorrectly yield DEFAULT_TIMEOUT_SECONDS here; the fix must not.
    """
    captured = _patch_client_recorder(monkeypatch)

    api_client.post_message("token", "chat-1", "hello", "agent", timeout_seconds=0)

    assert captured["timeout"] == 0
