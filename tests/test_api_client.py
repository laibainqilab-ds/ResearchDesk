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
