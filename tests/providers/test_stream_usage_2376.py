"""Regression tests for #2376 — request token usage on streaming openai models.

OpenAI-compatible streaming returns no token usage unless the client requests
``stream_options.include_usage`` (langchain_openai: ``stream_usage=True``).
Native Moonshot/Kimi does not volunteer usage, so the API reported 0 tokens.
``create_chat_model`` must default ``stream_usage=True`` for streaming openai-type
models, leave it off otherwise, and respect an explicit override.
"""

from __future__ import annotations

from unittest.mock import patch

import cogtrix_core.providers as providers_pkg
from cogtrix_core.providers import create_chat_model


def _factory_kwargs(provider_type: str = "openai", **call_kwargs) -> dict:
    """Return the kwargs ``create_chat_model`` forwards to the provider module
    (the provider's own ``create_chat_model`` is mocked, so no real chat model
    is built)."""
    with patch.object(providers_pkg, "_load_provider") as load:
        mod = load.return_value
        create_chat_model(
            provider_type,
            model="x",
            api_key="k",
            base_url="http://host/v1",
            **call_kwargs,
        )
        return mod.create_chat_model.call_args.kwargs


def test_streaming_openai_defaults_stream_usage_true() -> None:
    assert _factory_kwargs(streaming=True).get("stream_usage") is True


def test_non_streaming_openai_does_not_request_stream_usage() -> None:
    assert "stream_usage" not in _factory_kwargs(streaming=False)


def test_explicit_stream_usage_is_respected() -> None:
    # An operator-supplied stream_usage (e.g. via model_kwargs) must win.
    assert _factory_kwargs(streaming=True, stream_usage=False).get("stream_usage") is False


def test_non_openai_streaming_does_not_request_stream_usage() -> None:
    # Only the openai-compatible type needs the include_usage opt-in.
    assert "stream_usage" not in _factory_kwargs(provider_type="anthropic", streaming=True)
