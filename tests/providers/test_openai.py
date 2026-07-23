"""Regression tests for ``cogtrix_core/providers/openai.create_embeddings`` (#2067).

OpenAI-compatible third-party embedding endpoints (e.g. OpenRouter, vLLM) reject
langchain's default token-ID embedding input (``check_embedding_ctx_length=True``),
returning an empty ``data`` array — which surfaces as RAG ingestion failing with
"No embedding data received". ``create_embeddings`` must disable that for custom
``base_url`` endpoints so raw text is sent, while leaving native OpenAI untouched.
"""

from __future__ import annotations

from unittest.mock import patch

from cogtrix_core.providers import openai as openai_provider


def _captured_kwargs(**call_kwargs):
    """Call create_embeddings with OpenAIEmbeddings mocked; return its kwargs."""
    with (
        patch.object(openai_provider, "EMBEDDINGS_AVAILABLE", True),
        patch.object(openai_provider, "OpenAIEmbeddings") as mock_emb,
    ):
        openai_provider.create_embeddings(**call_kwargs)
    return mock_emb.call_args.kwargs


def test_custom_base_url_disables_ctx_length_check() -> None:
    """#2067: a custom / OpenAI-compatible base_url must send raw text."""
    kwargs = _captured_kwargs(
        model="qwen/qwen3-embedding-4b",
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs.get("check_embedding_ctx_length") is False
    assert kwargs.get("openai_api_base") == "https://openrouter.ai/api/v1"


def test_native_openai_keeps_langchain_default() -> None:
    """No custom base_url -> we don't force the flag (langchain default preserved)."""
    kwargs = _captured_kwargs(model="text-embedding-3-small", api_key="sk-test")
    assert "check_embedding_ctx_length" not in kwargs


def test_explicit_kwarg_overrides_the_default() -> None:
    """An explicit check_embedding_ctx_length wins over the #2067 default."""
    kwargs = _captured_kwargs(
        model="m",
        api_key="k",
        base_url="https://compat.example/v1",
        check_embedding_ctx_length=True,
    )
    assert kwargs.get("check_embedding_ctx_length") is True
