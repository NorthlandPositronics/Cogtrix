"""Anthropic Claude LLM provider."""

from __future__ import annotations

from typing import Any

from src.providers.defaults import CHAT_MODELS

# ── Lazy imports ─────────────────────────────────────────────────────

try:
    from langchain_anthropic import ChatAnthropic

    CHAT_AVAILABLE = True
except ImportError:
    ChatAnthropic = None  # type: ignore[misc, assignment]
    CHAT_AVAILABLE = False

# Anthropic does not offer a dedicated embedding API.
EMBEDDINGS_AVAILABLE = False


def create_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    **kwargs: Any,
) -> Any:
    """Create an Anthropic Claude chat model.

    Args:
        model: Model name (default: claude-sonnet-4-5).
        api_key: API key (``None`` → falls back to ``ANTHROPIC_API_KEY`` env var).
        base_url: Custom endpoint (``None`` → Anthropic default).
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to ``ChatAnthropic``.

    Returns:
        ``ChatAnthropic`` instance.

    Raises:
        ImportError: If ``langchain-anthropic`` is not installed.
    """
    if not CHAT_AVAILABLE:
        raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")

    llm_kwargs: dict[str, Any] = {
        "model": model or CHAT_MODELS["anthropic"],
        "temperature": temperature,
    }
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm_kwargs.update(kwargs)
    return ChatAnthropic(**llm_kwargs)


def create_embeddings(**kwargs: Any) -> Any:
    """Anthropic does not provide an embedding API.

    Raises:
        NotImplementedError: Always — use a different provider for embeddings.
    """
    raise NotImplementedError(
        "Anthropic does not offer a dedicated embedding API. "
        "Use 'openai', 'ollama', or 'google' for embeddings."
    )
