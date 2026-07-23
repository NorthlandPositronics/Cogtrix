"""OpenAI and OpenAI-compatible LLM provider (xAI, vLLM, Groq, Together, etc.)."""

from __future__ import annotations

from typing import Any

from src.providers.defaults import CHAT_MODELS, EMBEDDING_MODELS

# ── Lazy imports ─────────────────────────────────────────────────────

try:
    from langchain_openai import ChatOpenAI

    CHAT_AVAILABLE = True
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]
    CHAT_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[misc, assignment]
    EMBEDDINGS_AVAILABLE = False


def create_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    **kwargs: Any,
) -> Any:
    """Create an OpenAI (or compatible) chat model.

    Args:
        model: Model name (default: gpt-4.1-mini).
        api_key: API key (``None`` → falls back to ``OPENAI_API_KEY`` env var).
        base_url: Custom endpoint (``None`` → OpenAI default).
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to ``ChatOpenAI``.

    Returns:
        ``ChatOpenAI`` instance.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    if not CHAT_AVAILABLE:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    llm_kwargs: dict[str, Any] = {
        "model": model or CHAT_MODELS["openai"],
        "temperature": temperature,
        "max_retries": 3,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
        # OpenAI-compatible endpoints (vLLM, LM Studio, etc.) often require no
        # authentication, but the SDK unconditionally rejects a missing api_key.
        # Use the caller-supplied key when present; fall back to a placeholder so
        # the SDK's client-option check passes without forcing users to invent a
        # key.  "not-required" is deliberately descriptive so it never appears as
        # a confusing literal in SDK error messages (BUG-231).
        llm_kwargs["api_key"] = api_key or "not-required"
    elif api_key:
        llm_kwargs["api_key"] = api_key
    llm_kwargs.update(kwargs)
    return ChatOpenAI(**llm_kwargs)


def create_embeddings(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create OpenAI embeddings.

    Args:
        model: Embedding model name (default: text-embedding-3-small).
        api_key: API key.
        base_url: Custom endpoint.

    Returns:
        ``OpenAIEmbeddings`` instance.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    emb_kwargs: dict[str, Any] = {"model": model or EMBEDDING_MODELS["openai"]}
    if api_key:
        emb_kwargs["api_key"] = api_key
    if base_url:
        emb_kwargs["openai_api_base"] = base_url
    emb_kwargs.update(kwargs)
    return OpenAIEmbeddings(**emb_kwargs)
