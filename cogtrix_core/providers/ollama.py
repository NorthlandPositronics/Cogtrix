"""Ollama LLM provider for local model inference."""

from __future__ import annotations

from typing import Any

from cogtrix_core.providers.defaults import BASE_URLS, CHAT_MODELS, EMBEDDING_MODELS

# ── Lazy imports ─────────────────────────────────────────────────────

try:
    from langchain_ollama import ChatOllama

    CHAT_AVAILABLE = True
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]
    CHAT_AVAILABLE = False

try:
    from langchain_ollama import OllamaEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    OllamaEmbeddings = None  # type: ignore[misc, assignment]
    EMBEDDINGS_AVAILABLE = False


def create_chat_model(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.5,
    num_ctx: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create an Ollama chat model.

    Args:
        model: Model name (default: qwen3:8b).
        base_url: Ollama server URL (default: http://localhost:11434).
        temperature: Sampling temperature.
        num_ctx: Context window size in tokens.
        **kwargs: Extra keyword arguments forwarded to ``ChatOllama``.

    Returns:
        ``ChatOllama`` instance.

    Raises:
        ImportError: If ``langchain-ollama`` is not installed.
    """
    if not CHAT_AVAILABLE:
        raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")

    llm_kwargs: dict[str, Any] = {
        "model": model or CHAT_MODELS["ollama"],
        "base_url": base_url or BASE_URLS["ollama"],
        "temperature": temperature,
    }
    if num_ctx is not None:
        llm_kwargs["num_ctx"] = num_ctx
    llm_kwargs.update(kwargs)
    return ChatOllama(**llm_kwargs)  # type: ignore[arg-type]


def create_embeddings(
    model: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create Ollama embeddings.

    Args:
        model: Embedding model name (default: nomic-embed-text).
        base_url: Ollama server URL.

    Returns:
        ``OllamaEmbeddings`` instance.

    Raises:
        ImportError: If ``langchain-ollama`` is not installed.
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")

    emb_kwargs: dict[str, Any] = {
        "model": model or EMBEDDING_MODELS["ollama"],
        "base_url": base_url or BASE_URLS["ollama"],
    }
    emb_kwargs.update(kwargs)
    return OllamaEmbeddings(**emb_kwargs)
