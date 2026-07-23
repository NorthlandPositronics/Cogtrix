"""Google Gemini LLM provider."""

from __future__ import annotations

from typing import Any

from cogtrix_core.logging_config import get_logger
from cogtrix_core.providers import _redact_url
from cogtrix_core.providers.defaults import CHAT_MODELS, EMBEDDING_MODELS

log = get_logger()

# ── Lazy imports ─────────────────────────────────────────────────────

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    CHAT_AVAILABLE = True
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]
    CHAT_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    GoogleGenerativeAIEmbeddings = None  # type: ignore[misc, assignment]
    EMBEDDINGS_AVAILABLE = False


def create_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    **kwargs: Any,
) -> Any:
    """Create a Google Gemini chat model.

    Args:
        model: Model name (default: gemini-2.5-flash).
        api_key: API key (``None`` → falls back to ``GOOGLE_API_KEY`` env var).
        base_url: Must be ``None``.  Google API does not support custom
            endpoints; a non-``None`` value raises ``ValueError``.
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to
            ``ChatGoogleGenerativeAI``.

    Returns:
        ``ChatGoogleGenerativeAI`` instance.

    Raises:
        ImportError: If ``langchain-google-genai`` is not installed.
    """
    if not CHAT_AVAILABLE:
        raise ImportError(
            "langchain-google-genai not installed. Run: pip install langchain-google-genai"
        )

    if base_url:
        raise ValueError(
            "Google provider does not support custom base_url. "
            f"Received base_url={_redact_url(base_url)!r}. "
            "If you need a proxy or Vertex AI endpoint, configure GOOGLE_API_KEY to "
            "point to your instance or use the 'google' provider's transport= kwarg."
        )

    llm_kwargs: dict[str, Any] = {
        "model": model or CHAT_MODELS["google"],
        "temperature": temperature,
        "max_retries": 3,
    }
    if api_key:
        llm_kwargs["google_api_key"] = api_key
    llm_kwargs.update(kwargs)
    return ChatGoogleGenerativeAI(**llm_kwargs)


def create_embeddings(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create Google Generative AI embeddings.

    Args:
        model: Embedding model name (default: text-embedding-004).
        api_key: API key.
        base_url: Must be ``None``.  Google API does not support custom
            endpoints; a non-``None`` value raises ``ValueError``.

    Returns:
        ``GoogleGenerativeAIEmbeddings`` instance.

    Raises:
        ImportError: If ``langchain-google-genai`` is not installed.
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "langchain-google-genai not installed. Run: pip install langchain-google-genai"
        )

    if base_url:
        raise ValueError(
            "Google provider does not support custom base_url. "
            f"Received base_url={_redact_url(base_url)!r}. "
            "If you need a proxy or Vertex AI endpoint, configure GOOGLE_API_KEY to "
            "point to your instance or use the 'google' provider's transport= kwarg."
        )

    resolved_model = model or EMBEDDING_MODELS["google"]
    if not resolved_model.startswith("models/"):
        resolved_model = f"models/{resolved_model}"
    emb_kwargs: dict[str, Any] = {
        "model": resolved_model,
    }
    if api_key:
        emb_kwargs["google_api_key"] = api_key
    emb_kwargs.update(kwargs)
    return GoogleGenerativeAIEmbeddings(**emb_kwargs)
