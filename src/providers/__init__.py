"""Modular LLM provider registry.

Centralizes chat model and embedding creation so that ``core.py``,
``delegate.py``, ``deep_think.py``, and RAG code all use the same
dispatch logic.  Adding a new provider means adding a module under
``src/providers/`` and registering it here.

Usage::

    from src.providers import create_chat_model, create_embeddings

    llm = create_chat_model("openai", model="gpt-4.1-mini", api_key="sk-...")
    emb = create_embeddings("ollama", model="nomic-embed-text")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from src.providers.defaults import (
    BASE_URLS,
    CHAT_MODELS,
    EMBEDDING_MODELS,
    PROVIDER_TYPES,
)

if TYPE_CHECKING:
    from src.config import ProviderConfig

# Re-export for convenience
__all__ = [
    "PROVIDER_TYPES",
    "CHAT_MODELS",
    "EMBEDDING_MODELS",
    "BASE_URLS",
    "create_chat_model",
    "create_chat_model_from_config",
    "create_embeddings",
    "create_embeddings_from_config",
    "get_default_model",
    "get_default_embedding_model",
    "get_default_base_url",
    "is_chat_available",
    "is_embeddings_available",
]

# ── Provider module map ──────────────────────────────────────────────

_MODULES: dict[str, str] = {
    "openai": "src.providers.openai",
    "ollama": "src.providers.ollama",
    "anthropic": "src.providers.anthropic",
    "google": "src.providers.google",
}


def _load_provider(provider_type: str) -> Any:
    """Import and return a provider module by type name.

    Raises:
        ValueError: If *provider_type* is not a known provider type.
    """
    module_path = _MODULES.get(provider_type)
    if module_path is None:
        supported = ", ".join(sorted(PROVIDER_TYPES))
        raise ValueError(f"Unknown provider type: '{provider_type}'. Supported: {supported}")
    return importlib.import_module(module_path)


# ── Public helpers ───────────────────────────────────────────────────


def get_default_model(provider_type: str) -> str:
    """Return the default chat model for *provider_type*."""
    return CHAT_MODELS.get(provider_type, CHAT_MODELS["openai"])


def get_default_embedding_model(provider_type: str) -> str | None:
    """Return the default embedding model (``None`` if not supported)."""
    return EMBEDDING_MODELS.get(provider_type)


def get_default_base_url(provider_type: str) -> str | None:
    """Return the default base URL (``None`` = use SDK default)."""
    return BASE_URLS.get(provider_type)


def is_chat_available(provider_type: str) -> bool:
    """Check whether the required packages for chat are installed."""
    try:
        mod = _load_provider(provider_type)
        return bool(getattr(mod, "CHAT_AVAILABLE", False))
    except ValueError:
        return False


def is_embeddings_available(provider_type: str) -> bool:
    """Check whether the required packages for embeddings are installed."""
    try:
        mod = _load_provider(provider_type)
        return bool(getattr(mod, "EMBEDDINGS_AVAILABLE", False))
    except ValueError:
        return False


# ── Chat model creation ──────────────────────────────────────────────


def create_chat_model(
    provider_type: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    num_ctx: int | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create a chat model for *provider_type*.

    This is the **low-level** factory — pass explicit parameters.
    For the higher-level variant that reads from a ``ProviderConfig``
    dataclass, use :func:`create_chat_model_from_config`.

    Args:
        provider_type: One of ``PROVIDER_TYPES``.
        model: Model name (``None`` → provider default).
        api_key: API key (``None`` → env-var fallback).
        base_url: Custom endpoint.
        temperature: Sampling temperature.
        num_ctx: Context window size (Ollama only).
        max_tokens: Max output tokens (``None`` → API default).
        **kwargs: Extra arguments forwarded to the provider module.

    Returns:
        A LangChain chat-model instance.

    Raises:
        ValueError: If *provider_type* is unknown.
        ImportError: If the required LangChain package is missing.
    """
    mod = _load_provider(provider_type)
    kw: dict[str, Any] = {"model": model, "temperature": temperature, **kwargs}
    if api_key is not None:
        kw["api_key"] = api_key
    if base_url is not None:
        kw["base_url"] = base_url
    if num_ctx is not None and provider_type == "ollama":
        kw["num_ctx"] = num_ctx
    if max_tokens is not None:
        # Each provider uses a different kwarg name for output token limits
        if provider_type == "ollama":
            kw["num_predict"] = max_tokens
        elif provider_type == "google":
            kw["max_output_tokens"] = max_tokens
        else:  # openai, anthropic
            kw["max_tokens"] = max_tokens
    return mod.create_chat_model(**kw)


def create_chat_model_from_config(provider_config: ProviderConfig) -> Any:
    """Create a chat model from a :class:`ProviderConfig` dataclass.

    This is the **high-level** factory used by ``core.py`` and other
    modules that already have a resolved ``ProviderConfig``.

    Args:
        provider_config: Resolved provider configuration.

    Returns:
        A LangChain chat-model instance.
    """
    return create_chat_model(
        provider_config.type,
        model=provider_config.get_model(),
        api_key=provider_config.api_key,
        base_url=provider_config.get_base_url(),
        temperature=provider_config.temperature if provider_config.temperature is not None else 0,
        num_ctx=provider_config.num_ctx if provider_config.type == "ollama" else None,
        max_tokens=provider_config.max_tokens,
    )


# ── Embeddings creation ──────────────────────────────────────────────


def create_embeddings(
    provider_type: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create an embeddings instance for *provider_type*.

    Args:
        provider_type: One of ``PROVIDER_TYPES``.
        model: Embedding model name (``None`` → provider default).
        api_key: API key.
        base_url: Custom endpoint (used by Ollama and OpenAI-compatible).
        **kwargs: Extra arguments forwarded to the provider module.

    Returns:
        A LangChain embeddings instance.

    Raises:
        ValueError: If *provider_type* is unknown.
        ImportError: If the required LangChain package is missing.
        NotImplementedError: If the provider has no embedding support.
    """
    mod = _load_provider(provider_type)
    kw: dict[str, Any] = {"model": model, **kwargs}
    if api_key is not None:
        kw["api_key"] = api_key
    if base_url is not None:
        kw["base_url"] = base_url
    return mod.create_embeddings(**kw)


def create_embeddings_from_config(
    provider_type: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[Any, str]:
    """Create an embeddings instance from resolved embedding config.

    This is the **high-level** factory for embeddings, parallel to
    :func:`create_chat_model_from_config` for chat models.  Accepts the
    fields returned by ``Config.resolve_embedding_config()``.

    Returns:
        A ``(embeddings_instance, tag)`` tuple.  The *tag* is a
        human-readable string like ``"ollama/nomic-embed-text"`` suitable
        for logging and vector-store metadata.

    Raises:
        ValueError: If *provider_type* is unknown.
        ImportError: If the required LangChain package is missing.
        NotImplementedError: If the provider has no embedding support.
    """
    fn = create_embeddings(
        provider_type,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    resolved_model = model or get_default_embedding_model(provider_type) or "unknown"
    tag = f"{provider_type}/{resolved_model}"
    return fn, tag
