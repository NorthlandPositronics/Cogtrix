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
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlparse

from src.providers.defaults import (
    BASE_URLS,
    CHAT_MODELS,
    EMBEDDING_MODELS,
    PROVIDER_TYPES,
)

if TYPE_CHECKING:
    from src.config import ModelConfig, ProviderConfig

_log = logging.getLogger("cogtrix.providers")


def _redact_url(url: str | None) -> str:
    """Strip sensitive information from a URL for safe logging.

    Removes userinfo (username:password) and strips common sensitive query
    parameters like api_key, password, and token. Returns the original URL
    if parsing fails, but logs a warning.

    Args:
        url: The URL to redact (``None`` → returns a placeholder).

    Returns:
        A safe string for logging, with credentials and sensitive query
        params replaced by ``[redacted]``.
    """
    if not url:
        return "<unparseable URL>"

    try:
        parsed = urlparse(url)
    except Exception as exc:
        return f"<unparseable URL> ({exc})"

    # Rebuild netloc without userinfo
    hostname = parsed.hostname
    if hostname is None:
        return "<unparseable URL>"

    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    # Rebuild path
    path = parsed.path or "/"

    # Strip sensitive query params
    sensitive_keys = {
        "api_key",
        "apikey",
        "password",
        "token",
        "auth_token",
        "key",
        "secret",
        "access_token",
        "api_secret",
        "private_token",
        "refresh_token",  # OAuth refresh token; #1508
        "client_secret",  # OAuth client credentials flow; #1508
    }
    if parsed.query:
        try:
            params = parse_qsl(parsed.query, keep_blank_values=True)
            safe_params = [
                (k, "[redacted]" if k.lower() in sensitive_keys else v) for k, v in params
            ]
            query = "&".join(f"{k}={v}" for k, v in safe_params)
        except (ValueError, TypeError):
            query = parsed.query
    else:
        query = ""

    # Rebuild URL
    scheme = parsed.scheme or "http"
    return f"{scheme}://{netloc}{path}?{query}" if query else f"{scheme}://{netloc}{path}"


def _sanitize_auth_error_message(message: str) -> str:
    """Strip potential API key fragments from an authentication error message.

    Replaces known key patterns (OpenAI ``sk-*``, Anthropic ``sk-ant-*``,
    Google ``AIza*``, etc.) with ``[redacted]`` to prevent credential
    material from leaking into user-facing output or application logs.

    Args:
        message: The raw error message from the provider SDK.

    Returns:
        A sanitized message with key fragments replaced.
    """
    # Common API key prefixes and patterns that may appear in SDK errors.
    # Keep provider-specific prefixes before generic fallbacks to avoid
    # partial-substring leakage.
    #
    # Each specific prefix uses a negative lookbehind so that fragments like
    # "task-skills-dev" are not falsely redacted.
    patterns = [
        # OpenAI-style keys (also covers DeepSeek, which uses sk- prefix)
        r"(?<![a-zA-Z0-9])sk-[a-zA-Z0-9_-]+",
        # Anthropic-style keys
        r"(?<![a-zA-Z0-9])sk-ant-[a-zA-Z0-9_-]+",
        # Google / Gemini API keys
        r"(?<![a-zA-Z0-9])AIza[a-zA-Z0-9_-]+",
        # xAI (Grok) API keys
        r"(?<![a-zA-Z0-9])xai-[a-zA-Z0-9_-]+",
        # Groq API keys
        r"(?<![a-zA-Z0-9])gsk_[a-zA-Z0-9_-]+",
        # HuggingFace API keys
        r"(?<![a-zA-Z0-9])hf_[a-zA-Z0-9_-]+",
        # Generic "API key: <value>" fragments
        r"(?i)api\s*key[:\s]+[a-zA-Z0-9_-]+",
        # Generic "key: <value>" fragments in auth contexts
        r"(?i)(?:invalid\s+|incorrect\s+)?(?:api\s*key|x-api-key|authorization)\s*[:\s]+[a-zA-Z0-9_-]+",
    ]

    sanitized = message
    for pattern in patterns:
        sanitized = re.sub(pattern, "[redacted]", sanitized)

    return sanitized


# ── Exception types for provider errors ──────────────────────────────


class ProviderAuthError(Exception):
    """Raised when provider authentication fails or API key is invalid.

    This wraps authentication failures from the underlying provider SDK
    and provides a user-readable message for API responses.
    """

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message)
        self.provider = provider


class RateLimitError(Exception):
    """Raised when the provider returns a 429 Rate Limit error.

    Includes the Retry-After value if available from the response headers.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(Exception):
    """Raised when the provider returns a 401/403 Authentication error.

    This is a lower-level wrapper for SDK-specific auth errors.
    """


# ── Retryable chat model wrapper ─────────────────────────────────────
#
# This wrapper adds custom retry logic with exponential backoff for
# rate limits (429), reading Retry-After headers when available.


def _extract_retry_after(response: Any) -> float | None:
    """Extract Retry-After value from an API error response.

    Handles both numeric seconds and HTTP-date formats.
    Returns None if no Retry-After header is present.

    Args:
        response: The exception's response object (from SDK).

    Returns:
        Retry-After value in seconds, or None if not present.
    """
    if response is None:
        return None

    # Try to get from response.headers if available
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                # Try parsing as numeric seconds first
                return float(retry_after)
            except ValueError:
                # If not numeric, it might be an HTTP-date
                pass

    # Try response.text for JSON bodies with retry info
    try:
        text = getattr(response, "text", None) or str(response)
        if isinstance(text, str):
            import json as _json

            try:
                data = _json.loads(text)
                if "retry_after" in data:
                    return float(data["retry_after"])
                if "Retry-After" in data:
                    return float(data["Retry-After"])
            except (ValueError, TypeError):
                pass
    except Exception as exc:
        _log.debug("_extract_retry_after: failed to parse response: %s", exc)

    return None


class RetryableChatModel:
    """Wrapper around LangChain ChatModel with custom retry logic.

    Adds exponential backoff for rate limits (429) with Retry-After
    header support, and converts auth errors to ProviderAuthError.

    Args:
        model: The underlying LangChain chat model instance.
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial backoff delay in seconds (default: 1.0).
        max_delay: Maximum delay between retries in seconds (default: 30.0).
    """

    def __init__(
        self,
        model: Any,
        max_retries: int | None = None,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self._model = model
        self._max_retries = max_retries if max_retries is not None else 3
        self._initial_delay = initial_delay
        self._max_delay = max_delay

    def _should_retry(self, error: Exception) -> tuple[bool, float | None]:
        """Determine if error is retryable and extract retry delay.

        Returns:
            Tuple of (should_retry, retry_delay_or_none).
        """
        error_str = str(error).lower()

        # Check for RateLimitError
        if "rate_limit" in error_str or "rate limit" in error_str:
            retry_after = _extract_retry_after(getattr(error, "response", None))
            if retry_after is None:
                retry_after = getattr(error, "retry_after", None)
            return True, retry_after

        # Check for 429 status code in error (type-checking safe)
        status_code = getattr(error, "status_code", None)
        if status_code is not None and int(status_code) == 429:
            retry_after = _extract_retry_after(getattr(error, "response", None))
            if retry_after is None:
                retry_after = getattr(error, "retry_after", None)
            return True, retry_after

        # Check for 5xx transient server errors (502, 503, 504)
        if status_code is not None and int(status_code) in (502, 503, 504):
            return True, None

        return False, None

    def _is_auth_error(self, error: Exception) -> bool:
        """Check if error indicates authentication failure.

        Handles both string-pattern matching and direct exception type checks
        for LangChain SDK exceptions like openai.AuthenticationError.
        """
        # Check exception type directly
        error_type = type(error)
        if "AuthenticationError" in error_type.__name__:
            return True
        if "AuthError" in error_type.__name__:
            return True

        error_str = str(error).lower()

        if "authentication" in error_str:
            return True
        if "invalid_api_key" in error_str:
            return True
        if "unauthorized" in error_str:
            return True
        # NB: deliberately no bare ``"auth" in error_str`` fallback (#2147).
        # The specific strings above plus the 401/403 status check below cover
        # genuine auth failures; a bare "auth" substring also matched unrelated
        # transient errors ("oauth", "author", "authoritative", or a 5xx body
        # echoing such text), short-circuiting the retry path and surfacing a
        # retryable failure as a permanent ProviderAuthError.

        # Type-checking safe status code check
        status_code = getattr(error, "status_code", None)
        if status_code is not None and int(status_code) in (401, 403):
            return True

        return False

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the model with retry logic for rate limits.

        Raises:
            ProviderAuthError: For authentication failures.
            RateLimitError: If all retries are exhausted for rate limits.
        """
        # Internal flag used by _invoke_with_timeout to avoid nested retry
        # loops that would block a scarce ThreadPoolExecutor worker.
        _disable_retries = kwargs.pop("_cogtrix_disable_retries", False)
        if _disable_retries:
            return self._model.invoke(*args, **kwargs)

        last_error: Exception | None = None
        delay = self._initial_delay

        for attempt in range(self._max_retries + 1):
            try:
                return self._model.invoke(*args, **kwargs)
            except Exception as exc:
                last_error = exc

                # Check if this is an auth error - fail immediately
                if self._is_auth_error(exc):
                    safe_message = _sanitize_auth_error_message(str(exc))
                    raise ProviderAuthError(
                        f"Authentication failed: {safe_message}",
                        provider=getattr(self._model, "provider_name", None),
                    ) from exc

                # Check if this is a rate limit - retry if attempts remain
                should_retry, retry_delay = self._should_retry(exc)
                if should_retry:
                    if attempt < self._max_retries:
                        # Server-provided Retry-After takes precedence on first
                        # attempt; exponential backoff is used otherwise.
                        actual_delay = retry_delay if retry_delay is not None else delay
                        wait_time = min(actual_delay, self._max_delay)
                        time.sleep(wait_time)
                        delay = min(delay * 2, self._max_delay)
                        continue
                    else:
                        raise RateLimitError(
                            f"Rate limit exhausted after {self._max_retries + 1} attempts",
                            retry_after=retry_delay,
                        ) from exc

                # Non-retryable error - fail immediately
                raise

        # Should never reach here, but just in case
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected retry loop exit")

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate bind_tools to the wrapped model and re-wrap the result.

        Without this override, ``llm.bind_tools()`` returns the raw
        underlying model, causing all subsequent invocations to bypass
        ``RetryableChatModel.invoke()`` — including the
        ``_cogtrix_disable_retries`` flag used by
        ``_invoke_with_timeout()`` to prevent nested retry loops from
        blocking scarce ThreadPoolExecutor workers.
        """
        bound = self._model.bind_tools(*args, **kwargs)
        return RetryableChatModel(
            bound,
            max_retries=self._max_retries,
            initial_delay=self._initial_delay,
            max_delay=self._max_delay,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped model for compatibility."""
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"RetryableChatModel({self._model!r})"

    def __str__(self) -> str:
        return f"RetryableChatModel({self._model!s})"


if TYPE_CHECKING:
    from src.config import ModelConfig, ProviderConfig

# Re-export for convenience
__all__ = [
    "PROVIDER_TYPES",
    "CHAT_MODELS",
    "EMBEDDING_MODELS",
    "BASE_URLS",
    "ProviderAuthError",
    "RateLimitError",
    "AuthenticationError",
    "RetryableChatModel",
    "create_chat_model",
    "create_chat_model_from_configs",
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

_provider_cache: dict[str, Any] = {}
_provider_cache_lock = threading.Lock()


def _load_provider(provider_type: str) -> Any:
    """Import and return a provider module by type name.

    Uses double-checked locking so that slow imports do not block
    concurrent cache lookups for already-loaded providers.

    Raises:
        ValueError: If *provider_type* is not a known provider type.
    """
    with _provider_cache_lock:
        if provider_type in _provider_cache:
            return _provider_cache[provider_type]
        module_path = _MODULES.get(provider_type)
        if module_path is None:
            supported = ", ".join(sorted(PROVIDER_TYPES))
            raise ValueError(f"Unknown provider type: '{provider_type}'. Supported: {supported}")

    module = importlib.import_module(module_path)

    with _provider_cache_lock:
        if provider_type not in _provider_cache:
            _provider_cache[provider_type] = module
        return _provider_cache[provider_type]


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
    temperature: float = 0.5,
    num_ctx: int | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create a chat model for *provider_type*.

    This is the **low-level** factory — pass explicit parameters.
    For the higher-level variant that takes separate provider and model
    configs, use :func:`create_chat_model_from_configs`.

    Args:
        provider_type: One of ``PROVIDER_TYPES``.
        model: Model name (``None`` → provider default).
        api_key: API key (``None`` → env-var fallback).
        base_url: Custom endpoint.
        temperature: Sampling temperature.
        num_ctx: Context window size (Ollama only).
        max_tokens: Max output tokens (``None`` → API default).
        max_retries: Override default max retries (``None`` → use provider default).
        **kwargs: Extra arguments forwarded to the provider module.

    Returns:
        A LangChain chat-model instance wrapped with retry logic.

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
    model_instance = mod.create_chat_model(**kw)

    # Wrap with retry logic if not already wrapped
    if not isinstance(model_instance, RetryableChatModel):
        return RetryableChatModel(
            model_instance,
            max_retries=max_retries if max_retries is not None else 3,
        )
    return model_instance


def create_chat_model_from_configs(
    provider_config: ProviderConfig,
    model_config: ModelConfig,
    *,
    streaming: bool = False,
    max_retries: int | None = None,
) -> Any:
    """Create a chat model from separate provider and model configs.

    This is the **new** factory that takes connection info from
    ``provider_config`` and model settings from ``model_config``.

    Args:
        provider_config: Connection info (type, base_url, api_key).
        model_config: Model settings (model, temperature, context_window, max_tokens).
        streaming: Enable token-level streaming callbacks (for API mode).
        max_retries: Override default max retries (``None`` → use provider default).

    Returns:
        A LangChain chat-model instance wrapped with retry logic.
    """
    # Per-model sampling passthrough (#2122): forward operator-supplied kwargs
    # (e.g. frequency_penalty, presence_penalty, top_p, extra_body) to the
    # underlying chat model.  Strip keys we already pass explicitly so a stray
    # entry can't raise a duplicate-keyword TypeError; the dedicated config
    # fields win.
    _RESERVED = {
        "model",
        "api_key",
        "base_url",
        "temperature",
        "num_ctx",
        "max_tokens",
        "streaming",
        "max_retries",
    }
    extra_kwargs = {
        k: v for k, v in (model_config.model_kwargs or {}).items() if k not in _RESERVED
    }
    if extra_kwargs.keys() != (model_config.model_kwargs or {}).keys():
        dropped = sorted(set((model_config.model_kwargs or {}).keys()) - extra_kwargs.keys())
        _log.warning(
            "Ignoring reserved key(s) in model_kwargs for model %r: %s "
            "(set these via the dedicated config fields instead)",
            model_config.model,
            ", ".join(dropped),
        )
    return create_chat_model(
        provider_config.type,
        model=model_config.model,
        api_key=provider_config.api_key,
        base_url=provider_config.get_base_url(),
        temperature=(
            model_config.temperature
            if model_config.temperature is not None
            else model_config.DEFAULT_TEMPERATURE
        ),
        num_ctx=(
            (model_config.context_window or model_config.DEFAULT_CONTEXT_WINDOW)
            if provider_config.type == "ollama"
            else None
        ),
        max_tokens=model_config.max_tokens,
        streaming=streaming,
        max_retries=max_retries,
        **extra_kwargs,
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
    :func:`create_chat_model_from_configs` for chat models.  Accepts the
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
