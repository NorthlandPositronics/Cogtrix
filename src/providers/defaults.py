"""Default models, embedding models, and presets for each provider type."""

from __future__ import annotations

from typing import Any

# ── Default chat models per provider type ────────────────────────────
# Used when no model is specified in config.

CHAT_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash",
}

# ── Default embedding models per provider type ───────────────────────
# None means the provider does not offer a dedicated embedding API.

EMBEDDING_MODELS: dict[str, str | None] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "anthropic": None,
    "google": "text-embedding-004",
}

# ── Default base URLs (None = use SDK default) ──────────────────────

BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "ollama": "http://localhost:11434",
    "anthropic": None,
    "google": None,
}

# ── Environment variable names for API keys ──────────────────────────

ENV_KEY_NAMES: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    # xAI uses XAI_API_KEY but is an openai-compatible preset
}

# ── OpenAI-compatible provider presets ───────────────────────────────
# These use the "openai" provider type with a custom base URL.
# The setup wizard offers them as named choices; under the hood they
# produce a ProviderConfig with type="openai".

OPENAI_PRESETS: dict[str, dict[str, Any]] = {
    "xai": {
        "label": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4.1-fast",
        "env_key": "XAI_API_KEY",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
}

# ── All supported native provider types ──────────────────────────────

PROVIDER_TYPES: frozenset[str] = frozenset(CHAT_MODELS.keys())
