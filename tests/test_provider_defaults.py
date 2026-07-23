"""Tests for provider defaults registry."""

from __future__ import annotations

import pytest

from cogtrix_core.providers.defaults import (
    BASE_URLS,
    CHAT_MODELS,
    EMBEDDING_MODELS,
    ENV_KEY_NAMES,
    OPENAI_PRESETS,
    PROVIDER_TYPES,
)


class TestProviderDefaultsRegistry:
    """Verify the provider defaults data structures are correct and consistent."""

    def test_chat_models_contains_all_providers(self):
        """CHAT_MODELS must contain entries for all 4 native provider types."""
        assert set(CHAT_MODELS.keys()) == {"openai", "ollama", "anthropic", "google"}

    def test_chat_models_values_are_non_empty_strings(self):
        """Every chat model default must be a non-empty string."""
        for provider, model in CHAT_MODELS.items():
            assert isinstance(model, str), f"{provider}: model must be a string"
            assert model, f"{provider}: model must not be empty"

    def test_ollama_chat_model_default(self):
        """Ollama default chat model is qwen3:8b."""
        assert CHAT_MODELS["ollama"] == "qwen3:8b"

    def test_openai_chat_model_default(self):
        """OpenAI default chat model is gpt-4.1-mini."""
        assert CHAT_MODELS["openai"] == "gpt-4.1-mini"

    def test_anthropic_chat_model_default(self):
        """Anthropic default chat model is claude-sonnet-4-5."""
        assert CHAT_MODELS["anthropic"] == "claude-sonnet-4-5"

    def test_google_chat_model_default(self):
        """Google default chat model is gemini-2.5-flash."""
        assert CHAT_MODELS["google"] == "gemini-2.5-flash"


class TestEmbeddingModelsRegistry:
    """Verify the embedding models registry."""

    def test_embedding_models_contains_all_providers(self):
        """EMBEDDING_MODELS must cover all 4 native provider types."""
        assert set(EMBEDDING_MODELS.keys()) == {"openai", "ollama", "anthropic", "google"}

    def test_anthropic_has_no_embedding_model(self):
        """Anthropic does not offer a dedicated embedding API."""
        assert EMBEDDING_MODELS["anthropic"] is None

    def test_ollama_embedding_model_default(self):
        """Ollama default embedding model is nomic-embed-text."""
        assert EMBEDDING_MODELS["ollama"] == "nomic-embed-text"

    def test_openai_embedding_model_default(self):
        """OpenAI default embedding model is text-embedding-3-small."""
        assert EMBEDDING_MODELS["openai"] == "text-embedding-3-small"


class TestBaseUrlsRegistry:
    """Verify the base URL defaults."""

    def test_base_urls_contains_all_providers(self):
        """BASE_URLS must cover all 4 native provider types."""
        assert set(BASE_URLS.keys()) == {"openai", "ollama", "anthropic", "google"}

    def test_ollama_base_url_is_localhost(self):
        """Ollama default base URL points to localhost."""
        assert BASE_URLS["ollama"] == "http://localhost:11434"

    def test_cloud_providers_use_sdk_default(self):
        """Cloud providers (openai, anthropic, google) use None for SDK default."""
        for provider in ("openai", "anthropic", "google"):
            assert BASE_URLS[provider] is None


class TestEnvKeyNamesRegistry:
    """Verify the API key environment variable names."""

    def test_openai_env_key(self):
        assert ENV_KEY_NAMES["openai"] == "OPENAI_API_KEY"

    def test_anthropic_env_key(self):
        assert ENV_KEY_NAMES["anthropic"] == "ANTHROPIC_API_KEY"

    def test_google_env_key(self):
        assert ENV_KEY_NAMES["google"] == "GEMINI_API_KEY"

    def test_ollama_not_in_env_keys(self):
        """Ollama is local and does not require an API key env var."""
        assert "ollama" not in ENV_KEY_NAMES


class TestOpenAiPresets:
    """Verify the OpenAI-compatible provider presets."""

    def test_presets_are_non_empty(self):
        """OPENAI_PRESETS must contain at least one preset."""
        assert len(OPENAI_PRESETS) > 0

    def test_xai_preset(self):
        """xAI preset has correct label, base_url, model, and env_key."""
        preset = OPENAI_PRESETS["xai"]
        assert preset["label"] == "xAI (Grok)"
        assert preset["base_url"] == "https://api.x.ai/v1"
        assert preset["model"] == "grok-4.1-fast"
        assert preset["env_key"] == "XAI_API_KEY"

    def test_groq_preset(self):
        """Groq preset has correct label, base_url, model, and env_key."""
        preset = OPENAI_PRESETS["groq"]
        assert preset["label"] == "Groq"
        assert preset["base_url"] == "https://api.groq.com/openai/v1"
        assert preset["model"] == "llama-3.3-70b-versatile"
        assert preset["env_key"] == "GROQ_API_KEY"

    def test_deepseek_preset(self):
        """DeepSeek preset has correct label, base_url, model, and env_key."""
        preset = OPENAI_PRESETS["deepseek"]
        assert preset["label"] == "DeepSeek"
        assert preset["base_url"] == "https://api.deepseek.com/v1"
        assert preset["model"] == "deepseek-chat"
        assert preset["env_key"] == "DEEPSEEK_API_KEY"

    def test_all_presets_have_required_fields(self):
        """Every preset must have label, base_url, model, and env_key."""
        required = {"label", "base_url", "model", "env_key"}
        for name, preset in OPENAI_PRESETS.items():
            missing = required - set(preset.keys())
            assert not missing, f"{name} preset missing fields: {missing}"


class TestProviderTypes:
    """Verify the PROVIDER_TYPES frozenset."""

    def test_provider_types_matches_chat_models_keys(self):
        """PROVIDER_TYPES must exactly match CHAT_MODELS.keys()."""
        assert PROVIDER_TYPES == frozenset(CHAT_MODELS.keys())

    def test_provider_types_is_frozen(self):
        """PROVIDER_TYPES must be immutable."""
        with pytest.raises(AttributeError):
            PROVIDER_TYPES.add("new_provider")
