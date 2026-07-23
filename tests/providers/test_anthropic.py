"""Tests for the Anthropic Claude provider adapter.

Coverage for issue #1212 — anthropic.py and google.py have zero test coverage.

Anthropic is a primary provider alongside OpenAI and Ollama. A regression in
model resolution, API key forwarding, or import guard would silently break
every Anthropic user on the next release.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCreateChatModel:
    """Tests for anthropic.create_chat_model()."""

    def test_default_model_from_chat_models_registry(self) -> None:
        """Default model is pulled from CHAT_MODELS (claude-sonnet-4-5)."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model()

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "claude-sonnet-4-5"

    def test_custom_model_name_forwarded(self) -> None:
        """Custom model name is forwarded to ChatAnthropic."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(model="claude-3-5-sonnet-20240620")

        assert mock_cls.call_args.kwargs["model"] == "claude-3-5-sonnet-20240620"

    def test_api_key_forwarded_as_api_key(self) -> None:
        """api_key kwarg is forwarded as 'api_key' to ChatAnthropic."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(api_key="sk-ant-secret")

        assert mock_cls.call_args.kwargs["api_key"] == "sk-ant-secret"

    def test_api_key_omitted_when_none(self) -> None:
        """api_key is not passed when None — SDK falls back to env var."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(api_key=None)

        assert "api_key" not in mock_cls.call_args.kwargs

    def test_base_url_forwarded(self) -> None:
        """Custom base_url is forwarded to ChatAnthropic."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(base_url="https://api.anthropic.com/v1")

        assert mock_cls.call_args.kwargs["base_url"] == "https://api.anthropic.com/v1"

    def test_temperature_default_is_zero(self) -> None:
        """Default temperature is 0."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model()

        assert mock_cls.call_args.kwargs["temperature"] == 0

    def test_temperature_forwarded(self) -> None:
        """Custom temperature is forwarded to ChatAnthropic."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(temperature=0.7)

        assert mock_cls.call_args.kwargs["temperature"] == 0.7

    def test_extra_kwargs_forwarded(self) -> None:
        """Extra kwargs are forwarded to ChatAnthropic."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(top_p=0.9, max_tokens=4096)

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_tokens"] == 4096

    def test_all_kwargs_combined(self) -> None:
        """All parameters are correctly combined and forwarded."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model(
                model="claude-3-opus-20240229",
                api_key="sk-ant-key",
                base_url="https://proxy.example.com/anthropic",
                temperature=0.5,
                top_p=0.95,
                seed=42,
            )

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "claude-3-opus-20240229"
        assert kwargs["api_key"] == "sk-ant-key"
        assert kwargs["base_url"] == "https://proxy.example.com/anthropic"
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.95
        assert kwargs["seed"] == 42

    def test_max_retries_default_is_three(self) -> None:
        """max_retries defaults to 3 when not overridden."""
        from src.providers.anthropic import create_chat_model

        mock_cls = MagicMock()
        with patch("src.providers.anthropic.ChatAnthropic", mock_cls):
            create_chat_model()

        assert mock_cls.call_args.kwargs["max_retries"] == 3

    def test_import_error_raised_when_library_not_installed(self) -> None:
        """ImportError is raised when langchain-anthropic is not installed."""
        from src.providers import anthropic as anthropic_mod

        original = anthropic_mod.CHAT_AVAILABLE
        try:
            anthropic_mod.CHAT_AVAILABLE = False
            with pytest.raises(ImportError, match="langchain-anthropic not installed"):
                anthropic_mod.create_chat_model()
        finally:
            anthropic_mod.CHAT_AVAILABLE = original


class TestCreateEmbeddings:
    """Tests for anthropic.create_embeddings()."""

    def test_raises_not_implemented_error(self) -> None:
        """Anthropic does not provide a dedicated embedding API."""
        from src.providers.anthropic import create_embeddings

        with pytest.raises(
            NotImplementedError, match="Anthropic does not offer a dedicated embedding API"
        ):
            create_embeddings()
