"""Comprehensive tests for the Google Gemini provider adapter.

Coverage for issue #1212 — anthropic.py and google.py have zero test coverage.

Google Gemini is a primary provider alongside OpenAI, Ollama, and Anthropic.
A regression in model resolution, google_api_key forwarding, or the models/
prefix logic would silently break every Google user on the next release.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCreateChatModel:
    """Tests for google.create_chat_model()."""

    def test_default_model_from_chat_models_registry(self) -> None:
        """Default model is pulled from CHAT_MODELS (gemini-2.5-flash)."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model()

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "gemini-2.5-flash"

    def test_custom_model_name_forwarded(self) -> None:
        """Custom model name is forwarded to ChatGoogleGenerativeAI."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(model="gemini-2.0-flash")

        assert mock_cls.call_args.kwargs["model"] == "gemini-2.0-flash"

    def test_api_key_forwarded_as_google_api_key(self) -> None:
        """api_key kwarg is forwarded as 'google_api_key' to ChatGoogleGenerativeAI."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(api_key="AIza...")

        assert mock_cls.call_args.kwargs["google_api_key"] == "AIza..."

    def test_api_key_omitted_when_none(self) -> None:
        """google_api_key is not passed when None — SDK falls back to env var."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(api_key=None)

        assert "google_api_key" not in mock_cls.call_args.kwargs

    def test_temperature_default_is_zero(self) -> None:
        """Default temperature is 0."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model()

        assert mock_cls.call_args.kwargs["temperature"] == 0

    def test_temperature_forwarded(self) -> None:
        """Custom temperature is forwarded to ChatGoogleGenerativeAI."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(temperature=0.7)

        assert mock_cls.call_args.kwargs["temperature"] == 0.7

    def test_extra_kwargs_forwarded(self) -> None:
        """Extra kwargs are forwarded to ChatGoogleGenerativeAI."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(top_p=0.9, max_output_tokens=4096)

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_output_tokens"] == 4096

    def test_all_kwargs_combined(self) -> None:
        """All parameters are correctly combined and forwarded."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model(
                model="gemini-1.5-pro",
                api_key="AIza...",
                temperature=0.5,
                top_p=0.95,
                seed=42,
            )

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "gemini-1.5-pro"
        assert kwargs["google_api_key"] == "AIza..."
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.95
        assert kwargs["seed"] == 42

    def test_max_retries_default_is_three(self) -> None:
        """max_retries defaults to 3 when not overridden."""
        from cogtrix_core.providers.google import create_chat_model

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.ChatGoogleGenerativeAI", mock_cls):
            create_chat_model()

        assert mock_cls.call_args.kwargs["max_retries"] == 3

    def test_import_error_raised_when_library_not_installed(self) -> None:
        """ImportError is raised when langchain-google-genai is not installed."""
        from cogtrix_core.providers import google as google_mod

        original = google_mod.CHAT_AVAILABLE
        try:
            google_mod.CHAT_AVAILABLE = False
            with pytest.raises(ImportError, match="langchain-google-genai not installed"):
                google_mod.create_chat_model()
        finally:
            google_mod.CHAT_AVAILABLE = original

    def test_base_url_rejected_with_value_error(self) -> None:
        """Google provider does not support custom base_url."""
        from cogtrix_core.providers.google import create_chat_model

        with pytest.raises(ValueError, match="does not support custom base_url"):
            create_chat_model(base_url="http://proxy.example.com:8080")


class TestCreateEmbeddings:
    """Tests for google.create_embeddings()."""

    def test_default_model_from_embedding_models_registry(self) -> None:
        """Default embedding model is pulled from EMBEDDING_MODELS (text-embedding-004)."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            result = create_embeddings()

        mock_cls.assert_called_once()
        # Default model text-embedding-004 is added to models/ prefix
        assert mock_cls.call_args.kwargs["model"] == "models/text-embedding-004"
        assert result is mock_cls.return_value

    def test_custom_model_without_models_prefix_gets_prefix_added(self) -> None:
        """Model names without 'models/' prefix are prefixed with 'models/'."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            create_embeddings(model="text-embedding-005")

        assert mock_cls.call_args.kwargs["model"] == "models/text-embedding-005"

    def test_custom_model_with_models_prefix_passthrough(self) -> None:
        """Model names already prefixed with 'models/' are passed through unchanged."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            create_embeddings(model="models/text-embedding-004")

        assert mock_cls.call_args.kwargs["model"] == "models/text-embedding-004"

    def test_api_key_forwarded_as_google_api_key(self) -> None:
        """api_key kwarg is forwarded as 'google_api_key' to GoogleGenerativeAIEmbeddings."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            create_embeddings(api_key="AIza...")

        assert mock_cls.call_args.kwargs["google_api_key"] == "AIza..."

    def test_api_key_omitted_when_none(self) -> None:
        """google_api_key is not passed when None — SDK falls back to env var."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            create_embeddings(api_key=None)

        assert "google_api_key" not in mock_cls.call_args.kwargs

    def test_extra_kwargs_forwarded(self) -> None:
        """Extra kwargs are forwarded to GoogleGenerativeAIEmbeddings."""
        from cogtrix_core.providers.google import create_embeddings

        mock_cls = MagicMock()
        with patch("cogtrix_core.providers.google.GoogleGenerativeAIEmbeddings", mock_cls):
            create_embeddings(batch_size=256)

        assert mock_cls.call_args.kwargs["batch_size"] == 256

    def test_base_url_rejected_with_value_error(self) -> None:
        """Google embeddings does not support custom base_url."""
        from cogtrix_core.providers.google import create_embeddings

        with pytest.raises(ValueError, match="does not support custom base_url"):
            create_embeddings(base_url="http://proxy.example.com:8080")

    def test_import_error_raised_when_library_not_installed(self) -> None:
        """ImportError is raised when langchain-google-genai is not installed."""
        from cogtrix_core.providers import google as google_mod

        original = google_mod.EMBEDDINGS_AVAILABLE
        try:
            google_mod.EMBEDDINGS_AVAILABLE = False
            with pytest.raises(ImportError, match="langchain-google-genai not installed"):
                google_mod.create_embeddings()
        finally:
            google_mod.EMBEDDINGS_AVAILABLE = original
