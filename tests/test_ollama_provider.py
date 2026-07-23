"""Tests for the Ollama LLM provider adapter.

Ollama is the only local/self-hosted provider and is the default for new users.
A regression in model name resolution or URL configuration would break all
local/Ollama users.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCreateChatModel:
    """Tests for ollama.create_chat_model()."""

    def test_create_chat_model_with_defaults(self):
        """Default model and base_url are pulled from the defaults registry."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            result = create_chat_model()

        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:8b"
        assert call_kwargs["base_url"] == "http://localhost:11434"
        assert call_kwargs["temperature"] == 0.5
        assert result is mock_chat_ollama.return_value

    def test_create_chat_model_with_custom_model(self):
        """Custom model name is forwarded to ChatOllama."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(model="llama3:8b")

        assert mock_chat_ollama.call_args.kwargs["model"] == "llama3:8b"

    def test_create_chat_model_with_custom_base_url(self):
        """Custom base_url is forwarded to ChatOllama."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(base_url="http://192.168.1.100:11434")

        assert mock_chat_ollama.call_args.kwargs["base_url"] == "http://192.168.1.100:11434"

    def test_create_chat_model_temperature_forwarded(self):
        """Temperature is forwarded to ChatOllama."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(temperature=0.7)

        assert mock_chat_ollama.call_args.kwargs["temperature"] == 0.7

    def test_create_chat_model_top_p_forwarded(self):
        """Extra kwargs like top_p are forwarded to ChatOllama."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(top_p=0.9)

        assert mock_chat_ollama.call_args.kwargs["top_p"] == 0.9

    def test_create_chat_model_num_ctx_forwarded(self):
        """num_ctx is forwarded to ChatOllama when provided."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(num_ctx=32768)

        assert mock_chat_ollama.call_args.kwargs["num_ctx"] == 32768

    def test_create_chat_model_num_ctx_omitted_when_none(self):
        """num_ctx is not passed to ChatOllama when None (default)."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model()

        assert "num_ctx" not in mock_chat_ollama.call_args.kwargs

    def test_create_chat_model_all_kwargs_combined(self):
        """All parameters are correctly combined and forwarded."""
        from src.providers.ollama import create_chat_model

        mock_chat_ollama = MagicMock()
        with patch("src.providers.ollama.ChatOllama", mock_chat_ollama):
            create_chat_model(
                model="mistral:7b",
                base_url="http://ollama.local:11434",
                temperature=0.2,
                num_ctx=65536,
                top_p=0.95,
                seed=42,
            )

        kwargs = mock_chat_ollama.call_args.kwargs
        assert kwargs["model"] == "mistral:7b"
        assert kwargs["base_url"] == "http://ollama.local:11434"
        assert kwargs["temperature"] == 0.2
        assert kwargs["num_ctx"] == 65536
        assert kwargs["top_p"] == 0.95
        assert kwargs["seed"] == 42

    def test_create_chat_model_raises_import_error(self):
        """ImportError is raised when langchain-ollama is not installed."""
        from src.providers import ollama as ollama_mod

        original = ollama_mod.CHAT_AVAILABLE
        try:
            ollama_mod.CHAT_AVAILABLE = False
            with pytest.raises(ImportError, match="langchain-ollama not installed"):
                ollama_mod.create_chat_model()
        finally:
            ollama_mod.CHAT_AVAILABLE = original


class TestCreateEmbeddings:
    """Tests for ollama.create_embeddings()."""

    def test_create_embeddings_with_defaults(self):
        """Default model and base_url are pulled from the defaults registry."""
        from src.providers.ollama import create_embeddings

        mock_emb = MagicMock()
        with patch("src.providers.ollama.OllamaEmbeddings", mock_emb):
            result = create_embeddings()

        mock_emb.assert_called_once()
        call_kwargs = mock_emb.call_args.kwargs
        assert call_kwargs["model"] == "nomic-embed-text"
        assert call_kwargs["base_url"] == "http://localhost:11434"
        assert result is mock_emb.return_value

    def test_create_embeddings_with_custom_model(self):
        """Custom embedding model name is forwarded."""
        from src.providers.ollama import create_embeddings

        mock_emb = MagicMock()
        with patch("src.providers.ollama.OllamaEmbeddings", mock_emb):
            create_embeddings(model="mxbai-embed-large")

        assert mock_emb.call_args.kwargs["model"] == "mxbai-embed-large"

    def test_create_embeddings_with_custom_base_url(self):
        """Custom base_url is forwarded."""
        from src.providers.ollama import create_embeddings

        mock_emb = MagicMock()
        with patch("src.providers.ollama.OllamaEmbeddings", mock_emb):
            create_embeddings(base_url="http://remote:11434")

        assert mock_emb.call_args.kwargs["base_url"] == "http://remote:11434"

    def test_create_embeddings_extra_kwargs_forwarded(self):
        """Extra kwargs are forwarded to OllamaEmbeddings."""
        from src.providers.ollama import create_embeddings

        mock_emb = MagicMock()
        with patch("src.providers.ollama.OllamaEmbeddings", mock_emb):
            create_embeddings(num_ctx=512)

        assert mock_emb.call_args.kwargs["num_ctx"] == 512

    def test_create_embeddings_raises_import_error(self):
        """ImportError is raised when langchain-ollama is not installed."""
        from src.providers import ollama as ollama_mod

        original = ollama_mod.EMBEDDINGS_AVAILABLE
        try:
            ollama_mod.EMBEDDINGS_AVAILABLE = False
            with pytest.raises(ImportError, match="langchain-ollama not installed"):
                ollama_mod.create_embeddings()
        finally:
            ollama_mod.EMBEDDINGS_AVAILABLE = original
