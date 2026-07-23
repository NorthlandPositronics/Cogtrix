"""Regression tests for the Google Gemini provider.

ROOT CAUSE: ``create_chat_model()`` and ``create_embeddings()`` in
``src/providers/google.py`` accepted a ``base_url`` parameter but silently
ignored it, logging only a warning.  Users configuring a proxy or custom
endpoint believed traffic was routed through it while connections went
directly to the default Google API endpoint.

FIX: Raise ``ValueError`` when ``base_url`` is provided, forcing the caller
to fix their configuration instead of silently misrouting traffic.
"""

from __future__ import annotations

import pytest


class TestGoogleBaseUrlRejection:
    """base_url must raise ValueError instead of being silently ignored."""

    def test_chat_model_rejects_base_url(self) -> None:
        """create_chat_model must raise ValueError when base_url is set."""
        from src.providers.google import create_chat_model

        with pytest.raises(ValueError, match="does not support custom base_url"):
            create_chat_model(
                model="gemini-2.5-flash",
                base_url="http://proxy.example.com:8080",
            )

    def test_embeddings_rejects_base_url(self) -> None:
        """create_embeddings must raise ValueError when base_url is set."""
        from src.providers.google import create_embeddings

        with pytest.raises(ValueError, match="does not support custom base_url"):
            create_embeddings(
                model="text-embedding-004",
                base_url="http://proxy.example.com:8080",
            )

    def test_chat_model_allows_none_base_url(self) -> None:
        """create_chat_model must work normally when base_url is None."""
        from unittest.mock import MagicMock, patch

        from src.providers.google import create_chat_model

        with patch("src.providers.google.ChatGoogleGenerativeAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = create_chat_model(
                model="gemini-2.5-flash",
                base_url=None,
            )
            assert result is not None
            mock_cls.assert_called_once()

    def test_embeddings_allows_none_base_url(self) -> None:
        """create_embeddings must work normally when base_url is None."""
        from unittest.mock import MagicMock, patch

        from src.providers.google import create_embeddings

        with patch("src.providers.google.GoogleGenerativeAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = create_embeddings(
                model="text-embedding-004",
                base_url=None,
            )
            assert result is not None
            mock_cls.assert_called_once()
