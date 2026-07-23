"""Tests for RetryableChatModel — rate limits, auth errors, and 5xx transient errors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from src.providers import (  # noqa: F401
        ProviderAuthError,
        RateLimitError,
        RetryableChatModel,
        create_chat_model,
    )
else:
    try:
        from src.providers import (  # noqa: F811
            ProviderAuthError,
            RateLimitError,
            RetryableChatModel,
            create_chat_model,
        )
    except ImportError:
        pytest.skip(
            "RetryableChatModel not yet available (requires PR #694)",
            allow_module_level=True,
        )
        raise  # unreachable — satisfies pyright that names are defined


class MockLangChainModel:
    """Mock LangChain chat model for testing."""

    def __init__(self, name: str = "mock-model"):
        self.provider_name = name
        self.max_retries = 3

    def invoke(self, *args, **kwargs):
        """Mock invoke method."""
        raise NotImplementedError("Mock invoke not implemented")

    def __repr__(self):
        return f"MockLangChainModel({self.provider_name!r})"


class TestProviderAuthError:
    """Tests for ProviderAuthError exception."""

    def test_provider_auth_error_with_provider(self):
        """ProviderAuthError can include provider name."""
        error = ProviderAuthError("Auth failed", provider="openai")
        assert str(error) == "Auth failed"
        assert error.provider == "openai"

    def test_provider_auth_error_without_provider(self):
        """ProviderAuthError works without provider name."""
        error = ProviderAuthError("Auth failed")
        assert str(error) == "Auth failed"
        assert error.provider is None


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_with_retry_after(self):
        """RateLimitError can include retry_after value."""
        error = RateLimitError("Rate limited", retry_after=5.0)
        assert str(error) == "Rate limited"
        assert error.retry_after == 5.0

    def test_rate_limit_error_without_retry_after(self):
        """RateLimitError works without retry_after."""
        error = RateLimitError("Rate limited")
        assert str(error) == "Rate limited"
        assert error.retry_after is None


class TestRetryableChatModel:
    """Tests for RetryableChatModel wrapper."""

    def test_invoke_succeeds_on_first_try(self):
        """Success on first attempt doesn't retry."""
        mock_model = MockLangChainModel()
        mock_model.invoke = MagicMock(return_value="success")

        wrapper = RetryableChatModel(mock_model)
        result = wrapper.invoke("test input")

        assert result == "success"
        mock_model.invoke.assert_called_once()

    def test_invoke_retries_on_rate_limit(self):
        """Rate limit errors trigger retries with exponential backoff."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=0.1)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise rate_limit_error
            return "success after retries"

        mock_model.invoke = MagicMock(side_effect=side_effect)

        wrapper = RetryableChatModel(mock_model, max_retries=3, initial_delay=0.01)
        result = wrapper.invoke("test input")

        assert result == "success after retries"
        assert call_count == 3

    def test_invoke_fails_after_max_retries(self):
        """After max retries, RateLimitError is raised."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=0.01)

        def side_effect(*args, **kwargs):
            raise rate_limit_error

        mock_model.invoke = MagicMock(side_effect=side_effect)

        wrapper = RetryableChatModel(mock_model, max_retries=2, initial_delay=0.01)
        with pytest.raises(RateLimitError) as exc_info:
            wrapper.invoke("test input")

        assert "exhausted after 3 attempts" in str(exc_info.value)
        assert exc_info.value.retry_after == 0.01

    def test_auth_error_raises_immediately(self):
        """Authentication errors fail immediately without retry."""
        mock_model = MockLangChainModel()
        auth_error = Exception("Authentication failed: Invalid API key")

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError) as exc_info:
            wrapper.invoke("test input")

        assert "Authentication failed" in str(exc_info.value)
        mock_model.invoke.assert_called_once()

    def test_auth_error_from_status_code_401(self):
        """401 status code triggers auth error."""
        mock_model = MockLangChainModel()
        auth_error = Exception()
        auth_error.status_code = 401  # type: ignore[attr-defined]

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError):
            wrapper.invoke("test input")

    def test_auth_error_from_status_code_403(self):
        """403 status code triggers auth error."""
        mock_model = MockLangChainModel()
        auth_error = Exception()
        auth_error.status_code = 403  # type: ignore[attr-defined]

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError):
            wrapper.invoke("test input")

    def test_500_not_retried(self) -> None:
        """500 Internal Server Error is NOT retried — permanent, not transient."""
        mock_model = MagicMock()
        error_500 = Exception("500 Internal Server Error")
        error_500.status_code = 500  # type: ignore[attr-defined]

        mock_model.invoke = MagicMock(side_effect=error_500)

        wrapper = RetryableChatModel(mock_model, initial_delay=0.01)
        with pytest.raises(Exception) as exc_info:
            wrapper.invoke("test input")

        assert "500 Internal Server Error" in str(exc_info.value)
        mock_model.invoke.assert_called_once()

    def test_non_retryable_error_fails_immediately(self):
        """Non-retryable errors fail immediately."""
        mock_model = MockLangChainModel()
        other_error = ValueError("Some other error")

        mock_model.invoke = MagicMock(side_effect=other_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ValueError) as exc_info:
            wrapper.invoke("test input")

        assert "Some other error" in str(exc_info.value)
        mock_model.invoke.assert_called_once()

    def test_default_max_retries(self):
        """Default max_retries is 3."""
        mock_model = MockLangChainModel()
        wrapper = RetryableChatModel(mock_model)
        assert wrapper._max_retries == 3

    def test_custom_max_retries(self):
        """Custom max_retries is respected."""
        mock_model = MockLangChainModel()
        wrapper = RetryableChatModel(mock_model, max_retries=5)
        assert wrapper._max_retries == 5

    def test_delegates_attribute_access(self):
        """Attribute access is delegated to wrapped model."""
        mock_model = MockLangChainModel()
        mock_model.some_attr = "test_value"

        wrapper = RetryableChatModel(mock_model)
        assert wrapper.some_attr == "test_value"

    def test_repr_includes_wrapped_model(self):
        """__repr__ shows the wrapped model."""
        mock_model = MockLangChainModel("test-model")
        wrapper = RetryableChatModel(mock_model)

        repr_str = repr(wrapper)
        assert "RetryableChatModel" in repr_str
        assert "test-model" in repr_str


class TestRetryableChatModel5xx:
    """Tests for 5xx transient server error retry behavior."""

    def test_retries_on_503_and_succeeds(self) -> None:
        """503 Service Unavailable triggers retry, succeeds on next attempt."""
        mock_model = MagicMock()
        error_503 = Exception("503 Service Unavailable")
        error_503.status_code = 503  # type: ignore[attr-defined]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error_503
            return "success after 503"

        mock_model.invoke = MagicMock(side_effect=side_effect)

        wrapper = RetryableChatModel(mock_model, initial_delay=0.01)
        result = wrapper.invoke("test input")

        assert result == "success after 503"
        assert call_count == 2

    def test_exhausts_retries_on_repeated_502(self) -> None:
        """Repeated 502 Bad Gateway exhausts retries and raises RateLimitError."""
        mock_model = MagicMock()
        error_502 = Exception("502 Bad Gateway")
        error_502.status_code = 502  # type: ignore[attr-defined]

        mock_model.invoke = MagicMock(side_effect=error_502)

        wrapper = RetryableChatModel(mock_model, max_retries=2, initial_delay=0.01)
        with pytest.raises(RateLimitError) as exc_info:
            wrapper.invoke("test input")

        assert "exhausted after 3 attempts" in str(exc_info.value)


class TestCreateChatModelWrapping:
    """Tests that create_chat_model wraps models with RetryableChatModel."""

    @patch("src.providers._load_provider")
    def test_create_chat_model_returns_wrapped_model(self, mock_load):
        """create_chat_model wraps model with RetryableChatModel."""
        mock_module = MagicMock()
        mock_module.CHAT_AVAILABLE = True
        mock_model = MockLangChainModel()
        mock_module.create_chat_model = MagicMock(return_value=mock_model)
        mock_load.return_value = mock_module

        with patch("src.providers.CHAT_MODELS", {"openai": "gpt-4"}):
            result = create_chat_model("openai", model="gpt-4")

        assert isinstance(result, RetryableChatModel)
        assert result._model == mock_model

    @patch("src.providers._load_provider")
    def test_create_chat_model_respects_max_retries(self, mock_load):
        """create_chat_model passes max_retries to wrapper."""
        mock_module = MagicMock()
        mock_module.CHAT_AVAILABLE = True
        mock_model = MockLangChainModel()
        mock_module.create_chat_model = MagicMock(return_value=mock_model)
        mock_load.return_value = mock_module

        with patch("src.providers.CHAT_MODELS", {"openai": "gpt-4"}):
            result = create_chat_model("openai", model="gpt-4", max_retries=5)

        assert isinstance(result, RetryableChatModel)
        assert result._max_retries == 5


class TestExtractRetryAfter:
    """Tests for _extract_retry_after helper."""

    def test_extract_from_headers(self):
        """Extracts from response.headers."""
        from src.providers import _extract_retry_after

        response = MagicMock()
        response.headers = {"Retry-After": "5.5"}

        result = _extract_retry_after(response)
        assert result == 5.5

    def test_extract_from_none_response(self):
        """Returns None for None response."""
        from src.providers import _extract_retry_after

        assert _extract_retry_after(None) is None

    def test_extract_falls_back_to_none(self):
        """Returns None if no Retry-After found."""
        from src.providers import _extract_retry_after

        response = MagicMock()
        response.headers = {}

        result = _extract_retry_after(response)
        assert result is None

    def test_extract_logs_debug_on_exception(self, caplog):
        """Logs debug message when response parsing raises an exception."""
        from src.providers import _extract_retry_after

        class BadResponse:
            headers = {}
            text = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        response = BadResponse()

        with caplog.at_level("DEBUG", logger="cogtrix.providers"):
            result = _extract_retry_after(response)

        assert result is None
        assert "_extract_retry_after: failed to parse response" in caplog.text
        assert "boom" in caplog.text
