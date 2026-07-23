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

    def bind_tools(self, tools, **kwargs):
        """Mock bind_tools — returns a new model with tools bound."""
        bound = MockLangChainModel(self.provider_name)
        bound._bound_tools = tools
        # Copy the invoke mock so tests can assert on it
        bound.invoke = self.invoke  # type: ignore[method-assign]
        return bound

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


class TestRetryableChatModelDisableRetries:
    """Tests for _cogtrix_disable_retries kwarg (issue #1069).

    When _invoke_with_timeout submits a model call to the shared
    ThreadPoolExecutor, the inner retry loop must be disabled so that
    rate-limit backoff does not block a scarce worker thread.
    """

    def test_disable_retries_skips_inner_retry(self):
        """_cogtrix_disable_retries=True bypasses retry logic entirely."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=0.1)
        mock_model.invoke = MagicMock(side_effect=rate_limit_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3, initial_delay=0.01)
        with pytest.raises(RateLimitError):
            wrapper.invoke("test input", _cogtrix_disable_retries=True)

        # Should call the underlying model exactly once (no retries)
        mock_model.invoke.assert_called_once()
        # Verify the flag is NOT passed through to the underlying model
        _, kwargs = mock_model.invoke.call_args
        assert "_cogtrix_disable_retries" not in kwargs

    def test_disable_retries_returns_success(self):
        """_cogtrix_disable_retries=True works for successful calls."""
        mock_model = MockLangChainModel()
        mock_model.invoke = MagicMock(return_value="success")

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        result = wrapper.invoke("test input", _cogtrix_disable_retries=True)

        assert result == "success"
        mock_model.invoke.assert_called_once()

    def test_disable_retries_passes_through_kwargs(self):
        """Other kwargs are preserved when _cogtrix_disable_retries is used."""
        mock_model = MockLangChainModel()
        mock_model.invoke = MagicMock(return_value="success")

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        wrapper.invoke(
            "test input",
            temperature=0.7,
            max_tokens=100,
            _cogtrix_disable_retries=True,
        )

        _, kwargs = mock_model.invoke.call_args
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("max_tokens") == 100
        assert "_cogtrix_disable_retries" not in kwargs

    def test_default_behavior_retries_still_work(self):
        """Without the flag, normal retry behavior is preserved."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=0.01)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise rate_limit_error
            return "success"

        mock_model.invoke = MagicMock(side_effect=side_effect)

        wrapper = RetryableChatModel(mock_model, max_retries=3, initial_delay=0.01)
        result = wrapper.invoke("test input")

        assert result == "success"
        assert call_count == 3


class TestRetryableChatModelBindTools:
    """Tests for bind_tools re-wrapping (issue #1069 follow-up).

    Without an explicit ``bind_tools`` override, ``llm.bind_tools()``
    delegates through ``__getattr__`` to the raw underlying model.
    The returned bound model bypasses ``RetryableChatModel.invoke()``,
    causing the ``_cogtrix_disable_retries`` flag to leak to the API
    client and trigger a TypeError.
    """

    def test_bind_tools_returns_retryable_chat_model(self):
        """bind_tools wraps the result in RetryableChatModel."""
        mock_model = MockLangChainModel()
        wrapper = RetryableChatModel(mock_model, max_retries=5, initial_delay=2.0)

        bound = wrapper.bind_tools(["tool_a"])

        assert isinstance(bound, RetryableChatModel)
        assert bound._max_retries == 5
        assert bound._initial_delay == 2.0

    def test_bound_model_pops_disable_retries(self):
        """Bound model still handles _cogtrix_disable_retries correctly."""
        mock_model = MockLangChainModel()
        mock_model.invoke = MagicMock(return_value="success")

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        bound = wrapper.bind_tools(["tool_a"])

        result = bound.invoke("test input", _cogtrix_disable_retries=True)

        assert result == "success"
        mock_model.invoke.assert_called_once()
        _, kwargs = mock_model.invoke.call_args
        assert "_cogtrix_disable_retries" not in kwargs


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


class TestRetryableChatModelExponentialBackoff:
    """Regression tests for exponential backoff in RetryableChatModel (issue #1511)."""

    def test_exponential_backoff_increases_sleep_duration(self):
        """Sleep duration must double on each retry when no Retry-After header."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise rate_limit_error

        mock_model.invoke = MagicMock(side_effect=side_effect)
        wrapper = RetryableChatModel(mock_model, max_retries=3, initial_delay=1.0, max_delay=30.0)

        sleep_durations = []

        def capture_sleep(duration):
            sleep_durations.append(duration)

        with patch("src.providers.time.sleep", side_effect=capture_sleep):
            with pytest.raises(RateLimitError):
                wrapper.invoke("test input")

        # 3 retries = 3 sleeps before exhaustion
        assert len(sleep_durations) == 3
        assert sleep_durations[0] == 1.0  # initial delay
        assert sleep_durations[1] == 2.0  # doubled
        assert sleep_durations[2] == 4.0  # doubled again

    def test_server_retry_after_used_on_first_attempt(self):
        """Server-provided Retry-After is used on first retry, then backoff."""
        mock_model = MockLangChainModel()

        # First call: no Retry-After header → use exponential backoff
        # Second call: server says Retry-After=5.0 → use that
        call_count = 0

        class ErrorWithResponse(Exception):
            pass

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            exc = ErrorWithResponse("Rate limit exceeded")
            exc.status_code = 429
            if call_count == 2:
                # Second failure has Retry-After header
                exc.response = MagicMock()
                exc.response.headers = {"Retry-After": "5.0"}
            else:
                exc.response = None
            raise exc

        mock_model.invoke = MagicMock(side_effect=side_effect)
        wrapper = RetryableChatModel(mock_model, max_retries=3, initial_delay=1.0, max_delay=30.0)

        sleep_durations = []

        def capture_sleep(duration):
            sleep_durations.append(duration)

        with patch("src.providers.time.sleep", side_effect=capture_sleep):
            with pytest.raises(RateLimitError):
                wrapper.invoke("test input")

        # 3 retries = 3 sleeps
        assert len(sleep_durations) == 3
        # First retry: no Retry-After → use initial delay (1.0)
        assert sleep_durations[0] == 1.0
        # Second retry: server says 5.0 → use that
        assert sleep_durations[1] == 5.0
        # Third retry: no Retry-After again → use doubled delay (4.0)
        # (delay was doubled to 2.0 after first sleep, then to 4.0 after second)
        assert sleep_durations[2] == 4.0

    def test_max_delay_caps_backoff(self):
        """Exponential backoff is capped at max_delay."""
        mock_model = MockLangChainModel()
        rate_limit_error = RateLimitError("Rate limit exceeded")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise rate_limit_error

        mock_model.invoke = MagicMock(side_effect=side_effect)
        wrapper = RetryableChatModel(mock_model, max_retries=5, initial_delay=2.0, max_delay=5.0)

        sleep_durations = []

        def capture_sleep(duration):
            sleep_durations.append(duration)

        with patch("src.providers.time.sleep", side_effect=capture_sleep):
            with pytest.raises(RateLimitError):
                wrapper.invoke("test input")

        # 5 retries = 5 sleeps
        assert len(sleep_durations) == 5
        assert sleep_durations[0] == 2.0  # initial
        assert sleep_durations[1] == 4.0  # doubled
        assert sleep_durations[2] == 5.0  # capped at max_delay
        assert sleep_durations[3] == 5.0  # still capped
        assert sleep_durations[4] == 5.0  # still capped


class TestRedactUrl:
    """Regression tests for _redact_url — ensure credential query params are redacted."""

    def _call(self, url: str) -> str:
        from src.providers import _redact_url

        return _redact_url(url)

    def test_redacts_api_key(self):
        assert self._call("https://api.example.com/path?api_key=sk-abc123") == (
            "https://api.example.com/path?api_key=[redacted]"
        )

    def test_redacts_apikey(self):
        assert self._call("https://api.example.com/path?apikey=sk-xyz") == (
            "https://api.example.com/path?apikey=[redacted]"
        )

    def test_redacts_password(self):
        assert self._call("https://api.example.com/path?password=secret") == (
            "https://api.example.com/path?password=[redacted]"
        )

    def test_redacts_token(self):
        assert self._call("https://api.example.com/path?token=mytoken") == (
            "https://api.example.com/path?token=[redacted]"
        )

    def test_redacts_auth_token(self):
        assert self._call("https://api.example.com/path?auth_token=bearer123") == (
            "https://api.example.com/path?auth_token=[redacted]"
        )

    def test_redacts_key(self):
        """#1071 — 'key' is a common credential param missing from original sensitive_keys."""
        assert self._call("https://api.example.com/path?key=sk-abc123") == (
            "https://api.example.com/path?key=[redacted]"
        )

    def test_redacts_secret(self):
        """#1071 — 'secret' is used by AWS-style credential embedding."""
        assert self._call("https://api.example.com/path?secret=mysecret") == (
            "https://api.example.com/path?secret=[redacted]"
        )

    def test_redacts_access_token(self):
        """#1071 — 'access_token' is used by OAuth-style flows."""
        assert self._call("https://api.example.com/path?access_token=atk_456") == (
            "https://api.example.com/path?access_token=[redacted]"
        )

    def test_redacts_api_secret(self):
        """#1071 — 'api_secret' is paired with api_key in some APIs."""
        assert self._call("https://api.example.com/path?api_secret=apisecret") == (
            "https://api.example.com/path?api_secret=[redacted]"
        )

    def test_redacts_private_token(self):
        """#1071 — 'private_token' is used by GitLab-style APIs."""
        assert self._call("https://api.example.com/path?private_token=pt_789") == (
            "https://api.example.com/path?private_token=[redacted]"
        )

    def test_redacts_refresh_token(self):
        """#1508 — 'refresh_token' used in OAuth query-param flows."""
        assert self._call("https://auth.example.com/oauth?refresh_token=rft_abc") == (
            "https://auth.example.com/oauth?refresh_token=[redacted]"
        )

    def test_redacts_client_secret(self):
        """#1508 — 'client_secret' used in OAuth client credentials flow."""
        assert self._call("https://auth.example.com/token?client_secret=cs_xyz") == (
            "https://auth.example.com/token?client_secret=[redacted]"
        )

    def test_non_sensitive_params_preserved(self):
        """Params not in sensitive_keys are not redacted."""
        result = self._call("https://api.example.com/path?key=sk-abc&page=2&limit=10")
        assert "key=[redacted]" in result
        assert "page=2" in result
        assert "limit=10" in result

    def test_empty_query_preserved(self):
        """URLs without query strings are returned safely."""
        assert self._call("https://api.example.com/path") == ("https://api.example.com/path")

    def test_none_url_returns_placeholder(self):
        assert self._call(None) == "<unparseable URL>"

    def test_case_insensitive_match(self):
        """Sensitive key matching is case-insensitive."""
        assert self._call("https://api.example.com/path?KEY=sk-abc") == (
            "https://api.example.com/path?KEY=[redacted]"
        )
        assert self._call("https://api.example.com/path?Secret=topsecret") == (
            "https://api.example.com/path?Secret=[redacted]"
        )


class TestSanitizeAuthErrorMessage:
    """Regression tests for _sanitize_auth_error_message (issue #1105)."""

    def _call(self, message: str) -> str:
        from src.providers import _sanitize_auth_error_message

        return _sanitize_auth_error_message(message)

    def test_redacts_openai_sk_key(self):
        """OpenAI-style sk- prefix is redacted."""
        raw = "Authentication failed: Incorrect API key: sk-test123abc"
        result = self._call(raw)
        assert "sk-test123abc" not in result
        assert "[redacted]" in result

    def test_redacts_anthropic_sk_ant_key(self):
        """Anthropic-style sk-ant- prefix is redacted."""
        raw = "Auth error: sk-ant-api03-test-key-xyz"
        result = self._call(raw)
        assert "sk-ant-api03-test-key-xyz" not in result
        assert "[redacted]" in result

    def test_redacts_google_aiza_key(self):
        """Google-style AIza prefix is redacted."""
        raw = "Authentication failed: Invalid API key AIzaSyA12345"
        result = self._call(raw)
        assert "AIzaSyA12345" not in result
        assert "[redacted]" in result

    def test_leaves_safe_messages_intact(self):
        """Messages without key patterns are unchanged."""
        safe = "Authentication failed: Invalid credentials"
        assert self._call(safe) == safe

    def test_redacts_multiple_keys_in_one_message(self):
        """Multiple key fragments are all redacted."""
        raw = "Keys sk-aaa and AIzaBBB found"
        result = self._call(raw)
        assert "sk-aaa" not in result
        assert "AIzaBBB" not in result
        assert result.count("[redacted]") == 2

    def test_idempotent_on_already_sanitized_message(self):
        """Re-sanitizing an already-redacted message is a no-op."""
        once = self._call("Auth failed: sk-secret123")
        twice = self._call(once)
        assert once == twice
        assert "[redacted]" in twice

    def test_no_false_positive_on_substring_skills(self):
        """Substrings like 'task-skills-dev' must not be redacted."""
        safe = "Deploy task-skills-dev to staging"
        result = self._call(safe)
        assert result == safe
        assert "[redacted]" not in result

    def test_redacts_xai_key(self):
        """xAI-style xai- prefix is redacted."""
        raw = "Auth error: xai-secret-token-abc"
        result = self._call(raw)
        assert "xai-secret-token-abc" not in result
        assert "[redacted]" in result

    def test_redacts_groq_key(self):
        """Groq-style gsk_ prefix is redacted."""
        raw = "Invalid API key: gsk_test_12345"
        result = self._call(raw)
        assert "gsk_test_12345" not in result
        assert "[redacted]" in result

    def test_redacts_huggingface_key(self):
        """HuggingFace-style hf_ prefix is redacted."""
        raw = "Auth failed: hf_abcdef123456"
        result = self._call(raw)
        assert "hf_abcdef123456" not in result
        assert "[redacted]" in result


class TestRetryableChatModelAuthErrorSanitization:
    """Regression tests for auth-error key sanitization in RetryableChatModel (issue #1105)."""

    def test_auth_error_sanitizes_sk_key(self):
        """ProviderAuthError message does not contain sk- key fragments."""
        mock_model = MockLangChainModel()
        auth_error = Exception("Authentication failed: Incorrect API key: sk-secret123")

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError) as exc_info:
            wrapper.invoke("test input")

        assert "sk-secret123" not in str(exc_info.value)
        assert "[redacted]" in str(exc_info.value)
        assert "Authentication failed" in str(exc_info.value)

    def test_auth_error_sanitizes_google_key(self):
        """ProviderAuthError message does not contain AIza key fragments."""
        mock_model = MockLangChainModel()
        auth_error = Exception("Authentication failed: Invalid API key: AIzaSyA12345")

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError) as exc_info:
            wrapper.invoke("test input")

        assert "AIzaSyA12345" not in str(exc_info.value)
        assert "[redacted]" in str(exc_info.value)

    def test_auth_error_without_key_unchanged(self):
        """Safe auth error messages are preserved."""
        mock_model = MockLangChainModel()
        auth_error = Exception("Authentication failed: Invalid credentials")

        mock_model.invoke = MagicMock(side_effect=auth_error)

        wrapper = RetryableChatModel(mock_model, max_retries=3)
        with pytest.raises(ProviderAuthError) as exc_info:
            wrapper.invoke("test input")

        assert "Invalid credentials" in str(exc_info.value)
        assert "[redacted]" not in str(exc_info.value)
