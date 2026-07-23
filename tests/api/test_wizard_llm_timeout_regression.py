"""Regression tests for wizard LLM timeout wrapping (issue #1568).

Issue #1558 Phase 2 Rank 7 — two bare ``llm.invoke()`` calls in
``src/api/routes/config.py`` (``_wizard_test_connection``,
``_wizard_invoke_llm``) were wrapped in ThreadPoolExecutor with 60s timeout.

Tests verify:
- Normal success paths are unchanged
- Timeout raises ``FuturesTimeoutError`` and is handled gracefully
- Pool is always shut down regardless of outcome
"""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch


class _SlowLLM:
    """Fake LLM whose ``invoke`` hangs past the 60s timeout."""

    def invoke(self, messages: list) -> MagicMock:
        time.sleep(120)  # deliberately longer than the 60s timeout
        return MagicMock(content="ok")


class _FastLLM:
    """Fake LLM whose ``invoke`` returns immediately."""

    def invoke(self, messages: list) -> MagicMock:
        return MagicMock(content="wizard response text")


class _ErrorLLM:
    """Fake LLM whose ``invoke`` raises an exception."""

    def invoke(self, messages: list) -> MagicMock:
        raise RuntimeError("provider unreachable")


# ---------------------------------------------------------------------------
# _wizard_invoke_llm tests
# ---------------------------------------------------------------------------


def test_returns_content_on_success() -> None:
    """Normal path: ``_wizard_invoke_llm`` returns the LLM response content."""
    from src.api.routes.config import _wizard_invoke_llm

    result = _wizard_invoke_llm(_FastLLM(), [])
    assert result == "wizard response text"


def test_returns_empty_on_timeout() -> None:
    """Timeout: ``_wizard_invoke_llm`` returns ``''`` on ``FuturesTimeoutError`` and logs a warning."""
    from src.api.routes.config import _wizard_invoke_llm, log

    with (
        patch.object(log, "warning") as mock_warning,
        patch("src.api.routes.config.ThreadPoolExecutor") as MockPool,
    ):
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_executor.submit.return_value = mock_future

        result = _wizard_invoke_llm(MagicMock(), [])

    assert result == ""
    mock_warning.assert_called_once()
    assert "timed out after" in mock_warning.call_args[0][0]


def test_pool_shutdown_called_on_timeout() -> None:
    """Timeout: pool.shutdown(wait=False, cancel_futures=True) is always called."""
    from src.api.routes.config import _wizard_invoke_llm

    with patch("src.api.routes.config.ThreadPoolExecutor") as MockPool:
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.side_effect = FuturesTimeoutError()

        _wizard_invoke_llm(MagicMock(), [])

    mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


def test_pool_shutdown_called_on_success() -> None:
    """Success: pool.shutdown(wait=False, cancel_futures=True) is always called."""
    from src.api.routes.config import _wizard_invoke_llm

    with patch("src.api.routes.config.ThreadPoolExecutor") as MockPool:
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_future = MagicMock()
        mock_future.result.return_value = MagicMock(content="ok")
        mock_executor.submit.return_value = mock_future

        _wizard_invoke_llm(MagicMock(), [])

    mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


def test_returns_empty_on_llm_exception() -> None:
    """Non-timeout exception: ``_wizard_invoke_llm`` returns ``''`` (unchanged behavior)."""
    from src.api.routes.config import _wizard_invoke_llm

    with patch("src.api.routes.config.ThreadPoolExecutor") as MockPool:
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.side_effect = RuntimeError("boom")

        result = _wizard_invoke_llm(MagicMock(), [])

    assert result == ""


# ---------------------------------------------------------------------------
# _wizard_test_connection tests
# ---------------------------------------------------------------------------


def test_probe_returns_no_warning_on_success() -> None:
    """Normal path: ``_wizard_test_connection`` probe succeeds with no warning."""
    from src.api.routes.config import _wizard_test_connection

    with patch("src.agent.core.create_llm_from_provider_config") as mock_create:
        mock_create.return_value = _FastLLM()

        llm, probe_warning = _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    assert probe_warning is None
    assert llm is not None


def test_probe_returns_warning_on_exception() -> None:
    """Non-timeout exception: ``_wizard_test_connection`` captures warning and proceeds."""
    from src.api.routes.config import _wizard_test_connection

    with patch("src.agent.core.create_llm_from_provider_config") as mock_create:
        mock_create.return_value = _ErrorLLM()

        llm, probe_warning = _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    assert probe_warning == "provider unreachable"
    assert llm is not None


def test_probe_returns_warning_on_timeout() -> None:
    """Timeout: ``_wizard_test_connection`` captures warning, logs, and proceeds."""
    from src.api.routes.config import _wizard_test_connection

    with (
        patch("src.agent.core.create_llm_from_provider_config") as mock_create,
        patch("src.api.routes.config.ThreadPoolExecutor") as MockPool,
    ):
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.side_effect = FuturesTimeoutError()
        mock_create.return_value = MagicMock()

        llm, probe_warning = _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    assert probe_warning is not None
    assert "timeout" in probe_warning
    assert llm is not None
    mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


def test_probe_pool_shutdown_called_on_timeout() -> None:
    """Timeout: pool.shutdown(wait=False, cancel_futures=True) is always called."""
    from src.api.routes.config import _wizard_test_connection

    with (
        patch("src.agent.core.create_llm_from_provider_config") as mock_create,
        patch("src.api.routes.config.ThreadPoolExecutor") as MockPool,
    ):
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.side_effect = FuturesTimeoutError()
        mock_create.return_value = MagicMock()

        _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


def test_probe_pool_shutdown_called_on_success() -> None:
    """Success: pool.shutdown(wait=False, cancel_futures=True) is always called."""
    from src.api.routes.config import _wizard_test_connection

    with (
        patch("src.agent.core.create_llm_from_provider_config") as mock_create,
        patch("src.api.routes.config.ThreadPoolExecutor") as MockPool,
    ):
        mock_executor = MagicMock()
        MockPool.return_value.__enter__.return_value = mock_executor
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_executor.submit.return_value = mock_future
        mock_create.return_value = _FastLLM()

        _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
