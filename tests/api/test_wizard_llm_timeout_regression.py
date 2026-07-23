"""Regression tests for wizard LLM timeout wrapping (#1568, #1186).

Two bare ``llm.invoke()`` calls in ``cogtrix_core/api/routes/config.py``
(``_wizard_test_connection``, ``_wizard_invoke_llm``) are bounded by a hard
timeout. Under #1186 the raw ``with ThreadPoolExecutor(...) as pool`` block was
replaced by the centralized ``src.concurrency.invoke_with_timeout`` — the ``with``
form's ``__exit__`` calls ``shutdown(wait=True)``, which joins (and therefore
BLOCKS on) a worker still hung inside ``llm.invoke`` against an unresponsive
provider, defeating the timeout it wraps.

Tests verify BEHAVIOR (not the internal pool mechanics):
- Normal success paths are unchanged.
- A timeout returns gracefully AND — critically — returns quickly rather than
  blocking on the hung worker (the #1186 contract).
- A non-timeout exception is handled gracefully.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch


class _HangingLLM:
    """Fake LLM whose ``invoke`` blocks until released (or a hard cap) — models a
    provider that is not responding, to prove the caller is not blocked with it."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def invoke(self, messages: list) -> MagicMock:
        # Unblocked by the test after it asserts; the cap keeps a leaked thread
        # from lingering the whole session if an assertion fails first.
        self.release.wait(timeout=30)
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
    from cogtrix_core.api.routes.config import _wizard_invoke_llm

    result = _wizard_invoke_llm(_FastLLM(), [])
    assert result == "wizard response text"


def test_returns_empty_on_timeout_without_blocking() -> None:
    """Timeout: returns ``''``, logs a warning, and does NOT block on the hung
    worker (returns in ~timeout, not ~30s) — the #1186 contract."""
    from cogtrix_core.api.routes.config import _wizard_invoke_llm, log

    hanging = _HangingLLM()
    try:
        with (
            patch.object(log, "warning") as mock_warning,
            patch("cogtrix_core.api.routes.config._WIZARD_LLM_TIMEOUT_SECONDS", 0.2),
        ):
            start = time.monotonic()
            result = _wizard_invoke_llm(hanging, [])
            elapsed = time.monotonic() - start
    finally:
        hanging.release.set()

    assert result == ""
    assert elapsed < 5, f"caller blocked {elapsed:.1f}s on a hung invoke (footgun back?)"
    mock_warning.assert_called_once()
    assert "timed out after" in mock_warning.call_args[0][0]


def test_returns_empty_on_llm_exception_and_logs_warning() -> None:
    """Non-timeout exception: returns ``''`` and logs a warning with the exception."""
    from cogtrix_core.api.routes.config import _wizard_invoke_llm, log

    with patch.object(log, "warning") as mock_warning:
        result = _wizard_invoke_llm(_ErrorLLM(), [])

    assert result == ""
    mock_warning.assert_called_once()
    assert "raised exception: %s" in mock_warning.call_args[0][0]
    assert isinstance(mock_warning.call_args[0][1], RuntimeError)
    assert mock_warning.call_args[1].get("exc_info") is True


# ---------------------------------------------------------------------------
# _wizard_test_connection tests
# ---------------------------------------------------------------------------


def test_probe_returns_no_warning_on_success() -> None:
    """Normal path: ``_wizard_test_connection`` probe succeeds with no warning."""
    from cogtrix_core.api.routes.config import _wizard_test_connection

    with patch("cogtrix_core.agent.core.create_llm_from_provider_config") as mock_create:
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
    from cogtrix_core.api.routes.config import _wizard_test_connection

    with patch("cogtrix_core.agent.core.create_llm_from_provider_config") as mock_create:
        mock_create.return_value = _ErrorLLM()

        llm, probe_warning = _wizard_test_connection(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
            base_url=None,
        )

    assert probe_warning == "provider unreachable"
    assert llm is not None


def test_probe_returns_warning_on_timeout_without_blocking() -> None:
    """Timeout: the probe captures a timeout warning, proceeds, and does NOT block
    on the hung worker (the #1186 contract)."""
    from cogtrix_core.api.routes.config import _wizard_test_connection

    hanging = _HangingLLM()
    try:
        with (
            patch("cogtrix_core.agent.core.create_llm_from_provider_config") as mock_create,
            patch("cogtrix_core.api.routes.config._WIZARD_LLM_TIMEOUT_SECONDS", 0.2),
        ):
            mock_create.return_value = hanging
            start = time.monotonic()
            llm, probe_warning = _wizard_test_connection(
                provider_type="openai",
                model="gpt-4",
                api_key="sk-test",
                base_url=None,
            )
            elapsed = time.monotonic() - start
    finally:
        hanging.release.set()

    assert probe_warning is not None
    assert "timeout" in probe_warning
    assert llm is not None
    assert elapsed < 5, f"probe blocked {elapsed:.1f}s on a hung invoke (footgun back?)"
