"""Tests for cogtrix_core/orchestration/reflection_delegate.py."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.orchestration.reflection_delegate import (
    CounterPlanEvaluator,
    PlanGenerator,
    _call_llm,
)


class TestCallLlmTimeout:
    """Regression tests for #1558: _call_llm must timeout instead of hanging.

    Tests pre-#1903 mocked ``concurrent.futures.ThreadPoolExecutor`` to
    inject the timeout/exception.  After the migration to
    :func:`src.concurrency.invoke_with_timeout` (#1903), the pool
    construction moved out of this module into the centralized helper;
    these tests now mock ``invoke_with_timeout`` at its import site
    (``src.orchestration.reflection_delegate.invoke_with_timeout``) so
    the behavioural contract — timeout / exception → empty string with
    WARNING log — is verified independently of the pool plumbing.

    The two pre-#1903 ``test_pool_shutdown_called*`` tests were deleted:
    they pinned the *implementation* detail that ``_call_llm`` shut down
    its own pool, which is no longer this function's responsibility.
    The shared pool's shutdown contract is covered by
    ``tests/test_concurrency.py::TestSharedPoolContract``.
    """

    def _make_llm(self, content="ok"):
        """Create a fake LLM that returns *content*."""
        llm = MagicMock()
        result = MagicMock()
        result.content = content
        llm.invoke.return_value = result
        return llm

    def test_returns_content_on_success(self):
        llm = self._make_llm("plan text")
        result = _call_llm(llm, "generate a plan")
        assert result == "plan text"
        llm.invoke.assert_called_once()

    def test_returns_empty_on_timeout(self, caplog: pytest.LogCaptureFixture):
        """When ``invoke_with_timeout`` raises ``TimeoutError``, return empty string."""
        llm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch(
                "cogtrix_core.orchestration.reflection_delegate.invoke_with_timeout",
                side_effect=TimeoutError("timed out"),
            ):
                result = _call_llm(llm, "generate a plan")

        assert result == ""
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected WARNING log on timeout"
        assert any("timed out" in r.getMessage().lower() for r in warning_records)

    def test_returns_empty_on_llm_exception(self, caplog: pytest.LogCaptureFixture):
        """Non-timeout exceptions still return empty string with a warning."""
        llm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch(
                "cogtrix_core.orchestration.reflection_delegate.invoke_with_timeout",
                side_effect=RuntimeError("model exploded"),
            ):
                result = _call_llm(llm, "generate a plan")

        assert result == ""
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected WARNING log on exception"
        assert any("failed" in r.getMessage().lower() for r in warning_records)


class TestPlanGeneratorTimeout:
    """PlanGenerator.generate_plan must not hang when _call_llm times out."""

    def test_generate_plan_returns_fallback_on_timeout(self):
        """When _call_llm returns empty string, generate_plan falls back gracefully."""
        llm = MagicMock()

        with patch("cogtrix_core.orchestration.reflection_delegate._call_llm", return_value=""):
            generator = PlanGenerator(llm=llm)
            snapshot = generator.generate_plan("do something complex")

        assert snapshot["plan"] == "Plan for: do something complex"
        assert snapshot["assumptions"] == ["No explicit assumptions identified"]
        assert snapshot["evidence"] == ["No explicit evidence cited"]


class TestCounterPlanEvaluatorTimeout:
    """CounterPlanEvaluator must not hang when _call_llm times out."""

    def test_evaluate_plan_returns_fallback_on_timeout(self):
        """When _call_llm returns empty string, counter-plan falls back gracefully."""
        llm = MagicMock()
        plan = {
            "plan": "test plan",
            "assumptions": ["a1"],
            "evidence": ["e1"],
            "confidence": 8.0,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        with patch("cogtrix_core.orchestration.reflection_delegate._call_llm", return_value=""):
            evaluator = CounterPlanEvaluator(llm=llm)
            justification = evaluator.evaluate_plan(plan, "test task")

        assert justification["counter_plan"] == "Counter-plan unavailable for: test plan"
        assert justification["flaws"] == []
        assert justification["should_proceed"] is True  # 8.0 + 0 >= 7.0
