"""Tests for src/orchestration/reflection_delegate.py."""

import concurrent.futures
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.reflection_delegate import (
    _REFLECTION_LLM_TIMEOUT_SECONDS,
    CounterPlanEvaluator,
    PlanGenerator,
    _call_llm,
)


class TestCallLlmTimeout:
    """Regression tests for #1558: _call_llm must timeout instead of hanging."""

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
        """When future.result(timeout=60) raises TimeoutError, return empty string."""
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = concurrent.futures.TimeoutError("timed out")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        llm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch("concurrent.futures.ThreadPoolExecutor", return_value=fake_pool) as mock_tpe:
                result = _call_llm(llm, "generate a plan")

        assert result == ""
        mock_tpe.assert_called_once_with(max_workers=1)
        fake_future.result.assert_called_once_with(timeout=_REFLECTION_LLM_TIMEOUT_SECONDS)
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected WARNING log on timeout"
        assert any("timed out" in r.getMessage().lower() for r in warning_records)

    def test_returns_empty_on_plain_timeout_error(self, caplog: pytest.LogCaptureFixture):
        """Built-in TimeoutError must also be caught and return empty string."""
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = TimeoutError("plain timeout")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        llm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch("concurrent.futures.ThreadPoolExecutor", return_value=fake_pool):
                result = _call_llm(llm, "generate a plan")

        assert result == ""
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected WARNING log on plain TimeoutError"

    def test_returns_empty_on_llm_exception(self, caplog: pytest.LogCaptureFixture):
        """Non-timeout exceptions still return empty string with a warning."""
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = RuntimeError("model exploded")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        llm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch("concurrent.futures.ThreadPoolExecutor", return_value=fake_pool):
                result = _call_llm(llm, "generate a plan")

        assert result == ""
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected WARNING log on exception"
        assert any("failed" in r.getMessage().lower() for r in warning_records)

    def test_pool_shutdown_called(self):
        """Pool.shutdown(wait=False) must be called even on success."""
        fake_future = MagicMock(spec=concurrent.futures.Future)
        result = MagicMock()
        result.content = "ok"
        fake_future.result.return_value = result

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        llm = MagicMock()

        with patch("concurrent.futures.ThreadPoolExecutor", return_value=fake_pool):
            _call_llm(llm, "generate a plan")

        fake_pool.shutdown.assert_called_once_with(wait=False)

    def test_pool_shutdown_called_on_timeout(self):
        """Pool.shutdown(wait=False) must be called even on timeout."""
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = concurrent.futures.TimeoutError("timed out")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        llm = MagicMock()

        with patch("concurrent.futures.ThreadPoolExecutor", return_value=fake_pool):
            _call_llm(llm, "generate a plan")

        fake_pool.shutdown.assert_called_once_with(wait=False)


class TestPlanGeneratorTimeout:
    """PlanGenerator.generate_plan must not hang when _call_llm times out."""

    def test_generate_plan_returns_fallback_on_timeout(self):
        """When _call_llm returns empty string, generate_plan falls back gracefully."""
        llm = MagicMock()

        with patch("src.orchestration.reflection_delegate._call_llm", return_value=""):
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

        with patch("src.orchestration.reflection_delegate._call_llm", return_value=""):
            evaluator = CounterPlanEvaluator(llm=llm)
            justification = evaluator.evaluate_plan(plan, "test task")

        assert justification["counter_plan"] == "Counter-plan unavailable for: test plan"
        assert justification["flaws"] == []
        assert justification["should_proceed"] is True  # 8.0 + 0 >= 7.0
