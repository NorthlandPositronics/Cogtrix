"""Tests for cogtrix_core/tools/calculator.py — factorial cap and safe evaluation."""

import pytest

from cogtrix_core.tools.calculator import _safe_factorial, calculate


class TestSafeFactorial:
    def test_factorial_cap_raises_for_1559(self) -> None:
        with pytest.raises(ValueError, match="factorial argument too large"):
            _safe_factorial(1559)

    def test_factorial_cap_raises_for_value_above_limit(self) -> None:
        with pytest.raises(ValueError):
            _safe_factorial(99999)

    def test_factorial_exactly_at_limit_succeeds(self) -> None:
        result = _safe_factorial(1558)
        assert result > 0

    def test_factorial_small_value_returns_correct(self) -> None:
        assert _safe_factorial(5) == 120

    def test_factorial_zero_returns_one(self) -> None:
        assert _safe_factorial(0) == 1

    def test_factorial_one_returns_one(self) -> None:
        assert _safe_factorial(1) == 1


class TestCalculateFactorial:
    def test_factorial_via_calculate_small(self) -> None:
        result = calculate("factorial(5)")
        assert result == "120"

    def test_factorial_via_calculate_cap_returns_error(self) -> None:
        result = calculate("factorial(1559)")
        assert result.startswith("Error")
        assert "factorial" in result.lower() or "large" in result.lower()
