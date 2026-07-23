"""Tests for the CLI query-complexity heuristic."""

from __future__ import annotations

from cogtrix import _classify_query_complexity


class TestClassifyQueryComplexity:
    def test_simple_casual_prompt_stays_simple(self) -> None:
        assert _classify_query_complexity("How are you today?") == "simple"

    def test_reasoning_prompt_does_not_route_as_simple(self) -> None:
        assert _classify_query_complexity("Explain how functions work.") == "complex"

    def test_analysis_prompt_does_not_route_as_simple(self) -> None:
        assert _classify_query_complexity("Analyze this.") == "complex"

    def test_code_prompt_does_not_route_as_simple(self) -> None:
        assert _classify_query_complexity("Fix the bug in shell.py.") == "complex"
