"""Tests for tests/quality/judge.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.quality.judge import _build_judge_prompt, _fallback_score, _parse_score, score_scenario


@pytest.fixture
def result() -> dict[str, object]:
    return {
        "scenario": {
            "description": "Create a purchase order from a supplier quote.",
            "expected_outcome": "PO created and routed for approval.",
            "criteria": ["PO created", "Approval routed"],
            "tools_required": ["create_po", "route_approval"],
            "tools_called": ["create_po", "route_approval"],
        },
        "final_response": "Purchase order #1234 created and sent to approver.",
        "tool_calls_made": ["create_po", "route_approval"],
    }


class TestBuildJudgePrompt:
    def test_includes_description(self, result: dict[str, object]) -> None:
        prompt = _build_judge_prompt(result)
        assert "Create a purchase order" in prompt

    def test_includes_expected_outcome(self, result: dict[str, object]) -> None:
        prompt = _build_judge_prompt(result)
        assert "PO created and routed for approval." in prompt

    def test_includes_criteria(self, result: dict[str, object]) -> None:
        prompt = _build_judge_prompt(result)
        assert "- PO created" in prompt
        assert "- Approval routed" in prompt

    def test_includes_tools_called(self, result: dict[str, object]) -> None:
        prompt = _build_judge_prompt(result)
        assert "create_po, route_approval" in prompt

    def test_caps_final_response(self, result: dict[str, object]) -> None:
        r = dict(result)
        r["final_response"] = "x" * 5000
        prompt = _build_judge_prompt(r)
        assert len(prompt) < 6000


class TestParseScore:
    def test_valid_json(self) -> None:
        assert _parse_score('{"score": 0.75, "reason": "partial"}') == 0.75

    def test_json_in_markdown_fence(self) -> None:
        raw = '```json\n{"score": 0.5, "reason": "ok"}\n```'
        assert _parse_score(raw) == 0.5

    def test_regex_fallback(self) -> None:
        assert _parse_score('"score" : 0.8') == 0.8

    def test_no_score_returns_none(self) -> None:
        assert _parse_score("I think it did well.") is None


class TestFallbackScore:
    def test_all_tools_called(self, result: dict[str, object]) -> None:
        assert _fallback_score(result) == 1.0

    def test_partial_tools(self, result: dict[str, object]) -> None:
        r = dict(result)
        r["tool_calls_made"] = ["create_po"]
        assert _fallback_score(r) == 0.5

    def test_no_tools(self, result: dict[str, object]) -> None:
        r = dict(result)
        r["tool_calls_made"] = []
        assert _fallback_score(r) == 0.0

    def test_error_returns_zero(self, result: dict[str, object]) -> None:
        r = dict(result)
        r["error"] = "timeout"
        assert _fallback_score(r) == 0.0

    def test_no_required_tools(self, result: dict[str, object]) -> None:
        r = dict(result)
        r["scenario"] = {"final_response": "something"}
        assert _fallback_score(r) == 1.0


class TestScoreScenario:
    def test_happy_path(self, result: dict[str, object]) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 1.0, "reason": "perfect"}')

        with patch("tests.quality.judge.get_model", return_value=MagicMock()) as mock_get:
            with patch("tests.quality.judge._build_llm", return_value=mock_llm):
                score = score_scenario(result, judge_model="claude-sonnet-4-6")

        assert score == 1.0
        mock_get.assert_called_once_with("claude-sonnet-4-6")
        mock_llm.invoke.assert_called_once()

    def test_unknown_model_falls_back(self, result: dict[str, object]) -> None:
        with patch("tests.quality.judge.get_model", side_effect=KeyError("unknown")):
            score = score_scenario(result, judge_model="unknown")
        assert score == 1.0

    def test_missing_api_key_falls_back(self, result: dict[str, object]) -> None:
        with patch("tests.quality.judge.get_model", return_value=MagicMock()):
            with patch("tests.quality.judge._build_llm", side_effect=OSError("no key")):
                score = score_scenario(result)
        assert score == 1.0

    def test_malformed_response_falls_back(self, result: dict[str, object]) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I think it did well.")

        with patch("tests.quality.judge.get_model", return_value=MagicMock()):
            with patch("tests.quality.judge._build_llm", return_value=mock_llm):
                score = score_scenario(result)
        assert score == 1.0
