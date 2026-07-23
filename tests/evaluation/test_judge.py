"""Tests for tests/evaluation/judge.py."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.evaluation.judge import (
    _build_judge_prompt,
    _fallback_score,
    _parse_score,
    judge_response,
    judge_result,
)
from tests.evaluation.runner import EvalResult, EvalScenario

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def scenario() -> EvalScenario:
    return EvalScenario(
        id="po_basic",
        domain="procurement",
        title="PO Approval",
        description="Create a purchase order from a supplier quote.",
        user_prompt="Quote is $487.50 for 50 units.",
        system_prompt="You are a procurement assistant.",
        tools_required=["create_po", "route_approval"],
        expected_outcome="PO created and routed for approval.",
        success_criteria=["PO created", "Approval routed"],
    )


@pytest.fixture
def result() -> EvalResult:
    return EvalResult(
        scenario_id="po_basic",
        model_id="gpt-4o",
        model_display_name="gpt-4o",
        passed=True,
        tool_calls_made=["create_po", "route_approval"],
        tool_calls_required=["create_po", "route_approval"],
        turns_used=3,
        elapsed_seconds=4.5,
        final_response="Purchase order #1234 created and sent to approver.",
    )


# ── _build_judge_prompt ──────────────────────────────────────────────────────


class TestBuildJudgePrompt:
    def test_includes_description(self, scenario: EvalScenario, result: EvalResult) -> None:
        prompt = _build_judge_prompt(scenario, result)
        assert scenario.description in prompt

    def test_includes_expected_outcome(self, scenario: EvalScenario, result: EvalResult) -> None:
        prompt = _build_judge_prompt(scenario, result)
        assert scenario.expected_outcome in prompt

    def test_includes_criteria(self, scenario: EvalScenario, result: EvalResult) -> None:
        prompt = _build_judge_prompt(scenario, result)
        assert "- PO created" in prompt
        assert "- Approval routed" in prompt

    def test_includes_tools_called(self, scenario: EvalScenario, result: EvalResult) -> None:
        prompt = _build_judge_prompt(scenario, result)
        assert "create_po, route_approval" in prompt

    def test_caps_final_response(self, scenario: EvalScenario, result: EvalResult) -> None:
        long_response = "x" * 5000
        r = replace(result, final_response=long_response)
        prompt = _build_judge_prompt(scenario, r)
        assert len(prompt) < 6000  # capped + template overhead

    def test_none_tools(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, tool_calls_made=[])
        prompt = _build_judge_prompt(scenario, r)
        assert "TOOLS CALLED: none" in prompt


# ── _parse_score ─────────────────────────────────────────────────────────────


class TestParseScore:
    def test_valid_json(self) -> None:
        assert _parse_score('{"score": 0.75, "reason": "partial"}') == 0.75

    def test_valid_json_integer(self) -> None:
        assert _parse_score('{"score": 1, "reason": "good"}') == 1.0

    def test_json_in_markdown_fence(self) -> None:
        raw = '```json\n{"score": 0.5, "reason": "ok"}\n```'
        assert _parse_score(raw) == 0.5

    def test_json_in_plain_fence(self) -> None:
        raw = '```\n{"score": 0.25}\n```'
        assert _parse_score(raw) == 0.25

    def test_clamps_above_one(self) -> None:
        assert _parse_score('{"score": 1.5}') == 1.0

    def test_clamps_below_zero(self) -> None:
        assert _parse_score('{"score": -0.3}') == 0.0

    def test_regex_fallback(self) -> None:
        assert _parse_score('"score" : 0.8') == 0.8
        assert _parse_score('"score"=0.9') == 0.9

    def test_plain_number_fallback(self) -> None:
        assert _parse_score("The score is 0.6 overall.") == 0.6
        assert _parse_score("score 0.4") == 0.4

    def test_no_score_returns_none(self) -> None:
        assert _parse_score("I think it did well.") is None

    def test_invalid_json_no_score_key(self) -> None:
        assert _parse_score('{"grade": 0.8}') is None


# ── _fallback_score ──────────────────────────────────────────────────────────


class TestFallbackScore:
    def test_all_tools_called(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, tool_calls_made=["create_po", "route_approval"], error=None)
        assert _fallback_score(scenario, r) == 1.0

    def test_partial_tools(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, tool_calls_made=["create_po"], error=None)
        assert _fallback_score(scenario, r) == 0.5

    def test_no_tools(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, tool_calls_made=[], error=None)
        assert _fallback_score(scenario, r) == 0.0

    def test_error_returns_zero(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, error="timeout", tool_calls_made=["create_po", "route_approval"])
        assert _fallback_score(scenario, r) == 0.0

    def test_no_required_tools(self, scenario: EvalScenario, result: EvalResult) -> None:
        s = replace(scenario, tools_required=[])
        r = replace(result, tool_calls_made=[], final_response="something")
        assert _fallback_score(s, r) == 1.0

    def test_no_required_tools_empty_response(
        self, scenario: EvalScenario, result: EvalResult
    ) -> None:
        s = replace(scenario, tools_required=[])
        r = replace(result, tool_calls_made=[], final_response="")
        assert _fallback_score(s, r) == 0.0


# ── judge_response ───────────────────────────────────────────────────────────


class TestJudgeResponse:
    def test_happy_path(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 1.0, "reason": "perfect"}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()) as mock_get:
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                score = judge_response(scenario, result, judge_model="claude-sonnet-4-6")

        assert score == 1.0
        mock_get.assert_called_once_with("claude-sonnet-4-6")
        mock_llm.invoke.assert_called_once()

    def test_unknown_model_falls_back(self, scenario: EvalScenario, result: EvalResult) -> None:
        with patch("tests.evaluation.judge.get_model", side_effect=KeyError("unknown")):
            score = judge_response(scenario, result, judge_model="unknown")
        # fallback: all tools called → 1.0
        assert score == 1.0

    def test_missing_api_key_falls_back(self, scenario: EvalScenario, result: EvalResult) -> None:
        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch(
                "tests.evaluation.judge._build_llm",
                side_effect=OSError("no key"),
            ):
                score = judge_response(scenario, result)
        assert score == 1.0

    def test_llm_exception_falls_back(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("model error")

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                score = judge_response(scenario, result)
        assert score == 1.0

    def test_malformed_response_falls_back(
        self, scenario: EvalScenario, result: EvalResult
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I think it did well.")

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                score = judge_response(scenario, result)
        # fallback heuristic
        assert score == 1.0

    def test_temperature_passed_to_build_llm(
        self, scenario: EvalScenario, result: EvalResult
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.8}')
        mock_model_cfg = MagicMock()

        with patch("tests.evaluation.judge.get_model", return_value=mock_model_cfg):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm) as mock_build:
                judge_response(scenario, result)

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("temperature") == 0.0

    def test_score_clamped(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 2.5}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                score = judge_response(scenario, result)
        assert score == 1.0

    def test_partial_credit(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.5, "reason": "partial"}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                score = judge_response(scenario, result)
        assert score == 0.5


# ── judge_result ─────────────────────────────────────────────────────────────


class TestJudgeResult:
    def test_passed_when_score_ge_half(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.75}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judged = judge_result(scenario, result)

        assert judged.passed is True
        assert "judge_score=0.75" in judged.notes

    def test_failed_when_score_lt_half(self, scenario: EvalScenario, result: EvalResult) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.25}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judged = judge_result(scenario, result)

        assert judged.passed is False
        assert "judge_score=0.25" in judged.notes

    def test_preserves_existing_notes(self, scenario: EvalScenario, result: EvalResult) -> None:
        r = replace(result, notes="already noted")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 1.0}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judged = judge_result(scenario, r)

        assert judged.notes == "already noted | judge_score=1.00"
