"""Tests for tests/evaluation/judge.py."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.evaluation.judge import (
    _build_judge_prompt,
    _build_judge_prompt_for_turn,
    _fallback_score,
    _parse_score,
    judge_response,
    judge_result,
)
from tests.evaluation.runner import EvalResult, EvalScenario, Turn, TurnResult

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


# ── Multi-turn judging (issue #1545, PR 2 of 3) ──────────────────────────────


def _multi_turn_scenario(weights: list[float] | None = None) -> EvalScenario:
    """Build a 3-turn scenario with author-controlled judge weights."""
    weights = weights or [1.0, 1.0, 1.0]
    return EvalScenario(
        id="mt_scenario",
        domain="test",
        title="t",
        description="A three-turn related-sequence workflow.",
        system_prompt="",
        tools_required=[],
        expected_outcome="All three turns succeed in sequence.",
        turns=[
            Turn(
                user_prompt=f"turn {i + 1} prompt",
                success_criteria=[f"contains: turn{i + 1}"],
                judge_weight=weights[i],
            )
            for i in range(3)
        ],
    )


def _multi_turn_result(turn_finals: list[str]) -> EvalResult:
    """Build an EvalResult whose turn_results aligns with the scenario above."""
    return EvalResult(
        scenario_id="mt_scenario",
        model_id="m",
        model_display_name="M",
        passed=True,
        tool_calls_made=[],
        tool_calls_required=[],
        turns_used=3,
        elapsed_seconds=1.0,
        final_response=turn_finals[-1],
        turn_results=[TurnResult(final_response=f, tool_calls_made=[]) for f in turn_finals],
    )


class TestJudgeResponseMultiTurn:
    def test_invokes_judge_once_per_turn(self) -> None:
        """Multi-turn dispatch calls the judge LLM once per turn — not once for the session."""
        scenario = _multi_turn_scenario()
        result = _multi_turn_result(["turn1 done", "turn2 done", "turn3 done"])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 1.0}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judge_response(scenario, result)

        assert mock_llm.invoke.call_count == 3

    def test_weighted_aggregate(self) -> None:
        """Aggregate = sum(score_i * weight_i) / sum(weight_i).

        Three turns with weights [1, 1, 3] and scores [0.5, 0.5, 1.0]:
        weighted = (0.5*1 + 0.5*1 + 1.0*3) / (1 + 1 + 3) = 4.0 / 5 = 0.80
        """
        scenario = _multi_turn_scenario(weights=[1.0, 1.0, 3.0])
        result = _multi_turn_result(["t1", "t2", "t3"])

        mock_llm = MagicMock()
        # Three judge invocations in order: 0.5, 0.5, 1.0.
        mock_llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.5}'),
            MagicMock(content='{"score": 0.5}'),
            MagicMock(content='{"score": 1.0}'),
        ]

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                aggregate = judge_response(scenario, result)

        assert aggregate == pytest.approx(0.80)

    def test_equal_weights_unweighted_mean(self) -> None:
        """Equal weights collapse to a simple mean (1+0+0.5)/3 = 0.5."""
        scenario = _multi_turn_scenario(weights=[1.0, 1.0, 1.0])
        result = _multi_turn_result(["t1", "t2", "t3"])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content='{"score": 1.0}'),
            MagicMock(content='{"score": 0.0}'),
            MagicMock(content='{"score": 0.5}'),
        ]

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                aggregate = judge_response(scenario, result)

        assert aggregate == pytest.approx(0.5)

    def test_any_turn_failure_falls_back_to_heuristic(self) -> None:
        """If any per-turn judge call errors or returns malformed output, the
        whole result falls back to the heuristic.  Same all-or-nothing
        semantics as the single-turn path — keeps the behaviour contract
        consistent across multi-turn rollout."""
        scenario = _multi_turn_scenario()
        result = _multi_turn_result(["t1", "t2", "t3"])

        mock_llm = MagicMock()
        # First two turns score 1.0, third returns malformed output → heuristic kicks in.
        mock_llm.invoke.side_effect = [
            MagicMock(content='{"score": 1.0}'),
            MagicMock(content='{"score": 1.0}'),
            MagicMock(content="I think it did well overall."),
        ]

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                aggregate = judge_response(scenario, result)

        # tools_required=[] → fallback path: 1.0 if final_response else 0.0.
        # Final response is "t3", so heuristic gives 1.0.
        assert aggregate == 1.0

    def test_single_turn_unaffected_when_turn_results_is_length_one(self) -> None:
        """A scenario with one turn and turn_results of length 1 still
        takes the single-call path — the multi-turn dispatch requires
        ``len(turn_results) > 1``."""
        scenario = EvalScenario(
            id="single",
            domain="test",
            title="t",
            description="d",
            system_prompt="",
            tools_required=[],
            expected_outcome="",
            turns=[Turn(user_prompt="only", success_criteria=[])],
        )
        result = EvalResult(
            scenario_id="single",
            model_id="m",
            model_display_name="M",
            passed=True,
            tool_calls_made=[],
            tool_calls_required=[],
            turns_used=1,
            elapsed_seconds=1.0,
            final_response="response",
            turn_results=[TurnResult(final_response="response", tool_calls_made=[])],
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.7}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                aggregate = judge_response(scenario, result)

        assert aggregate == pytest.approx(0.7)
        # Crucially: invoked exactly once (single-call path).
        assert mock_llm.invoke.call_count == 1

    def test_mismatched_turn_counts_falls_back_to_single_call(self) -> None:
        """If scenario.turns and result.turn_results disagree in length —
        e.g. a stale or hand-constructed EvalResult — the dispatch
        cannot align per-turn pairs, so it falls back to the single-call
        path instead of misaligning data."""
        scenario = _multi_turn_scenario()  # 3 turns
        # Only 2 turn_results (mismatch).
        result = EvalResult(
            scenario_id="mt_scenario",
            model_id="m",
            model_display_name="M",
            passed=True,
            tool_calls_made=[],
            tool_calls_required=[],
            turns_used=2,
            elapsed_seconds=1.0,
            final_response="last",
            turn_results=[
                TurnResult(final_response="t1", tool_calls_made=[]),
                TurnResult(final_response="t2", tool_calls_made=[]),
            ],
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.9}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                aggregate = judge_response(scenario, result)

        assert aggregate == pytest.approx(0.9)
        assert mock_llm.invoke.call_count == 1


class TestJudgeResultNotesMultiTurn:
    def test_notes_include_per_turn_breakdown(self) -> None:
        """Multi-turn notes show the aggregate + per-turn scores."""
        scenario = _multi_turn_scenario(weights=[1.0, 1.0, 3.0])
        result = _multi_turn_result(["t1", "t2", "t3"])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.5}'),
            MagicMock(content='{"score": 0.5}'),
            MagicMock(content='{"score": 1.0}'),
        ]

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judged = judge_result(scenario, result)

        assert judged.passed is True
        assert "judge_score=0.80" in judged.notes
        assert "(turns: 0.50, 0.50, 1.00)" in judged.notes

    def test_notes_single_turn_unchanged(self) -> None:
        """Single-turn note format is unchanged: ``judge_score=X.YY``
        with no ``(turns: ...)`` suffix — backward-compat for dashboards
        that pattern-match on this string."""
        scenario = EvalScenario(
            id="single",
            domain="test",
            title="t",
            description="d",
            system_prompt="",
            tools_required=[],
            expected_outcome="",
            turns=[Turn(user_prompt="only", success_criteria=[])],
        )
        result = EvalResult(
            scenario_id="single",
            model_id="m",
            model_display_name="M",
            passed=True,
            tool_calls_made=[],
            tool_calls_required=[],
            turns_used=1,
            elapsed_seconds=1.0,
            final_response="ok",
            turn_results=[TurnResult(final_response="ok", tool_calls_made=[])],
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.9}')

        with patch("tests.evaluation.judge.get_model", return_value=MagicMock()):
            with patch("tests.evaluation.judge._build_llm", return_value=mock_llm):
                judged = judge_result(scenario, result)

        assert "judge_score=0.90" in judged.notes
        assert "(turns:" not in judged.notes


class TestBuildJudgePromptForTurn:
    def test_includes_turn_position(self) -> None:
        scenario = _multi_turn_scenario()
        prompt = _build_judge_prompt_for_turn(
            scenario,
            scenario.turns[1],
            TurnResult(final_response="t2 response", tool_calls_made=["foo"]),
            turn_idx=1,
            total_turns=3,
        )
        assert "turn 2 of 3" in prompt

    def test_includes_turn_user_prompt_and_criteria(self) -> None:
        scenario = _multi_turn_scenario()
        prompt = _build_judge_prompt_for_turn(
            scenario,
            scenario.turns[0],
            TurnResult(final_response="t1 response", tool_calls_made=[]),
            turn_idx=0,
            total_turns=3,
        )
        assert "turn 1 prompt" in prompt
        assert "- contains: turn1" in prompt

    def test_falls_back_to_none_for_empty_tools(self) -> None:
        scenario = _multi_turn_scenario()
        prompt = _build_judge_prompt_for_turn(
            scenario,
            scenario.turns[0],
            TurnResult(final_response="t1", tool_calls_made=[]),
            turn_idx=0,
            total_turns=3,
        )
        assert "TOOLS CALLED ON THIS TURN: none" in prompt
