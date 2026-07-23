"""Behavioral coverage for the deep_think 3-phase engine.

Issue #421 scopes Phase 1 coverage for cogtrix_core/tools/deep_think.py:
- single iteration through branch → develop → converge
- high-confidence early convergence
- max-iteration exhaustion
- malformed JSON recovery
- empty-context safety
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from cogtrix_core.tools.deep_think import deep_think


@dataclass
class _ScriptedLLM:
    """Thread-safe scripted LLM for deterministic deep_think tests."""

    responses: list[str]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.prompts: list[str] = []

    def __copy__(self) -> _ScriptedLLM:
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        prompt = messages[0].content if messages else ""
        with self._lock:
            self.prompts.append(str(prompt))
            if not self.responses:
                raise AssertionError("unexpected deep_think LLM call")
            content = self.responses.pop(0)
        return AIMessage(content=content)


def _branch_payload(*branches: dict[str, str]) -> str:
    return json.dumps(list(branches))


def _develop_payload(
    *,
    solution: str,
    confidence: float,
    plan: str = "Plan the approach.",
    execution: str = "Execute the plan.",
    observation: str = "Observe the outcome.",
    reflection: str = "Reflect on the result.",
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "plan": plan,
            "execution": execution,
            "solution": solution,
            "observation": observation,
            "reflection": reflection,
            "confidence": confidence,
            "strengths": strengths or ["clear structure"],
            "weaknesses": weaknesses or ["still needs validation"],
        }
    )


def _converge_payload(
    *,
    scores: list[tuple[str, float, str]],
    solution: str,
    reasoning: str,
    confidence: float,
    should_continue: bool,
    next_focus: str = "",
    patterns: str = "Common pattern.",
    mistakes: str = "Minor weaknesses.",
    missed: str = "No major misses.",
    insights: str = "Carry the best branch forward.",
    improvements: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "evaluations": [
                {"name": name, "score": score, "verdict": verdict}
                for name, score, verdict in scores
            ],
            "reflection": {
                "patterns": patterns,
                "mistakes": mistakes,
                "missed": missed,
                "insights": insights,
            },
            "synthesis": {
                "solution": solution,
                "reasoning": reasoning,
                "improvements_made": improvements or ["kept the strongest branch"],
            },
            "confidence": confidence,
            "should_continue": should_continue,
            "next_focus": next_focus,
        }
    )


class TestDeepThinkBehavior:
    def test_single_iteration_runs_branch_develop_converge(self) -> None:
        """A clean scripted run should execute all three phases exactly once."""
        llm = _ScriptedLLM(
            responses=[
                _branch_payload(
                    {
                        "name": "Breadth-first",
                        "strategy": "Explore the task broadly.",
                        "rationale": "Useful when the space is uncertain.",
                        "risks": "May miss the best direct answer.",
                    },
                    {
                        "name": "Focused",
                        "strategy": "Concentrate on the safest path.",
                        "rationale": "Reduces unnecessary branching.",
                        "risks": "Could under-explore alternatives.",
                    },
                    {
                        "name": "Fallback",
                        "strategy": "Work backward from the expected result.",
                        "rationale": "Good for constraint-heavy tasks.",
                        "risks": "Depends on a correct end-state guess.",
                    },
                ),
                _develop_payload(
                    solution="Breadth-first solution",
                    confidence=6.0,
                ),
                _develop_payload(
                    solution="Focused solution",
                    confidence=7.5,
                ),
                _develop_payload(
                    solution="Fallback solution",
                    confidence=5.5,
                ),
                _converge_payload(
                    scores=[
                        ("Breadth-first", 7.0, "Good coverage."),
                        ("Focused", 8.5, "Best balance."),
                        ("Fallback", 6.0, "Too speculative."),
                    ],
                    solution="Focused solution, refined with breadth-first checks.",
                    reasoning="The focused branch won on practicality.",
                    confidence=8.4,
                    should_continue=False,
                ),
            ]
        )

        report = deep_think(
            task="Decide the best path for a bounded analysis task.",
            context="Customer notes and constraints for the current task.",
            max_iterations=1,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )

        assert "## Iteration 1" in report
        assert "Breadth-first" in report
        assert "Focused solution, refined with breadth-first checks." in report
        assert "confidence: 8.4/10" in report
        assert llm.prompts[0].startswith("You are a strategic problem-solver")
        assert llm.prompts[0].count("CONTEXT:") == 1
        assert len(llm.prompts) == 5

    def test_high_confidence_stops_after_second_iteration(self) -> None:
        """High confidence should stop the loop once iteration 2 is complete."""
        llm = _ScriptedLLM(
            responses=[
                _branch_payload(
                    {
                        "name": "Plan A",
                        "strategy": "Approach A.",
                        "rationale": "Baseline.",
                        "risks": "Known tradeoffs.",
                    },
                    {
                        "name": "Plan B",
                        "strategy": "Approach B.",
                        "rationale": "Alternative.",
                        "risks": "Different tradeoffs.",
                    },
                    {
                        "name": "Plan C",
                        "strategy": "Approach C.",
                        "rationale": "Fallback.",
                        "risks": "More complex.",
                    },
                ),
                *[
                    _develop_payload(solution=f"Iteration 1 branch {i}", confidence=7.0 + i / 10)
                    for i in range(3)
                ],
                _converge_payload(
                    scores=[
                        ("Plan A", 7.4, "Promising."),
                        ("Plan B", 8.0, "Best first pass."),
                        ("Plan C", 6.8, "Least attractive."),
                    ],
                    solution="Iteration 1 synthesis.",
                    reasoning="Good but not enough to stop early yet.",
                    confidence=9.6,
                    should_continue=True,
                    next_focus="Tighten the strongest branch.",
                ),
                _branch_payload(
                    {
                        "name": "Plan A v2",
                        "strategy": "Refined A.",
                        "rationale": "Improve the first attempt.",
                        "risks": "May still miss edge cases.",
                    },
                    {
                        "name": "Plan B v2",
                        "strategy": "Refined B.",
                        "rationale": "Carry forward best ideas.",
                        "risks": "Requires more synthesis.",
                    },
                    {
                        "name": "Plan C v2",
                        "strategy": "Refined C.",
                        "rationale": "Keep a fallback.",
                        "risks": "More work.",
                    },
                ),
                *[
                    _develop_payload(solution=f"Iteration 2 branch {i}", confidence=7.6 + i / 10)
                    for i in range(3)
                ],
                _converge_payload(
                    scores=[
                        ("Plan A v2", 8.2, "Better."),
                        ("Plan B v2", 9.0, "Best final answer."),
                        ("Plan C v2", 6.7, "Still fallback."),
                    ],
                    solution="Iteration 2 synthesis.",
                    reasoning="This is strong enough to stop.",
                    confidence=9.7,
                    should_continue=True,
                    next_focus="",
                ),
            ]
        )

        report = deep_think(
            task="Refine the best approach until confidence is high.",
            context="Only the task-specific facts for this turn.",
            max_iterations=5,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )

        assert report.count("## Iteration ") == 2
        assert "Iteration 2 synthesis." in report
        assert "confidence: 9.7/10" in report
        assert len(llm.prompts) == 10

    def test_max_iterations_returns_best_result(self) -> None:
        """When confidence stays low, the engine should run until max_iterations."""
        llm = _ScriptedLLM(
            responses=[
                _branch_payload(
                    {
                        "name": "One",
                        "strategy": "Try one.",
                        "rationale": "Simple.",
                        "risks": "Limited.",
                    },
                    {
                        "name": "Two",
                        "strategy": "Try two.",
                        "rationale": "Alternate.",
                        "risks": "Still limited.",
                    },
                    {
                        "name": "Three",
                        "strategy": "Try three.",
                        "rationale": "Backup.",
                        "risks": "Same risk.",
                    },
                ),
                *[
                    _develop_payload(solution=f"Iteration 1 solution {i}", confidence=4.0 + i)
                    for i in range(3)
                ],
                _converge_payload(
                    scores=[
                        ("One", 5.0, "Okay."),
                        ("Two", 6.0, "Best of the first round."),
                        ("Three", 4.0, "Weak."),
                    ],
                    solution="Iteration 1 best result.",
                    reasoning="Keep going.",
                    confidence=4.9,
                    should_continue=True,
                    next_focus="Improve the strongest branch.",
                ),
                _branch_payload(
                    {
                        "name": "One v2",
                        "strategy": "Try one again.",
                        "rationale": "Sharper.",
                        "risks": "Still limited.",
                    },
                    {
                        "name": "Two v2",
                        "strategy": "Try two again.",
                        "rationale": "Improved.",
                        "risks": "Some risk.",
                    },
                    {
                        "name": "Three v2",
                        "strategy": "Try three again.",
                        "rationale": "Fallback.",
                        "risks": "Still the fallback.",
                    },
                ),
                *[
                    _develop_payload(solution=f"Iteration 2 solution {i}", confidence=5.0 + i)
                    for i in range(3)
                ],
                _converge_payload(
                    scores=[
                        ("One v2", 6.0, "Improved."),
                        ("Two v2", 7.5, "Best final."),
                        ("Three v2", 4.5, "Still weak."),
                    ],
                    solution="Iteration 2 best result.",
                    reasoning="Max iterations reached.",
                    confidence=5.4,
                    should_continue=True,
                    next_focus="",
                ),
            ]
        )

        report = deep_think(
            task="Stop after the configured iteration budget if confidence stays low.",
            context="Task-specific facts only.",
            max_iterations=2,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )

        assert report.count("## Iteration ") == 2
        assert "Iteration 2 best result." in report
        assert "2 iterations, 6 branches explored" in report
        assert len(llm.prompts) == 10

    def test_malformed_json_recovers_with_fallback_branch(self) -> None:
        """Malformed JSON should fall back instead of failing the whole tool."""
        llm = _ScriptedLLM(
            responses=[
                "this is not json",
                "still not json",
                "neither is this",
            ]
        )

        report = deep_think(
            task="Recover from a malformed model response.",
            context="",
            max_iterations=1,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )

        assert "Direct approach" in report
        assert "neither is this" in report
        assert len(llm.prompts) == 3

    def test_empty_context_safe_default_behavior(self) -> None:
        """An empty context should still produce a full report with no crash."""
        llm = _ScriptedLLM(
            responses=[
                _branch_payload(
                    {
                        "name": "Minimal",
                        "strategy": "Proceed directly.",
                        "rationale": "No context means fewer moving parts.",
                        "risks": "May miss nuance.",
                    },
                    {
                        "name": "Alternative",
                        "strategy": "Check a second path.",
                        "rationale": "Useful when context is absent.",
                        "risks": "Can still be overcautious.",
                    },
                    {
                        "name": "Fallback",
                        "strategy": "Use the simplest safe answer.",
                        "rationale": "Good for empty inputs.",
                        "risks": "Might be too generic.",
                    },
                ),
                _develop_payload(solution="Minimal solution", confidence=6.2),
                _develop_payload(solution="Alternative solution", confidence=6.5),
                _develop_payload(solution="Fallback solution", confidence=5.8),
                _converge_payload(
                    scores=[
                        ("Minimal", 7.6, "Best for empty context."),
                        ("Alternative", 7.0, "Reasonable."),
                        ("Fallback", 6.2, "Generic."),
                    ],
                    solution="Minimal solution.",
                    reasoning="Empty context should not break the engine.",
                    confidence=8.0,
                    should_continue=False,
                ),
            ]
        )

        report = deep_think(
            task="Produce a safe answer when there is no prior context.",
            context="",
            max_iterations=1,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )

        assert "## Iteration 1" in report
        assert "Minimal solution." in report
        assert "\nCONTEXT:\n" not in llm.prompts[0]
        assert len(llm.prompts) == 5
