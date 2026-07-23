"""Regression tests for the run_scenario timeout path.

Previously the ``with ThreadPoolExecutor(...) as executor:`` pattern in
``run_scenario`` hung at ``__exit__`` when ``graph.invoke`` ran past the
per-scenario timeout — Python threads cannot be force-killed, and the
context manager's default ``shutdown(wait=True)`` blocks until the LLM
thread finishes its provider-side request.  In Gate 2 this manifested as
``ci_gate2`` emitting zero per-scenario result lines after many minutes
when even one model looped on tool calls.

This test pins the fix: a hanging graph must surface as a TimeoutError
within the scenario's ``timeout_seconds`` plus a small overhead, never
blocking the surrounding loop indefinitely.
"""

from __future__ import annotations

import signal
import time

from tests.evaluation.runner import EvalScenario, ModelConfig, run_scenario


def test_run_scenario_timeout_does_not_hang_at_exit(monkeypatch) -> None:
    class _HangingGraph:
        def invoke(self, *_args: object, **_kwargs: object) -> dict:
            time.sleep(60)
            return {"messages": []}

    monkeypatch.setattr(
        "src.orchestration.graph.build_agent_graph",
        lambda **_kw: _HangingGraph(),
    )
    monkeypatch.setattr(
        "tests.evaluation.runner._build_llm",
        lambda *_a, **_kw: object(),
    )

    scenario = EvalScenario(
        id="hang-test",
        domain="test",
        title="hang test",
        description="",
        user_prompt="hi",
        system_prompt="",
        tools_required=["classify_invoice"],
        expected_outcome="",
        success_criteria=[],
        max_turns=5,
        timeout_seconds=1,
    )
    model = ModelConfig(
        id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Claude",
        tier="A",
        smoke=True,
        env_key="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-6",
    )

    def _watchdog(*_args: object) -> None:
        raise AssertionError(
            "run_scenario did not return within 5s — ThreadPoolExecutor " "hang has regressed"
        )

    old = signal.signal(signal.SIGALRM, _watchdog)
    signal.alarm(5)
    try:
        result = run_scenario(scenario, model)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    assert not result.passed
    assert result.error is not None
    assert "timed out" in result.error.lower()
