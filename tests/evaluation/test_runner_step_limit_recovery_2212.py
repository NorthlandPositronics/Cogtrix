"""#2212 — the Gate-2 harness recovers at the recursion cap like production.

`regression_persist_before_refusing` (and other scenarios) intermittently blew
the LangGraph recursion cap on weak smoke models; the old harness called
``graph.invoke`` raw, so a ``GraphRecursionError`` propagated and the turn was
scored as a hard crash (``tools=0 turns=0 error=Recursion limit…``) — a recurring
flaky false-red that blocked unrelated PRs.

Production's ``run_agent`` does NOT crash there: ``recover_from_step_limit``
re-invokes once with a tight "answer now, no more tools" nudge and finalizes a
best-effort turn. ``run_scenario`` now mirrors that (same fix as the
role_sysadmin #2368 and role_swe harnesses): stream to keep the latest state,
and on ``GraphRecursionError`` re-invoke once under ``_STEP_LIMIT_RECOVERY_LIMIT``
instead of crashing. The recovered turn is still scored on its content, and the
recovery is flagged (``recovered_from_step_limit``) but never gates the pass.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from tests.evaluation.runner import (
    _STEP_LIMIT_RECOVERY_LIMIT,
    EvalScenario,
    ModelConfig,
    run_scenario,
)

_MODEL = ModelConfig(
    id="deepseek-v4-flash",
    provider="deepseek",
    display_name="DeepSeek V4 Flash",
    tier="B",
    smoke=True,
    env_key="DEEPSEEK_API_KEY",
    model_id="deepseek-v4-flash",
)


def _scenario(**over: Any) -> EvalScenario:
    base = dict(
        id="persist-recovery-test",
        domain="test",
        title="persist recovery",
        description="",
        user_prompt="find the answer or say you can't",
        system_prompt="",
        tools_required=[],
        expected_outcome="",
        success_criteria=[],
        max_turns=12,
        timeout_seconds=10,
    )
    base.update(over)
    return EvalScenario(**base)  # type: ignore[arg-type]


def _patch(monkeypatch, graph: Any) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("cogtrix_core.orchestration.graph.build_agent_graph", lambda **_kw: graph)
    monkeypatch.setattr("tests.evaluation.runner._build_llm", lambda *_a, **_kw: object())


class _RecursionThenRecoverGraph:
    """Loops to the cap on the main run, then finalizes on the recovery re-invoke.

    The recovery call is recognised by ``recursion_limit == _STEP_LIMIT_RECOVERY_LIMIT``
    (the tight budget the finalize nudge runs under).
    """

    def __init__(self) -> None:
        self.recovery_seen_limit: int | None = None

    def stream(self, inputs: dict, config: dict | None = None, stream_mode: str | None = None):
        limit = (config or {}).get("recursion_limit")
        msgs = list(inputs.get("messages", []))
        if limit == _STEP_LIMIT_RECOVERY_LIMIT:
            # Recovery re-invoke: emit a graceful terminal answer, no crash.
            self.recovery_seen_limit = limit
            yield {
                "messages": msgs
                + [AIMessage(content="I could not find that information after searching.")]
            }
            return
        # Main run: make a real tool call (so the partial trail is non-empty),
        # then blow the recursion cap.
        yield {
            "messages": msgs
            + [
                AIMessage(content="", tool_calls=[{"name": "search_web", "args": {}, "id": "t1"}]),
                ToolMessage(content="no results", tool_call_id="t1", name="search_web"),
            ]
        }
        raise GraphRecursionError("Recursion limit of 60 reached without hitting a stop condition.")


class _AlwaysRecursesGraph:
    """Blows the cap on BOTH the main run and the recovery re-invoke."""

    def stream(self, inputs: dict, config: dict | None = None, stream_mode: str | None = None):
        yield {"messages": list(inputs.get("messages", []))}
        raise GraphRecursionError("Recursion limit reached")


class _CleanGraph:
    """Completes normally — no recursion cap involved."""

    def stream(self, inputs: dict, config: dict | None = None, stream_mode: str | None = None):
        yield {
            "messages": list(inputs.get("messages", []))
            + [AIMessage(content="Here is the answer.")]
        }


def test_recursion_cap_recovers_instead_of_crashing(monkeypatch) -> None:
    graph = _RecursionThenRecoverGraph()
    _patch(monkeypatch, graph)

    result = run_scenario(_scenario(), _MODEL)

    # No crash: the run finalized rather than surfacing the recursion error.
    assert result.error is None
    # Flagged as recovered (reported), and the recovery re-invoke actually ran
    # under the tight finalize budget.
    assert result.recovered_from_step_limit is True
    assert graph.recovery_seen_limit == _STEP_LIMIT_RECOVERY_LIMIT
    # The recovered turn is scored on its content — the finalize answer is present…
    assert "could not find" in result.final_response.lower()
    # …and the pre-cap trail (the search_web call) is preserved, not read as 0.
    assert "search_web" in result.tool_calls_made


def test_recovery_that_also_caps_still_does_not_crash(monkeypatch) -> None:
    _patch(monkeypatch, _AlwaysRecursesGraph())

    result = run_scenario(_scenario(), _MODEL)

    # Even when the recovery re-invoke ALSO hits the cap, we keep the trail and
    # never propagate the crash to the scenario result.
    assert result.error is None
    assert result.recovered_from_step_limit is True


def test_normal_completion_is_not_flagged(monkeypatch) -> None:
    _patch(monkeypatch, _CleanGraph())

    result = run_scenario(_scenario(), _MODEL)

    assert result.error is None
    assert result.recovered_from_step_limit is False
    assert "answer" in result.final_response.lower()
