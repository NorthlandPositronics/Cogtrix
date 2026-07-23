"""#2258 — a pending extend_run(delegate) must run even on NORMAL turn completion.

``_handle_extend_run`` used to fire only when the agent exhausted its step budget
(``hit_recursion_limit``). But the extend_run tool tells the agent a ``delegate``
request "will execute after this run" and to "continue with any remaining
sequential work", so the agent legitimately wraps up and ends the turn without
hitting the limit — and the delegated subtasks were silently dropped, making the
agent's "results coming later" message a false promise.

The runner now honours a queued ``delegate`` extension on normal completion too.
A ``continue`` extension is NOT honoured on normal completion (it only adds budget
"when the current limit is reached"; finishing within budget means it wasn't
needed).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from cogtrix_core.orchestration.run_config import AgentRunConfig
from cogtrix_core.orchestration.runner import run_agent


def _make_mock_llm(responses: list[AIMessage]) -> MagicMock:
    # Yield the scripted responses in order, then keep returning the final one.
    # Repeating the last (a no-tool-call answer) keeps the mock robust if the
    # graph invokes once more than expected — e.g. a verification-recovery
    # re-prompt — instead of raising StopIteration mid-stream.
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    queue = list(responses)
    last = responses[-1]

    def _invoke(*_args, **_kwargs) -> AIMessage:
        return queue.pop(0) if queue else last

    mock_llm.invoke.side_effect = _invoke
    return mock_llm


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.requires_confirmation.return_value = False
    return reg


def _config(mock_llm: MagicMock) -> AgentRunConfig:
    return AgentRunConfig(
        llm=mock_llm,
        system_prompt="You are helpful.",
        available_tools={},
        active_tools_list=[],
    )


def _extend_call(mode: str, subtasks: list[str] | None = None) -> AIMessage:
    args: dict = {"mode": mode, "reason": "needs more"}
    if subtasks is not None:
        args["subtasks"] = subtasks
    return AIMessage(
        content="",
        tool_calls=[{"name": "extend_run", "args": args, "id": "tc1"}],
        id="m1",
    )


class TestExtendRunOnNormalCompletion:
    """The pending-extension gate must consider normal completion, not just the
    recursion-limit path (#2258)."""

    def teardown_method(self) -> None:
        # Drain any background compression jobs so state doesn't leak between tests.
        from cogtrix_core.orchestration import runner as runner_mod

        for _ in range(100):
            runner_mod._drain_background_compression_jobs()
            with runner_mod._cache_lock:
                if not runner_mod._pending_background_compression_jobs:
                    break
            time.sleep(0.01)

    def test_delegate_extension_runs_on_normal_completion(self) -> None:
        # Agent registers a delegate extension, then finishes normally (no
        # recursion limit). The delegation must still run.
        mock_llm = _make_mock_llm(
            [
                _extend_call("delegate", subtasks=["check node A", "check node B"]),
                AIMessage(content="The checks are running; I'll synthesize soon.", id="m2"),
            ]
        )
        with patch(
            "cogtrix_core.orchestration.runner._handle_extend_run", return_value="SYNTHESIZED"
        ) as mock_handle:
            result = run_agent(
                "check resources", [], _make_registry(), set(), config=_config(mock_llm)
            )

        mock_handle.assert_called_once()
        # The synthesized delegation result is returned, not the bare announcement.
        assert result == "SYNTHESIZED"

    def test_continue_extension_run_on_normal_completion(self) -> None:
        # #2267: a continue extension is now ALSO honoured on normal completion —
        # otherwise the agent can register continue, announce "results coming
        # later", and stop with no deferred work done (the recurring #2258 false
        # promise). The runner re-invokes via _handle_extend_run.
        mock_llm = _make_mock_llm(
            [
                _extend_call("continue"),
                AIMessage(content="The checks are running; results next turn.", id="m2"),
            ]
        )
        with patch(
            "cogtrix_core.orchestration.runner._handle_extend_run", return_value="CONTINUED"
        ) as mock_handle:
            result = run_agent("do a thing", [], _make_registry(), set(), config=_config(mock_llm))

        mock_handle.assert_called_once()
        assert result == "CONTINUED"

    def test_no_extension_normal_completion_unaffected(self) -> None:
        # No extend_run call at all: the normal response path is untouched.
        mock_llm = _make_mock_llm([AIMessage(content="Simple answer.", id="m1")])
        with patch("cogtrix_core.orchestration.runner._handle_extend_run") as mock_handle:
            result = run_agent("hi", [], _make_registry(), set(), config=_config(mock_llm))

        mock_handle.assert_not_called()
        assert "Simple answer" in result
