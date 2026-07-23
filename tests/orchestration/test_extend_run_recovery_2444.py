"""Regression tests for the extend-run → recover_from_step_limit fallback (#2444).

``_handle_extend_run`` (cogtrix_core/orchestration/runner.py, continue mode) re-invokes
the compiled graph with a raised ``recursion_limit`` after ``extend_run`` was
called mid-turn. If that continuation ALSO exhausts its (higher) recursion
budget, the function falls back to ``recover_from_step_limit``:

    return recover_from_step_limit(graph, result, input_messages, invoke_config, log)

Both the graph and ``recover_from_step_limit`` only need a ``.stream()``
surface, so the composite path is fully unit-testable with a hand-rolled fake
graph — no real LangGraph build required.

#2463 (fixed): the fallback now recovers from the FULLEST state —
``continue_result`` (which accumulates the continuation's ``graph.stream()``
progress on top of the pre-extension ``result`` messages) — so when the
continuation raises ``RecursionError`` too, tool-call progress made *strictly
during the continuation* survives into the fallback instead of being dropped.
The tests below pin the fallback firing, the pre-extension state surviving, and
the continuation progress being preserved (the last was flipped from a gap-pin
once #2463 landed).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.orchestration.runner import _handle_extend_run
from cogtrix_core.tools.extend_run import ExtendRunState


class _DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[object, ...]] = []
        self.warnings: list[tuple[object, ...]] = []
        self.errors: list[tuple[object, ...]] = []

    def info(self, *args: object) -> None:
        self.infos.append(args)

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def error(self, *args: object) -> None:
        self.errors.append(args)

    def debug(self, *args: object) -> None:
        pass

    def isEnabledFor(self, level: int) -> bool:
        return False


def _make_graph(phase_b_message: ToolMessage) -> tuple[SimpleNamespace, list[list[Any]]]:
    """A fake compiled graph whose ``.stream()`` makes real progress on its
    FIRST call (the extend-run continuation) before also raising
    RecursionError, then raises RecursionError immediately on every
    subsequent call (the recovery retry inside recover_from_step_limit)."""
    calls: list[list[Any]] = []

    def fake_stream(inputs: dict, config: dict, stream_mode: str) -> Any:
        calls.append(list(inputs["messages"]))
        call_number = len(calls)
        if call_number == 1:
            yield {"messages": [*inputs["messages"], phase_b_message]}
            raise RecursionError("continuation also hit the recursion limit")
        raise RecursionError("recovery retry also hit the recursion limit")

    return SimpleNamespace(stream=fake_stream), calls


def _make_original_result() -> tuple[dict, ToolMessage, ToolMessage]:
    phase_a_tool_msg = ToolMessage(
        content="PHASE_A_FINDING: initial run gathered this before extension",
        tool_call_id="tc1",
        name="search_tool",
    )
    phase_b_tool_msg = ToolMessage(
        content="PHASE_B_FINDING: gathered only during the continuation",
        tool_call_id="tc2",
        name="search_tool",
    )
    original_result = {
        "messages": [
            HumanMessage(content="research topic X"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_tool", "args": {}, "id": "tc1", "type": "tool_call"}],
            ),
            phase_a_tool_msg,
        ]
    }
    return original_result, phase_a_tool_msg, phase_b_tool_msg


class TestHandleExtendRunRecoveryFallback:
    """continue-mode `_handle_extend_run` → `recover_from_step_limit` fallback."""

    def test_fallback_fires_and_surfaces_pre_extension_progress(self) -> None:
        """When the extended continuation ALSO raises RecursionError, the
        function must fall back to recover_from_step_limit (not crash, not
        return nothing) and that fallback must still surface whatever tool
        progress the ORIGINAL pre-extension run gathered."""
        original_result, phase_a_tool_msg, phase_b_tool_msg = _make_original_result()
        graph, calls = _make_graph(phase_b_tool_msg)
        extend_state = ExtendRunState()
        extend_state.request_extension(mode="continue", reason="needs more steps")
        log = _DummyLogger()

        response = _handle_extend_run(
            extend_state,
            graph,
            original_result,
            original_result["messages"],
            {"recursion_limit": 60},
            SimpleNamespace(),
            None,
            log,
        )

        assert response, "recover_from_step_limit fallback must produce a response"
        assert "PHASE_A_FINDING" in response, (
            "Fallback must surface tool findings the ORIGINAL pre-extension "
            f"run gathered (via build_tool_results_response). Got: {response!r}"
        )
        # Sanity: the continuation really ran (called graph.stream at least
        # once) before the recovery cascade took over.
        assert len(calls) >= 1

    def test_progress_made_during_continuation_is_preserved_on_fallback(self) -> None:
        """#2463 fix: progress made STRICTLY during the extend-run continuation
        survives the fallback.

        `_handle_extend_run` now recovers from the FULLEST state — `continue_result`
        (which accumulates the continuation's progress on top of the pre-extension
        `result` messages), not the pre-extension `result` — so tool findings the
        continuation gathered before it also hit RecursionError are surfaced by the
        recovery instead of being dropped. (This test previously pinned the gap; per
        its own instruction it was flipped to assert the fix once #2463 landed.)
        """
        original_result, phase_a_tool_msg, phase_b_tool_msg = _make_original_result()
        graph, calls = _make_graph(phase_b_tool_msg)
        extend_state = ExtendRunState()
        extend_state.request_extension(mode="continue", reason="needs more steps")
        log = _DummyLogger()

        response = _handle_extend_run(
            extend_state,
            graph,
            original_result,
            original_result["messages"],
            {"recursion_limit": 60},
            SimpleNamespace(),
            None,
            log,
        )

        assert "PHASE_B_FINDING" in (response or ""), (
            "#2463: findings gathered during the continuation must survive the "
            "fallback (recover_from_step_limit is now called with continue_result, "
            f"not the pre-extension result). Got: {response!r}"
        )
        # And the pre-extension progress is still there too.
        assert "PHASE_A_FINDING" in (response or "")

    def test_fallback_returns_recovery_failed_message_when_nothing_survives(self) -> None:
        """When neither the pre-extension run NOR the continuation produced any
        usable tool results or partial content, the fallback must still degrade to
        the fixed RECOVERY_FAILED_MESSAGE sentinel rather than raising or returning
        an empty string."""
        from cogtrix_core.orchestration.phases import RECOVERY_FAILED_MESSAGE

        bare_result = {"messages": [HumanMessage(content="impossible task")]}

        # The continuation makes NO progress — its stream is empty, so
        # continue_result carries no tool results either (otherwise, post-#2463,
        # that progress would rightly be surfaced). The recovery retry (which also
        # calls .stream()) gets the same empty stream, so nothing survives.
        graph = SimpleNamespace(stream=lambda *_a, **_k: iter(()))
        extend_state = ExtendRunState()
        extend_state.request_extension(mode="continue", reason="needs more steps")
        log = _DummyLogger()

        response = _handle_extend_run(
            extend_state,
            graph,
            bare_result,
            bare_result["messages"],
            {"recursion_limit": 60},
            SimpleNamespace(),
            None,
            log,
        )

        assert response == RECOVERY_FAILED_MESSAGE
