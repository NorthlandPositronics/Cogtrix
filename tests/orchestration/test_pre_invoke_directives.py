"""Unit tests for the extracted pre_invoke_directives module (forge A2).

These tests exercise :func:`apply_pre_invoke_directives` and
:func:`apply_late_directives` in isolation — they do NOT build the full
``call_model`` node. The point of the extraction is that these functions
own a well-defined slice of the orchestration contract; tests should
hit them directly so a regression in directive ordering, side-effect
bookkeeping, or arm/clear semantics is caught without the noise of
``build_call_model_node`` plumbing.

Behavioural contracts under test
================================

* **Source ordering** of the appended directive messages is preserved
  exactly (see source-line table in ``pre_invoke_directives.py``).
* **Force-thinking-break flag semantics**: P0 sets/clears for the
  CURRENT round (consumed immediately by the thinking-break consumer);
  P2 arms for the NEXT round (consumer in the next call_model
  invocation).
* **Bug #1717** — a new checkpoint must clear a polling-loop arm set
  in the prior round, so a substantive synthesised answer is not
  immediately re-stripped down to a degraded re-summary.
* **Bug #1510** — when every recent same-tool result is a "not loaded"
  stub, the polling-loop advisory is still injected but the next-round
  thinking-break arm is suppressed (the agent is still discovering the
  tool state via request_tools and must not be punished).
* **Stuck-threshold calibration** runs exactly once (only when
  ``call_count == 1`` and ``stuck_threshold_calibrated[0] is False``).
* **No LLM invocation** during pre-invoke prep — the function must
  purely shape ``msgs`` and bookkeep state.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.orchestration.nodes.call_model import CallModelContext
from src.orchestration.nodes.pre_invoke_directives import (
    apply_late_directives,
    apply_pre_invoke_directives,
)


class _DummyLogger:
    """Minimal logger stub that records every call so tests can assert
    on log content without depending on the real logging hierarchy."""

    def __init__(self) -> None:
        self.infos: list[tuple[Any, ...]] = []
        self.warnings: list[tuple[Any, ...]] = []
        self.debugs: list[tuple[Any, ...]] = []

    def info(self, *args: Any) -> None:
        self.infos.append(args)

    def warning(self, *args: Any) -> None:
        self.warnings.append(args)

    def debug(self, *args: Any) -> None:
        self.debugs.append(args)


def _make_context(**overrides: Any) -> CallModelContext:
    """Build a CallModelContext with permissive defaults.

    Defaults mirror ``_make_node`` in ``test_call_model.py`` so the two
    test files stay in sync on the contract surface.
    """
    defaults: dict[str, Any] = {
        "llm": MagicMock(),
        "tools_ready": None,
        "active_tools_list": [],
        "active_names": set(),
        "bound_cache": OrderedDict(),
        "bound_cache_lock": MagicMock(),
        "cached_fingerprint": [()],
        "compression_cache": {},
        "tool_version": [0],
        "last_tool_version": [0],
        "call_count": [1],  # default: first round of turn
        "last_input_tokens": [0],
        "max_context_tokens": None,
        "context_max_messages": 0,
        "context_max_tokens": 0,
        "model_max_tokens": None,
        "compression_llm": None,
        "memory_manager": None,
        "checkpoint_store": None,
        "calls_since_last_checkpoint": [0],
        "last_checkpoint_count": [0],
        "rounds_since_checkpoint": [0],
        "force_thinking_break": [False],
        "consecutive_errors": [0],
        "last_identical_error_signature": [None],
        "consecutive_identical_error_count": [0],
        "last_reflection_at": [0],
        "tool_health_check_interval": 0,
        "last_tool_health_check_at": [0],
        "tool_quality_gate_enabled": False,
        "topic_switch_detection_enabled": False,
        "stuck_threshold": 5,
        "stuck_no_checkpoint_threshold": [20],
        "stuck_threshold_calibrated": [False],
        "checkpoint_nudge_interval": 10,
        "reflection_interval": 20,
        "max_request_tools_noops": 3,
        "sys_msg": None,
        "model_timeout": 120,
        "tool_context_limit_pct": 0.5,
        "da_enabled": False,
        "da_report_uncertainty": False,
        "da_min_confidence": 5.0,
        "apply_context_message_cap": lambda msgs, max_msgs, max_tokens: msgs,
        "maybe_compress": lambda msgs: msgs,
        "invoke_with_timeout": lambda llm, msgs, config, timeout: AIMessage(content="ok"),
        "all_tool_results_substanceless": lambda msgs: False,
    }
    defaults.update(overrides)
    return CallModelContext(**defaults)  # type: ignore[arg-type]


class _StubCheckpointStore:
    """Minimal stand-in for the real CheckpointStore.

    Implements just the surface used by P0: ``__len__`` and
    ``summary()``. The summary text is configurable so we can verify
    it's appended verbatim.
    """

    def __init__(self, count: int = 0, summary: str = "") -> None:
        self._count = count
        self._summary = summary

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._count

    def summary(self) -> str:
        return self._summary


# ─────────────────────────────────────────────────────────────────────
# Test 1 — directive ordering when all P0 + P1 branches are armed
# ─────────────────────────────────────────────────────────────────────


class TestDirectiveOrdering:
    """Asserts that the appended directives appear in source-defined
    order. Any reordering by a future refactor will break this test."""

    def test_directive_ordering_all_branches_armed(self):
        """Fixture: checkpoint store populated, topic-switch true,
        polling-loop true (3 identical tool calls), reflection-due
        (call_count == 20, REFLECTION_INTERVAL == 20), tool-quality
        gate true.

        Expected order (P1 → P0 → P2):
          1. topic-switch nudge (P1, SystemMessage with 'changed topic')
          2. checkpoint summary (P0, HumanMessage)
          3. tool-verification (P2, SystemMessage 'Tool-state verification')
          4. reflection (P2, HumanMessage 'Work cycle check')
          5. polling-loop advisory (P2, SystemMessage 'in a row')
          6. tool-quality gate (P2, SystemMessage 'tools returned')
        """
        log = _DummyLogger()
        memory_manager = MagicMock()
        memory_manager.reset_summary_state = MagicMock()
        memory_manager._reset_summary_state = None
        ckpt_store = _StubCheckpointStore(count=2, summary="[CHECKPOINTS]\n- did X")

        context = _make_context(
            call_count=[20],  # reflection-due, NOT first round so stuck-conclusion skipped
            stuck_threshold_calibrated=[True],  # skip calibration noise
            memory_manager=memory_manager,
            topic_switch_detection_enabled=True,
            checkpoint_store=ckpt_store,
            last_checkpoint_count=[2],  # not new → no arm/clear of force_thinking_break
            tool_health_check_interval=20,  # fires at call_count % 20 == 0
            reflection_interval=20,
            active_names={"checkpoint", "search_web"},
            tool_quality_gate_enabled=True,
            all_tool_results_substanceless=lambda msgs: True,
        )

        # 3 consecutive identical tool calls to trigger polling-loop advisory.
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(
                content="r1",
                tool_calls=[{"name": "search_web", "args": {}, "id": "t1"}],
            ),
            ToolMessage(content="result1", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="r2",
                tool_calls=[{"name": "search_web", "args": {}, "id": "t2"}],
            ),
            ToolMessage(content="result2", tool_call_id="t2", name="search_web"),
            AIMessage(
                content="r3",
                tool_calls=[{"name": "search_web", "args": {}, "id": "t3"}],
            ),
            ToolMessage(content="result3", tool_call_id="t3", name="search_web"),
        ]
        msgs = list(repaired)
        state_messages = list(repaired)

        with patch(
            "src.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
            return_value=True,
        ):
            msgs = apply_pre_invoke_directives(context, state_messages, repaired, msgs, log)
        msgs = apply_late_directives(context, state_messages, repaired, msgs, log)

        # Extract the markers in the order they appear and assert the
        # sequence matches the documented ordering.
        ordered_markers: list[str] = []
        for m in msgs:
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                continue
            if isinstance(m, SystemMessage) and "changed topic" in content:
                ordered_markers.append("topic_switch")
            elif isinstance(m, HumanMessage) and content.startswith("[CHECKPOINTS]"):
                ordered_markers.append("ckpt_summary")
            elif isinstance(m, SystemMessage) and "Tool-state verification" in content:
                ordered_markers.append("tool_verification")
            elif isinstance(m, HumanMessage) and "Work cycle check" in content:
                ordered_markers.append("reflection")
            elif isinstance(m, SystemMessage) and "in a row" in content:
                ordered_markers.append("polling_loop")
            elif isinstance(m, SystemMessage) and "tools returned no data" in content:
                ordered_markers.append("tool_quality")

        assert ordered_markers == [
            "topic_switch",
            "ckpt_summary",
            "tool_verification",
            "reflection",
            "polling_loop",
            "tool_quality",
        ], f"directive ordering must be preserved 1:1 from source; got {ordered_markers}"


# ─────────────────────────────────────────────────────────────────────
# Test 2 — bug #1717 reproducer (clear arm on new checkpoint)
# ─────────────────────────────────────────────────────────────────────


class TestNewCheckpointClearsThinkingBreakArm:
    """Bug #1717: when the agent records a new checkpoint, any
    previously-armed thinking break (e.g. from the polling-loop branch
    in the prior round) must be CLEARED. Otherwise the next round
    truncates the substantive answer down to a degraded re-summary.

    Code comment lives at ``pre_invoke_directives.py:_phase_p0_
    calibration_and_checkpoint`` (no dedicated test before this PR).
    """

    def test_arm_cleared_when_checkpoint_count_increases(self):
        log = _DummyLogger()
        # Simulate: previous round had a polling-loop event that armed
        # force_thinking_break for THIS round. Between rounds, the
        # agent recorded a checkpoint. The checkpoint clear in P0 must
        # fire before the thinking-break consumer runs.
        ckpt_store = _StubCheckpointStore(count=1, summary="[CKPT]\n- progress")
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            checkpoint_store=ckpt_store,
            last_checkpoint_count=[0],  # was 0, now ckpt_store has 1
            force_thinking_break=[True],  # armed by previous round
        )
        msgs: list[Any] = [HumanMessage(content="hello")]

        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)

        assert (
            context.force_thinking_break[0] is False
        ), "new checkpoint must clear the thinking-break arm (bug #1717)"
        assert context.last_checkpoint_count[0] == 1
        assert context.rounds_since_checkpoint[0] == 0
        assert context.calls_since_last_checkpoint[0] == 0

    def test_arm_preserved_when_checkpoint_count_unchanged(self):
        """Control: when no new checkpoint was recorded, an existing
        arm must NOT be silently cleared (bookkeeping only — the arm
        is the upstream signal we must honor)."""
        log = _DummyLogger()
        ckpt_store = _StubCheckpointStore(count=1, summary="[CKPT]\n- progress")
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            checkpoint_store=ckpt_store,
            last_checkpoint_count=[1],  # unchanged
            force_thinking_break=[True],  # armed by previous round
        )
        msgs: list[Any] = [HumanMessage(content="hello")]
        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)

        assert context.force_thinking_break[0] is True, (
            "no-progress round must NOT silently clear an existing arm; "
            "only a fresh checkpoint earns the clear"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 3 — bug #1510 reproducer (suppress arm on all-stub results)
# ─────────────────────────────────────────────────────────────────────


class TestPollingLoopSuppressedOnAllStubResults:
    """Bug #1510: when every consecutive identical tool call returned a
    "not loaded" stub, the polling-loop advisory MUST still be injected
    (the agent must know it should not keep calling the unloaded tool)
    but the next-round thinking-break arm MUST be suppressed (the agent
    is still discovering the tool state via request_tools and the
    arm would punish the correct recovery move)."""

    def test_advisory_injected_arm_suppressed(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
        )
        # 3 consecutive ToolMessages all returning "not loaded" stubs.
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(content="r1", tool_calls=[{"name": "search_web", "args": {}, "id": "t1"}]),
            ToolMessage(
                content="Tool 'search_web' is in the catalog but not loaded.",
                tool_call_id="t1",
                name="search_web",
            ),
            AIMessage(content="r2", tool_calls=[{"name": "search_web", "args": {}, "id": "t2"}]),
            ToolMessage(
                content="Tool 'search_web' is in the catalog but not loaded.",
                tool_call_id="t2",
                name="search_web",
            ),
            AIMessage(content="r3", tool_calls=[{"name": "search_web", "args": {}, "id": "t3"}]),
            ToolMessage(
                content="Tool 'search_web' is in the catalog but not loaded.",
                tool_call_id="t3",
                name="search_web",
            ),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)

        # Advisory present
        assert any(
            isinstance(m, SystemMessage) and "in a row" in m.content and "search_web" in m.content
            for m in msgs
        ), "advisory must still be injected so the agent knows to stop"

        # Arm suppressed
        assert (
            context.force_thinking_break[0] is False
        ), "thinking-break arm must be suppressed when all results were 'not loaded' stubs (#1510)"


# ─────────────────────────────────────────────────────────────────────
# Test 4 — stuck-threshold calibration runs exactly once
# ─────────────────────────────────────────────────────────────────────


class TestStuckThresholdCalibration:
    """Calibration is keyed on (``call_count == 1`` AND not yet
    calibrated). After the first round, the flag is True and the
    block is skipped — even if call_count rolls back to 1 in a future
    turn (the calibration is per-session, not per-turn)."""

    def test_runs_on_first_round_only(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[1],
            stuck_threshold_calibrated=[False],
        )
        msgs: list[Any] = [HumanMessage(content="build me a complex web app from scratch")]
        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)

        assert context.stuck_threshold_calibrated[0] is True
        # Default for MODERATE/COMPLEX_RESEARCH is 20, COMPLEX_ACTION is 35.
        # Whichever classification fires, the value MUST have been set.
        assert context.stuck_no_checkpoint_threshold[0] in (20, 35)

    def test_does_not_rerun_when_already_calibrated(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[1],
            stuck_threshold_calibrated=[True],  # already calibrated
            stuck_no_checkpoint_threshold=[99],  # sentinel — must not be overwritten
        )
        msgs: list[Any] = [HumanMessage(content="hello")]
        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)

        assert (
            context.stuck_no_checkpoint_threshold[0] == 99
        ), "calibration must not re-run when stuck_threshold_calibrated is True"

    def test_does_not_run_on_subsequent_rounds(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[False],
            stuck_no_checkpoint_threshold=[99],
        )
        msgs: list[Any] = [HumanMessage(content="hello")]
        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)

        assert (
            context.stuck_no_checkpoint_threshold[0] == 99
        ), "calibration must not run when call_count != 1"
        assert (
            context.stuck_threshold_calibrated[0] is False
        ), "calibrated flag must not flip when calibration didn't fire"


# ─────────────────────────────────────────────────────────────────────
# Test 5 — no LLM invocation during prep
# ─────────────────────────────────────────────────────────────────────


class TestApplyPreInvokeDirectivesDoesNotInvokeLLM:
    """Guard: the pre-invoke directive phases must not invoke the LLM.
    The whole point of separating P1+P0 from the main invoke is so the
    LLM round happens exactly once per call_model invocation (or twice
    when the thinking break fires, which is in the OTHER module). A
    future refactor that accidentally calls ``invoke_with_timeout``
    from inside the prep phase would be a contract violation."""

    def test_invoke_with_timeout_not_called(self):
        log = _DummyLogger()
        invoke_mock = MagicMock(return_value=AIMessage(content="should not be called"))
        # Use call_count=20 so the reflection branch is also exercised
        # via apply_late_directives — and even then the LLM must not
        # be called.
        ckpt_store = _StubCheckpointStore(count=2, summary="[CKPT]")
        context = _make_context(
            call_count=[20],
            stuck_threshold_calibrated=[True],
            checkpoint_store=ckpt_store,
            last_checkpoint_count=[2],
            reflection_interval=20,
            invoke_with_timeout=invoke_mock,
        )
        msgs: list[Any] = [HumanMessage(content="hello")]
        apply_pre_invoke_directives(context, list(msgs), list(msgs), msgs, log)
        apply_late_directives(context, list(msgs), list(msgs), msgs, log)

        invoke_mock.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Test 6 — apply_late_directives arms for NEXT round (not consumed)
# ─────────────────────────────────────────────────────────────────────


class TestApplyLateDirectivesArmsThinkingBreakForNextRound:
    """The polling-loop branch in P2 arms ``force_thinking_break[0]``
    for the NEXT call_model round — the current round's thinking-break
    consumer has already run by the time P2 executes.

    Verify the flag IS set after apply_late_directives. The consumer
    in the next round is responsible for clearing it (handled by
    :mod:`thinking_break_policy`)."""

    def test_polling_loop_arms_for_next_round(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
        )
        # 3 consecutive non-stub identical tool calls.
        repaired = [
            HumanMessage(content="hello"),
            AIMessage(content="r1", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1", name="merge_pr"),
            AIMessage(content="r2", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t2"}]),
            ToolMessage(content="ok", tool_call_id="t2", name="merge_pr"),
            AIMessage(content="r3", tool_calls=[{"name": "merge_pr", "args": {}, "id": "t3"}]),
            ToolMessage(content="ok", tool_call_id="t3", name="merge_pr"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)

        # Advisory was injected
        assert any(
            isinstance(m, SystemMessage) and "in a row" in m.content and "merge_pr" in m.content
            for m in msgs
        )

        # Arm set for NEXT round — the consumer is the thinking-break
        # policy module called from the NEXT call_model invocation.
        assert context.force_thinking_break[0] is True, (
            "polling-loop must arm force_thinking_break for the NEXT call_model "
            "round (the current round's consumer has already run by P2 time)"
        )
