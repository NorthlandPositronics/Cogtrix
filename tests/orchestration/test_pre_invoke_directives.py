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

from cogtrix_core.orchestration.nodes.call_model import CallModelContext
from cogtrix_core.orchestration.nodes.pre_invoke_directives import (
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
        "budget_stopped_tools": set(),
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
        "apply_context_message_cap": lambda msgs, max_msgs, max_tokens, **kw: msgs,
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
            "cogtrix_core.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
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
# Test 3b — #1943 Fix #4 reproducer (suppress on distinct args)
# ─────────────────────────────────────────────────────────────────────


class TestPollingLoopSuppressedOnDistinctArgs:
    """#1943 Fix #4: when the last N same-tool calls used PAIRWISE-
    DISTINCT arguments, the agent is iterating (sequential reads of N
    different files, diversified web_search queries, etc.), not
    polling.  The polling-loop advisory + thinking-break arm must be
    SUPPRESSED — both punish legitimate iteration."""

    def test_sequential_read_file_with_distinct_paths_suppressed(self):
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
        )
        # 3 consecutive read_file calls — each with a DIFFERENT path
        # argument.  This is exactly the prompt shape from the #1943
        # reproducer: "read these 6 files in order".
        repaired = [
            HumanMessage(content="Read and summarise the three files in order."),
            AIMessage(
                content="r1",
                tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1"}],
            ),
            ToolMessage(content="contents of a.py", tool_call_id="t1", name="read_file"),
            AIMessage(
                content="r2",
                tool_calls=[{"name": "read_file", "args": {"path": "b.py"}, "id": "t2"}],
            ),
            ToolMessage(content="contents of b.py", tool_call_id="t2", name="read_file"),
            AIMessage(
                content="r3",
                tool_calls=[{"name": "read_file", "args": {"path": "c.py"}, "id": "t3"}],
            ),
            ToolMessage(content="contents of c.py", tool_call_id="t3", name="read_file"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)

        # No advisory injected.
        assert not any(
            isinstance(m, SystemMessage) and "in a row" in (m.content or "") for m in msgs
        ), (
            "Polling-loop advisory must NOT fire when same-tool calls had "
            "distinct args — that's iteration, not polling (#1943 Fix #4)"
        )

        # Thinking-break arm not set.
        assert (
            context.force_thinking_break[0] is False
        ), "Thinking-break arm must NOT be set when calls were iteration"

        # Observability log fired so operators can trace the suppression.
        assert any(
            "Polling-loop signal suppressed" in str(args) for args in log.infos
        ), "INFO log must surface that the suppression discriminator fired"

    def test_diversified_web_search_queries_suppressed(self):
        """Same-tool calls with distinct query args (diversified
        research) must not trigger the polling-loop signal."""
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
        )
        repaired = [
            HumanMessage(content="Research the WASM ecosystem."),
            AIMessage(
                content="r1",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "wasmer download"}, "id": "t1"}
                ],
            ),
            ToolMessage(content="hits 1", tool_call_id="t1", name="web_search"),
            AIMessage(
                content="r2",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "wasmtime ABI spec"}, "id": "t2"}
                ],
            ),
            ToolMessage(content="hits 2", tool_call_id="t2", name="web_search"),
            AIMessage(
                content="r3",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "wasi-preview2 docs"}, "id": "t3"}
                ],
            ),
            ToolMessage(content="hits 3", tool_call_id="t3", name="web_search"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)
        assert not any(
            isinstance(m, SystemMessage) and "in a row" in (m.content or "") for m in msgs
        )
        assert context.force_thinking_break[0] is False

    def test_same_tool_same_args_still_triggers_polling_loop(self):
        """Genuine polling — same tool, IDENTICAL args — must still
        trip the detector.  Pins that the fix is targeted: only
        iteration is suppressed, not real polling."""
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
            active_names={"web_search"},
        )
        # 3 consecutive identical args — the classic polling shape.
        repaired = [
            HumanMessage(content="..."),
            AIMessage(
                content="r1",
                tool_calls=[{"name": "web_search", "args": {"query": "X"}, "id": "t1"}],
            ),
            ToolMessage(content="hits", tool_call_id="t1", name="web_search"),
            AIMessage(
                content="r2",
                tool_calls=[{"name": "web_search", "args": {"query": "X"}, "id": "t2"}],
            ),
            ToolMessage(content="hits", tool_call_id="t2", name="web_search"),
            AIMessage(
                content="r3",
                tool_calls=[{"name": "web_search", "args": {"query": "X"}, "id": "t3"}],
            ),
            ToolMessage(content="hits", tool_call_id="t3", name="web_search"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)
        assert any(
            isinstance(m, SystemMessage) and "in a row" in (m.content or "") for m in msgs
        ), (
            "Genuine same-tool-same-args polling must STILL trip the detector — "
            "the fix is targeted to iteration only, not all consecutive same-tool calls."
        )
        # And the thinking-break arm should be set (real polling case).
        assert context.force_thinking_break[0] is True

    def test_cap_hit_responses_force_polling_loop_to_fire(self):
        """#1984 regression: when recent ToolMessages are action-tier
        cap-hit responses (Bug F #1712), the distinct-args exemption
        must NOT suppress the polling-loop signal.  Cap-hit iteration
        is thrashing, not progress — the agent needs the advisory to
        stop calling the tool and produce a final response.

        Reproducer:
        ``regression_web_search_no_external_url_recommendation_on_low_yield``
        in Gate 2 shard B, which recursed to recursion_limit after 25
        cap-hits because Fix #4 suppressed the polling-loop signal on
        the distinct queries.
        """
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
            active_names={"web_search"},
        )
        # 3 consecutive web_search calls with distinct queries, but the
        # dispatcher's action-tier cap blocked each one — recent
        # ToolMessages carry the cap signature.
        cap_response = (
            "You have called 'web_search' 6 times in succession this turn. "
            "Further 'web_search' calls are blocked for the remainder of this "
            "turn. Choose ONE: (a) If the results already gathered are "
            "sufficient, produce a final text answer now. (b) ..."
        )
        repaired = [
            HumanMessage(content="Research."),
            AIMessage(
                content="r1",
                tool_calls=[{"name": "web_search", "args": {"query": "alpha topic"}, "id": "t1"}],
            ),
            ToolMessage(content=cap_response, tool_call_id="t1", name="web_search"),
            AIMessage(
                content="r2",
                tool_calls=[{"name": "web_search", "args": {"query": "beta topic"}, "id": "t2"}],
            ),
            ToolMessage(content=cap_response, tool_call_id="t2", name="web_search"),
            AIMessage(
                content="r3",
                tool_calls=[{"name": "web_search", "args": {"query": "gamma topic"}, "id": "t3"}],
            ),
            ToolMessage(content=cap_response, tool_call_id="t3", name="web_search"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)

        # Polling-loop advisory MUST fire — agent is thrashing the cap.
        assert any(
            isinstance(m, SystemMessage) and "in a row" in (m.content or "") for m in msgs
        ), (
            "Polling-loop advisory must fire when recent responses are "
            "action-tier cap-hits — Fix #4's distinct-args exemption cannot "
            "swallow the cap-hit signal (#1984 regression)."
        )
        # And the thinking-break arm should be set so the agent is forced
        # to a text-only response next round.
        assert (
            context.force_thinking_break[0] is True
        ), "Thinking-break arm must be set when iteration is cap-thrashing"

    def test_unresolvable_args_falls_back_to_legacy_behavior(self):
        """When the args dict can't be resolved (e.g. AIMessage was
        filtered out by message repair, or tool_call_id has no
        matching args entry), default to LEGACY polling-loop behaviour.
        Conservative — don't let a missing lookup turn off the safety
        net."""
        log = _DummyLogger()
        context = _make_context(
            call_count=[5],
            stuck_threshold_calibrated=[True],
            force_thinking_break=[False],
        )
        # ToolMessages present but NO AIMessages — no args lookup is
        # possible.  The detector should still fire.
        repaired = [
            HumanMessage(content="..."),
            ToolMessage(content="r1", tool_call_id="t1", name="read_file"),
            ToolMessage(content="r2", tool_call_id="t2", name="read_file"),
            ToolMessage(content="r3", tool_call_id="t3", name="read_file"),
        ]
        msgs = list(repaired)
        msgs = apply_late_directives(context, list(repaired), repaired, msgs, log)
        assert any(
            isinstance(m, SystemMessage) and "in a row" in (m.content or "") for m in msgs
        ), (
            "When args resolution fails, fall back to legacy polling-loop "
            "behaviour — conservative safety-net default."
        )


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


# ─────────────────────────────────────────────────────────────────────
# #1943 PR #1 — compression runs BEFORE the cap (ordering swap)
# ─────────────────────────────────────────────────────────────────────


class TestCompressionRunsBeforeCap:
    """Eviction is destructive — it drops messages without preserving
    their content.  Compression is lossy-but-recoverable.  The pipeline
    MUST run compression first so the cap below has less to drop.

    Before #1943 PR #1 the order was reversed: the cap evicted old
    ToolMessages whose content could later be impossible to recover,
    then compression ran on the survivors — which couldn't help with
    the data that was already gone.  Reproducer: #1943 / verify-1919
    context-overflow run.
    """

    def test_maybe_compress_invoked_before_cap(self):
        """Track the invocation order of ``maybe_compress`` and
        ``apply_context_message_cap``.  ``maybe_compress`` MUST run
        first so it can summarise before the cap evicts."""
        call_order: list[str] = []

        def _cap(msgs, max_msgs, max_tokens, **kw):
            call_order.append("cap")
            return msgs

        def _compress(msgs):
            call_order.append("compress")
            return msgs

        context = _make_context(
            apply_context_message_cap=_cap,
            maybe_compress=_compress,
            # Enable the cap so it's actually called (rather than
            # short-circuited at the ``context_max_messages > 0`` gate).
            context_max_messages=10,
        )
        repaired = [HumanMessage(content="hi"), AIMessage(content="ok")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert call_order == [
            "compress",
            "cap",
        ], f"Compression must run before the cap (#1943 PR #1); got {call_order!r}"

    def test_cap_runs_even_when_compression_disabled(self):
        """Disabling compression (by leaving the stub a no-op) does NOT
        skip the cap — the cap still fires when its budget is set."""
        call_order: list[str] = []

        def _cap(msgs, max_msgs, max_tokens, **kw):
            call_order.append("cap")
            return msgs

        def _compress(msgs):
            # No-op — simulates compression disabled or short-circuited.
            return msgs

        context = _make_context(
            apply_context_message_cap=_cap,
            maybe_compress=_compress,
            context_max_messages=10,
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert "cap" in call_order, "cap must still fire when budget is set"

    def test_neither_runs_when_cap_budget_is_zero(self):
        """When ``context_max_messages == 0`` AND ``context_max_tokens
        == 0``, the cap is disabled — but compression still runs
        unconditionally as the lossy-but-recoverable safety net."""
        call_order: list[str] = []

        def _cap(msgs, max_msgs, max_tokens, **kw):
            call_order.append("cap")
            return msgs

        def _compress(msgs):
            call_order.append("compress")
            return msgs

        context = _make_context(
            apply_context_message_cap=_cap,
            maybe_compress=_compress,
            context_max_messages=0,
            context_max_tokens=0,
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        # Compression always runs; cap is gated on the budget being set.
        assert call_order == ["compress"]


class TestRollingSummaryPlumbing:
    """#1943 PR #3: when a memory manager carries a rolling summary,
    the pre-invoke phase fetches it (under the manager's hybrid lock,
    short timeout) and threads it into the cap call so the eviction
    marker can embed the summary as a semantic anchor.

    The fetch is defensive — broken managers, locked-out managers, and
    missing managers all degrade to ``evicted_summary=None``.  The cap
    falls back to the PR #1 prose in that case; this test class verifies
    only the orchestration plumbing, not the marker prose itself (which
    is exercised in ``tests/test_context_message_cap.py``).
    """

    def test_summary_plumbed_through_when_manager_has_summary(self):
        """``MemoryManager._summary`` is fetched and passed as
        ``evicted_summary=`` to the cap."""
        seen_kwargs: dict[str, Any] = {}

        def _cap(msgs, max_msgs, max_tokens, **kw):
            seen_kwargs.update(kw)
            return msgs

        class _FakeMemoryManager:
            _hybrid_lock = MagicMock()
            _summary = "Earlier the user asked about deployments."

            def __init__(self) -> None:
                self._hybrid_lock.acquire.return_value = True

        manager = _FakeMemoryManager()
        context = _make_context(
            apply_context_message_cap=_cap,
            context_max_messages=10,
            memory_manager=manager,
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert seen_kwargs.get("evicted_summary") == ("Earlier the user asked about deployments.")

    def test_summary_is_none_when_manager_is_absent(self):
        """CLI direct path with no memory plumbing: ``memory_manager`` is
        ``None`` and ``evicted_summary=None`` reaches the cap."""
        seen_kwargs: dict[str, Any] = {}

        def _cap(msgs, max_msgs, max_tokens, **kw):
            seen_kwargs.update(kw)
            return msgs

        context = _make_context(
            apply_context_message_cap=_cap,
            context_max_messages=10,
            memory_manager=None,
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert seen_kwargs.get("evicted_summary") is None

    def test_summary_is_none_when_summary_attribute_missing(self):
        """A memory manager that doesn't expose ``_summary`` (very old
        subclass / mock) must not raise — the plumbing degrades to
        ``None`` and the cap falls back to PR #1 prose."""
        seen_kwargs: dict[str, Any] = {}

        def _cap(msgs, max_msgs, max_tokens, **kw):
            seen_kwargs.update(kw)
            return msgs

        class _ManagerWithoutSummary:
            # Deliberately no ``_summary`` attribute, no ``_hybrid_lock``.
            pass

        context = _make_context(
            apply_context_message_cap=_cap,
            context_max_messages=10,
            memory_manager=_ManagerWithoutSummary(),
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert seen_kwargs.get("evicted_summary") is None

    def test_summary_fetch_skipped_when_lock_contended(self):
        """When the manager's ``_hybrid_lock`` cannot be acquired within
        the short timeout, the plumbing degrades to ``None`` rather
        than blocking the cascade waiting on the background summarizer."""
        seen_kwargs: dict[str, Any] = {}

        def _cap(msgs, max_msgs, max_tokens, **kw):
            seen_kwargs.update(kw)
            return msgs

        class _ManagerWithLockedLock:
            _hybrid_lock = MagicMock()
            _summary = "should never be read because the lock is held"

            def __init__(self) -> None:
                # acquire() returns False → simulates contention.
                self._hybrid_lock.acquire.return_value = False

        context = _make_context(
            apply_context_message_cap=_cap,
            context_max_messages=10,
            memory_manager=_ManagerWithLockedLock(),
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert seen_kwargs.get("evicted_summary") is None

    def test_summary_fetch_swallows_exceptions(self):
        """If the summary fetch raises (broken memory layer), the cascade
        must continue with ``evicted_summary=None`` — memory must never
        crash the orchestration cascade."""
        seen_kwargs: dict[str, Any] = {}

        def _cap(msgs, max_msgs, max_tokens, **kw):
            seen_kwargs.update(kw)
            return msgs

        class _BrokenManager:
            @property
            def _hybrid_lock(self):
                raise RuntimeError("memory subsystem in a bad state")

            @property
            def _summary(self):
                raise RuntimeError("memory subsystem in a bad state")

        context = _make_context(
            apply_context_message_cap=_cap,
            context_max_messages=10,
            memory_manager=_BrokenManager(),
        )
        repaired = [HumanMessage(content="hi")]
        log = _DummyLogger()

        # Must not raise.
        apply_pre_invoke_directives(context, list(repaired), repaired, list(repaired), log)

        assert seen_kwargs.get("evicted_summary") is None


# ─────────────────────────────────────────────────────────────────────
# #2054 — stuck-conclusion nudge must skip short conversational replies
# ─────────────────────────────────────────────────────────────────────


def _has_stuck_nudge(msgs: list[Any]) -> bool:
    return any(
        isinstance(m, HumanMessage)
        and isinstance(getattr(m, "content", ""), str)
        and "[Stuck-conclusion check]" in m.content
        for m in msgs
    )


class TestStuckConclusionLengthGuard:
    """#2054 — the Bug-G nudge fires only for substantial repeated answers, so
    short conversational acknowledgments don't get force-rewritten into
    duplicate replies on chat channels."""

    def _ctx(self) -> Any:
        # Isolate the stuck-conclusion path: first round, calibration done,
        # topic-switch off.
        return _make_context(
            call_count=[1],
            stuck_threshold_calibrated=[True],
            topic_switch_detection_enabled=False,
        )

    def test_short_near_identical_replies_do_not_trip_nudge(self) -> None:
        short = "No rush, take your time."
        msgs = [AIMessage(content=short), AIMessage(content=short)]
        out = apply_pre_invoke_directives(self._ctx(), list(msgs), list(msgs), msgs, _DummyLogger())
        assert not _has_stuck_nudge(out), "short acks must not trigger the stuck-conclusion nudge"

    def test_substantial_near_identical_replies_still_trip_nudge(self) -> None:
        long_answer = (
            "Based on the available evidence the project timeline cannot be accelerated "
            "without descoping at least one milestone; the critical path runs through "
            "vendor delivery which is fixed."
        )
        assert len(long_answer) > 80
        msgs = [AIMessage(content=long_answer), AIMessage(content=long_answer)]
        out = apply_pre_invoke_directives(self._ctx(), list(msgs), list(msgs), msgs, _DummyLogger())
        assert _has_stuck_nudge(out), "substantial repeated answers must still trigger the nudge"

    def test_short_replies_skip_sequencematcher(self, monkeypatch) -> None:
        # #2199: the cheap length guard must run BEFORE the O(n·m) SequenceMatcher,
        # so short prior responses short-circuit without paying the diff cost.
        import difflib

        calls = {"n": 0}
        real = difflib.SequenceMatcher

        def _spy(*args: Any, **kwargs: Any):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(difflib, "SequenceMatcher", _spy)
        short = "No rush, take your time."
        msgs = [AIMessage(content=short), AIMessage(content=short)]
        apply_pre_invoke_directives(self._ctx(), list(msgs), list(msgs), msgs, _DummyLogger())
        assert calls["n"] == 0, (
            "SequenceMatcher must be skipped for short prior responses — the length "
            "guard should short-circuit before the O(n·m) diff (#2199)"
        )
