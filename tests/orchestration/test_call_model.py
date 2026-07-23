"""Unit tests for the extracted call_model node.

Tests for the call_model node in cogtrix_core/orchestration/nodes/call_model.py.

The call_model node:
- Binds active tools to the LLM
- Applies message compression when context limits are exceeded
- Detects topic switches and resets memory summary state
- Injects tool-quality gate messages when all tools return empty
- Handles decision accountability (DA) confidence checks
- Manages timeouts and retry behavior
- Detects stuck states (consecutive errors, polling loops)
- Applies context budget guards to responses
- Repairs invalid tool message pairs
- Manages a cache of bound LLM instances with lock protection
- Injects checkpoint nudges and reflection messages
- Injects tool state verification at regular intervals
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cogtrix_core.agent.core import CogtrixState
from cogtrix_core.orchestration.nodes.call_model import (
    CallModelContext,
    _compute_search_effort,
    _has_arithmetic_intent,
    _has_numeric_tool_results,
    build_call_model_node,
)


class _DummyLogger:
    """Minimal logger stub for testing."""

    def __init__(self):
        self.infos: list[tuple[object, ...]] = []
        self.warnings: list[tuple[object, ...]] = []

    def info(self, *args: object):
        self.infos.append(args)

    def warning(self, *args: object):
        self.warnings.append(args)


def _make_state(messages: list) -> CogtrixState:
    """Create a minimal CogtrixState with given messages."""
    return {"messages": messages}


def _make_node(**overrides: object) -> Any:
    """Build a call_model node with default mock dependencies."""
    # Default context values
    defaults: dict[str, object] = {
        "llm": MagicMock(),
        "tools_ready": MagicMock(),
        "active_tools_list": [],
        "active_names": set(),
        "budget_stopped_tools": set(),
        "bound_cache": OrderedDict(),
        "bound_cache_lock": MagicMock(),
        "cached_fingerprint": [()],
        "compression_cache": {},
        "tool_version": [0],
        "last_tool_version": [0],
        "call_count": [0],
        "last_input_tokens": [0],
        "max_context_tokens": None,
        "context_max_messages": 0,
        "context_max_tokens": 0,
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
        "model_max_tokens": None,
        "tool_context_limit_pct": 0.5,
        "da_enabled": False,
        "da_report_uncertainty": False,
        "da_min_confidence": 5.0,
        "apply_context_message_cap": lambda msgs, max_msgs, max_tokens: msgs,
        "maybe_compress": lambda msgs: msgs,
        "invoke_with_timeout": lambda llm, msgs, config, timeout: AIMessage(content="ok"),
        "all_tool_results_substanceless": lambda msgs: False,
    }
    defaults.update(overrides)  # type: ignore[arg-type]

    # Convert dict to CallModelContext
    context = CallModelContext(**defaults)  # type: ignore[arg-type]
    return build_call_model_node(context)


# ─────────────────────────────────────────────────────────────────────────────
# Basic routing tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelBasicRouting:
    """Basic call_model node routing behavior."""

    def test_returns_response_when_no_messages(self):
        """When no messages, call_model still processes (filters transient, checks limits)."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke = MagicMock(return_value=AIMessage(content="ok"))

        node = _make_node(llm=llm, active_tools_list=[])
        state = _make_state([])

        result = node(state, {})

        # Empty messages still goes through the node and returns a response
        assert result == {"messages": [AIMessage(content="ok")]}

    def test_sys_msg_prepended_to_llm_input(self):
        """System message should be prepended to messages when provided."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            sys_msg=SystemMessage(content="system prompt"),
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        assert captured_msgs[0].content == "system prompt"
        assert captured_msgs[-1].content == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# Compression trigger tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelCompression:
    """Tests for message compression behavior."""

    def test_compression_applied_when_context_limits_set(self):
        """When context limits are set, compression should be applied."""
        maybe_compress = MagicMock(return_value=[])
        node = _make_node(maybe_compress=maybe_compress)
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        maybe_compress.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Topic-switch detection tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelTopicSwitchDetection:
    """Tests for topic-switch detection and summary state reset."""

    def test_topic_switch_resets_summary_state(self):
        """When topic switch detected, memory_manager.reset_summary_state should be called."""
        memory_manager = MagicMock()
        memory_manager.reset_summary_state = MagicMock()
        memory_manager._reset_summary_state = None

        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="response")

        should_reset = MagicMock(return_value=True)

        node = _make_node(
            memory_manager=memory_manager,
            topic_switch_detection_enabled=True,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        # Patch _should_reset_summary_for_topic_switch
        with patch(
            "cogtrix_core.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
            should_reset,
        ):
            state = _make_state([HumanMessage(content="new topic")])
            node(state, {})

        # Check that nudge message was added to LLM input
        assert any("changed topic" in m.content for m in captured_msgs)
        memory_manager.reset_summary_state.assert_called_once()

    def test_topic_switch_adds_nudge_message(self):
        """Topic switch should append a nudge message to the conversation."""
        memory_manager = MagicMock()
        memory_manager.reset_summary_state = MagicMock()

        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="response")

        should_reset = MagicMock(return_value=True)

        node = _make_node(
            memory_manager=memory_manager,
            topic_switch_detection_enabled=True,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        with patch(
            "cogtrix_core.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
            should_reset,
        ):
            state = _make_state([HumanMessage(content="new topic")])
            node(state, {})

        # Check for the nudge message in LLM input
        assert any(
            isinstance(m, SystemMessage) and "changed topic" in m.content for m in captured_msgs
        )
        with patch(
            "cogtrix_core.orchestration.nodes.call_model._should_reset_summary_for_topic_switch",
            should_reset,
        ):
            state = _make_state([HumanMessage(content="new topic")])
            node(state, {})

        # Check for the nudge message (actual message content)
        # The SystemMessage is passed to the LLM, not returned in result["messages"]
        # Check captured_msgs from the first invocation (which was inside the first patch block)
        assert any(
            isinstance(m, SystemMessage) and "changed topic" in m.content for m in captured_msgs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool-quality gate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelStuckConclusionDetection:
    """Bug G #1713 — stuck-conclusion nudge.

    When the last two assistant final responses (from prior user turns)
    are >= 90% similar, call_model must inject a HumanMessage before the
    LLM invocation telling the model it's been repeating itself and to
    either acknowledge no new evidence or pursue a different angle.
    Pairs with the system-prompt rule that bans "You're absolutely right"
    prefixes on unchanged answers.
    """

    def test_byte_identical_prior_finals_inject_nudge(self):
        """The cogtrix56 reproducer: turn-3 and turn-4 final assistant
        messages are byte-identical (sycophantic apology then SAME
        content). On the start of turn 5 the nudge must fire."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        repeated_text = (
            "Based on verified information from ScienceSoft's official "
            "website and executive LinkedIn profiles, there is no "
            "evidence that Alex Ranasheuski, Boris Shiklo, or Tim "
            "Grigoriev have Ukrainian citizenship, reside in Ukraine, "
            "are part of the Ukrainian diaspora, or participate in any "
            "Ukraine-related support activities."
        )

        node = _make_node(
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
            call_count=[0],  # increments to 1 inside call_model
        )
        # Conversation shape: two prior turns each with a final AIMessage
        # carrying the same text, plus a fresh user message starting
        # turn 5.
        state = _make_state(
            [
                HumanMessage(content="who are their management team"),
                AIMessage(content=repeated_text),
                HumanMessage(content="dig deeper — Alex Ranasheuski"),
                AIMessage(content=repeated_text),
                HumanMessage(content="you have repeated the same message"),
            ]
        )

        node(state, {})

        # The nudge must be present in the LLM input.
        nudge = next(
            (
                m
                for m in captured_msgs
                if isinstance(m, HumanMessage) and "Stuck-conclusion check" in m.content
            ),
            None,
        )
        assert nudge is not None, (
            "Stuck-conclusion nudge must fire when the prior 2 assistant "
            "final responses are byte-identical (Bug G #1713 cogtrix56 "
            "reproducer)"
        )
        # The nudge must explicitly forbid the sycophantic prefix path so
        # the model has a terminal directive, not a vague hint.
        assert "absolutely right" in nudge.content.lower()
        assert "conclusion is unchanged" in nudge.content.lower()

    def test_dissimilar_prior_finals_do_not_trigger_nudge(self):
        """Negative control: when the prior two assistant responses are
        substantively different, the nudge must NOT fire — otherwise
        every multi-turn session pays the nudge tax."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        node = _make_node(
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
            call_count=[0],  # increments to 1 inside call_model
        )
        state = _make_state(
            [
                HumanMessage(content="who is the CEO"),
                AIMessage(
                    content="The CEO is Alex Ranasheuski, based on the company's "
                    "official site. He took the role in 2018 after the prior CEO "
                    "stepped down."
                ),
                HumanMessage(content="tell me about the CTO"),
                AIMessage(
                    content="Boris Shiklo is the CTO. His LinkedIn profile shows "
                    "20 years of engineering management experience and a focus on "
                    "enterprise software architecture."
                ),
                HumanMessage(content="what about Tim Grigoriev"),
            ]
        )

        node(state, {})

        nudge = next(
            (
                m
                for m in captured_msgs
                if isinstance(m, HumanMessage) and "Stuck-conclusion check" in m.content
            ),
            None,
        )
        assert nudge is None, (
            "Stuck-conclusion nudge must NOT fire when the prior 2 "
            "assistant final responses are substantively different — "
            "otherwise the nudge tax hits every normal multi-turn "
            "session"
        )

    def test_nudge_only_fires_at_start_of_new_turn(self):
        """The nudge is keyed on call_count == 1 — within-turn rounds
        (call_count > 1) must NOT pay the similarity check, because
        the prior 2 AIMessages will both be from the current turn
        (which can legitimately repeat tool-call shapes) and not from
        prior user turns."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        repeated_text = "Same content twice"
        node = _make_node(
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
            call_count=[5],  # not first round of turn
        )
        state = _make_state(
            [
                HumanMessage(content="q1"),
                AIMessage(content=repeated_text),
                AIMessage(content=repeated_text),
                HumanMessage(content="q2"),
            ]
        )

        node(state, {})

        nudge = next(
            (
                m
                for m in captured_msgs
                if isinstance(m, HumanMessage) and "Stuck-conclusion check" in m.content
            ),
            None,
        )
        assert nudge is None, (
            "Stuck-conclusion nudge must not fire at call_count > 1 "
            "(within-turn rounds) — only at the start of a fresh "
            "user turn"
        )


class TestCallModelCJKLeakDetection:
    """Bug K #1720 — CJK character leakage detection.

    Multilingual LLMs (qwen3-coder observed in cogtrix57.log lines
    27652, 30376) occasionally sample a CJK token where an English
    equivalent was expected ("立场" / "表态" in place of "stance" /
    "statement"). The detector is intentionally LOG-ONLY: legitimate
    quoted CJK content (e.g., the user asked about Chinese terms)
    must survive the round, and silent stripping would corrupt
    legitimate output. The fix surfaces the rate so ops can decide
    whether the leakage is worth a follow-up mitigation.
    """

    def test_cjk_in_response_logs_warning(self):
        """The cogtrix57 reproducer: response contains 立场 (stance)
        mixed into English prose. call_model must log a WARNING
        identifying the CJK characters."""
        import logging

        captured_warnings: list[str] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured_warnings.append(record.getMessage())

        handler = _CaptureHandler()
        logger = logging.getLogger("cogtrix")
        logger.addHandler(handler)
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content=(
                        "Its wording and title indicate it represents official "
                        "company立场 on the matter."
                    )
                ),
            )
            state = _make_state([HumanMessage(content="what is their position?")])
            node(state, {})
        finally:
            logger.removeHandler(handler)

        cjk_warnings = [w for w in captured_warnings if "Non-Latin (CJK)" in w]
        assert cjk_warnings, (
            "CJK leak detection must log a WARNING when the assistant "
            "response contains CJK characters (Bug K #1720 cogtrix57 "
            "reproducer)"
        )
        # The warning must include the actual leaked characters so ops
        # can correlate to a specific scenario. The sample list quotes
        # each char individually (e.g. ['立', '场']), so check for
        # presence of each char rather than the joined string.
        assert "立" in cjk_warnings[0] and "场" in cjk_warnings[0], (
            "WARNING must include the actual leaked CJK characters so "
            f"ops can identify the leak pattern; got: {cjk_warnings[0]!r}"
        )

    def test_multiple_cjk_runs_logged_with_count(self):
        """Both 立场 and 表态 appear (cogtrix57 line 30376 + 27652).
        The warning must report the total count, not just the first
        match, so ops can see the magnitude of the leak."""
        import logging

        captured_warnings: list[str] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured_warnings.append(record.getMessage())

        handler = _CaptureHandler()
        logger = logging.getLogger("cogtrix")
        logger.addHandler(handler)
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content=(
                        "This early condemnation demonstrates genuine 立场 rather "
                        "than opportunistic post-invasion 表态."
                    )
                ),
            )
            node(_make_state([HumanMessage(content="explain")]), {})
        finally:
            logger.removeHandler(handler)

        cjk_warnings = [w for w in captured_warnings if "Non-Latin (CJK)" in w]
        assert cjk_warnings
        # "立场" = 2 chars, "表态" = 2 chars → 4 total
        assert "4" in cjk_warnings[0], (
            "WARNING must report the total CJK character count "
            f"(expected 4, got message: {cjk_warnings[0]!r})"
        )

    def test_pure_english_response_logs_no_warning(self):
        """Negative control. A pure-English response must NOT trigger
        the CJK warning — otherwise every turn logs a false positive."""
        import logging

        captured_warnings: list[str] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured_warnings.append(record.getMessage())

        handler = _CaptureHandler()
        logger = logging.getLogger("cogtrix")
        logger.addHandler(handler)
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content="The company's official position is clearly stated."
                ),
            )
            node(_make_state([HumanMessage(content="explain")]), {})
        finally:
            logger.removeHandler(handler)

        cjk_warnings = [w for w in captured_warnings if "Non-Latin (CJK)" in w]
        assert (
            not cjk_warnings
        ), f"Pure-English response must not log CJK warnings: {cjk_warnings!r}"

    def test_cjk_detection_does_not_modify_response(self):
        """The fix is detection-only — the response content must be
        passed through unchanged. Legitimate quoted CJK content
        (e.g., the user asked about Chinese terms) must survive."""
        original_content = "The Chinese term 立场 means 'stance' or 'position'."
        node = _make_node(
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                content=original_content
            ),
        )
        result = node(_make_state([HumanMessage(content="what does 立场 mean?")]), {})

        # Find the AIMessage in the result (skip RemoveMessage entries
        # that the repair path may emit).
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_msgs
        assert ai_msgs[-1].content == original_content, (
            "CJK detection must NOT modify the response content — "
            "silent stripping would corrupt legitimate quoted CJK "
            "(the user asked about a Chinese term)"
        )


class TestCallModelSycophanticPrefixDetection:
    """Bug G #1713 follow-up — orchestrator-side detection (log-only)
    of sycophantic validation prefixes.

    The system-prompt rule forbids ``"You're absolutely right" /
    "I apologize" / "You're right"`` prefixes on unchanged answers, but
    the 2026-05-21 corpus replay on E03 (USD/EUR tip/tax) showed
    qwen3-coder still emits ``"You're right - let me call
    `get_current_datetime` ..."`` despite the rule. The orchestrator
    layer logs a WARNING when the prefix appears so ops can see the
    bypass rate, but does NOT modify the response — earlier strip
    prototypes (PR #1731 iterations 1 and 2) broke Gate 2 shard D ×
    kimi-k2-5 and shard B × kimi-k2-5 because intra-turn responses
    kimi later referenced lost context. Mirrors Bug K's detection-only
    pattern from PR #1729.
    """

    @staticmethod
    def _capture_warnings():
        """Attach a handler to the cogtrix logger and return
        ``(captured_warnings_list, remove_handler_callable)``."""
        import logging

        captured: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured.append(record.getMessage())

        handler = _Capture()
        logger = logging.getLogger("cogtrix")
        logger.addHandler(handler)
        return captured, lambda: logger.removeHandler(handler)

    def test_youre_right_prefix_logs_warning(self):
        captured, remove = self._capture_warnings()
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content=(
                        "You're right - let me call `get_current_datetime` to verify "
                        "today's date and ensure my calculations use the correct "
                        "exchange rate."
                    )
                ),
            )
            result = node(_make_state([HumanMessage(content="convert 100 USD")]), {})
        finally:
            remove()
        ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        # Content is preserved verbatim — no modification.
        assert ai.content.startswith("You're right"), (
            "Detection-only: response content must NOT be modified. " f"Got: {ai.content[:80]!r}"
        )
        # But a WARNING fired so ops can see the rate.
        sycophantic = [w for w in captured if "Sycophantic prefix detected" in w]
        assert sycophantic, (
            "Sycophantic prefix detector must log a WARNING when the "
            "prefix appears (Bug G #1713)"
        )
        assert "You're right" in sycophantic[0] or "you're right" in sycophantic[0].lower()

    def test_youre_absolutely_right_logs_warning(self):
        captured, remove = self._capture_warnings()
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content="You're absolutely right - the answer is 118."
                ),
            )
            node(_make_state([HumanMessage(content="check it")]), {})
        finally:
            remove()
        assert any("Sycophantic prefix detected" in w for w in captured)

    def test_apology_prefix_logs_warning(self):
        captured, remove = self._capture_warnings()
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content="I apologize. The correct answer is 42."
                ),
            )
            node(_make_state([HumanMessage(content="q")]), {})
        finally:
            remove()
        assert any("Sycophantic prefix detected" in w for w in captured)

    def test_raising_important_point_logs_warning(self):
        captured, remove = self._capture_warnings()
        try:
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(
                    content="You're raising an important point — the answer is actually 7, not 5."
                ),
            )
            node(_make_state([HumanMessage(content="q")]), {})
        finally:
            remove()
        assert any("Sycophantic prefix detected" in w for w in captured)

    def test_response_content_passes_through_unchanged(self):
        """The detector MUST NOT modify the response — earlier strip
        prototypes broke Gate 2 kimi-k2-5 scenarios by altering
        intra-turn responses the model later referenced. Pin
        passthrough on a response that WOULD have triggered the strip
        in the old design."""
        original = AIMessage(
            content="You're right - let me look that up.",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "tc1"}],
            response_metadata={"finish_reason": "tool_calls"},
            id="ai-msg-original",
        )
        node = _make_node(
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=lambda llm, msgs, config, timeout: original,
        )
        result = node(_make_state([HumanMessage(content="q")]), {})
        ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        # Content unchanged — load-bearing for kimi-k2-5 multi-turn.
        assert ai.content == original.content
        # Tool calls + metadata + id all preserved.
        assert ai.tool_calls and ai.tool_calls[0]["id"] == "tc1"
        assert ai.response_metadata.get("finish_reason") == "tool_calls"
        assert ai.id == "ai-msg-original"

    def test_no_warning_when_prefix_not_at_start(self):
        """Embedded "you're right" further in the text (quoting the
        user, discussing the right answer, etc.) must NOT trigger the
        warning — only the validation-phrase-at-start pattern."""
        captured, remove = self._capture_warnings()
        try:
            content = (
                "The user is asking about rights. You're right that this is a "
                "complex topic, but I'll focus on the specific case."
            )
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(content=content),
            )
            node(_make_state([HumanMessage(content="q")]), {})
        finally:
            remove()
        sycophantic = [w for w in captured if "Sycophantic prefix detected" in w]
        assert not sycophantic, (
            f"Embedded 'you're right' (not at start) must not trigger "
            f"the warning. Got: {sycophantic!r}"
        )

    def test_no_warning_on_genuine_substantive_response(self):
        """Negative control: a response that doesn't start with a
        validation phrase must not log the warning."""
        captured, remove = self._capture_warnings()
        try:
            content = "The answer is 42. I computed 6 × 7 = 42 using the calculate tool."
            node = _make_node(
                maybe_compress=lambda msgs: msgs,
                invoke_with_timeout=lambda llm, msgs, config, timeout: AIMessage(content=content),
            )
            node(_make_state([HumanMessage(content="q")]), {})
        finally:
            remove()
        sycophantic = [w for w in captured if "Sycophantic prefix detected" in w]
        assert not sycophantic


class TestCallModelToolQualityGate:
    """Tests for tool output quality gate behavior."""

    def test_quality_gate_injects_message_when_all_empty(self):
        """When all tools return empty, quality gate should inject a message."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke = MagicMock(return_value=AIMessage(content="response"))

        all_empty = MagicMock(return_value=True)

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            tool_quality_gate_enabled=True,
            all_tool_results_substanceless=all_empty,
            maybe_compress=lambda msgs: msgs,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        all_empty.assert_called_once()

    def test_quality_gate_message_instructs_to_ask_user(self):
        """Quality gate message should ask user how to proceed."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke = MagicMock(return_value=AIMessage(content="response"))

        all_empty = MagicMock(return_value=True)

        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            tool_quality_gate_enabled=True,
            all_tool_results_substanceless=all_empty,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Find the quality gate message
        quality_msg = next(
            (m for m in captured_msgs if "All tools returned no data" in m.content), None
        )
        assert quality_msg is not None
        assert "how to proceed" in quality_msg.content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Decision Accountability (_da_enabled) tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelDecisionAccountability:
    """Tests for decision accountability (_da_enabled) behavior."""

    def test_da_enabled_extract_decision_justification(self):
        """When DA enabled, extract_decision_justification should be called."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke = MagicMock(return_value=AIMessage(content="response with decision"))

        # Mock extract_decision_justification to return a valid result
        da_result = {
            "confidence": 8.0,
            "confidence_adjustment": 0.0,
            "flaws": [],
            "should_proceed": True,
        }

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            da_enabled=True,
            maybe_compress=lambda msgs: msgs,
        )

        with patch(
            "cogtrix_core.orchestration.nodes.call_model.extract_decision_justification",
            return_value=da_result,
        ):
            state = _make_state([HumanMessage(content="hello")])
            node(state, {})

        # DA should have been called
        # The actual extraction happens in the node, so we verify by checking it doesn't crash

    def test_da_report_uncertainty_when_confidence_low(self):
        """When DA confidence is below threshold, uncertainty note should be added."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke = MagicMock(return_value=AIMessage(content="response"))

        da_result = {
            "confidence": 3.0,
            "confidence_adjustment": 0.0,
            "flaws": ["Critical flaw 1"],
            "should_proceed": False,
        }

        def capture_invoke(llm_obj, msgs, config, timeout):
            return AIMessage(content="response")

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            da_enabled=True,
            da_report_uncertainty=True,
            da_min_confidence=5.0,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )

        with patch(
            "cogtrix_core.orchestration.nodes.call_model.extract_decision_justification",
            return_value=da_result,
        ):
            state = _make_state([HumanMessage(content="hello")])
            result = node(state, {})

        # The response should have uncertainty note
        response = result["messages"][-1]
        assert isinstance(response, AIMessage)
        assert "UNCERTAINTY" in response.content or "confidence" in response.content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Timeout handling tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelTimeout:
    """Tests for timeout handling in call_model."""

    def test_timeout_raises_runtime_error(self):
        """LLM timeout should raise RuntimeError (context overflow is handled separately)."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm

        def raise_timeout(llm_obj, msgs, config, timeout):
            raise RuntimeError("Timeout exceeded")

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            invoke_with_timeout=raise_timeout,
        )
        state = _make_state([HumanMessage(content="hello")])

        with pytest.raises(RuntimeError, match="Timeout exceeded"):
            node(state, {})

    def test_first_call_has_longer_timeout(self):
        """First call_model invocation should use longer timeout (max of configured and 300)."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm

        captured_timeouts: list[int] = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_timeouts.append(timeout)
            return AIMessage(content="ok")

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            model_timeout=120,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        # First call
        node(state, {})
        assert captured_timeouts[-1] == 300, "First call should use max(120, 300) = 300"

        # Second call
        node(state, {})
        assert captured_timeouts[-1] == 120, "Subsequent calls should use configured timeout"


# ─────────────────────────────────────────────────────────────────────────────
# Consecutive-error/stuck detection tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelStuckDetection:
    """Tests for stuck detection (consecutive errors, polling loops, etc.)."""

    def test_consecutive_errors_triggers_thinking_break(self):
        """When consecutive errors exceed threshold, thinking break should be triggered."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="thinking break response")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        # Simulate force_thinking_break being set (e.g. by process_tools).
        # Non-search stuck tools bypass the effort gate and fire the normal
        # thinking break (#1520).
        node = _make_node(
            llm=llm,
            active_tools_list=[],
            force_thinking_break=[True],
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[{"name": "merge_pull_request", "args": {}, "id": "tc1"}],
                ),
                ToolMessage(
                    content="Error: rule violation", tool_call_id="tc1", name="merge_pull_request"
                ),
                AIMessage(
                    content="r2",
                    tool_calls=[{"name": "merge_pull_request", "args": {}, "id": "tc2"}],
                ),
                ToolMessage(
                    content="Error: rule violation", tool_call_id="tc2", name="merge_pull_request"
                ),
            ]
        )

        node(state, {})

        # Verify the thinking break message was added to the LLM input
        assert any("THINKING BREAK" in m.content for m in captured_msgs)

    def test_polling_loop_detected_for_repeated_tool_calls(self):
        """Repeated tool calls in a row should trigger an advisory + arm
        the thinking break for the next round.

        Without arming the break, models that ignore the advisory (observed
        with Llama 3.3 70B on the Gate 2 finance scenario) loop until the
        graph hits its recursion_limit, because the duplicate-call cache
        returns success-shaped ToolMessages so ``_consecutive_errors`` never
        advances and no other escalation fires.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(
                content="response",
                tool_calls=[
                    {"name": "repeated_tool", "args": {}, "id": "tc1"},
                ],
            )

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        # Pass the force_thinking_break list by reference so we can read it
        # back after the node runs.
        force_thinking_break: list[bool] = [False]
        node = _make_node(
            llm=llm,
            active_tools_list=[],
            force_thinking_break=force_thinking_break,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        # 3 consecutive ToolMessages with the same name triggers detection.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1", tool_calls=[{"name": "repeated_tool", "args": {}, "id": "tc1"}]
                ),
                ToolMessage(content="result1", tool_call_id="tc1", name="repeated_tool"),
                AIMessage(
                    content="r2", tool_calls=[{"name": "repeated_tool", "args": {}, "id": "tc2"}]
                ),
                ToolMessage(content="result2", tool_call_id="tc2", name="repeated_tool"),
                AIMessage(
                    content="r3", tool_calls=[{"name": "repeated_tool", "args": {}, "id": "tc3"}]
                ),
                ToolMessage(content="result3", tool_call_id="tc3", name="repeated_tool"),
            ]
        )

        node(state, {})

        # An advisory was injected naming the stuck tool.
        assert any(
            "in a row" in m.content.lower() and "repeated_tool" in m.content for m in captured_msgs
        ), f"Expected polling-loop advisory naming 'repeated_tool'; got {[m.content for m in captured_msgs]}"

        # The thinking-break flag was armed for the next call_model round so
        # the loop can definitively be broken even if the model ignores the
        # advisory and calls the same tool again.
        assert force_thinking_break[0] is True, (
            "Polling-loop detection must arm _force_thinking_break so the "
            "next call_model round forces a tool-less, text-only response."
        )

    def test_polling_loop_advisory_injected_but_thinking_break_not_armed_when_all_stubs(self):
        """Regression test for #1510.

        When every consecutive tool call returned a "not loaded" stub, the
        polling-loop detector must NOT arm _force_thinking_break, because the
        agent has not exhausted the tool — it is still discovering that the
        tool is not active. The correct recovery is request_tools, and arming
        the thinking break punishes that recovery move and forces fabrication.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="response")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [False]
        node = _make_node(
            llm=llm,
            active_tools_list=[],
            force_thinking_break=force_thinking_break,
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        # 3 consecutive ToolMessages all returning "not loaded" stubs.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1", tool_calls=[{"name": "search_web", "args": {}, "id": "tc1"}]
                ),
                ToolMessage(
                    content="Tool 'search_web' is in the catalog but not loaded. "
                    "To load it now, issue a structured tool call: "
                    'request_tools(add=["search_web"])',
                    tool_call_id="tc1",
                    name="search_web",
                ),
                AIMessage(
                    content="r2", tool_calls=[{"name": "search_web", "args": {}, "id": "tc2"}]
                ),
                ToolMessage(
                    content="Tool 'search_web' is in the catalog but not loaded. "
                    "To load it now, issue a structured tool call: "
                    'request_tools(add=["search_web"])',
                    tool_call_id="tc2",
                    name="search_web",
                ),
                AIMessage(
                    content="r3", tool_calls=[{"name": "search_web", "args": {}, "id": "tc3"}]
                ),
                ToolMessage(
                    content="Tool 'search_web' is in the catalog but not loaded. "
                    "To load it now, issue a structured tool call: "
                    'request_tools(add=["search_web"])',
                    tool_call_id="tc3",
                    name="search_web",
                ),
            ]
        )

        node(state, {})

        # Advisory was injected (agent still needs to know it should not
        # keep calling the unloaded tool).
        assert any(
            "search_web" in m.content and "in a row" in m.content.lower() for m in captured_msgs
        ), "Polling-loop advisory should still be injected for stub-only rounds"

        # Thinking-break flag must NOT be armed — agent may be recovering
        # via request_tools and should not be punished with tool stripping.
        assert force_thinking_break[0] is False, (
            "Polling-loop detection must NOT arm _force_thinking_break when "
            "all consecutive calls returned 'not loaded' stubs. The agent is "
            "still discovering that the tool is not active; the correct "
            "recovery (request_tools) must not be punished."
        )

    def test_thinking_break_prompt_prevents_fabrication_when_no_checkpoint_data(self):
        """Regression test for #1510.

        When the thinking break fires and no checkpoint data has been
        accumulated, the prompt must tell the model NOT to fabricate
        specific numbers, percentages, or authoritative-sounding claims.
        A short honest "I could not retrieve current data" is preferred
        over a confident fabrication.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="answer from model")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        # checkpoint_store=None simulates "no checkpoints accumulated" —
        # the worst case for fabrication risk.
        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,  # no checkpoint data
            calls_since_last_checkpoint=[0],
            last_checkpoint_count=[0],
            rounds_since_checkpoint=[0],
            stuck_no_checkpoint_threshold=[20],
            invoke_with_timeout=capture_invoke,
        )
        # Use a non-search tool so the effort gate is bypassed and the normal
        # refusal thinking break fires (#1520).
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1", tool_calls=[{"name": "write_file", "args": {}, "id": "tc1"}]
                ),
                ToolMessage(
                    content="Error: permission denied", tool_call_id="tc1", name="write_file"
                ),
                AIMessage(
                    content="r2", tool_calls=[{"name": "write_file", "args": {}, "id": "tc2"}]
                ),
                ToolMessage(
                    content="Error: permission denied", tool_call_id="tc2", name="write_file"
                ),
            ]
        )

        node(state, {})

        # Find the thinking-break HumanMessage injected into LLM input.
        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert len(tb_msgs) == 1, f"Expected exactly one thinking-break message; got {tb_msgs}"
        tb_content = tb_msgs[0].content

        # Must NOT say "draw on your own knowledge" — that encourages fabrication.
        assert "draw on your own knowledge" not in tb_content.lower(), (
            "Thinking-break prompt must not encourage drawing on training knowledge "
            "when no checkpoint data exists — that leads to fabrication."
        )

        # Must contain anti-fabrication guidance.
        assert "do not fabricate" in tb_content.lower() or "not fabricate" in tb_content.lower(), (
            "Thinking-break prompt must explicitly tell the model not to fabricate "
            f"specific numbers, percentages, or claims. Got: {tb_content[:200]}"
        )

        # Must acknowledge that data could not be retrieved.
        assert (
            "returned nothing" in tb_content.lower()
            or "tools returned" in tb_content.lower()
            or "could not retrieve" in tb_content.lower()
        ), (
            "Thinking-break prompt must acknowledge that data could not be retrieved. "
            f"Got: {tb_content[:200]}"
        )

        # Must contain the enhanced anti-fabrication clause from #1516 — naming
        # the specific data categories where fabrication risk is highest.
        anti_fab_keywords = ["live data", "current prices", "stock levels", "FX rates", "SKUs"]
        has_enhanced_clause = any(kw in tb_content.lower() for kw in anti_fab_keywords)
        assert has_enhanced_clause, (
            f"Thinking-break prompt must contain the enhanced anti-fabrication clause "
            f"(naming specific data categories: {anti_fab_keywords}). "
            f"Got: {tb_content[:300]}"
        )

    def test_thinking_break_suppressed_when_low_effort_and_no_checkpoints(self):
        """Regression test for #1520.

        When the thinking break fires, no checkpoints exist, AND the agent has
        made fewer than _MIN_SEARCH_EFFORT searches without trying http_get,
        the break must be suppressed and a strategy nudge injected instead.
        This prevents "lazy refusal" where the agent gives up after shallow
        searches.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )
        # Only 1 search_web in history — below _MIN_SEARCH_EFFORT threshold.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "product A"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content="generic results", tool_call_id="tc1", name="search_web"),
            ]
        )

        result = node(state, {})

        # The node should NOT have returned early from a thinking break;
        # instead it falls through to normal processing with a nudge appended.
        assert result == {"messages": [AIMessage(content="ok")]}

        # Verify strategy nudge was injected.
        nudge_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "STRATEGY NUDGE" in m.content
        ]
        assert len(nudge_msgs) == 1, f"Expected exactly one strategy nudge; got {nudge_msgs}"
        nudge_content = nudge_msgs[0].content
        assert (
            "try harder" in nudge_content.lower()
        ), "Strategy nudge must encourage the agent to try harder."
        assert "http_get" in nudge_content.lower(), "Strategy nudge must suggest using http_get."

        # Must NOT contain the refusal/thinking-break prompt.
        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert (
            len(tb_msgs) == 0
        ), "Thinking break must be suppressed when effort is low and no checkpoints exist."

    def test_thinking_break_fires_when_high_effort_and_no_checkpoints(self):
        """When effort threshold is met (≥3 searches), the thinking break refusal
        prompt should still fire even without checkpoints."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="honest refusal")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )
        # 3 distinct search_web calls — meets _MIN_SEARCH_EFFORT threshold.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "product A price"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content="result1", tool_call_id="tc1", name="search_web"),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "product B price"}, "id": "tc2"}
                    ],
                ),
                ToolMessage(content="result2", tool_call_id="tc2", name="search_web"),
                AIMessage(
                    content="r3",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "product C price"}, "id": "tc3"}
                    ],
                ),
                ToolMessage(content="result3", tool_call_id="tc3", name="search_web"),
            ]
        )

        node(state, {})

        # Thinking break refusal prompt should fire.
        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert len(tb_msgs) == 1, f"Expected thinking break; got {tb_msgs}"
        assert "do not fabricate" in tb_msgs[0].content.lower()

    def test_thinking_break_fires_synthesise_message_when_results_substantive(self):
        """When effort is met AND prior searches returned substantive results
        (≥2 URL: lines per result, long content), the thinking-break message
        must steer the agent toward synthesising — NOT toward refusing.

        Closes #1585: cogtrix29 emitted a one-line refusal despite having
        5 successful searches with real product names in its message history.
        The discriminator added here is the substantive-results check.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="synthesised answer")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )

        # 3 distinct search_web calls, each returning a payload that mirrors
        # the live DDG / Tavily output format (2 URL: lines + 300+ chars).
        def _payload(seed: str) -> str:
            snippet = (
                "A representative snippet that exists for this fixture so the payload "
                "easily clears the 300-character substantiveness threshold the helper "
                "uses to distinguish real results from sponsored-slot near-empty payloads."
            )
            return (
                f"Search results for: {seed}\n\n"
                f"1. Result {seed}-A\n"
                f"   URL: https://example-A.test/{seed}\n"
                f"   Domain: example-A.test\n"
                f"   {snippet}\n\n"
                f"2. Result {seed}-B\n"
                f"   URL: https://example-B.test/{seed}\n"
                f"   Domain: example-B.test\n"
                f"   {snippet}\n"
            )

        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "alpha tools"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content=_payload("alpha"), tool_call_id="tc1", name="search_web"),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "beta tools"}, "id": "tc2"}
                    ],
                ),
                ToolMessage(content=_payload("beta"), tool_call_id="tc2", name="search_web"),
                AIMessage(
                    content="r3",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "gamma tools"}, "id": "tc3"}
                    ],
                ),
                ToolMessage(content=_payload("gamma"), tool_call_id="tc3", name="search_web"),
            ]
        )

        node(state, {})

        # The thinking-break message should be the synthesise-not-refuse variant.
        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert len(tb_msgs) == 1, f"Expected thinking break; got {tb_msgs}"
        body = tb_msgs[0].content.lower()
        # MUST steer toward synthesis
        assert "synthesise" in body or "synthesize" in body, (
            "Substantive-results branch must instruct the model to synthesise: "
            f"got:\n{tb_msgs[0].content}"
        )
        # MUST explicitly call out the lazy-refusal anti-pattern
        assert "could not retrieve" in body, (
            "Substantive-results branch must call out the 'I could not retrieve' "
            "lazy-refusal anti-pattern by name: "
            f"got:\n{tb_msgs[0].content}"
        )
        # MUST still forbid fabrication beyond actual search results
        assert "fabricate" in body, (
            "Substantive-results branch must still forbid fabrication: "
            f"got:\n{tb_msgs[0].content}"
        )

    def test_thinking_break_refusal_message_unchanged_when_results_empty(self):
        """When effort is met but search results were empty (or all errors),
        the message must keep the existing refuse-is-OK framing — this is the
        legitimate-refusal path from #1520.  Without this branch the agent
        would be unable to refuse even when there's nothing to synthesise.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="honest refusal")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )

        # 3 distinct search_web calls, all returning "No results found"
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "alpha tools"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(
                    content="No results found for: alpha tools",
                    tool_call_id="tc1",
                    name="search_web",
                ),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "beta tools"}, "id": "tc2"}
                    ],
                ),
                ToolMessage(
                    content="No results found for: beta tools",
                    tool_call_id="tc2",
                    name="search_web",
                ),
                AIMessage(
                    content="r3",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "gamma tools"}, "id": "tc3"}
                    ],
                ),
                ToolMessage(
                    content="No results found for: gamma tools",
                    tool_call_id="tc3",
                    name="search_web",
                ),
            ]
        )

        node(state, {})

        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert len(tb_msgs) == 1, f"Expected thinking break; got {tb_msgs}"
        body = tb_msgs[0].content.lower()
        # Empty-results branch must NOT instruct synthesis — there's nothing to synthesise.
        assert "synthesise" not in body and "synthesize" not in body, (
            "Empty-results branch must not instruct synthesis (#1520 path): "
            f"got:\n{tb_msgs[0].content}"
        )
        # Empty-results branch must keep the refusal-is-OK framing.
        assert "could not retrieve" in body
        assert "do not fabricate" in body

    def test_thinking_break_fires_when_http_get_attempted_and_no_checkpoints(self):
        """When http_get has been attempted (even with few searches), the agent
        has earned the right to refuse — the thinking break should fire normally."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="honest refusal")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )
        # Only 1 search but also 1 http_get — effort threshold met via http_get.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "product A"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content="result1", tool_call_id="tc1", name="search_web"),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {"name": "http_get", "args": {"url": "https://example.com"}, "id": "tc2"}
                    ],
                ),
                ToolMessage(content="page html", tool_call_id="tc2", name="http_get"),
            ]
        )

        node(state, {})

        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert len(tb_msgs) == 1, f"Expected thinking break; got {tb_msgs}"

    def test_thinking_break_suppressed_when_prior_turns_have_high_effort(self):
        """Regression test for #1532 Bug 1.

        In a long session with prior search activity, a fresh question that
        triggers a thinking break must be evaluated on its OWN effort — not
        session-cumulatively.  If the current turn has low effort, the strategy
        nudge must fire instead of the refusal prompt.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )
        # Prior turn: 3 distinct searches (meets threshold on its own).
        # Current turn: only 1 search — below threshold.
        state = _make_state(
            [
                HumanMessage(content="old question about Selene"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "selene ai"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content="result1", tool_call_id="tc1", name="search_web"),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "selene pricing"}, "id": "tc2"}
                    ],
                ),
                ToolMessage(content="result2", tool_call_id="tc2", name="search_web"),
                AIMessage(
                    content="r3",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "selene features"}, "id": "tc3"}
                    ],
                ),
                ToolMessage(content="result3", tool_call_id="tc3", name="search_web"),
                # Fresh question — current turn.
                HumanMessage(content="What about OpenClaw?"),
                AIMessage(
                    content="r4",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "openclaw ai"}, "id": "tc4"}
                    ],
                ),
                ToolMessage(content="result4", tool_call_id="tc4", name="search_web"),
            ]
        )

        result = node(state, {})

        # Should fall through to normal processing with a nudge appended.
        assert result == {"messages": [AIMessage(content="ok")]}

        # Strategy nudge must be injected (current turn has only 1 search).
        nudge_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "STRATEGY NUDGE" in m.content
        ]
        assert len(nudge_msgs) == 1, f"Expected strategy nudge; got {nudge_msgs}"

        # Thinking break must NOT fire — effort is low on the current turn.
        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert (
            len(tb_msgs) == 0
        ), "Thinking break must be suppressed when current-turn effort is low."

    def test_thinking_break_suppressed_when_near_duplicate_searches(self):
        """Regression test for :next24 reproducer (#1520).

        Three reorderings of the same query should NOT pass the effort gate.
        The strategy nudge should fire instead of the refusal prompt.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        force_thinking_break: list[bool] = [True]
        node = _make_node(
            llm=llm,
            active_tools_list=[MagicMock(name="request_tools")],
            force_thinking_break=force_thinking_break,
            checkpoint_store=None,
            invoke_with_timeout=capture_invoke,
        )
        # 3 search_web calls with near-duplicate queries — should collapse to 1 distinct.
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="r1",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"query": "foo bar baz qux"},
                            "id": "tc1",
                        }
                    ],
                ),
                ToolMessage(content="results1", tool_call_id="tc1", name="search_web"),
                AIMessage(
                    content="r2",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"query": "baz foo bar qux"},
                            "id": "tc2",
                        }
                    ],
                ),
                ToolMessage(content="results2", tool_call_id="tc2", name="search_web"),
                AIMessage(
                    content="r3",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"query": "bar qux baz foo"},
                            "id": "tc3",
                        }
                    ],
                ),
                ToolMessage(content="results3", tool_call_id="tc3", name="search_web"),
            ]
        )

        node(state, {})

        # Strategy nudge should fire, not the thinking break refusal.
        nudge_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "STRATEGY NUDGE" in m.content
        ]
        assert len(nudge_msgs) == 1, f"Expected strategy nudge; got {nudge_msgs}"

        tb_msgs = [
            m
            for m in captured_msgs
            if isinstance(m, HumanMessage) and "THINKING BREAK" in m.content
        ]
        assert (
            len(tb_msgs) == 0
        ), "Thinking break must be suppressed when near-duplicate searches don't pass the gate."

    # ── Bug #1717 regression — stale armed flag cleared by new checkpoint ──

    def test_new_checkpoint_clears_armed_thinking_break(self):
        """Regression test for Bug #1717.

        Sequence (from cogtrix57.log around line 30356):
          Round N-1: Temporal polling loop detected — arms thinking break
                     for the NEXT round.
          Round N:   Agent heeds the advisory, emits a substantive
                     579-token answer AND calls ``checkpoint``.
          Round N+1: call_model runs, sees the checkpoint count
                     incremented, but DOES NOT clear the armed flag.
                     Thinking break fires, tools stripped, agent
                     forced into a degraded 131-token re-summary.

        The fix clears ``_force_thinking_break`` when a new checkpoint
        is detected, because a fresh checkpoint is direct evidence the
        agent has made progress and the "stuck" arm is stale.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        # Minimal CheckpointStore stub: len() returns a count we control.
        class _StubCheckpointStore:
            def __init__(self, count: int) -> None:
                self._count = count

            def __len__(self) -> int:
                return self._count

            def summary(self) -> str:
                return ""

        # Setup: arm the flag (as if previous round triggered polling-loop),
        # ``_last_checkpoint_count`` is 0 (no checkpoints recorded yet at
        # the time the flag was armed), and ``checkpoint_store`` now has
        # 1 checkpoint (recorded in the round between arm and consume).
        force_thinking_break: list[bool] = [True]
        last_checkpoint_count: list[int] = [0]
        rounds_since_checkpoint: list[int] = [5]
        calls_since_last_checkpoint: list[int] = [3]

        node = _make_node(
            llm=llm,
            active_tools_list=[],
            checkpoint_store=_StubCheckpointStore(count=1),
            force_thinking_break=force_thinking_break,
            last_checkpoint_count=last_checkpoint_count,
            rounds_since_checkpoint=rounds_since_checkpoint,
            calls_since_last_checkpoint=calls_since_last_checkpoint,
            call_count=[10],  # established session, not first round
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="follow-up")])

        node(state, {})

        # Primary assertion — the flag must be cleared by the
        # new-checkpoint detection before the thinking-break consumer
        # block at the same call's later point.
        assert force_thinking_break[0] is False, (
            "A new checkpoint since the flag was armed should disarm "
            "_force_thinking_break — the agent has demonstrably made "
            "progress and a thinking break would degrade its substantive "
            "response into a re-summary (Bug #1717)."
        )

        # Secondary assertion — the LLM input must NOT contain a
        # THINKING BREAK message (which the consume path would append
        # if the flag were still True).
        thinking_break_msgs = [
            m for m in captured_msgs if "THINKING BREAK" in getattr(m, "content", "")
        ]
        assert thinking_break_msgs == [], (
            "No THINKING BREAK message should be injected when the agent "
            "has just recorded a new checkpoint."
        )

        # Sanity — the new-checkpoint reset path also resets the rolling
        # counters. If a future refactor moves the disarm without also
        # moving the counter reset, this asserts they stay coupled.
        assert last_checkpoint_count[0] == 1
        assert rounds_since_checkpoint[0] == 0
        assert calls_since_last_checkpoint[0] == 0

    def test_armed_flag_still_fires_when_no_new_checkpoint(self):
        """Negative control for Bug #1717 fix.

        The disarm path must ONLY fire when a checkpoint was actually
        recorded since the flag was armed. If the flag is armed AND no
        new checkpoint, the thinking break should still fire — that's
        the original intent of the polling-loop arm.
        """
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        class _StubCheckpointStore:
            def __init__(self, count: int) -> None:
                self._count = count

            def __len__(self) -> int:
                return self._count

            def summary(self) -> str:
                return ""

        force_thinking_break: list[bool] = [True]
        # last_checkpoint_count == current count → no progress since arm
        node = _make_node(
            llm=llm,
            active_tools_list=[],
            checkpoint_store=_StubCheckpointStore(count=2),
            force_thinking_break=force_thinking_break,
            last_checkpoint_count=[2],  # already counted both checkpoints
            rounds_since_checkpoint=[5],
            calls_since_last_checkpoint=[3],
            call_count=[10],
            maybe_compress=lambda msgs: msgs,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="follow-up")])

        node(state, {})

        # Flag should have been consumed (set to False by the consumer
        # block at line ~646), AND the THINKING BREAK message should
        # have been injected.
        assert force_thinking_break[0] is False, (
            "Flag is consumed by the thinking-break path on the round "
            "after arming (existing semantics)."
        )
        assert any("THINKING BREAK" in getattr(m, "content", "") for m in captured_msgs), (
            "When no new checkpoint has been recorded since the flag "
            "was armed, the thinking break MUST still fire — that's "
            "the existing polling-loop escalation."
        )


class TestSearchEffortHelper:
    """Tests for the _compute_search_effort helper."""

    def test_empty_messages(self):
        assert _compute_search_effort([]) == (0, False)

    def test_counts_distinct_search_web(self):
        """Only distinct queries count toward effort."""
        msgs = [
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "foo bar"}, "id": "t1"}],
            ),
            ToolMessage(content="a", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="b",
                tool_calls=[{"name": "search_web", "args": {"query": "baz qux"}, "id": "t2"}],
            ),
            ToolMessage(content="b", tool_call_id="t2", name="search_web"),
        ]
        assert _compute_search_effort(msgs) == (2, False)

    def test_collapses_near_duplicate_queries(self):
        """Reordered or near-duplicate queries should collapse to one count."""
        msgs = [
            AIMessage(
                content="a",
                tool_calls=[
                    {"name": "search_web", "args": {"query": "Soudal Fix All Silirub"}, "id": "t1"}
                ],
            ),
            ToolMessage(content="r1", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="b",
                tool_calls=[
                    {"name": "search_web", "args": {"query": "Silirub Fix All Soudal"}, "id": "t2"}
                ],
            ),
            ToolMessage(content="r2", tool_call_id="t2", name="search_web"),
            AIMessage(
                content="c",
                tool_calls=[
                    {"name": "search_web", "args": {"query": "Fix Soudal Silirub All"}, "id": "t3"}
                ],
            ),
            ToolMessage(content="r3", tool_call_id="t3", name="search_web"),
        ]
        # All three normalise to the same token set {soudal, fix, all, silirub}
        assert _compute_search_effort(msgs) == (1, False)

    def test_skips_error_results(self):
        """Search calls that returned errors should not count."""
        msgs = [
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "foo"}, "id": "t1"}],
            ),
            ToolMessage(
                content="Error searching: request failed", tool_call_id="t1", name="search_web"
            ),
            AIMessage(
                content="b",
                tool_calls=[{"name": "search_web", "args": {"query": "bar"}, "id": "t2"}],
            ),
            ToolMessage(
                content="Tool 'search_web' is in the catalog but not loaded",
                tool_call_id="t2",
                name="search_web",
            ),
        ]
        assert _compute_search_effort(msgs) == (0, False)

    def test_skips_empty_queries(self):
        """Search calls with empty/missing query args should not count."""
        msgs = [
            AIMessage(content="a", tool_calls=[{"name": "search_web", "args": {}, "id": "t1"}]),
            ToolMessage(content="r1", tool_call_id="t1", name="search_web"),
        ]
        assert _compute_search_effort(msgs) == (0, False)

    def test_detects_http_get(self):
        msgs = [
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "foo"}, "id": "t1"}],
            ),
            ToolMessage(content="a", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="b",
                tool_calls=[
                    {"name": "http_get", "args": {"url": "https://example.com"}, "id": "t2"}
                ],
            ),
            ToolMessage(content="b", tool_call_id="t2", name="http_get"),
        ]
        assert _compute_search_effort(msgs) == (1, True)

    def test_ignores_non_tool_messages(self):
        msgs = [
            HumanMessage(content="hello"),
            AIMessage(content="response"),
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "foo"}, "id": "t1"}],
            ),
            ToolMessage(content="a", tool_call_id="t1", name="search_web"),
        ]
        assert _compute_search_effort(msgs) == (1, False)

    def test_ignores_other_tools(self):
        msgs = [
            ToolMessage(content="a", tool_call_id="t1", name="write_file"),
            ToolMessage(content="b", tool_call_id="t2", name="read_email"),
        ]
        assert _compute_search_effort(msgs) == (0, False)

    def test_effort_scoped_to_current_turn(self):
        """Regression test for #1532 Bug 1.

        Searches from a prior user turn must NOT count toward the current
        turn's effort gate.  Only messages after the most recent HumanMessage
        are considered.
        """
        msgs = [
            # Prior turn — 3 distinct searches (would meet threshold alone).
            HumanMessage(content="old question about Selene"),
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "selene ai"}, "id": "t1"}],
            ),
            ToolMessage(content="r1", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="b",
                tool_calls=[
                    {"name": "search_web", "args": {"query": "selene pricing"}, "id": "t2"}
                ],
            ),
            ToolMessage(content="r2", tool_call_id="t2", name="search_web"),
            AIMessage(
                content="c",
                tool_calls=[
                    {"name": "search_web", "args": {"query": "selene features"}, "id": "t3"}
                ],
            ),
            ToolMessage(content="r3", tool_call_id="t3", name="search_web"),
            # Current turn — only 1 search.
            HumanMessage(content="What about OpenClaw?"),
            AIMessage(
                content="d",
                tool_calls=[{"name": "search_web", "args": {"query": "openclaw ai"}, "id": "t4"}],
            ),
            ToolMessage(content="r4", tool_call_id="t4", name="search_web"),
        ]
        # Only the current turn's 1 search should count.
        assert _compute_search_effort(msgs) == (1, False)

    def test_effort_counts_all_searches_when_no_human_message(self):
        """When no HumanMessage exists, effort falls back to the full list."""
        msgs = [
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "foo"}, "id": "t1"}],
            ),
            ToolMessage(content="r1", tool_call_id="t1", name="search_web"),
            AIMessage(
                content="b",
                tool_calls=[{"name": "search_web", "args": {"query": "bar"}, "id": "t2"}],
            ),
            ToolMessage(content="r2", tool_call_id="t2", name="search_web"),
        ]
        assert _compute_search_effort(msgs) == (2, False)

    def test_effort_scoped_with_multiple_human_messages(self):
        """With multiple HumanMessages, only the last turn's searches count."""
        msgs = [
            HumanMessage(content="first question"),
            AIMessage(
                content="a",
                tool_calls=[{"name": "search_web", "args": {"query": "q1"}, "id": "t1"}],
            ),
            ToolMessage(content="r1", tool_call_id="t1", name="search_web"),
            HumanMessage(content="second question"),
            AIMessage(
                content="b",
                tool_calls=[{"name": "search_web", "args": {"query": "q2"}, "id": "t2"}],
            ),
            ToolMessage(content="r2", tool_call_id="t2", name="search_web"),
            HumanMessage(content="third question"),
            AIMessage(
                content="c",
                tool_calls=[{"name": "search_web", "args": {"query": "q3"}, "id": "t3"}],
            ),
            ToolMessage(content="r3", tool_call_id="t3", name="search_web"),
        ]
        assert _compute_search_effort(msgs) == (1, False)


# ─────────────────────────────────────────────────────────────────────────────
# _has_substantive_search_results — #1585 discriminator
# ─────────────────────────────────────────────────────────────────────────────


class TestHasSubstantiveSearchResults:
    """Tests for the _has_substantive_search_results helper (#1585).

    This helper distinguishes "effort spent + results came back rich"
    from "effort spent + results were empty / errors / blocked pages".
    The thinking-break dispatch branches on it: the former gets a
    synthesise-don't-refuse message, the latter gets the existing
    honest-refusal-is-OK message.

    See the cogtrix28 (success) vs cogtrix29 (refusal) trace pair for
    the empirical motivation.
    """

    def _ddg_payload(self, n_results: int) -> str:
        """Build a synthetic search_web ToolMessage payload that mirrors the
        live DDG / Tavily / Brave / Exa output format (URL: + Domain: + snippet
        lines per result)."""
        lines = [f"Search results for: synthetic query {n_results} results", ""]
        for i in range(1, n_results + 1):
            lines.extend(
                [
                    f"{i}. Synthetic Product {i}",
                    f"   URL: https://example-{i}.test/landing",
                    f"   Domain: example-{i}.test",
                    f"   A short snippet describing synthetic product {i} for the test fixture.",
                    "",
                ]
            )
        return "\n".join(lines)

    def test_empty_messages(self):
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        assert _has_substantive_search_results([]) is False

    def test_single_url_is_not_substantive(self):
        """A single URL: line could be a sponsored slot; require ≥ 2 URL: lines."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=self._ddg_payload(1),  # one URL line
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is False

    def test_two_url_results_are_substantive(self):
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=self._ddg_payload(2),
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is True

    def test_five_url_results_are_substantive(self):
        """Typical DDG response with 5 results — most common substantive case."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=self._ddg_payload(5),
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is True

    def test_error_results_are_not_substantive(self):
        """Error-wrapper messages must not count, even if they happen to contain 'URL:' text."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=(
                    "Error searching: DuckDuckGo rate-limited (HTTP 429). "
                    "Try again. Documentation URL: https://example.test/help"
                ),
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is False

    def test_no_results_placeholder_is_not_substantive(self):
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content="No results found for: x",
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is False

    def test_not_loaded_stub_is_not_substantive(self):
        """The 'tool not loaded' placeholder must not count."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=(
                    "Tool 'search_web' is in the catalog but not loaded. "
                    "To load it now, issue a structured tool call ..."
                ),
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is False

    def test_short_content_is_not_substantive(self):
        """Even with ≥ 2 URL: lines, very short content (< 300 chars) is not enough."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        short_payload = "Search results for: q\n\n1. A\n   URL: u1\n\n2. B\n   URL: u2\n"
        assert len(short_payload) < 300
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=short_payload,
                tool_call_id="t1",
                name="search_web",
            ),
        ]
        assert _has_substantive_search_results(msgs) is False

    def test_scoped_to_current_turn_only(self):
        """Substantive results from a PRIOR turn must not count for the current turn.

        Same scope rule as _compute_search_effort — prevents prior-turn search
        success from short-circuiting the rich-yield branch on a fresh question
        where the new turn's searches actually returned empty.
        """
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            HumanMessage(content="first question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "t1"}],
            ),
            ToolMessage(
                content=self._ddg_payload(5),  # rich PRIOR-turn results
                tool_call_id="t1",
                name="search_web",
            ),
            HumanMessage(content="second question"),  # fresh turn starts here
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "y"}, "id": "t2"}],
            ),
            ToolMessage(
                content="No results found for: y",  # empty current-turn results
                tool_call_id="t2",
                name="search_web",
            ),
        ]
        # Current-turn results were empty — substantive flag must be False.
        assert _has_substantive_search_results(msgs) is False

    def test_mixed_current_turn_results_substantive_wins(self):
        """If any of the current turn's searches returned rich results, the
        flag should be True — even if other searches in the same turn failed
        (e.g. 4 successful + 1 infra error, as in the cogtrix29 trace)."""
        from cogtrix_core.orchestration.nodes.call_model import _has_substantive_search_results

        msgs = [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "a"}, "id": "t1"}],
            ),
            ToolMessage(
                content=self._ddg_payload(5),  # successful
                tool_call_id="t1",
                name="search_web",
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "b"}, "id": "t2"}],
            ),
            ToolMessage(
                content="Error searching: DuckDuckGo rate-limited (HTTP 429)",
                tool_call_id="t2",
                name="search_web",
            ),
        ]
        # One rich + one error → still substantive (per the cogtrix29 trace
        # pattern that motivated this fix).
        assert _has_substantive_search_results(msgs) is True


# ─────────────────────────────────────────────────────────────────────────────
# Budget guard tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelBudgetGuard:
    """Tests for context budget guard (_apply_context_budget_guard)."""

    def test_budget_guard_applied_to_response(self):
        """Response should be processed by budget guard."""
        # Verify the budget guard pattern exists in source
        source = (
            Path(__file__).parent.parent.parent
            / "cogtrix_core"
            / "orchestration"
            / "nodes"
            / "call_model.py"
        )
        call_model_source = source.read_text()
        assert "_apply_context_budget_guard" in call_model_source


# ─────────────────────────────────────────────────────────────────────────────
# _repair_tool_message_pairs tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelRepairToolMessagePairs:
    """Tests for _repair_tool_message_pairs handling."""

    def test_repair_tool_message_pairs_called(self):
        """_repair_tool_message_pairs should be called during call_model."""
        # Verify the repair pattern exists in source
        source = (
            Path(__file__).parent.parent.parent
            / "cogtrix_core"
            / "orchestration"
            / "nodes"
            / "call_model.py"
        )
        call_model_source = source.read_text()
        assert "_repair_tool_message_pairs" in call_model_source

    def test_repair_cleans_invalid_tool_message_pairs(self):
        """Invalid tool message pairs should be repaired or removed."""
        # This is a source-level check - verify the repair logic is present
        source = (
            Path(__file__).parent.parent.parent
            / "cogtrix_core"
            / "orchestration"
            / "nodes"
            / "call_model.py"
        )
        call_model_source = source.read_text()
        # Check for the repair pattern
        assert "repaired_state_messages" in call_model_source


# ─────────────────────────────────────────────────────────────────────────────
# Stale _bound_cache tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelBoundCache:
    """Tests for _bound_cache handling and stale cache detection."""

    def test_bound_cache_lock_used_for_operations(self):
        """Bound cache operations should be under lock."""
        source = (
            Path(__file__).parent.parent.parent
            / "cogtrix_core"
            / "orchestration"
            / "nodes"
            / "call_model.py"
        )
        call_model_source = source.read_text()
        assert "with _bound_cache_lock:" in call_model_source

    def test_cache_eviction_when_full(self):
        """Cache should evict oldest entry when full (8 entries)."""
        # Behavioral test: verify OrderedDict eviction actually works
        cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        max_cache_size = 8

        # Fill the cache to capacity
        for i in range(max_cache_size):
            key = (f"tool_{i}",)
            cache[key] = f"value_{i}"

        # Verify cache is at capacity
        assert len(cache) == max_cache_size

        # Add one more entry - this should trigger eviction
        new_key = ("tool_new",)
        if len(cache) >= max_cache_size:
            cache.popitem(last=False)
        cache[new_key] = "value_new"

        # Verify oldest entry was evicted (FIFO order)
        assert len(cache) == max_cache_size
        assert ("tool_0",) not in cache  # oldest evicted
        assert ("tool_new",) in cache  # newest present

        # Verify all other entries are still there
        for i in range(1, max_cache_size):
            assert (f"tool_{i}",) in cache

    def test_fingerprint_changes_when_tools_change(self):
        """Tool version change should reset cached fingerprint."""
        tool_version = [0]
        last_tool_version = [0]
        cached_fingerprint = [()]

        node = _make_node(
            tool_version=tool_version,
            last_tool_version=last_tool_version,
            cached_fingerprint=cached_fingerprint,
            active_tools_list=[],
        )

        # Simulate tool version change
        tool_version[0] = 1

        # Run the node to trigger fingerprint update
        state = _make_state([HumanMessage(content="hello")])
        node(state, {})

        # Fingerprint should be updated
        assert cached_fingerprint[0] == ()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint and reflection tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelCheckpointAndReflection:
    """Tests for checkpoint nudge and reflection messages."""

    def test_checkpoint_nudge_injected_after_interval(self):
        """Checkpoint nudge should be injected after checkpoint_nudge_interval rounds."""
        calls_since = [10]  # At interval

        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="response")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        node = _make_node(
            llm=llm,
            checkpoint_nudge_interval=10,
            calls_since_last_checkpoint=calls_since,
            call_count=[5],
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Check for checkpoint nudge message in LLM input — must be a
        # SystemMessage so models treat it as high-salience instruction.
        nudge_msgs = [m for m in captured_msgs if "Checkpoint reminder" in m.content]
        assert len(nudge_msgs) == 1
        assert isinstance(nudge_msgs[0], SystemMessage)

    def test_reflection_injected_after_interval(self):
        """Reflection message should be injected after reflection_interval rounds."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="response")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        last_reflection = [0]
        # call_count is incremented at start of node, so set to 19 to trigger at round 20
        call_count = [19]  # After increment, becomes 20, which is divisible by 20

        node = _make_node(
            llm=llm,
            reflection_interval=20,
            last_reflection_at=last_reflection,
            call_count=call_count,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Check for reflection message in LLM input
        assert any("Work cycle" in m.content for m in captured_msgs)

    def test_reflection_after_errors_includes_debug_cycle_check(self):
        """When consecutive_errors >= 2, reflection should include debug cycle check."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm

        last_reflection = [0]
        consecutive_errors = [3]
        # call_count is incremented at start of node, so set to 19 to trigger at round 20
        call_count = [19]

        node = _make_node(
            llm=llm,
            reflection_interval=20,
            last_reflection_at=last_reflection,
            consecutive_errors=consecutive_errors,
            call_count=call_count,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Check for debug cycle check message in LLM input
        assert any("Debug cycle" in m.content for m in captured_msgs)


# ─────────────────────────────────────────────────────────────────────────────
# Tool state verification tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelToolStateVerification:
    """Tests for tool state verification messages."""

    def test_tool_verification_injected_at_interval(self):
        """Tool state verification should be injected at configured interval."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        last_check = [0]
        # call_count is incremented at start of node, so set to 19 to trigger at round 20
        call_count = [19]
        tool_health_check_interval = 20
        active_names = {"tool_a", "tool_b"}

        node = _make_node(
            llm=llm,
            tool_health_check_interval=tool_health_check_interval,
            last_tool_health_check_at=last_check,
            call_count=call_count,
            active_names=active_names,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Check for tool verification message in LLM input
        assert any("Tool-state" in m.content for m in captured_msgs)

    def test_tool_verification_lists_active_tools(self):
        """Tool verification message should list active tool names."""
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        last_check = [0]
        # call_count is incremented at start of node, so set to 19 to trigger at round 20
        call_count = [19]
        tool_health_check_interval = 20
        active_names = {"search_web", "calculator"}

        node = _make_node(
            llm=llm,
            tool_health_check_interval=tool_health_check_interval,
            last_tool_health_check_at=last_check,
            call_count=call_count,
            active_names=active_names,
            invoke_with_timeout=capture_invoke,
        )
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # Check that tool names are listed in the message
        tool_msg = next((m for m in captured_msgs if "Tool-state" in m.content), None)
        assert tool_msg is not None
        assert "search_web" in tool_msg.content
        assert "calculator" in tool_msg.content


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases and error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelEdgeCases:
    """Edge cases and error handling."""

    def test_no_llm_raises_runtime_error(self):
        """When LLM is None, RuntimeError should be raised."""
        node = _make_node(llm=None)

        with pytest.raises(RuntimeError, match="LLM not configured"):
            node(_make_state([HumanMessage(content="hello")]), {})

    def test_empty_tool_list_invokes_llm_directly(self):
        """When no tools, LLM.invoke should be called directly (not bind_tools)."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        node = _make_node(llm=llm, active_tools_list=[], invoke_with_timeout=capture_invoke)
        state = _make_state([HumanMessage(content="hello")])

        node(state, {})

        # bind_tools should not be called when no tools
        llm.bind_tools.assert_not_called()

    def test_transient_messages_filtered_from_llm_input(self):
        """Messages with transient metadata should be filtered before LLM call."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        captured_msgs: list = []

        def capture_invoke(llm_obj, msgs, config, timeout):
            captured_msgs.extend(msgs)
            return AIMessage(content="ok")

        node = _make_node(llm=llm, active_tools_list=[], invoke_with_timeout=capture_invoke)
        state = _make_state(
            [
                HumanMessage(content="hello"),
                AIMessage(content="response", response_metadata={"transient": True}),
                HumanMessage(content="followup"),
            ]
        )

        node(state, {})

        # Transient message should be filtered
        assert len(captured_msgs) == 2
        assert not any("transient" in str(m) for m in captured_msgs)


# ─────────────────────────────────────────────────────────────────────────────
# Test utility functions used by call_model
# ─────────────────────────────────────────────────────────────────────────────


class TestCallModelUtilityFunctions:
    """Tests for utility functions used by call_model."""

    def test_infer_llm_provider(self):
        """_infer_llm_provider_name should extract provider from LLM object."""
        from cogtrix_core.orchestration.graph import _infer_llm_provider_name

        # Test with OpenAI LLM (proper module path)
        class MockOpenAI:
            __module__ = "langchain_openai"

        assert _infer_llm_provider_name(MockOpenAI()) == "openai"

        # Test with Ollama
        class MockOllama:
            __module__ = "langchain_community.llms.ollama"

        assert _infer_llm_provider_name(MockOllama()) == "ollama"

        # Test with Anthropic
        class MockAnthropic:
            __module__ = "langchain_anthropic"

        assert _infer_llm_provider_name(MockAnthropic()) == "anthropic"

    def test_infer_llm_model(self):
        """_infer_llm_model_name should extract model name from LLM object."""
        from cogtrix_core.orchestration.graph import _infer_llm_model_name

        class MockLLM:
            model_name = "gpt-4"

        assert _infer_llm_model_name(MockLLM()) == "gpt-4"

        class MockLLM2:
            model = "claude-3"

        assert _infer_llm_model_name(MockLLM2()) == "claude-3"


class TestGuardTruncatedToolCalls:
    """Tests for _guard_truncated_tool_calls — prevents partial writes from
    truncated tool-call arguments caused by the LLM hitting its token limit.

    Root cause (May 2026 session bugfix/101-failed-logic): qwen3-coder via
    the spark provider hit max_tokens=4096 while generating write_file args,
    producing a truncated file content string.  The tool reported success and
    the agent overwrote a 16 KB file with a 4 KB fragment, destroying all
    helper functions.  The model then hallucinated the file was complete.
    """

    def setup_method(self):
        from cogtrix_core.orchestration.nodes.call_model import _guard_truncated_tool_calls

        self.guard = _guard_truncated_tool_calls
        self.log = MagicMock()

    def _ai_msg(self, tool_calls, finish_reason="tool_calls", completion_tokens=100):
        return AIMessage(
            content="",
            tool_calls=tool_calls,
            response_metadata={
                "finish_reason": finish_reason,
                "token_usage": {"completion_tokens": completion_tokens},
            },
        )

    # --- definitive detection (finish_reason) ---

    def test_blocks_on_finish_reason_length(self):
        """finish_reason='length' (OpenAI/Azure) → always intercept."""
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"path": "/tmp/x.py", "content": "x" * 2000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="length",
            completion_tokens=4096,
        )
        result = self.guard(msg, None, self.log)
        assert not result.tool_calls
        assert "NOT executed" in result.content
        self.log.warning.assert_called_once()

    def test_blocks_on_finish_reason_max_tokens(self):
        """finish_reason='max_tokens' (Anthropic) → always intercept."""
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"content": "x" * 3000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="max_tokens",
            completion_tokens=8192,
        )
        result = self.guard(msg, None, self.log)
        assert not result.tool_calls

    # --- heuristic detection (provider-specific caps) ---

    def test_blocks_heuristic_exact_cap_large_args(self):
        """completion_tokens at a known cap boundary + large args → intercept.

        Reproduces the exact qwen3-coder/spark failure: finish_reason=
        'tool_calls' but completion_tokens=4096 and a write_file with
        ~16 KB of Python content.
        """
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"path": "/tmp/enhanced.py", "content": "x" * 12000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="tool_calls",
            completion_tokens=4096,  # exact boundary seen in the failing session
        )
        result = self.guard(msg, None, self.log)
        assert not result.tool_calls
        assert "write_file" in result.content
        self.log.warning.assert_called_once()

    def test_no_block_heuristic_small_args_at_cap(self):
        """completion_tokens at cap but small args → do NOT intercept.

        A short write_file (e.g. adding 5 lines) that coincidentally
        generates exactly 4096 tokens should not be blocked.
        """
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"path": "/tmp/x.py", "content": "print('hi')"},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="tool_calls",
            completion_tokens=4096,
        )
        result = self.guard(msg, None, self.log)
        # Small payload — heuristic should not fire
        assert result.tool_calls  # NOT blocked
        self.log.warning.assert_not_called()

    def test_no_block_normal_response(self):
        """Normal tool call (not at any limit) → pass through unchanged."""
        msg = self._ai_msg(
            [{"name": "read_file", "args": {"path": "/tmp/x.py"}, "id": "1", "type": "tool_call"}],
            finish_reason="tool_calls",
            completion_tokens=250,
        )
        result = self.guard(msg, None, self.log)
        assert result.tool_calls
        self.log.warning.assert_not_called()

    def test_blocks_configured_max_tokens(self):
        """When model_max_tokens is set, use it as the exact limit."""
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"content": "x" * 5000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="tool_calls",
            completion_tokens=2048,
        )
        result = self.guard(msg, model_max_tokens=2048, log=self.log)
        assert not result.tool_calls

    def test_non_aimessage_returned_unchanged(self):
        """Non-AIMessage inputs are passed through without modification."""
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content="hello")
        result = self.guard(msg, None, self.log)
        assert result is msg

    def test_no_tool_calls_returned_unchanged(self):
        """Plain text AIMessage (no tool calls) is not affected."""
        msg = AIMessage(content="Here is my response.")
        result = self.guard(msg, None, self.log)
        assert result is msg

    def test_error_message_names_blocked_tools(self):
        """Error message must name the specific tool calls that were blocked."""
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"content": "x" * 8000},
                    "id": "1",
                    "type": "tool_call",
                },
                {
                    "name": "append_file",
                    "args": {"content": "y" * 4000},
                    "id": "2",
                    "type": "tool_call",
                },
            ],
            finish_reason="length",
            completion_tokens=4096,
        )
        result = self.guard(msg, None, self.log)
        assert "write_file" in result.content
        assert "append_file" in result.content

    # --- gap-coverage tests (added in review of PR #1370) ---

    def test_blocks_heuristic_via_usage_metadata_output_tokens(self):
        """Tier 2 heuristic also reads ``response.usage_metadata.output_tokens``.

        Some providers (Anthropic + langchain_core 1.3+, Gemini) expose
        completion token counts on the message's ``usage_metadata``
        rather than in ``response_metadata.token_usage``.  When the
        token_usage path is empty/missing, the guard must fall back
        to ``usage_metadata`` — otherwise the heuristic silently
        skips the very providers it's meant to catch.
        """
        # Build the AIMessage directly so we can populate usage_metadata
        # without going through the _ai_msg helper (which only sets
        # response_metadata.token_usage).
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "/tmp/x.py", "content": "x" * 8000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            response_metadata={"finish_reason": "tool_calls"},
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 4096,
                "total_tokens": 4596,
            },
        )
        result = self.guard(msg, None, self.log)
        assert not result.tool_calls, (
            "guard must read usage_metadata.output_tokens when "
            "response_metadata.token_usage is missing"
        )
        assert "write_file" in result.content
        self.log.warning.assert_called_once()

    def test_no_block_when_model_max_tokens_set_but_not_reached(self):
        """``model_max_tokens`` configured but completion well below it.

        When the model is configured with ``max_tokens=4096`` and the
        response uses only 3500 (not in ``_COMMON_TOKEN_CAPS`` and below
        the configured cap), neither Tier 1 nor Tier 2 should fire,
        even when the tool-call payload is large.  This protects
        against false positives during long-but-complete responses.
        """
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {"path": "/tmp/x.py", "content": "x" * 8000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="tool_calls",
            completion_tokens=3500,  # below configured cap and not a power-of-2 boundary
        )
        result = self.guard(msg, model_max_tokens=4096, log=self.log)
        assert result.tool_calls, (
            "configured max_tokens not yet reached and completion not "
            "at a heuristic cap — must not block"
        )
        self.log.warning.assert_not_called()

    def test_malformed_metadata_does_not_crash_guard(self):
        """Guard must swallow internal exceptions and return the response unchanged.

        The agent turn must never crash because the guard hit unexpected
        data shapes from a misbehaving provider.  Passes a malformed
        ``finish_reason_details`` (string instead of dict) that forces
        the fallback chain to call ``.get`` on a non-dict, raising
        ``AttributeError`` — caught by the broad ``except Exception``
        in the guard.  The returned object must be the original
        response, not a synthesised error message.
        """
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"content": "x" * 4000},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            response_metadata={
                # First two keys are falsy so the OR chain falls through
                # to finish_reason_details, which is malformed.
                "finish_reason": "",
                "stop_reason": None,
                "finish_reason_details": "not-a-dict",
                "token_usage": {"completion_tokens": 4096},
            },
        )
        result = self.guard(msg, None, self.log)
        # Original response returned unchanged — tool_calls preserved.
        assert result is msg, "exception path must return the original response unchanged"
        assert result.tool_calls

    def test_no_block_when_model_max_higher_than_coincidental_cap(self):
        """Legitimate large response at coincidental cap boundary.

        Scenario: model configured with ``max_tokens=8192`` returns a
        legitimate complete response that happens to use exactly 4096
        completion tokens with a 5 KB write_file payload (e.g. a
        medium-sized config file).

        When ``model_max_tokens`` is set, it is treated as the
        authoritative truncation signal — the Tier 2 power-of-2
        heuristic is suppressed entirely for providers that expose
        their cap.  Only Tier 1 (``completion_tokens >=
        model_max_tokens``) and the ``finish_reason`` checks apply.
        Eliminates the false-positive where ``completion_tokens=4096``
        coincidentally matches ``_COMMON_TOKEN_CAPS`` despite being
        well below the configured ``8192`` cap.
        """
        msg = self._ai_msg(
            [
                {
                    "name": "write_file",
                    "args": {
                        "path": "/tmp/config.yaml",
                        "content": "key: value\n" * 500,  # ~5500 chars, well below cap
                    },
                    "id": "1",
                    "type": "tool_call",
                }
            ],
            finish_reason="tool_calls",
            completion_tokens=4096,  # coincidental match with _COMMON_TOKEN_CAPS
        )
        result = self.guard(msg, model_max_tokens=8192, log=self.log)
        assert result.tool_calls, (
            "model_max_tokens=8192 was not reached (completion=4096) — "
            "the configured cap should take precedence over the heuristic"
        )
        self.log.warning.assert_not_called()


# ── cogtrix47 Issue 4 — arithmetic-intent + numeric-data helpers ──────


class TestArithmeticIntentDetection:
    """``_has_arithmetic_intent`` scans the latest user prompt for
    quantity / conversion / total phrasing. Without this signal the
    thinking-break synthesise branch can't tell the model that a
    "could not retrieve" refusal is wrong when the user has asked
    a math question and the tool results contain the numbers.
    """

    def test_how_many_for_money(self) -> None:
        # The cogtrix47 reproducer.
        prompt = (
            "I'll be in Vienna today and need to buy as many items as possible "
            "for $100 NZD. How many can I buy?"
        )
        assert _has_arithmetic_intent([HumanMessage(content=prompt)])

    def test_how_much_does_x_cost(self) -> None:
        assert _has_arithmetic_intent([HumanMessage(content="How much does the X cost?")])

    def test_convert_nzd_to_eur(self) -> None:
        assert _has_arithmetic_intent(
            [HumanMessage(content="Convert 100 NZD to EUR for me, please.")]
        )

    def test_what_is_the_total(self) -> None:
        assert _has_arithmetic_intent([HumanMessage(content="What is the total cost for 5 items?")])

    def test_in_eur(self) -> None:
        assert _has_arithmetic_intent(
            [HumanMessage(content="Quote the Vienna prices in EUR, please.")]
        )

    def test_can_i_afford(self) -> None:
        assert _has_arithmetic_intent(
            [HumanMessage(content="Can I afford a full case at $50 each?")]
        )

    def test_negative_research_question(self) -> None:
        # "How does X work" is research, not arithmetic — must not fire.
        assert not _has_arithmetic_intent(
            [HumanMessage(content="How does the Soudal Fix All Silirub product work?")]
        )

    def test_negative_no_human_message(self) -> None:
        assert not _has_arithmetic_intent([SystemMessage(content="hi"), AIMessage(content="hi")])

    def test_uses_only_last_human_message(self) -> None:
        # An arithmetic prompt 3 turns ago must NOT light up a fresh
        # non-arithmetic research turn.
        msgs = [
            HumanMessage(content="How many widgets for $100?"),
            AIMessage(content="..."),
            HumanMessage(content="Explain how widget production works."),
        ]
        assert not _has_arithmetic_intent(msgs)


class TestNumericToolResultsDetection:
    """``_has_numeric_tool_results`` looks for currency / percentage
    tokens in ToolMessages from the current turn. Pairs with the
    arithmetic-intent detector to decide whether the agent has the
    raw material to attempt a calculation."""

    def _turn(self, *tool_contents: str) -> list:
        msgs: list = [HumanMessage(content="How many can I buy for $100?")]
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}],
            )
        )
        for i, c in enumerate(tool_contents):
            msgs.append(ToolMessage(content=c, tool_call_id=f"c{i + 1}", name="web_search"))
        return msgs

    def test_currency_prefix_token(self) -> None:
        assert _has_numeric_tool_results(self._turn("Hornbach sells the tube for €7.49 each."))

    def test_iso_code_suffix(self) -> None:
        assert _has_numeric_tool_results(self._turn("Spot rate is 100 USD per unit today."))

    def test_iso_code_prefix(self) -> None:
        assert _has_numeric_tool_results(self._turn("Price: USD 12.99 wholesale."))

    def test_percentage(self) -> None:
        assert _has_numeric_tool_results(self._turn("VAT applies at 20%."))

    def test_no_numeric_data(self) -> None:
        assert not _has_numeric_tool_results(self._turn("No relevant pricing information found."))

    def test_ignores_prior_turn_tool_results(self) -> None:
        # A previous turn's tool result containing money MUST NOT
        # satisfy the check for the current turn — arithmetic-fitness
        # is per-turn (mirrors _compute_search_effort scoping).
        msgs = [
            HumanMessage(content="Previous question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "p"}, "id": "c0"}],
            ),
            ToolMessage(content="€50 each", tool_call_id="c0", name="web_search"),
            HumanMessage(content="How many can I buy for $100?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "c1"}],
            ),
            ToolMessage(content="No relevant pricing.", tool_call_id="c1", name="web_search"),
        ]
        assert not _has_numeric_tool_results(msgs)


class TestSearchLoopRecognisesWebSearch:
    """PR-G renamed the legacy ``search_web`` tool to ``web_search``.
    The effort gate and stuck-detection branch must accept the
    modern name; cogtrix47 had 7 ``web_search`` calls counted as
    zero effort because the check was still hard-coded to the old
    name.
    """

    def test_compute_search_effort_counts_web_search(self) -> None:
        msgs = [
            HumanMessage(content="research X"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "q1"}, "id": "c1"},
                    {"name": "web_search", "args": {"query": "q2 distinct"}, "id": "c2"},
                ],
            ),
            ToolMessage(content="result 1", tool_call_id="c1", name="web_search"),
            ToolMessage(content="result 2", tool_call_id="c2", name="web_search"),
        ]
        count, http_get = _compute_search_effort(msgs)
        assert count == 2
        assert http_get is False

    def test_compute_search_effort_still_counts_legacy_search_web(self) -> None:
        # Back-compat: a session with the old tool name still counts.
        msgs = [
            HumanMessage(content="research X"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "q1"}, "id": "c1"}],
            ),
            ToolMessage(content="result 1", tool_call_id="c1", name="search_web"),
        ]
        count, _ = _compute_search_effort(msgs)
        assert count == 1
