"""Tests for context-overflow error handling in call_model (Issue #291).

Covers:
- _is_context_overflow_error() classification (Layer 1 — pure unit)
- Retry logic on overflow: compress + retry (Layer 2 — mock LLM)
- TinyContextLLM integration: end-to-end overflow recovery (Layer 3)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------------------------
# Layer 1 — Pure unit tests: _is_context_overflow_error
# ---------------------------------------------------------------------------


class TestIsContextOverflowError:
    def test_known_patterns(self):
        from src.orchestration.graph import _is_context_overflow_error

        assert _is_context_overflow_error(Exception("context_length_exceeded"))
        assert _is_context_overflow_error(
            Exception("This model's maximum context length is 4096 tokens")
        )
        assert _is_context_overflow_error(Exception("input is too long for this model"))
        assert _is_context_overflow_error(Exception("prompt is too long, reduce the length"))
        assert _is_context_overflow_error(Exception("context window exceeded"))

    def test_non_overflow_errors(self):
        from src.orchestration.graph import _is_context_overflow_error

        assert not _is_context_overflow_error(Exception("ConnectionError: timeout"))
        assert not _is_context_overflow_error(Exception("401 Unauthorized"))
        assert not _is_context_overflow_error(Exception("rate_limit_exceeded"))
        assert not _is_context_overflow_error(ValueError("bad input"))

    def test_case_insensitive(self):
        from src.orchestration.graph import _is_context_overflow_error

        assert _is_context_overflow_error(Exception("CONTEXT_LENGTH_EXCEEDED"))
        assert _is_context_overflow_error(Exception("Context Window Full"))

    def test_non_overflow_not_misclassified(self):
        from src.orchestration.graph import _is_context_overflow_error

        assert not _is_context_overflow_error(RuntimeError("internal server error"))
        assert not _is_context_overflow_error(Exception(""))


# ---------------------------------------------------------------------------
# Layer 2 — Mock LLM tests: retry logic
# ---------------------------------------------------------------------------


def _make_fake_invoke():
    """Fake invoke: raises overflow on first main call, returns ok on retry."""
    call_log: list[str] = []

    def fake_invoke(messages, config=None):
        is_compression = any("summarise" in str(m.content).lower() for m in messages)
        call_log.append("compression" if is_compression else "main")
        if is_compression:
            return AIMessage(content="[summary]")
        if call_log.count("main") == 1:
            raise Exception("context_length_exceeded")
        return AIMessage(content="Answer after retry.")

    fake_invoke.call_log = call_log  # type: ignore[attr-defined]
    return fake_invoke


def _build_graph_with_fake_llm(fake_invoke_fn, max_context_tokens: int = 16_384):
    """Build a graph with a mock LLM whose .invoke uses *fake_invoke_fn*."""
    from src.orchestration.graph import build_agent_graph

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = fake_invoke_fn
    return build_agent_graph(
        llm=mock_llm,
        system_prompt="",
        active_tools_list=[],
        available_tools={},
        max_context_tokens=max_context_tokens,
        context_compression=True,
    )


class TestOverflowRetryLogic:
    def test_overflow_triggers_retry_and_succeeds(self):
        """On context overflow, compression fires and retry succeeds."""
        fake_invoke = _make_fake_invoke()
        graph = _build_graph_with_fake_llm(fake_invoke)

        # Large history to give compression something to work on
        history = []
        for i in range(5):
            history.append(HumanMessage(content=f"User message {i}: " + "x" * 200))
            history.append(AIMessage(content=f"AI response {i}: " + "y" * 200))
        history.append(HumanMessage(content="Final question"))

        result = graph.invoke({"messages": history})
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Answer after retry." in m.content for m in ai_messages)

    def test_overflow_retry_failure_raises_clean_error(self):
        """If retry also overflows, RuntimeError with clear message is raised."""

        def always_overflow(messages, config=None):
            raise Exception("context_length_exceeded")

        from src.orchestration.graph import build_agent_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = always_overflow

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=16_384,
            context_compression=True,
        )

        with pytest.raises(RuntimeError) as exc_info:
            graph.invoke({"messages": [HumanMessage(content="hi")]})

        assert "Start a new session with /session new" in str(exc_info.value)

    def test_non_overflow_exception_propagates(self):
        """Non-overflow exceptions propagate unchanged — not wrapped in RuntimeError."""

        def connection_error(messages, config=None):
            raise ConnectionError("network unreachable")

        from src.orchestration.graph import build_agent_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = connection_error

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=16_384,
        )

        with pytest.raises(ConnectionError, match="network unreachable"):
            graph.invoke({"messages": [HumanMessage(content="hi")]})


class TestContextMessageCapInvocation:
    def test_cap_runs_before_model_call_and_keeps_tool_pair_intact(self):
        from src.orchestration.graph import build_agent_graph

        seen_messages = []

        class RecordingLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, config=None):
                seen_messages.append(messages)
                return AIMessage(content="final answer")

        graph = build_agent_graph(
            llm=RecordingLLM(),
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=16_384,
            context_compression=False,
            context_max_messages=3,
        )

        history = [
            HumanMessage(content="oldest"),
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "lookup", "args": {}}]),
            ToolMessage(content="ok", tool_call_id="call_1"),
            HumanMessage(content="latest"),
        ]

        graph.invoke({"messages": history})

        assert len(seen_messages) == 1
        invoked = seen_messages[0]
        assert len(invoked) == 3
        assert isinstance(invoked[0], AIMessage)
        assert invoked[0].tool_calls[0]["id"] == "call_1"
        assert isinstance(invoked[1], ToolMessage)
        assert invoked[1].tool_call_id == "call_1"
        assert isinstance(invoked[2], HumanMessage)
        assert invoked[2].content == "latest"

    def test_cap_uses_configured_token_budget(self):
        from src.orchestration.graph import build_agent_graph
        from src.orchestration.run_config import AgentRunConfig

        seen_messages = []

        class RecordingLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, config=None):
                seen_messages.append(messages)
                return AIMessage(content="final answer")

        graph = build_agent_graph(
            config=AgentRunConfig(
                llm=RecordingLLM(),
                system_prompt="",
                active_tools_list=[],
                available_tools={},
                context_max_messages=10,
                context_max_tokens=1,
                context_compression=False,
            ),
        )

        graph.invoke(
            {
                "messages": [
                    HumanMessage(content="oldest"),
                    HumanMessage(content="middle"),
                    HumanMessage(content="latest"),
                ]
            }
        )

        assert len(seen_messages) == 1
        invoked = seen_messages[0]
        assert len(invoked) == 1
        assert isinstance(invoked[0], HumanMessage)
        assert invoked[0].content == "latest"


# ---------------------------------------------------------------------------
# Layer 3 — TinyContextLLM integration
# ---------------------------------------------------------------------------


class TinyContextLLM:
    """Fake LLM that rejects messages exceeding max_chars total content length.

    Handles both list-of-messages (normal invocations) and string (compression
    invocations from compress_tool_message).
    """

    def __init__(self, max_chars: int = 500, compression_keyword: str = "summarise"):
        self.max_chars = max_chars
        self.compression_keyword = compression_keyword
        self.invoke_count = 0

    def _is_compression_call(self, messages) -> bool:
        if isinstance(messages, str):
            return self.compression_keyword in messages.lower()
        return any(
            self.compression_keyword in str(getattr(m, "content", "")).lower() for m in messages
        )

    def _total_chars(self, messages) -> int:
        if isinstance(messages, str):
            return len(messages)
        return sum(len(str(getattr(m, "content", ""))) for m in messages)

    def invoke(self, messages, config=None):
        self.invoke_count += 1
        if self._is_compression_call(messages):
            return AIMessage(content="[compressed summary]")
        total = self._total_chars(messages)
        if total > self.max_chars:
            raise Exception(f"context_length_exceeded: {total} chars > {self.max_chars}")
        return AIMessage(content="ok")

    def bind_tools(self, tools):
        return self


class TestTinyContextLLM:
    def test_end_to_end_overflow_recovery(self):
        """Large history causes overflow, compression fires, retry succeeds.

        Threshold for compression to run: total_chars >= int(16384 * 4 * 0.72) = 47186.
        We use 3 large ToolMessages (~18000 chars each) so total ≈ 54000 > 47186.
        TinyContextLLM rejects calls with total > 50000 chars.
        After compression, ToolMessages become "[compressed summary]" so total << 50000.
        """
        from src.orchestration.graph import build_agent_graph

        # max_chars=50_000: original 54000-char history overflows, compressed doesn't
        tiny_llm = TinyContextLLM(max_chars=50_000)
        graph = build_agent_graph(
            llm=tiny_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=16_384,
            context_compression=True,
        )

        # Build history: 3 ToolMessages of 18000 chars each → total ~54000 > threshold
        history = []
        for i in range(3):
            history.append(HumanMessage(content=f"q{i}"))
            history.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "t", "args": {}, "id": f"c{i}"}],
                )
            )
            history.append(
                ToolMessage(
                    content="x" * 18_000,
                    tool_call_id=f"c{i}",
                    name="some_tool",
                )
            )
            history.append(AIMessage(content=f"a{i}"))
        history.append(HumanMessage(content="Final question"))

        result = graph.invoke({"messages": history})
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        # Turn completed without exception
        assert len(ai_messages) >= 1
        assert any(m.content == "ok" for m in ai_messages)
        # Retry fired (invoke_count > 1)
        assert tiny_llm.invoke_count > 1

    def test_unrecoverable_overflow_raises_clean_error(self):
        """If even a single message exceeds context, clean RuntimeError is raised."""
        from src.orchestration.graph import build_agent_graph

        tiny_llm = TinyContextLLM(max_chars=10)
        graph = build_agent_graph(
            llm=tiny_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=16_384,
            context_compression=True,
        )

        # 50 chars > max_chars=10; compression won't help since there are no ToolMessages
        with pytest.raises(RuntimeError) as exc_info:
            graph.invoke({"messages": [HumanMessage(content="x" * 50)]})

        assert "Start a new session with /session new" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Layer 1 — compression.py: min_age_cycles=0 / min_chars=0 guard verification
# ---------------------------------------------------------------------------


class TestCompressionZeroFloors:
    def test_min_age_cycles_zero_compresses_all_eligible(self):
        """With min_age_cycles=0, all ToolMessages regardless of age are eligible."""
        from src.orchestration.compression import apply_message_compression

        # compress_tool_message requires result > 20 chars to avoid fallback truncation
        compressed_summary = "[compressed summary: key findings preserved]"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=compressed_summary)

        # max_context_tokens=16384 → threshold_chars = int(65536 * 0.72) = 47186
        # Build messages with total_chars >= 47186 so compression actually runs
        big_messages = [
            HumanMessage(content="q" * 20000),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
            ToolMessage(
                content="x" * 20000,
                tool_call_id="c1",
                name="some_tool",
            ),
            AIMessage(content="a" * 10000),
        ]
        cache: dict[str, str] = {}
        result = apply_message_compression(
            big_messages,
            call_count=1,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=16_384,
            min_age_cycles=0,
            min_chars=0,
            emergency_threshold=0.0,
        )
        # The ToolMessage should have been compressed
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0].content == compressed_summary


# ---------------------------------------------------------------------------
# Issue #308 — Mid-turn compression regression tests
# ---------------------------------------------------------------------------


class TestMidTurnCompressionThreshold:
    def test_mid_turn_threshold_is_lower_than_turn_start_threshold(self):
        """_MID_TURN_COMPRESSION_THRESHOLD (0.60) < _COMPRESSION_THRESHOLD_RATIO (0.72).

        REGRESSION: ensures the mid-turn guard fires sooner than the old
        turn-start token-based threshold.
        """
        from src.orchestration.compression import (
            _COMPRESSION_THRESHOLD_RATIO,
            _MID_TURN_COMPRESSION_THRESHOLD,
        )

        assert _MID_TURN_COMPRESSION_THRESHOLD < _COMPRESSION_THRESHOLD_RATIO

    def test_mid_turn_compression_fires_during_tool_loop(self, monkeypatch):
        """Compression fires mid-turn when char estimate exceeds 60% threshold.

        REGRESSION: without fix, compression only fires at turn start using
        the PREVIOUS turn's token count (stale). With the fix, _maybe_compress
        fires based on CURRENT message chars before every model.invoke().
        """
        from src.orchestration.compression import (
            _CHARS_PER_TOKEN,
            COMPRESSION_MIN_AGE_CYCLES,
        )
        from src.orchestration.graph import build_agent_graph

        compression_calls: list[dict] = []
        original_apply = __import__(
            "src.orchestration.compression", fromlist=["apply_message_compression"]
        ).apply_message_compression

        def tracking_apply(msgs, **kw):
            compression_calls.append(kw)
            return original_apply(msgs, **kw)

        monkeypatch.setattr("src.orchestration.graph.apply_message_compression", tracking_apply)

        max_context = 16_384
        # 65% by char estimate → above 60% threshold, below 72% old threshold
        target_chars = int(max_context * _CHARS_PER_TOKEN * 0.65)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Return short enough content to avoid fallback in compress_tool_message
        mock_llm.invoke.return_value = AIMessage(content="[compressed summary content here]")

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=max_context,
            context_compression=True,
            compression_min_age=COMPRESSION_MIN_AGE_CYCLES,
        )

        # Build history: ToolMessage at age >= 3 (3 AIMessages follow it)
        # so the age guard doesn't prevent compression when min_age_ovr=compression_min_age
        history = [
            HumanMessage(content="q1"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
            ToolMessage(content="x" * target_chars, tool_call_id="c1", name="tool"),
            AIMessage(content="a1"),
            AIMessage(content="a2"),
            AIMessage(content="a3"),
            HumanMessage(content="follow-up"),
        ]

        graph.invoke({"messages": history})

        assert compression_calls, (
            "apply_message_compression was not called — mid-turn compression did not fire. "
            "This is the Issue #308 regression: compression must trigger at 60% char estimate."
        )

    def test_emergency_compression_uses_min_age_override_zero(self, monkeypatch):
        """At 85%+ char estimate, _maybe_compress passes min_age_override=0.

        REGRESSION: emergency mode must compress ALL eligible ToolMessages
        regardless of age to prevent overflow during a long tool loop.
        """
        from src.orchestration.compression import _CHARS_PER_TOKEN
        from src.orchestration.graph import build_agent_graph

        captured_kwargs: dict = {}

        def tracking_apply(msgs, **kw):
            captured_kwargs.update(kw)
            return msgs  # return unchanged — we only care about the kwargs

        monkeypatch.setattr("src.orchestration.graph.apply_message_compression", tracking_apply)

        max_context = 16_384
        # 87% by char estimate → above 85% emergency threshold
        target_chars = int(max_context * _CHARS_PER_TOKEN * 0.87)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="done")

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=max_context,
            context_compression=True,
        )

        history = [HumanMessage(content="x" * target_chars)]
        graph.invoke({"messages": history})

        assert (
            "min_age_override" in captured_kwargs
        ), "_maybe_compress did not pass min_age_override to apply_message_compression"
        assert captured_kwargs["min_age_override"] == 0, (
            f"Expected min_age_override=0 for emergency compression, "
            f"got {captured_kwargs['min_age_override']}"
        )

    def test_no_infinite_loop_when_compression_changes_nothing(self, monkeypatch):
        """If compression returns messages unchanged, tool loop proceeds without retry.

        REGRESSION: _maybe_compress must not loop when apply_message_compression
        returns the same message list (e.g. no messages are old enough to compress).
        """
        from src.orchestration.compression import _CHARS_PER_TOKEN
        from src.orchestration.graph import build_agent_graph

        call_count_tracker = [0]

        def no_op_apply(msgs, **kw):
            call_count_tracker[0] += 1
            return msgs  # always unchanged

        monkeypatch.setattr("src.orchestration.graph.apply_message_compression", no_op_apply)

        max_context = 16_384
        target_chars = int(max_context * _CHARS_PER_TOKEN * 0.70)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="done")

        graph = build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            max_context_tokens=max_context,
            context_compression=True,
        )

        history = [HumanMessage(content="x" * target_chars)]
        # Must complete without infinite loop or RecursionError
        result = graph.invoke({"messages": history})

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any(
            "done" in m.content for m in ai_messages
        ), "Graph did not complete — possible infinite loop in _maybe_compress"
        # apply_message_compression called at most once per call_model invocation
        assert call_count_tracker[0] <= 3, (
            f"apply_message_compression called {call_count_tracker[0]} times — "
            "possible retry loop when compression changes nothing"
        )

    def test_min_age_override_bypasses_internal_threshold_in_compression(self):
        """min_age_override not None → apply_message_compression skips threshold check.

        When _maybe_compress passes min_age_override, apply_message_compression
        must run even when the char/token pressure is between 60-72%
        (below the internal 0.72 threshold but above the mid-turn 0.60 threshold).
        """
        from src.orchestration.compression import (
            _CHARS_PER_TOKEN,
            _COMPRESSION_THRESHOLD_RATIO,
            _MID_TURN_COMPRESSION_THRESHOLD,
            apply_message_compression,
        )

        compressed_summary = "[compressed: key facts preserved for mid-turn check]"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=compressed_summary)

        max_context = 16_384
        # Build messages at 65% — above mid-turn (0.60) but below old threshold (0.72)
        target_chars = int(max_context * _CHARS_PER_TOKEN * 0.65)
        assert _MID_TURN_COMPRESSION_THRESHOLD < _COMPRESSION_THRESHOLD_RATIO

        # 4 AIMessages to give age >= 3 for the ToolMessage
        messages = [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
            ToolMessage(content="x" * target_chars, tool_call_id="c1", name="tool"),
            AIMessage(content="a1"),
            AIMessage(content="a2"),
            AIMessage(content="a3"),
        ]

        cache: dict[str, str] = {}

        # Without min_age_override=None (normal call) at 65%: would return early.
        # With min_age_override=3 (from _maybe_compress): must NOT return early.
        result = apply_message_compression(
            messages,
            call_count=5,
            compression_cache=cache,
            llm=mock_llm,
            max_context_tokens=max_context,
            min_age_cycles=3,
            min_chars=100,
            min_age_override=3,  # non-None → bypass threshold check
        )

        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert tool_msgs, "No ToolMessages in result"
        assert tool_msgs[0].content == compressed_summary, (
            "ToolMessage was not compressed — min_age_override did not bypass "
            "the internal threshold check"
        )
