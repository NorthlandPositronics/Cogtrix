"""Tests for AIMessage compression support in apply_message_compression() (Issue #294)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.orchestration.compression import apply_message_compression

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")

# Threshold: total_chars >= max_context_tokens * _CHARS_PER_TOKEN * 0.72
# With max_context_tokens=16_384 and _CHARS_PER_TOKEN=3, threshold ≈ 35,389 chars.
# With 5 AI pairs (10 messages), each AI needs ~4 KB to reliably exceed threshold.
_TRIGGER_CONTEXT = 16_384
# 4 pairs (8 msgs) × 10 KB = 40 KB > threshold (16_384 * 3 * 0.72 ≈ 35,389 chars)
_LARGE_AI_CONTENT = "A" * 10_000  # well above ai_min_chars=500
_SMALL_AI_CONTENT = "short"


def _make_llm(summary: str = "summarised content") -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.content = summary
    llm.invoke.return_value = response
    return llm


def _apply(msgs, llm, min_age: int = 1, min_chars: int = 1, ai_min_chars: int = 500):
    return apply_message_compression(
        msgs,
        call_count=10,
        compression_cache={},
        llm=llm,
        max_context_tokens=_TRIGGER_CONTEXT,
        min_age_cycles=min_age,
        min_chars=min_chars,
        emergency_threshold=0.0,  # always trigger
        ai_min_chars=ai_min_chars,
    )


def _build_history(n_pairs: int, ai_content: str = _LARGE_AI_CONTENT) -> list:
    """Build alternating HumanMessage / AIMessage pairs."""
    msgs = []
    for i in range(n_pairs):  # noqa: B007
        msgs.append(HumanMessage(content=f"User question {i}"))
        msgs.append(AIMessage(content=ai_content))
    return msgs


# ---------------------------------------------------------------------------
# Core compression tests
# ---------------------------------------------------------------------------


class TestAIMessagesCompressedWhenOldAndLarge:
    """Old, large AIMessages are summarised; recent ones and human messages are untouched."""

    def test_old_large_ai_messages_are_compressed(self):
        """First 3 of 5 AIMessages (age >= 2) are compressed; last 2 protected."""
        msgs = _build_history(5)  # 5 pairs → 5 AIMessages
        llm = _make_llm("key finding")
        result = _apply(msgs, llm, min_age=2)

        ai_results = [m for m in result if isinstance(m, AIMessage)]
        assert len(ai_results) == 5

        # First 3 should be summarised
        for ai_msg in ai_results[:3]:
            assert ai_msg.content.startswith(
                "[Summary:"
            ), f"Expected summary prefix, got: {ai_msg.content[:60]}"

        # Last 2 should be unchanged (protected)
        for ai_msg in ai_results[3:]:
            assert ai_msg.content == _LARGE_AI_CONTENT

    def test_human_messages_never_compressed(self):
        """HumanMessages must never be modified regardless of age or content size."""
        msgs = _build_history(5)
        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1)

        for msg in result:
            if isinstance(msg, HumanMessage):
                assert not msg.content.startswith("[Summary:")
                assert "User question" in msg.content

    def test_small_ai_messages_not_compressed(self):
        """AIMessages shorter than ai_min_chars are skipped."""
        msgs = _build_history(5, ai_content=_SMALL_AI_CONTENT)
        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1, ai_min_chars=500)

        for msg in result:
            if isinstance(msg, AIMessage):
                assert msg.content == _SMALL_AI_CONTENT

        llm.invoke.assert_not_called()

    def test_compression_does_not_mutate_input(self):
        """Input messages list must not be mutated."""
        msgs = _build_history(4)
        original_contents = [m.content for m in msgs]
        llm = _make_llm("summary")
        _apply(msgs, llm, min_age=1)

        for original, msg in zip(original_contents, msgs, strict=False):
            assert msg.content == original


class TestAIMessagesProtectedWhenRecent:
    """Last 2 AIMessages are never compressed regardless of size or age."""

    def test_last_two_ai_messages_always_protected(self):
        """With 4 large old AIMessages, only first 2 are compressed."""
        msgs = _build_history(4)  # 4 AIMessages
        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1)

        ai_results = [m for m in result if isinstance(m, AIMessage)]
        assert len(ai_results) == 4

        # First 2 compressed
        assert ai_results[0].content.startswith("[Summary:")
        assert ai_results[1].content.startswith("[Summary:")

        # Last 2 protected
        assert ai_results[2].content == _LARGE_AI_CONTENT
        assert ai_results[3].content == _LARGE_AI_CONTENT

    def test_with_only_two_ai_messages_none_compressed(self):
        """When there are only 2 AIMessages both are protected."""
        msgs = _build_history(2)
        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1)

        for msg in result:
            if isinstance(msg, AIMessage):
                assert msg.content == _LARGE_AI_CONTENT

        llm.invoke.assert_not_called()

    def test_with_one_ai_message_not_compressed(self):
        """Single AIMessage is always protected."""
        msgs = _build_history(1)
        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1)

        assert result[1].content == _LARGE_AI_CONTENT
        llm.invoke.assert_not_called()


class TestHumanMessagesNeverCompressed:
    """HumanMessages are never touched regardless of age or size."""

    def test_large_old_human_messages_untouched(self):
        """Large, old HumanMessages should never receive summary treatment."""
        big_human_content = "H" * 5_000
        msgs = []
        for _ in range(5):
            msgs.append(HumanMessage(content=big_human_content))
            msgs.append(AIMessage(content=_LARGE_AI_CONTENT))

        llm = _make_llm("summary")
        result = _apply(msgs, llm, min_age=1)

        for msg in result:
            if isinstance(msg, HumanMessage):
                assert msg.content == big_human_content
                assert not msg.content.startswith("[Summary:")


class TestAICompressionDoesNotCrashOnLLMFailure:
    """AIMessage compression failure leaves original intact — no exception propagates."""

    def test_llm_failure_leaves_original_intact(self):
        """When LLM raises an exception, the original AIMessage is preserved."""
        msgs = _build_history(4)
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")

        # Must not raise
        result = _apply(msgs, llm, min_age=1)

        # All AIMessages should retain original content
        for msg in result:
            if isinstance(msg, AIMessage):
                assert msg.content == _LARGE_AI_CONTENT

    def test_partial_llm_failure_preserves_successes(self):
        """If LLM fails on some calls but succeeds on others, successful ones are updated."""
        msgs = _build_history(5)
        llm = MagicMock()

        call_count = [0]
        good_response = MagicMock()
        good_response.content = "summary ok"

        def side_effect(prompt_msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            return good_response

        llm.invoke.side_effect = side_effect
        result = _apply(msgs, llm, min_age=2)

        ai_results = [m for m in result if isinstance(m, AIMessage)]
        # At least some should be summarised, some might retain original
        summaries = [m for m in ai_results if m.content.startswith("[Summary:")]
        originals = [m for m in ai_results if m.content == _LARGE_AI_CONTENT]
        assert len(summaries) + len(originals) == 5  # all accounted for


class TestToolMessagesUnaffectedByAIPass:
    """ToolMessages are still compressed correctly alongside AIMessages."""

    def test_tool_and_ai_both_compressed(self):
        """ToolMessages and old AIMessages can both be compressed in one call."""
        big_tool_content = "T" * 3_000
        tool_msg = ToolMessage(content=big_tool_content, tool_call_id="tc1", name="shell")
        # 3 AI messages before the tool, 2 after (to protect last 2 AI)
        ai_old = [AIMessage(content=_LARGE_AI_CONTENT) for _ in range(3)]
        ai_recent = [AIMessage(content=_LARGE_AI_CONTENT), AIMessage(content=_LARGE_AI_CONTENT)]
        msgs = ai_old + [tool_msg] + ai_recent

        tool_response = MagicMock()
        tool_response.content = "compressed tool"
        ai_response = MagicMock()
        ai_response.content = "compressed ai"

        llm = MagicMock()
        llm.invoke.return_value = ai_response

        # For ToolMessage compression the llm is called differently (compress_tool_message)
        with patch(
            "cogtrix_core.orchestration.compression.compress_tool_message",
            return_value="compressed tool",
        ):
            result = _apply(msgs, llm, min_age=1)

        # ToolMessage should be compressed
        tool_results = [m for m in result if isinstance(m, ToolMessage)]
        assert tool_results[0].content == "compressed tool"

        # Old AIMessages should also be compressed
        ai_results = [m for m in result if isinstance(m, AIMessage)]
        for ai_msg in ai_results[:3]:
            assert ai_msg.content.startswith("[Summary:")


# ---------------------------------------------------------------------------
# /compact slash command integration tests
# ---------------------------------------------------------------------------


class TestCompactReportsAICompression:
    """/compact output reports AI message compression separately."""

    def test_compact_reports_ai_summarised(self, capsys):
        """When only AIMessages change, output mentions 'assistant response'."""
        try:
            from cogtrix import _build_slash_commands
        except ImportError:
            pytest.skip("cogtrix not importable")

        reg = _build_slash_commands()

        orig = AIMessage(content="long response " * 100)
        compressed_msg = AIMessage(content="[Summary: short]")

        mm = MagicMock()
        mm._messages = [orig]
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        with patch(
            "cogtrix_core.cli.commands.apply_message_compression", return_value=[compressed_msg]
        ):
            reg.dispatch("/compact")

        out, _ = capsys.readouterr()
        assert "assistant response" in out.lower()

    def test_compact_reports_tool_and_ai_separately(self, capsys):
        """When both ToolMessages and AIMessages change, both are mentioned."""
        try:
            from cogtrix import _build_slash_commands
        except ImportError:
            pytest.skip("cogtrix not importable")

        reg = _build_slash_commands()

        orig_tool = ToolMessage(content="long tool " * 100, tool_call_id="tc1", name="shell")
        orig_ai = AIMessage(content="long response " * 100)

        comp_tool = ToolMessage(content="[Summary: tool]", tool_call_id="tc1", name="shell")
        comp_ai = AIMessage(content="[Summary: ai]")

        mm = MagicMock()
        mm._messages = [orig_tool, orig_ai]
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        with patch(
            "cogtrix_core.cli.commands.apply_message_compression", return_value=[comp_tool, comp_ai]
        ):
            reg.dispatch("/compact")

        out, _ = capsys.readouterr()
        assert "tool result" in out.lower() or "tool" in out.lower()
        assert "assistant response" in out.lower()


class TestCompactNothingWhenTrulyEmpty:
    """/compact reports 'Nothing to compress' only when neither ToolMessages
    nor AIMessages qualify."""

    def test_nothing_to_compress_when_nothing_changes(self, capsys):
        """When apply_message_compression returns the same content, prints Nothing."""
        try:
            from cogtrix import _build_slash_commands
        except ImportError:
            pytest.skip("cogtrix not importable")

        reg = _build_slash_commands()

        msg = MagicMock()
        msg.content = "short"
        mm = MagicMock()
        mm._messages = [msg]
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        with patch("cogtrix_core.cli.commands.apply_message_compression", return_value=[msg]):
            reg.dispatch("/compact")

        out, _ = capsys.readouterr()
        assert "Nothing to compress" in out

    def test_nothing_to_compress_when_messages_empty(self, capsys):
        """When _messages is empty or falsy, prints Nothing to compress."""
        try:
            from cogtrix import _build_slash_commands
        except ImportError:
            pytest.skip("cogtrix not importable")

        reg = _build_slash_commands()

        mm = MagicMock()
        mm._messages = []
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        reg.dispatch("/compact")

        out, _ = capsys.readouterr()
        assert "Nothing to compress" in out or "no messages" in out.lower()


class TestToolCallsPreservedOnCompression:
    """#2365: compressing an AIMessage that carries tool_calls must keep the
    tool-call declarations, so the ToolMessages answering them are not orphaned
    (the upstream of the provider-400 tool-pair corruption under heavy
    compression)."""

    def test_tool_calls_survive_ai_compression(self):
        # An OLD, large AIMessage that ALSO declares a tool call, with its answer.
        # Enough large AIMessages follow it that the context exceeds the trigger
        # threshold and it is old enough to be eligible (not one of the last two).
        msgs = [
            HumanMessage(content="do the task"),
            AIMessage(
                content=_LARGE_AI_CONTENT,
                tool_calls=[{"id": "tc1", "name": "execute_shell_command", "args": {}}],
            ),
            ToolMessage(content="result", tool_call_id="tc1", name="execute_shell_command"),
            HumanMessage(content="q1"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="q2"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="q3"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="last"),
            AIMessage(content="recent reply"),  # protected (one of the last two)
        ]

        result = _apply(msgs, _make_llm("summary text"), min_age=1)

        # The tool-call AIMessage was summarised (content replaced) ...
        target = next(
            m
            for m in result
            if isinstance(m, AIMessage)
            and any(tc.get("id") == "tc1" for tc in (getattr(m, "tool_calls", None) or []))
        )
        assert str(target.content).startswith("[Summary"), "content should be summarised"
        # ... but the tool_call declaration survived, so tc1 is still declared.
        assert [tc["id"] for tc in target.tool_calls] == ["tc1"]
        # And the answering ToolMessage is still present (not orphaned/stripped).
        tool_ids = [getattr(m, "tool_call_id", None) for m in result if isinstance(m, ToolMessage)]
        assert "tc1" in tool_ids

    def test_additional_kwargs_preserved_but_raw_tool_calls_dropped(self):
        # additional_kwargs (e.g. provider cache metadata) survive compression, but
        # any RAW tool_calls copy inside it is dropped — .tool_calls is the source of
        # truth, and a stale/divergent raw copy could re-orphan a ToolMessage via the
        # repair (forge audit; matches message_repair/phases convention).
        msgs = [
            HumanMessage(content="do the task"),
            AIMessage(
                content=_LARGE_AI_CONTENT,
                tool_calls=[{"id": "tc1", "name": "execute_shell_command", "args": {}}],
                additional_kwargs={
                    "cache_control": {"type": "ephemeral"},
                    "tool_calls": [{"id": "STALE", "type": "function"}],
                },
            ),
            ToolMessage(content="ok", tool_call_id="tc1", name="execute_shell_command"),
            HumanMessage(content="q1"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="q2"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="q3"),
            AIMessage(content=_LARGE_AI_CONTENT),
            HumanMessage(content="last"),
            AIMessage(content="recent reply"),
        ]
        result = _apply(msgs, _make_llm("summary text"), min_age=1)

        target = next(
            m
            for m in result
            if isinstance(m, AIMessage)
            and any(tc.get("id") == "tc1" for tc in (getattr(m, "tool_calls", None) or []))
        )
        # non-tool_calls metadata kept ...
        assert target.additional_kwargs.get("cache_control") == {"type": "ephemeral"}
        # ... but the stale raw tool_calls copy is gone (no divergent declaration).
        assert "tool_calls" not in target.additional_kwargs
