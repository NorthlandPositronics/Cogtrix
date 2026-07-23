"""Regression tests for Round 4 deferred audit findings — ARCH-037-09."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# ---------------------------------------------------------------------------
# 1.  sanitize_history — orphaned trailing HumanMessage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")
class TestSanitizeHistoryOrphanedHuman:
    """sanitize_history must not crash or silently drop a trailing HumanMessage."""

    def _sanitize(self, msgs):
        from src.memory.manager import BaseMemoryManager

        return BaseMemoryManager.sanitize_history(msgs)

    def test_trailing_human_message_preserved(self) -> None:
        """A HumanMessage at the end of history with no following AI message
        is NOT the 'bad pair' case (no next message to inspect).  It must be
        preserved so the next agent turn sees the full context."""
        msgs = [
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="second question"),
        ]
        result = self._sanitize(msgs)
        assert len(result) == 3
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "second question"

    def test_human_then_bad_ai_both_dropped(self) -> None:
        """A HumanMessage followed immediately by a bad-content AIMessage must
        remove BOTH messages."""
        msgs = [
            HumanMessage(content="ok question"),
            AIMessage(content="An error occurred: connection refused"),
        ]
        result = self._sanitize(msgs)
        assert len(result) == 0

    def test_orphaned_human_after_tool_chain_preserved(self) -> None:
        """A HumanMessage preceded by a complete tool chain (AI+ToolMessage) and
        followed by nothing must survive sanitization."""
        msgs = [
            HumanMessage(content="do the thing"),
            AIMessage(content="", tool_calls=[{"name": "shell", "id": "t1", "args": {}}]),
            ToolMessage(content="done", tool_call_id="t1"),
            AIMessage(content="all done"),
            HumanMessage(content="next request"),
        ]
        result = self._sanitize(msgs)
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "next request"


# ---------------------------------------------------------------------------
# 2.  compression fallback cap when LLM fails
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")
class TestCompressionFallbackCap:
    """compress_tool_message must cap output at _FALLBACK_MAX_CHARS on LLM failure."""

    def test_llm_failure_caps_output(self) -> None:
        from src.orchestration.compression import (
            _FALLBACK_MAX_CHARS,
            compress_tool_message,
        )

        long_content = "x" * (_FALLBACK_MAX_CHARS + 5000)

        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = compress_tool_message(long_content, "my_tool", failing_llm)

        assert len(result) <= _FALLBACK_MAX_CHARS + 200
        assert "truncated" in result

    def test_llm_failure_short_content_unchanged(self) -> None:
        """Content shorter than _FALLBACK_MAX_CHARS is returned as-is on LLM failure."""
        from src.orchestration.compression import compress_tool_message

        short_content = "short output"
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = compress_tool_message(short_content, "my_tool", failing_llm)
        assert result == short_content

    def test_empty_llm_response_uses_truncation(self) -> None:
        """An LLM that returns empty string triggers the tiny-result fallback."""
        from src.orchestration.compression import _FALLBACK_MAX_CHARS, compress_tool_message

        long_content = "y" * (_FALLBACK_MAX_CHARS + 1000)
        mock_response = MagicMock()
        mock_response.content = ""
        llm = MagicMock()
        llm.invoke.return_value = mock_response

        result = compress_tool_message(long_content, "my_tool", llm)
        assert len(result) <= _FALLBACK_MAX_CHARS + 200


# ---------------------------------------------------------------------------
# 3.  agent_performed_writes — empty exec_msgs write detection
# ---------------------------------------------------------------------------


class TestAgentPerformedWritesEmptyMsgs:
    """agent_performed_writes([]) must return False without raising."""

    def test_empty_msgs_returns_false(self) -> None:
        from src.orchestration.phases import agent_performed_writes

        assert agent_performed_writes([]) is False

    def test_none_action_tool_returns_false(self) -> None:
        """A ToolMessage from a non-action tool must not count as a write."""
        if not _HAS_LANGCHAIN:
            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [ToolMessage(content="output", tool_call_id="t1", name="read_file")]
        assert agent_performed_writes(msgs) is False

    def test_write_file_tool_returns_true(self) -> None:
        """A successful write_file ToolMessage must be detected as a write."""
        if not _HAS_LANGCHAIN:
            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [ToolMessage(content="Written successfully", tool_call_id="t1", name="write_file")]
        assert agent_performed_writes(msgs) is True

    def test_errored_write_not_counted(self) -> None:
        """A write_file ToolMessage that starts with 'Error' must NOT count as a write."""
        if not _HAS_LANGCHAIN:
            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [
            ToolMessage(content="Error: permission denied", tool_call_id="t1", name="write_file")
        ]
        assert agent_performed_writes(msgs) is False
