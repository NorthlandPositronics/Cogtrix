"""Tests for src/orchestration/compression.py — non-string content guard and tool name sanitization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.orchestration.compression import apply_message_compression, truncate_tool_output

try:
    from langchain_core.messages import AIMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")

# Threshold: total_chars >= max_context_tokens * 4 * 0.72
# With _TRIGGER_CONTEXT = 16_384, threshold = 47_186 chars.
# Content of _TRIGGER_CHARS bytes (50 K) easily exceeds this.
_TRIGGER_CONTEXT = 16_384
_TRIGGER_CHARS = 50_000


def _make_old_ai_messages(n: int = 5) -> list:
    return [AIMessage(content="thinking step") for _ in range(n)]


def _apply(msgs, llm, min_age: int = 1, min_chars: int = 1):
    cache: dict[str, str] = {}
    return apply_message_compression(
        msgs,
        call_count=10,
        compression_cache=cache,
        llm=llm,
        max_context_tokens=_TRIGGER_CONTEXT,
        min_age_cycles=min_age,
        min_chars=min_chars,
    )


class TestNonStringContentGuard:
    """apply_message_compression must skip ToolMessages whose content is not a string."""

    def test_list_content_skipped_without_error(self) -> None:
        """LangChain keeps list content as list; the non-string guard must skip it."""
        old_ais = _make_old_ai_messages()
        # Use a list with a large text block so the str() total triggers the threshold
        big_text = "r" * _TRIGGER_CHARS
        tm = ToolMessage(
            content=[{"type": "text", "text": big_text}],
            tool_call_id="tc_list",
            name="my_tool",
        )
        assert isinstance(tm.content, list), "Precondition: list content stays as list"

        msgs = old_ais + [tm, AIMessage(content="done")]
        llm = MagicMock()
        result = _apply(msgs, llm)
        assert len(result) == len(msgs)
        llm.invoke.assert_not_called()

    def test_string_content_eligible_for_compression(self) -> None:
        old_ais = _make_old_ai_messages()
        long_content = "x" * _TRIGGER_CHARS
        tm = ToolMessage(
            content=long_content,
            tool_call_id="tc_str",
            name="my_tool",
        )
        msgs = old_ais + [tm, AIMessage(content="done")]

        compressed_mock = MagicMock()
        compressed_mock.content = "compressed result"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        result = _apply(msgs, llm, min_chars=100)
        assert len(result) == len(msgs)
        llm.invoke.assert_called_once()


class TestToolNameSanitization:
    """Tool names with control characters must be sanitized before use in prompts."""

    def _capture(self, name: str) -> list[str]:
        old_ais = _make_old_ai_messages()
        long_content = "y" * _TRIGGER_CHARS
        tm = ToolMessage(content=long_content, tool_call_id="tc_san", name=name)
        msgs = old_ais + [tm, AIMessage(content="done")]

        captured: list[str] = []

        def fake_invoke(prompt: str, *args, **kwargs):
            captured.append(prompt)
            m = MagicMock()
            m.content = "compressed"
            return m

        llm = MagicMock()
        llm.invoke.side_effect = fake_invoke
        _apply(msgs, llm, min_chars=100)
        return captured

    def test_newline_in_tool_name_sanitized(self) -> None:
        captured = self._capture("evil\ntool")
        assert len(captured) == 1
        # Verify no raw newline appears within the tool-name field on the "Tool:" line
        tool_line = captured[0].split("Tool:")[1].split("\n")[0]
        assert "\n" not in tool_line

    def test_null_byte_in_tool_name_sanitized(self) -> None:
        captured = self._capture("evil\x00tool")
        assert len(captured) == 1
        assert "\x00" not in captured[0]

    def test_carriage_return_in_tool_name_sanitized(self) -> None:
        captured = self._capture("evil\rtool")
        assert len(captured) == 1
        assert "\r" not in captured[0].split("Tool:")[1].split("\n")[0]


class TestTruncateToolOutput:
    def test_short_text_unchanged(self) -> None:
        text = "hello"
        assert truncate_tool_output(text, 100) == text

    def test_long_text_has_truncation_marker(self) -> None:
        text = "a" * 200
        result = truncate_tool_output(text, 100)
        assert "truncated" in result

    def test_long_text_keeps_start_and_end(self) -> None:
        text = "START" + "x" * 200 + "END"
        result = truncate_tool_output(text, 20)
        assert "START" in result
        assert "END" in result
