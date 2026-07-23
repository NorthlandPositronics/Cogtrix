"""Tests for pair-safe context message capping."""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.orchestration.graph import _apply_context_message_cap  # noqa: E402


def _ai(tool_call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "lookup", "args": {}}],
    )


def _tool(tool_call_id: str) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=tool_call_id)


class TestContextMessageCap:
    def test_trims_oldest_messages_without_splitting_tool_pair(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            _ai("call_1"),
            _tool("call_1"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, 3)

        assert len(result) == 3
        assert isinstance(result[0], AIMessage)
        assert result[0].tool_calls[0]["id"] == "call_1"
        assert isinstance(result[1], ToolMessage)
        assert result[1].tool_call_id == "call_1"
        assert result[2].content == "latest"

    def test_truncation_logs_warning(self, caplog) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            _ai("call_1"),
            _tool("call_1"),
            HumanMessage(content="latest"),
        ]

        with caplog.at_level("WARNING"):
            _apply_context_message_cap(msgs, 3)

        assert any("context_max_messages=3" in record.message for record in caplog.records)

    def test_zero_disables_cap(self) -> None:
        msgs = [HumanMessage(content="a"), _ai("call_2"), _tool("call_2")]

        result = _apply_context_message_cap(msgs, 0)

        assert result == msgs

    def test_trims_by_token_budget(self) -> None:
        msgs = [
            HumanMessage(content="oldest message"),
            HumanMessage(content="middle message"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, max_messages=10, max_tokens=3)

        assert result == [msgs[-1]]

    def test_combined_caps_use_stricter_limit(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="middle"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, max_messages=2, max_tokens=100)

        assert result == msgs[-2:]

    def test_oversized_latest_message_is_preserved(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="x" * 100),
        ]

        result = _apply_context_message_cap(msgs, max_messages=10, max_tokens=1)

        assert result == [msgs[-1]]
