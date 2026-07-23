"""Tests for src/api/schemas/message.py — chat messages, tool calls, sync turn."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.message import (
    ClearHistoryRequest,
    MessageOut,
    SendMessageRequest,
    SyncTurnOut,
    ToolCallRecord,
    ToolConfirmRequest,
)

# ---------------------------------------------------------------------------
# ToolCallRecord — embedded tool invocation sub-object
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_valid_in_progress(self) -> None:
        r = ToolCallRecord(
            tool_call_id="call_1",
            tool_name="web_search",
            input={"query": "weather Vienna"},
        )
        assert r.output is None
        assert r.duration_ms is None
        assert r.error is None

    def test_valid_completed(self) -> None:
        r = ToolCallRecord(
            tool_call_id="call_1",
            tool_name="web_search",
            input={"query": "x"},
            output="results...",
            duration_ms=340,
        )
        assert r.output == "results..."
        assert r.duration_ms == 340

    def test_valid_with_error(self) -> None:
        r = ToolCallRecord(
            tool_call_id="call_1",
            tool_name="http_get",
            input={"url": "https://x"},
            error="HTTP 404",
        )
        assert r.error == "HTTP 404"
        assert r.output is None

    def test_empty_input_dict_accepted(self) -> None:
        # Tools with zero args still need a dict, not None.
        r = ToolCallRecord(tool_call_id="call_1", tool_name="list_files", input={})
        assert r.input == {}

    def test_missing_required_input(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallRecord(tool_call_id="c1", tool_name="t")  # type: ignore[call-arg]

    def test_missing_required_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallRecord(tool_call_id="c1", input={})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MessageOut — role literal + ensure_utc on created_at
# ---------------------------------------------------------------------------


class TestMessageOut:
    def test_valid_user_message(self) -> None:
        m = MessageOut(
            id="m1",
            session_id="s1",
            role="user",
            content="Hello",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert m.tool_calls == []
        assert m.token_counts is None

    def test_valid_assistant_with_tool_calls(self) -> None:
        m = MessageOut(
            id="m2",
            session_id="s1",
            role="assistant",
            content="Let me check",
            tool_calls=[
                ToolCallRecord(tool_call_id="c1", tool_name="web_search", input={"q": "x"}),
            ],
            token_counts={"input": 320, "output": 88},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert len(m.tool_calls) == 1
        assert m.token_counts == {"input": 320, "output": 88}

    def test_all_four_roles_accepted(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            m = MessageOut(
                id="m",
                session_id="s",
                role=role,  # type: ignore[arg-type]
                content="x",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            assert m.role == role

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageOut(
                id="m",
                session_id="s",
                role="bogus",  # type: ignore[arg-type]
                content="x",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_naive_datetime_gets_utc(self) -> None:
        m = MessageOut(
            id="m",
            session_id="s",
            role="user",
            content="x",
            created_at=datetime(2026, 1, 1),  # naive
        )
        assert m.created_at.tzinfo is UTC

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            MessageOut(  # type: ignore[call-arg]
                session_id="s",
                role="user",
                content="x",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_nested_tool_call_serializes(self) -> None:
        """Nested ToolCallRecord round-trips through model_dump for OpenAPI."""
        m = MessageOut(
            id="m",
            session_id="s",
            role="assistant",
            content="x",
            tool_calls=[
                ToolCallRecord(
                    tool_call_id="c1",
                    tool_name="web_search",
                    input={"q": "x"},
                    output="y",
                    duration_ms=42,
                ),
            ],
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        dumped = m.model_dump()
        assert dumped["tool_calls"][0]["tool_call_id"] == "c1"
        assert dumped["tool_calls"][0]["duration_ms"] == 42


# ---------------------------------------------------------------------------
# SendMessageRequest — content bounds + mode enum
# ---------------------------------------------------------------------------


class TestSendMessageRequest:
    def test_valid_minimal(self) -> None:
        r = SendMessageRequest(content="Hello")
        assert r.mode == "normal"
        assert r.optimize_prompt is None

    def test_all_three_modes_accepted(self) -> None:
        for mode in ("normal", "think", "delegate"):
            r = SendMessageRequest(content="x", mode=mode)  # type: ignore[arg-type]
            assert r.mode == mode

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SendMessageRequest(content="x", mode="ponder")  # type: ignore[arg-type]

    def test_optimize_prompt_override(self) -> None:
        assert SendMessageRequest(content="x", optimize_prompt=True).optimize_prompt is True
        assert SendMessageRequest(content="x", optimize_prompt=False).optimize_prompt is False

    def test_content_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            SendMessageRequest(content="")

    def test_content_at_max_length(self) -> None:
        r = SendMessageRequest(content="x" * 65536)
        assert len(r.content) == 65536

    def test_content_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 65536"):
            SendMessageRequest(content="x" * 65537)

    def test_missing_required_content(self) -> None:
        with pytest.raises(ValidationError):
            SendMessageRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SyncTurnOut — sync-mode response with token + duration accounting
# ---------------------------------------------------------------------------


class TestSyncTurnOut:
    def test_valid(self) -> None:
        out = SyncTurnOut(
            message_id="m1",
            text="The capital is Paris.",
            total_tokens=1800,
            input_tokens=1420,
            output_tokens=380,
            duration_ms=4200,
            tool_calls=3,
        )
        assert out.text == "The capital is Paris."
        assert out.total_tokens == 1800

    def test_zero_tool_calls(self) -> None:
        out = SyncTurnOut(
            message_id="m1",
            text="hi",
            total_tokens=10,
            input_tokens=5,
            output_tokens=5,
            duration_ms=200,
            tool_calls=0,
        )
        assert out.tool_calls == 0

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            SyncTurnOut(  # type: ignore[call-arg]
                message_id="m1",
                text="x",
                input_tokens=5,
                output_tokens=5,
                duration_ms=100,
                tool_calls=0,
            )


# ---------------------------------------------------------------------------
# ClearHistoryRequest — keep_last optional, must be >= 0
# ---------------------------------------------------------------------------


class TestClearHistoryRequest:
    def test_empty_clears_all(self) -> None:
        r = ClearHistoryRequest()
        assert r.keep_last is None

    def test_keep_last_zero_accepted(self) -> None:
        assert ClearHistoryRequest(keep_last=0).keep_last == 0

    def test_keep_last_positive_accepted(self) -> None:
        assert ClearHistoryRequest(keep_last=10).keep_last == 10

    def test_keep_last_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ClearHistoryRequest(keep_last=-1)


# ---------------------------------------------------------------------------
# ToolConfirmRequest — six-action enum dispatched over WebSocket
# ---------------------------------------------------------------------------


class TestToolConfirmRequest:
    def test_all_six_actions_accepted(self) -> None:
        for action in ("allow", "deny", "allow_all", "disable", "forbid_all", "cancel"):
            r = ToolConfirmRequest(
                confirmation_id="conf_1", action=action  # type: ignore[arg-type]
            )
            assert r.action == action

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolConfirmRequest(confirmation_id="conf_1", action="maybe")  # type: ignore[arg-type]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolConfirmRequest(action="allow")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ToolConfirmRequest(confirmation_id="c1")  # type: ignore[call-arg]
