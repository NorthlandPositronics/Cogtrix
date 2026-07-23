"""Tests for src/api/schemas/memory.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.memory import MemoryModeSwitchRequest, MemoryStateOut


class TestMemoryStateOut:
    """MemoryStateOut schema construction and validation."""

    def test_memory_state_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        state = MemoryStateOut(
            session_id="sess-123",
            mode="conversation",
            summary="Previous discussion about APIs.",
            window_messages=20,
            summarized_messages=45,
            tokens_used=8400,
            context_window=131072,
            vector_recall_enabled=True,
            mode_meta={"entities": ["Alice", "Bob"]},
            updated_at=now,
        )
        assert state.session_id == "sess-123"
        assert state.mode == "conversation"
        assert state.tokens_used == 8400

    def test_memory_state_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        state = MemoryStateOut(
            session_id="sess-123",
            mode="code",
            window_messages=10,
            summarized_messages=0,
            tokens_used=1200,
            context_window=32768,
            vector_recall_enabled=False,
            updated_at=naive,
        )
        assert state.updated_at.tzinfo is not None

    def test_memory_state_out_empty_mode_meta(self) -> None:
        """Empty mode_meta uses default_factory."""
        state = MemoryStateOut(
            session_id="sess-123",
            mode="reasoning",
            window_messages=5,
            summarized_messages=0,
            tokens_used=500,
            context_window=8192,
            vector_recall_enabled=False,
            updated_at=datetime.now(UTC),
        )
        assert state.mode_meta == {}

    def test_memory_state_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            MemoryStateOut(
                session_id="sess-123",
                mode="conversation",
                window_messages=10,
                summarized_messages=0,
                tokens_used=500,
                context_window=8192,
                vector_recall_enabled=False,
                # updated_at missing
            )

    def test_memory_state_out_invalid_mode(self) -> None:
        """Invalid memory mode raises ValidationError."""
        with pytest.raises(ValidationError):
            MemoryStateOut(
                session_id="sess-123",
                mode="invalid_mode",
                window_messages=10,
                summarized_messages=0,
                tokens_used=500,
                context_window=8192,
                vector_recall_enabled=False,
                updated_at=datetime.now(UTC),
            )


class TestMemoryModeSwitchRequest:
    """MemoryModeSwitchRequest schema construction and validation."""

    def test_memory_mode_switch_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = MemoryModeSwitchRequest(mode="reasoning")
        assert req.mode == "reasoning"

    def test_memory_mode_switch_request_invalid_mode(self) -> None:
        """Invalid mode raises ValidationError."""
        with pytest.raises(ValidationError):
            MemoryModeSwitchRequest(mode="invalid")

    def test_memory_mode_switch_request_missing_mode(self) -> None:
        """Missing mode raises ValidationError."""
        with pytest.raises(ValidationError):
            MemoryModeSwitchRequest()
