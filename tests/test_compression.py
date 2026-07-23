"""Tests for src/orchestration/compression.py — non-string content guard and tool name sanitization."""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock

import pytest

from src.orchestration.compression import (
    _CHARS_PER_TOKEN,
    _COMPRESSION_THRESHOLD_RATIO,
    apply_message_compression,
    truncate_tool_output,
)

try:
    from langchain_core.messages import AIMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")

# Threshold: total_chars >= max_context_tokens * _CHARS_PER_TOKEN * 0.72
# With _TRIGGER_CONTEXT = 16_384 and _CHARS_PER_TOKEN = 3, threshold = 35_389 chars.
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


class TestCharsPerTokenMultiplier:
    def test_compression_triggers_at_lower_char_count(self) -> None:
        """Compression fires at 2x multiplier where 4x would not.

        max_context_tokens = 16_384 (minimum for compression to run at all)
        With *4: threshold = 16_384 * 4 * 0.72 = 47_185 chars — would NOT trigger at 25_000 chars
        With *2: threshold = 16_384 * 2 * 0.72 = 23_593 chars — DOES trigger at 25_000 chars
        """
        assert _CHARS_PER_TOKEN == 2, "constant must be 2 for this test to be meaningful"

        max_tokens = 16_384
        # 25_000 chars: above *2 threshold (23_593) but below *4 threshold (47_185)
        content = "z" * 25_000
        old_ais = [AIMessage(content="step") for _ in range(4)]
        tm = ToolMessage(content=content, tool_call_id="tc_cpt", name="my_tool")
        msgs = old_ais + [tm, AIMessage(content="done")]

        compressed_mock = MagicMock()
        compressed_mock.content = "compressed summary of tool output"  # > 20 chars, < original
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        cache: dict[str, str] = {}
        result = apply_message_compression(
            msgs,
            call_count=10,
            compression_cache=cache,
            llm=llm,
            max_context_tokens=max_tokens,
            min_age_cycles=1,
            min_chars=100,
        )

        # Compression must have fired — at least one message differs from input
        assert result != msgs, "compression should have triggered but did not"
        llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Regression: token-based trigger (BUG #302)
# ---------------------------------------------------------------------------


class TestTokenBasedCompressionTrigger:
    """BUG #302 — compression must fire on actual_input_tokens pressure, not only char count."""

    _MAX_CTX = 100_000  # large enough that char threshold is far above small message content

    def _small_messages(self) -> list:
        """Build messages with total chars << char threshold but eligible for compression."""
        # 5 old AIMessages + 1 ToolMessage (2500 chars) + 1 final AIMessage
        # total_chars ≈ 5*13 + 2500 + 4 ≈ 2569  (well below char threshold of ~216 K)
        old_ais = [AIMessage(content="thinking step") for _ in range(5)]
        tm = ToolMessage(
            content="x" * 2_500,
            tool_call_id="tc_tok",
            name="my_tool",
        )
        return old_ais + [tm, AIMessage(content="done")]

    def _make_llm(self) -> MagicMock:
        compressed_mock = MagicMock()
        compressed_mock.content = "compressed result (token trigger)"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock
        return llm

    def _apply_with_tokens(self, msgs, actual_input_tokens: int, min_chars: int = 100) -> list:
        return apply_message_compression(
            msgs,
            call_count=10,
            compression_cache={},
            llm=self._make_llm(),
            max_context_tokens=self._MAX_CTX,
            min_age_cycles=1,
            min_chars=min_chars,
            actual_input_tokens=actual_input_tokens,
        )

    def test_compression_triggers_on_token_pressure_not_chars(self) -> None:
        """
        REGRESSION: compression fires when actual_input_tokens/max_context >= 0.72,
        even when total working-memory chars are far below the char threshold.
        This test FAILS on unpatched code (char check short-circuits before token check).
        """
        msgs = self._small_messages()
        # Confirm total chars are well below the char threshold
        char_threshold = int(self._MAX_CTX * _CHARS_PER_TOKEN * _COMPRESSION_THRESHOLD_RATIO)
        total_chars = sum(len(getattr(m, "content", "") or "") for m in msgs)
        assert total_chars < char_threshold, "precondition: chars must be below threshold"

        # actual_input_tokens = 80% of max context — above the 72% trigger
        actual_tokens = int(self._MAX_CTX * 0.80)
        llm = self._make_llm()
        result = apply_message_compression(
            msgs,
            call_count=10,
            compression_cache={},
            llm=llm,
            max_context_tokens=self._MAX_CTX,
            min_age_cycles=1,
            min_chars=100,
            actual_input_tokens=actual_tokens,
        )
        assert result != msgs, "compression should have fired on token pressure but did not"
        assert llm.invoke.called, "LLM must have been invoked for compression"

    def test_compression_skips_on_low_token_pressure(self) -> None:
        """When actual_input_tokens < threshold, compression does not run."""
        msgs = self._small_messages()
        # 50% token pressure — below the 72% threshold
        actual_tokens = int(self._MAX_CTX * 0.50)
        result = self._apply_with_tokens(msgs, actual_tokens)
        assert result == msgs, "compression must be skipped at low token pressure"

    def test_compression_falls_back_to_chars_when_no_token_data(self) -> None:
        """When actual_input_tokens=0, char-based threshold is used as fallback."""
        # Build messages above the char threshold
        max_ctx = _TRIGGER_CONTEXT  # 16_384 — small context so char threshold is low
        old_ais = [AIMessage(content="thinking step") for _ in range(5)]
        tm = ToolMessage(
            content="z" * _TRIGGER_CHARS,  # 50 K chars, well above char threshold
            tool_call_id="tc_fallback",
            name="my_tool",
        )
        msgs = old_ais + [tm, AIMessage(content="done")]

        compressed_mock = MagicMock()
        compressed_mock.content = "compressed result"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        result = apply_message_compression(
            msgs,
            call_count=10,
            compression_cache={},
            llm=llm,
            max_context_tokens=max_ctx,
            min_age_cycles=1,
            min_chars=100,
            actual_input_tokens=0,  # no token data — must fall back to char check
        )
        assert result != msgs, "char-based fallback must still trigger compression"
        assert llm.invoke.called

    def test_emergency_compression_at_90pct(self) -> None:
        """At >= 90% token pressure, emergency compression runs with min_age=0."""
        # Build a ToolMessage that is young (age < normal min_age_cycles=5) so it
        # would NOT be compressed under normal rules, but SHOULD be under emergency.
        # age = 0 AIMessages after it (it's the last message before the final AI)
        young_tm = ToolMessage(
            content="y" * 3_000,  # above min_chars=100
            tool_call_id="tc_emerg",
            name="my_tool",
        )
        msgs = [young_tm, AIMessage(content="done")]

        compressed_mock = MagicMock()
        compressed_mock.content = "emergency compressed result"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        # 92% token pressure — above the 90% emergency threshold
        actual_tokens = int(self._MAX_CTX * 0.92)
        result = apply_message_compression(
            msgs,
            call_count=10,
            compression_cache={},
            llm=llm,
            max_context_tokens=self._MAX_CTX,
            min_age_cycles=5,  # normally requires 5 AI messages after the tool message
            min_chars=100,
            actual_input_tokens=actual_tokens,
        )
        assert result != msgs, "emergency compression must fire at 90%+ token pressure"
        assert llm.invoke.called, "LLM must be invoked even for young messages in emergency mode"


# ---------------------------------------------------------------------------
# Regression: per-call timeout and overall deadline (Issue #304)
# ---------------------------------------------------------------------------


class TestCompressionTimeout:
    """BUG #304 — compression LLM calls must time out gracefully, not hang forever."""

    _TRIGGER_CTX = _TRIGGER_CONTEXT

    def _make_eligible_tool_messages(self, n: int = 2) -> list:
        """Build n old+large ToolMessages that are eligible for compression."""
        old_ais = _make_old_ai_messages(5)
        tms = [
            ToolMessage(
                content="x" * _TRIGGER_CHARS,
                tool_call_id=f"tc_timeout_{k}",
                name="my_tool",
            )
            for k in range(n)
        ]
        return old_ais + tms + [AIMessage(content="done")]

    def test_compression_per_call_timeout_leaves_original(self, monkeypatch) -> None:
        """
        REGRESSION: When a compression LLM call raises TimeoutError, the original
        message is preserved and no exception propagates.

        Without fix: AIMessage loop calls llm.invoke() with no timeout → hangs.
        With fix: TimeoutError caught → original returned, no crash.
        """

        msgs = self._make_eligible_tool_messages(2)
        _ = [getattr(m, "content", None) for m in msgs]

        # Make llm.invoke raise TimeoutError (simulates a stalled LLM call)
        llm = MagicMock()
        llm.invoke.side_effect = concurrent.futures.TimeoutError("simulated timeout")

        # Must not raise
        result = _apply(msgs, llm)

        assert len(result) == len(msgs), "result length must match input"
        # No [Summary:] prefix — original or truncation, but never a summary prefix
        for msg in result:
            c = getattr(msg, "content", "")
            assert not str(c).startswith(
                "[Summary:"
            ), f"Timed-out call must not produce [Summary:] prefix, got: {str(c)[:60]}"

    def test_compression_overall_deadline_stops_loop(self, monkeypatch) -> None:
        """
        REGRESSION: When overall deadline is set to 0, compression stops immediately
        and returns without processing items.

        Without fix: no deadline code → all 5 items compressed (mock is fast).
        With fix: deadline=0 → items skipped, result unchanged.
        """
        import src.orchestration.compression as comp_mod

        # Force immediate deadline
        monkeypatch.setattr(comp_mod, "_COMPRESSION_TOTAL_TIMEOUT_SECS", 0)

        old_ais = _make_old_ai_messages(5)
        tms = [
            ToolMessage(
                content="z" * _TRIGGER_CHARS,
                tool_call_id=f"tc_dl_{k}",
                name="my_tool",
            )
            for k in range(5)
        ]
        msgs = old_ais + tms + [AIMessage(content="done")]
        _ = [getattr(m, "content", None) for m in msgs]

        # LLM mock that would compress successfully if invoked
        compressed_mock = MagicMock()
        compressed_mock.content = "compressed"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        result = _apply(msgs, llm)

        # With deadline=0, no tool messages should be summarised (deadline fires first)
        for msg in result:
            c = getattr(msg, "content", "")
            assert not str(c).startswith(
                "[Summary:"
            ), f"Deadline=0 must prevent summaries, got: {str(c)[:60]}"

    def test_compression_timeout_constant_exists(self) -> None:
        """_COMPRESSION_TOTAL_TIMEOUT_SECS is defined as a positive int."""
        from src.orchestration.compression import _COMPRESSION_TOTAL_TIMEOUT_SECS

        assert isinstance(_COMPRESSION_TOTAL_TIMEOUT_SECS, int)
        assert _COMPRESSION_TOTAL_TIMEOUT_SECS > 0
