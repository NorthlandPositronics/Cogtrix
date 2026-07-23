"""Tests for /compact slash command (Issue #292)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.cli.commands import _build_slash_commands

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry():
    reg = _build_slash_commands()
    return reg


def _dispatch(reg, cmd: str) -> str:
    result = reg.dispatch(cmd)
    return result or "continue"


def _captured(capsys) -> str:
    out, _ = capsys.readouterr()
    return out


def _make_memory_manager(messages: list) -> MagicMock:
    mm = MagicMock()
    mm._messages = messages
    mm._llm = MagicMock()
    mm.get_stats.return_value = {"total_messages": len(messages)}
    mm.get_message_count.return_value = len(messages)
    return mm


# ---------------------------------------------------------------------------
# Test 1: /compact with no compressible messages returns "Nothing to compress"
# ---------------------------------------------------------------------------


class TestCompactNothingToCompress:
    def test_returns_continue(self, capsys):
        """Command returns 'continue' when nothing is compressed."""
        reg = _make_registry()

        short_msgs = [MagicMock(content="hi"), MagicMock(content="hello")]
        reg.memory_manager = _make_memory_manager(short_msgs)
        reg.max_context_tokens = 16_384

        with patch("src.cli.commands.apply_message_compression", return_value=short_msgs):
            result = _dispatch(reg, "/compact")

        assert result == "continue"

    def test_prints_nothing_to_compress(self, capsys):
        """Output contains 'Nothing to compress' when function returns unchanged list."""
        reg = _make_registry()

        short_msgs = [MagicMock(content="hi"), MagicMock(content="hello")]
        reg.memory_manager = _make_memory_manager(short_msgs)
        reg.max_context_tokens = 16_384

        with patch("src.cli.commands.apply_message_compression", return_value=short_msgs):
            _dispatch(reg, "/compact")

        out = _captured(capsys)
        assert "Nothing to compress" in out

    def test_messages_unchanged(self, capsys):
        """Message list is not modified when nothing is compressed."""
        reg = _make_registry()

        short_msgs = [MagicMock(content="hi"), MagicMock(content="hello")]
        mm = _make_memory_manager(short_msgs)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        with patch("src.cli.commands.apply_message_compression", return_value=short_msgs):
            _dispatch(reg, "/compact")

        # save() should not be called since nothing changed
        mm.save.assert_not_called()

    def test_no_memory_manager(self, capsys):
        """Graceful output when memory manager is not available."""
        reg = _make_registry()
        reg.memory_manager = None

        result = _dispatch(reg, "/compact")

        assert result == "continue"
        out = _captured(capsys)
        assert "not available" in out.lower() or "Memory" in out

    def test_empty_messages(self, capsys):
        """Graceful output when _messages is empty."""
        reg = _make_registry()
        mm = _make_memory_manager([])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        result = _dispatch(reg, "/compact")

        assert result == "continue"
        out = _captured(capsys)
        assert "Nothing to compress" in out or "no messages" in out.lower()


# ---------------------------------------------------------------------------
# Test 2: /compact standard — compresses old large messages
# ---------------------------------------------------------------------------


class TestCompactStandard:
    def test_reports_summarised(self, capsys):
        """Output contains 'summarised' after compression."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact")

        out = _captured(capsys)
        assert "summarised" in out

    def test_updates_mm_messages(self, capsys):
        """Memory manager _messages is updated with compressed version."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact")

        assert mm._messages == [compressed_msg]

    def test_saves_after_compression(self, capsys):
        """Memory manager save() is called after successful compression."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact")

        mm.save.assert_called_once()

    def test_standard_params_passed(self):
        """Standard /compact passes COMPRESSION_MIN_AGE_CYCLES and COMPRESSION_MIN_CHARS."""
        from src.orchestration.compression import COMPRESSION_MIN_AGE_CYCLES, COMPRESSION_MIN_CHARS

        reg = _make_registry()
        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch(
            "src.cli.commands.apply_message_compression", return_value=[compressed_msg]
        ) as mock_compress:
            _dispatch_no_capsys(reg, "/compact")

        call_kwargs = mock_compress.call_args[1]
        assert call_kwargs.get("min_age_cycles") == COMPRESSION_MIN_AGE_CYCLES
        assert call_kwargs.get("min_chars") == COMPRESSION_MIN_CHARS

    def test_returns_continue(self, capsys):
        """Command always returns 'continue'."""
        reg = _make_registry()
        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            result = _dispatch(reg, "/compact")

        assert result == "continue"


def _dispatch_no_capsys(reg, cmd: str) -> str:
    result = reg.dispatch(cmd)
    return result or "continue"


# ---------------------------------------------------------------------------
# Test 3: /compact aggressive — compresses everything
# ---------------------------------------------------------------------------


class TestCompactAggressive:
    def test_aggressive_reports_summarised(self, capsys):
        """Output contains 'summarised' and mentions aggressive."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "short"  # would not qualify for standard
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact aggressive")

        out = _captured(capsys)
        assert "summarised" in out

    def test_aggressive_params_passed(self):
        """Aggressive /compact passes min_age_cycles=0, min_chars=0, emergency_threshold=0.0."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "short"
        messages = [orig_msg]
        mm = _make_memory_manager(messages)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch(
            "src.cli.commands.apply_message_compression", return_value=[compressed_msg]
        ) as mock_compress:
            _dispatch_no_capsys(reg, "/compact aggressive")

        call_kwargs = mock_compress.call_args[1]
        assert call_kwargs.get("min_age_cycles") == 0
        assert call_kwargs.get("min_chars") == 0
        assert call_kwargs.get("emergency_threshold") == 0.0

    def test_aggressive_updates_messages(self):
        """Memory manager _messages is updated after aggressive compression."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "short"
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch_no_capsys(reg, "/compact aggressive")

        assert mm._messages == [compressed_msg]

    def test_aggressive_prefix_in_output(self, capsys):
        """Output contains 'Aggressive' prefix for aggressive mode."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "short"
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact aggressive")

        out = _captured(capsys)
        assert "Aggressive" in out or "aggressive" in out


# ---------------------------------------------------------------------------
# Test 4: argument parsing is case-insensitive
# ---------------------------------------------------------------------------


class TestCompactArgParsing:
    @pytest.mark.parametrize(
        "cmd", ["/compact AGGRESSIVE", "/compact Aggressive", "/compact aggressive"]
    )
    def test_aggressive_case_insensitive(self, cmd):
        """'aggressive' argument is matched case-insensitively."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "short"
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch(
            "src.cli.commands.apply_message_compression", return_value=[compressed_msg]
        ) as mock_compress:
            _dispatch_no_capsys(reg, cmd)

        call_kwargs = mock_compress.call_args[1]
        assert call_kwargs.get("min_age_cycles") == 0
        assert call_kwargs.get("min_chars") == 0

    def test_unknown_arg_treated_as_standard(self):
        """Unrecognised argument falls through to standard compression params."""
        from src.orchestration.compression import COMPRESSION_MIN_AGE_CYCLES, COMPRESSION_MIN_CHARS

        reg = _make_registry()
        orig_msg = MagicMock()
        orig_msg.content = "x" * 3000
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        compressed_msg = MagicMock()
        compressed_msg.content = "summary"

        with patch(
            "src.cli.commands.apply_message_compression", return_value=[compressed_msg]
        ) as mock_compress:
            _dispatch_no_capsys(reg, "/compact whatever")

        call_kwargs = mock_compress.call_args[1]
        assert call_kwargs.get("min_age_cycles") == COMPRESSION_MIN_AGE_CYCLES
        assert call_kwargs.get("min_chars") == COMPRESSION_MIN_CHARS


# ---------------------------------------------------------------------------
# Test 5: /help compact shows correct usage and alias C
# ---------------------------------------------------------------------------


class TestCompactHelp:
    def test_help_compact_contains_usage(self, capsys):
        """'/help compact' output contains usage line."""
        reg = _make_registry()
        _dispatch(reg, "/help compact")
        out = _captured(capsys)
        assert "compact" in out.lower()
        assert "aggressive" in out.lower()

    def test_help_compact_mentions_alias(self, capsys):
        """'/help compact' output mentions alias C."""
        reg = _make_registry()
        _dispatch(reg, "/help compact")
        out = _captured(capsys)
        assert "compact" in out.lower()

    def test_short_help_describes_compression(self):
        """Command registry contains compact with appropriate short_help."""
        reg = _make_registry()
        cmd = reg._commands.get("compact")
        assert cmd is not None
        assert "compress" in cmd.short_help.lower() or "summarise" in cmd.short_help.lower()

    def test_compact_appears_in_help_output(self, capsys):
        """/compact appears in /help output with description."""
        import io
        import sys

        reg = _make_registry()
        # Capture output from _help_plain (no Rich console in tests)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            from cogtrix import _help_plain

            _help_plain(reg)
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "/compact" in out
        assert "compress" in out.lower() or "summarise" in out.lower()

    def test_compact_no_short_aliases(self):
        """Short aliases removed — tab completion makes them redundant."""
        reg = _make_registry()
        cmd = reg._commands.get("compact")
        assert cmd is not None
        assert not any(len(a) <= 2 for a in cmd.aliases)


# ---------------------------------------------------------------------------
# Test 6: progress_callback — Issue #303 regression tests
# ---------------------------------------------------------------------------


try:
    from langchain_core.messages import AIMessage, ToolMessage  # noqa: F401

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

_skip_no_langchain = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")


def _make_eligible_messages():
    """3 messages eligible for compression: 2 ToolMessages + 1 AIMessage.

    Total chars ~37,600 — exceeds the 35,389-char threshold for max_context_tokens=16_384.
    """
    if not _HAS_LANGCHAIN:
        return []
    # 5 old AIMessages to give ToolMessages enough age (min_age_cycles=1)
    ai_history = [AIMessage(content="step") for _ in range(5)]
    # 2 large ToolMessages (18 K each = 36 K chars total)
    tool1 = ToolMessage(content="x" * 18_000, tool_call_id="t1", name="shell")
    tool2 = ToolMessage(content="y" * 18_000, tool_call_id="t2", name="read_file")
    # 1 old AIMessage eligible for compression (>= 500 chars, enough age)
    ai_old = AIMessage(content="a" * 600)
    # Two recent AIMessages (protected — always last 2, not compressed)
    ai_recent1 = AIMessage(content="recent1")
    ai_recent2 = AIMessage(content="recent2")
    return [*ai_history, tool1, tool2, ai_old, ai_recent1, ai_recent2]


class TestCompactProgressCallback:
    @_skip_no_langchain
    def test_progress_callback_fires_per_item(self, monkeypatch):
        """
        REGRESSION: progress_callback(completed, total) fires once per compressed item.
        Without fix: parameter did not exist, callback never called.
        """
        from src.orchestration.compression import apply_message_compression

        calls = []
        messages = _make_eligible_messages()

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="short summary")

        _TRIGGER_CONTEXT = 16_384
        cache: dict = {}

        apply_message_compression(
            messages,
            call_count=10,
            compression_cache=cache,
            llm=fake_llm,
            max_context_tokens=_TRIGGER_CONTEXT,
            min_age_cycles=1,
            min_chars=1,
            progress_callback=lambda c, t: calls.append((c, t)),
        )

        assert len(calls) >= 1, "progress_callback should be called at least once"
        last_completed, last_total = calls[-1]
        assert last_completed == last_total, "final call should reach 100%"

    @_skip_no_langchain
    def test_progress_callback_optional(self):
        """apply_message_compression works without progress_callback (default None)."""
        from src.orchestration.compression import apply_message_compression

        messages = _make_eligible_messages()
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="short summary")

        result = apply_message_compression(
            messages,
            call_count=10,
            compression_cache={},
            llm=fake_llm,
            max_context_tokens=16_384,
            min_age_cycles=1,
            min_chars=1,
        )
        assert result is not None

    def test_compact_shows_reduction_pct(self, capsys):
        """Completion output shows char reduction percentage."""
        reg = _make_registry()
        orig_msg = MagicMock()
        orig_msg.content = "x" * 5000
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384
        reg.last_input_tokens = 0

        compressed_msg = MagicMock()
        compressed_msg.content = "short"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact")

        out = _captured(capsys)
        assert "%" in out, "Output should contain a percentage"
        assert "→" in out, "Output should contain the → char"
        assert "chars" in out, "Output should mention chars"

    def test_compact_nothing_shows_context_pct(self, capsys):
        """'Nothing to compress' includes current context pressure %."""
        reg = _make_registry()
        short_msgs = [MagicMock(content="hi"), MagicMock(content="hello")]
        mm = _make_memory_manager(short_msgs)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384
        reg.last_input_tokens = 8192  # 50% context pressure

        with patch("src.cli.commands.apply_message_compression", return_value=short_msgs):
            _dispatch(reg, "/compact")

        out = _captured(capsys)
        assert "Nothing to compress" in out
        assert "Context at" in out or "%" in out


# ---------------------------------------------------------------------------
# Test 7: /compact updates stats after compression — Issue #306 regression
# ---------------------------------------------------------------------------


class TestCompactUpdatesStats:
    def test_compact_updates_stats_after_compression(self, monkeypatch):
        """
        REGRESSION: After /compact, the token accumulator reflects post-compression estimate.
        Without fix: last_input_tokens unchanged after /compact (stale 100%).
        With fix: last_input_tokens updated to estimated reduced token count.
        """
        from src.orchestration.compression import _CHARS_PER_TOKEN

        reg = _make_registry()

        # Pre-compression: large context simulating 100% usage
        before_content = "x" * 900_000  # 900k chars → 300k tokens at _CHARS_PER_TOKEN=3
        orig_msg = MagicMock()
        orig_msg.content = before_content
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384

        # Set pre-compression token count to simulate 100% usage
        pre_compression_tokens = len(before_content) // _CHARS_PER_TOKEN
        reg.last_input_tokens = pre_compression_tokens

        # After compression: significantly shorter content
        after_content = "short summary"
        compressed_msg = MagicMock()
        compressed_msg.content = after_content

        monkeypatch.setattr(
            "src.cli.commands.apply_message_compression", lambda *a, **kw: [compressed_msg]
        )

        _dispatch_no_capsys(reg, "/compact")

        # After /compact, last_input_tokens must be lower than before
        assert reg.last_input_tokens < pre_compression_tokens, (
            f"Expected last_input_tokens < {pre_compression_tokens}, "
            f"got {reg.last_input_tokens}"
        )
        # Should match the char-based estimate
        expected = len(after_content) // _CHARS_PER_TOKEN
        assert reg.last_input_tokens == expected

    def test_compact_stats_message_includes_reduction(self, capsys):
        """Completion message includes char reduction summary after compression."""
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "x" * 5000
        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384
        reg.last_input_tokens = 0

        compressed_msg = MagicMock()
        compressed_msg.content = "short"

        with patch("src.cli.commands.apply_message_compression", return_value=[compressed_msg]):
            _dispatch(reg, "/compact")

        out = _captured(capsys)
        assert "Context reduced by" in out, f"Expected 'Context reduced by' in output, got: {out!r}"


# ---------------------------------------------------------------------------
# Test 8: /compact always produces output — Issue #309 regression
# ---------------------------------------------------------------------------


class TestCompactAlwaysProducesOutput:
    def test_compact_always_prints_output(self, monkeypatch, capsys):
        """
        REGRESSION #309: /compact always produces output — never returns silently.

        Root cause: ``except ImportError:`` in the Progress block was too narrow.
        Any non-ImportError exception raised inside ``with Progress(...)`` (e.g.
        a RuntimeError from a callback or terminal incompatibility) escaped the
        except clause, left ``compressed`` undefined, and propagated up without
        printing anything.

        Fix: broaden to ``except Exception:`` so any Progress failure falls back
        to running compression without the progress bar.

        This test FAILS on the broken code (RuntimeError propagates, no output)
        and PASSES after the fix (fallback path produces output).
        """
        reg = _make_registry()

        orig_msg = MagicMock()
        orig_msg.content = "x" * 5000
        compressed_msg = MagicMock()
        compressed_msg.content = "short"

        mm = _make_memory_manager([orig_msg])
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384
        reg.last_input_tokens = 0

        # Simulate: progress_callback triggers a non-ImportError inside Progress block.
        # With progress_callback → raise RuntimeError (Progress interaction failure).
        # Without progress_callback (fallback) → succeed.
        def _amc_side_effect(*args, **kwargs):
            if kwargs.get("progress_callback") is not None:
                raise RuntimeError("Progress update failed — non-TTY terminal")
            return [compressed_msg]

        monkeypatch.setattr("src.cli.commands.apply_message_compression", _amc_side_effect)

        exc_caught = None
        try:
            _dispatch(reg, "/compact")
        except Exception as e:
            exc_caught = e  # broken code raises here; fixed code does not

        out = _captured(capsys)
        assert len(out.strip()) > 0, (
            f"Expected /compact to produce output but got nothing. " f"Exception: {exc_caught!r}"
        )

    def test_compact_prints_nothing_to_compress_when_ineligible(self, monkeypatch, capsys):
        """
        REGRESSION #309: 'Nothing to compress' is printed even when Progress fails.

        When all messages are ineligible, apply_message_compression returns them
        unchanged.  With the broken ``except ImportError:``, a RuntimeError from
        the Progress block still silences all output.  After the fix the fallback
        path runs, detects no changes, and prints the "Nothing to compress" message.
        """
        reg = _make_registry()

        short_msgs = [MagicMock(content="hi"), MagicMock(content="hello")]
        mm = _make_memory_manager(short_msgs)
        reg.memory_manager = mm
        reg.max_context_tokens = 16_384
        reg.last_input_tokens = 0

        def _amc_side_effect(*args, **kwargs):
            if kwargs.get("progress_callback") is not None:
                raise RuntimeError("Progress update failed — non-TTY terminal")
            return short_msgs  # unchanged → changed == 0

        monkeypatch.setattr("src.cli.commands.apply_message_compression", _amc_side_effect)

        exc_caught = None
        try:
            _dispatch(reg, "/compact")
        except Exception as e:
            exc_caught = e

        out = _captured(capsys)
        assert "Nothing to compress" in out, (
            f"Expected 'Nothing to compress' in output but got: {out!r}. "
            f"Exception: {exc_caught!r}"
        )
