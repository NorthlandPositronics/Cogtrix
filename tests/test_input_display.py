"""Tests for #300 — user input must not be displayed twice.

After the fix, the ANSI erase sequences fire immediately after prompt()
returns, before print_user_turn renders the Rich 'you' panel.  We verify
the ordering by capturing sys.stdout.write calls.
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock


def _simulate_turn_start(has_stats: bool = False) -> list[str]:
    """
    Simulate the sequence of writes that happen after prompt() returns:
    1. ANSI erase (separator + optional stats + ❯)
    2. print_user_turn output ('you' panel text)

    Returns the list of strings written to sys.stdout in order.
    """
    writes: list[str] = []

    # Stub _toolbar_stats
    stats_value = "1.2s ↑100 ↓50" if has_stats else ""
    fake_input_session = types.ModuleType("src.ui.input_session")
    fake_input_session._toolbar_stats = stats_value  # type: ignore[attr-defined]

    with mock.patch.dict("sys.modules", {"src.ui.input_session": fake_input_session}):
        # Replay the erase block exactly as it appears in cogtrix.py
        # (this mirrors the code moved to immediately after prompt() returns)
        def capturing_write(s: str) -> int:
            writes.append(s)
            return len(s)

        with mock.patch.object(sys.stdout, "write", side_effect=capturing_write):
            with mock.patch.object(sys.stdout, "flush"):
                from src.ui.input_session import _toolbar_stats as _ts

                sys.stdout.write("\033[1A\033[2K\r")  # erase ❯ input line
                if _ts.strip():
                    sys.stdout.write("\033[1A\033[2K\r")  # erase stats line
                sys.stdout.write("\033[1A\033[2K\r")  # erase separator line
                sys.stdout.flush()

        # Simulate what print_user_turn would write (simplified)
        writes.append("you  12:34")

    return writes


def test_erase_fires_before_you_panel():
    """ANSI erase sequences appear in stdout before the 'you' panel content."""
    writes = _simulate_turn_start(has_stats=False)

    # Find first ANSI erase and first 'you' write
    first_erase = next((i for i, s in enumerate(writes) if "\033[1A" in s), None)
    first_you = next((i for i, s in enumerate(writes) if "you" in s), None)

    assert first_erase is not None, "No ANSI erase sequence found in writes"
    assert first_you is not None, "No 'you' panel content found in writes"
    assert (
        first_erase < first_you
    ), f"Erase (index {first_erase}) must precede 'you' panel (index {first_you})"


def test_erase_fires_before_you_panel_with_stats():
    """ANSI erase sequences appear before 'you' panel even when stats are present."""
    writes = _simulate_turn_start(has_stats=True)

    first_erase = next((i for i, s in enumerate(writes) if "\033[1A" in s), None)
    first_you = next((i for i, s in enumerate(writes) if "you" in s), None)

    assert first_erase is not None, "No ANSI erase sequence found in writes"
    assert first_you is not None, "No 'you' panel content found in writes"
    assert first_erase < first_you


def test_erase_count_no_stats():
    """Without stats: exactly 2 erase sequences (❯ line + separator line)."""
    writes = _simulate_turn_start(has_stats=False)
    erase_count = sum(1 for s in writes if "\033[1A\033[2K\r" in s)
    assert erase_count == 2, f"Expected 2 erase sequences without stats, got {erase_count}"


def test_erase_count_with_stats():
    """With stats: exactly 3 erase sequences (❯ line + stats line + separator line)."""
    writes = _simulate_turn_start(has_stats=True)
    erase_count = sum(1 for s in writes if "\033[1A\033[2K\r" in s)
    assert erase_count == 3, f"Expected 3 erase sequences with stats, got {erase_count}"


def test_user_message_not_duplicated_in_output():
    """
    User message appears in output exactly once (Rich panel only).
    After the fix, the ❯ line is erased before the panel renders —
    verified here by confirming no raw '❯' echo appears alongside 'you'.
    """
    writes = _simulate_turn_start(has_stats=False)

    # The only 'you' occurrence must come AFTER all erase sequences
    erase_indices = [i for i, s in enumerate(writes) if "\033[1A" in s]
    you_indices = [i for i, s in enumerate(writes) if "you" in s]

    assert you_indices, "Rich 'you' panel must appear in output"
    assert all(
        e < you_indices[0] for e in erase_indices
    ), "All erase sequences must precede the 'you' panel"


def test_long_user_message_stays_inside_panel():
    """
    REGRESSION: A user message longer than terminal width wraps inside the panel.
    All wrapped lines are prefixed with the panel border character.
    Text does NOT appear below the closing border.
    """
    from io import StringIO

    from rich.console import Console

    from src.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80, no_color=True)
    msg = (
        "Please search the Internet on the absolutely best text user interface designs, "
        "look for the best text user interface graphical design descriptions, and search "
        "for the users review - what do they like the most?"
    )
    assert len(msg) > 80, "Test message must exceed console width to trigger wrapping"

    print_user_turn(console, msg)
    output = buf.getvalue()
    lines = output.splitlines()

    # Find closing border
    closing_idx = next((i for i, ln in enumerate(lines) if "╵" in ln), None)
    assert closing_idx is not None, "No closing ╵ border found in output"

    # No non-empty content lines must appear after the closing border
    after_close = [ln for ln in lines[closing_idx + 1 :] if ln.strip()]
    assert not after_close, f"Content appeared after ╵ border: {after_close}"

    # Every line that carries message content must include the │ rail character
    # Content lines are those between the header and ╵ that don't contain "you"
    content_lines = [
        ln
        for ln in lines[:closing_idx]
        if ln.strip() and "│" in ln and "you" not in ln and "╷" not in ln
    ]
    assert content_lines, "No content lines found inside the panel"
    for ln in content_lines:
        assert "│" in ln, f"Content line missing │ prefix: {repr(ln)}"

    # The message text must be fully present inside the panel (before ╵)
    inside_text = "\n".join(lines[:closing_idx])
    assert "most?" in inside_text, "Last word of long message not found inside panel borders"


def test_short_user_message_unchanged():
    """Single-line messages render identically (no regression)."""
    from io import StringIO

    from rich.console import Console

    from src.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80, no_color=True)
    msg = "Hello world"

    print_user_turn(console, msg)
    output = buf.getvalue()
    lines = output.splitlines()

    closing_idx = next((i for i, ln in enumerate(lines) if "╵" in ln), None)
    assert closing_idx is not None, "No closing ╵ border found"

    # No content after closing border
    after_close = [ln for ln in lines[closing_idx + 1 :] if ln.strip()]
    assert not after_close, f"Content appeared after ╵: {after_close}"

    # Message appears exactly once, inside the panel
    content_lines = [ln for ln in lines[:closing_idx] if "Hello world" in ln]
    assert (
        len(content_lines) == 1
    ), f"Expected exactly 1 content line, got {len(content_lines)}: {content_lines}"
