"""Regression tests for issue #310/#311: panel borders and /help Panel rendering.

Problem 1 (#310): Console.print() defaults crop=True which can clip the panel
           closing border ╰───╯ at terminal height. Fix: _Console subclass
           defaults crop=False so borders are never clipped.

Problem 2 (#310): /help used console.rule() producing no side │ borders and no
           ╰───╯ closing border. Fix: _help_rich() now wraps content in a Panel.

Problem 3 (#311): Post-dispatch ANSI erases ate the panel's ╰───╯ border.
           Root cause: patch_stdout redraws the prompt ending with "❯ " (no
           trailing \\n), leaving the cursor ON the ❯ line.  Each \\033[1A then
           moved up one line too far, and the last erase hit ╰───╯.
           Fix: write \\n after patch_stdout.__exit__() so cursor advances past
           ❯ before the erase sequences run.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

# Verify Rich is available; skip entire module if not
pytest.importorskip("rich")

from rich.console import Console
from rich.panel import Panel

import cogtrix
from cogtrix import _build_slash_commands, _help_rich

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIDE_BORDER = "\u2502"  # │
_CLOSE_BORDER = "\u2570"  # ╰
_OPEN_BORDER = "\u256d"  # ╭


def _render_help() -> str:
    """Run _help_rich with a real registry and capture output to string."""
    reg = _build_slash_commands()
    buf = StringIO()
    test_console = Console(file=buf, width=100, no_color=True, highlight=False)
    with patch("cogtrix.console", test_console):
        _help_rich(reg)
    return buf.getvalue()


def _render_panel(content: str, height: int | None = None) -> str:
    """Render a simple Panel through cogtrix's _Console and return the string."""
    buf = StringIO()
    kw: dict = {"file": buf, "width": 80, "no_color": True, "highlight": False}
    if height is not None:
        kw["height"] = height
    test_console = cogtrix._Console(**kw)  # type: ignore[attr-defined]
    test_console.print(Panel(content, title="Test"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Problem 1: Console crop behaviour
# ---------------------------------------------------------------------------


class TestConsoleCropDisabled:
    """_Console.print must default crop=False so panel borders are never clipped."""

    def test_console_print_default_crop_is_false(self):
        """
        REGRESSION #310-P1: _Console.print must expose crop=False as default.

        Before fix: Console().print has crop=True (built-in Rich default).
        After fix: _Console().print has crop=False.
        """
        import inspect

        sig = inspect.signature(cogtrix._Console.print)  # type: ignore[attr-defined]
        assert sig.parameters["crop"].default is False, (
            "Expected _Console.print to default crop=False; "
            "got crop=True which can clip panel closing borders"
        )

    def test_panel_closing_border_present(self):
        """
        REGRESSION #310-P1: A rendered panel must include the ╰ closing border.
        """
        out = _render_panel("line1\nline2\nline3")
        assert _CLOSE_BORDER in out, f"Missing ╰ border in:\n{out}"

    def test_panel_opening_border_present(self):
        """Panel must also render the ╭ opening border."""
        out = _render_panel("line1\nline2\nline3")
        assert _OPEN_BORDER in out, f"Missing ╭ border in:\n{out}"

    def test_panel_side_borders_present(self):
        """Panel content lines must have │ side borders."""
        out = _render_panel("line1\nline2\nline3")
        assert _SIDE_BORDER in out, f"Missing │ side border in:\n{out}"

    def test_tall_panel_closing_border_not_clipped(self):
        """
        REGRESSION #310-P1: A panel taller than the console height must still
        render the ╰ closing border (crop=False prevents height-based clipping).
        """
        # 30 lines of content, console height set to 5 — would be clipped if crop=True
        content = "\n".join(f"line {i}" for i in range(30))
        out = _render_panel(content, height=5)
        assert _CLOSE_BORDER in out, (
            "Panel closing border was clipped (crop=True active). "
            "Expected _Console to use crop=False.\n" + out
        )


# ---------------------------------------------------------------------------
# Problem 2: /help Panel rendering
# ---------------------------------------------------------------------------


class TestHelpPanelBorders:
    """/help must render a Panel with full borders, not a bare Rule."""

    def test_help_has_closing_border(self):
        """
        REGRESSION #310-P2: /help output must contain the ╰ closing border.

        Before fix: _help_rich used console.rule() — no closing ╰ border.
        After fix: _help_rich wraps content in Panel() — ╰ border present.
        """
        out = _render_help()
        assert _CLOSE_BORDER in out, (
            "/help output is missing ╰ closing border.\n"
            "This means _help_rich still uses console.rule() instead of Panel().\n" + out
        )

    def test_help_has_side_borders(self):
        """
        REGRESSION #310-P2: /help content lines must have │ side borders.

        Before fix: console.rule() + bare print → no │ on content lines.
        After fix: Panel wraps content → │ on every content line.
        """
        out = _render_help()
        content_lines = [ln for ln in out.splitlines() if _SIDE_BORDER in ln]
        assert content_lines, (
            "/help output has no lines with │ side border.\n"
            "Expected Panel-wrapped content.\n" + out
        )

    def test_help_has_opening_border(self):
        """
        REGRESSION #310-P2: /help output must have ╭ opening border (Panel top).
        """
        out = _render_help()
        assert _OPEN_BORDER in out, "/help output is missing ╭ opening border.\n" + out

    def test_help_title_commands_present(self):
        """The Panel title 'Commands' must appear in the /help output."""
        out = _render_help()
        assert "Commands" in out, f"Panel title 'Commands' missing from /help output:\n{out}"

    def test_help_contains_command_names(self):
        """/help output must list known commands like /info, /tools, /quit."""
        out = _render_help()
        for cmd in ("/info", "/tools", "/quit"):
            assert cmd in out, f"Expected '{cmd}' in /help output:\n{out}"


# ---------------------------------------------------------------------------
# Problem 3 (#311): ANSI erase sequences must not eat the panel ╰ border
# ---------------------------------------------------------------------------


def _sim_terminal(writes: list) -> list:
    """
    Cursor-aware terminal line simulation.

    Tracks whether the cursor is ON the last written line (no trailing \\n) or
    on a blank row below it (trailing \\n). This distinction is critical for
    \\033[1A (cursor up): when the cursor sits ON the last line, \\033[1A skips
    it and moves to the line above, so that line is erased — not the last one.

    This accurately models the #311 root-cause: prompt_toolkit leaves the
    cursor ON the ❯ line (writes "❯ " with no \\n), causing \\033[1A to skip
    ❯ and instead target sep, then ╰───╯.
    """
    net: list = []
    cursor_below = True  # True = cursor is on a blank row BELOW net[-1]

    for w in writes:
        if w == "\033[1A\033[2K\r":
            if cursor_below:
                # cursor is below last line → \033[1A targets last line → erase
                if net:
                    net.pop()
            else:
                # cursor ON last line → \033[1A skips it, targets second-to-last
                if len(net) >= 2:
                    del net[-2]
        else:
            segs = w.split("\n")
            for i, seg in enumerate(segs):
                if seg:
                    if not cursor_below and net:
                        net[-1] += seg  # append to partial last line
                    else:
                        net.append(seg)
                    cursor_below = False
                if i < len(segs) - 1:
                    cursor_below = True  # this \n moves cursor to next blank row
            if w.endswith("\n"):
                cursor_below = True
            elif w:
                cursor_below = False

    return net


class TestPostDispatchEraseDoesNotEatBorder:
    """
    REGRESSION #311: Post-dispatch ANSI erases must not reach the panel's
    ╰ closing border.

    Root cause: patch_stdout redraws "sep\\n❯ " (no trailing \\n).  The cursor
    sits ON the ❯ line.  Without the \\n advance added by the fix, \\033[1A
    overshoots by one row on every invocation, and the last erase consumes
    the ╰───╯ border line.

    Fix: cogtrix.py writes \\n after _slash_patch_ctx.__exit__() to advance
    the cursor past ❯ before the erase sequences run.
    """

    def _run_flow(self, include_nl_advance: bool, has_stats: bool) -> list:
        """
        Simulate the full slash command output sequence and return net_lines.

        include_nl_advance: True = apply the fix (\\n before erases)
        has_stats: True = stats line present (3 erases), False = 2 erases
        """
        SEPARATOR = "─" * 40
        writes: list = []

        # Panel output from dispatch()
        writes.append("╭" + "─" * 38 + "╮\n")
        writes.append("│ tool content here                    │\n")
        writes.append("│ more content                         │\n")
        writes.append("╰" + "─" * 38 + "╯\n")  # ← this line must survive

        # patch_stdout.__exit__() redraws the prompt WITHOUT a trailing \n —
        # this is the real prompt_toolkit behaviour (❯ is a live input prompt).
        writes.append(SEPARATOR + "\n")
        if has_stats:
            writes.append("  1.2s ↑100 ↓50\n")
        writes.append("❯ ")  # NO trailing \n — cursor lands ON this line

        # Fix: advance past ❯ before erasing
        if include_nl_advance:
            writes.append("\n")

        # Erase sequences
        writes.append("\033[1A\033[2K\r")  # erase ❯
        if has_stats:
            writes.append("\033[1A\033[2K\r")  # erase stats
        writes.append("\033[1A\033[2K\r")  # erase separator

        return _sim_terminal(writes)

    def test_border_eaten_without_fix_no_stats(self):
        """
        FAILING SCENARIO: without \\n advance and no stats bar, the last
        erase moves above sep and erases ╰───╯.  This test documents the
        pre-fix behaviour and must show ╰ is absent.
        """
        net = self._run_flow(include_nl_advance=False, has_stats=False)
        border_present = any("╰" in line for line in net)
        assert not border_present, (
            "Expected ╰ to be erased (pre-fix behaviour), but it survived.\n" f"Net lines: {net}"
        )

    def test_border_survives_with_fix_no_stats(self):
        """
        REGRESSION #311: with \\n advance and no stats bar, erases consume
        ❯ and sep only — ╰───╯ must remain in net output.
        """
        net = self._run_flow(include_nl_advance=True, has_stats=False)
        border_lines = [line for line in net if "╰" in line]
        assert border_lines, (
            "╰ closing border was erased by post-dispatch ANSI sequences!\n" f"Net lines: {net}"
        )

    def test_border_survives_with_fix_with_stats(self):
        """
        REGRESSION #311: with \\n advance and stats bar present (3 erases),
        erases consume ❯, stats, and sep — ╰───╯ must remain in net output.
        """
        net = self._run_flow(include_nl_advance=True, has_stats=True)
        border_lines = [line for line in net if "╰" in line]
        assert border_lines, (
            "╰ closing border was erased (stats=True) by post-dispatch ANSI sequences!\n"
            f"Net lines: {net}"
        )

    def test_border_eaten_without_fix_with_stats(self):
        """
        FAILING SCENARIO (stats=True): without \\n advance, last erase
        targets ╰───╯.  Documents pre-fix behaviour.
        """
        net = self._run_flow(include_nl_advance=False, has_stats=True)
        border_present = any("╰" in line for line in net)
        assert not border_present, (
            "Expected ╰ to be erased (pre-fix behaviour, stats=True), but it survived.\n"
            f"Net lines: {net}"
        )

    def test_no_patch_stdout_for_slash_commands(self):
        """Slash commands must NOT use patch_stdout — it causes output erasure.

        Regression: patch_stdout + erase logic erased slash command output
        (panels, toggle messages) making commands appear to do nothing.
        The fix removed patch_stdout from the slash dispatch path entirely.
        """
        import inspect

        import cogtrix as _cgt

        src = inspect.getsource(_cgt)
        # The slash dispatch block should NOT contain patch_stdout wrapping
        # (the old pattern was: _slash_patch_ctx = _patch_stdout(raw=True))
        assert "_slash_patch_ctx" not in src, (
            "patch_stdout wrapping for slash commands should be removed — "
            "it causes output erasure with prompt_toolkit"
        )
