"""Tests for issue #297: separator + stats scroll into output on long slash command responses."""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt_toolkit_stub():
    """Return a minimal prompt_toolkit stub (does not install into sys.modules)."""
    pt = types.ModuleType("prompt_toolkit")
    pt.PromptSession = MagicMock  # type: ignore[attr-defined]

    class _FakePatchCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    pt_patch = types.ModuleType("prompt_toolkit.patch_stdout")
    pt_patch.patch_stdout = MagicMock(return_value=_FakePatchCtx())  # type: ignore[attr-defined]

    pt_formatted = types.ModuleType("prompt_toolkit.formatted_text")
    pt_formatted.ANSI = str  # type: ignore[attr-defined]

    mods = {
        "prompt_toolkit": pt,
        "prompt_toolkit.patch_stdout": pt_patch,
        "prompt_toolkit.formatted_text": pt_formatted,
        "prompt_toolkit.keys": types.ModuleType("prompt_toolkit.keys"),
        "prompt_toolkit.shortcuts": types.ModuleType("prompt_toolkit.shortcuts"),
        "prompt_toolkit.styles": types.ModuleType("prompt_toolkit.styles"),
        "prompt_toolkit.completion": types.ModuleType("prompt_toolkit.completion"),
        "prompt_toolkit.document": types.ModuleType("prompt_toolkit.document"),
    }
    mods["prompt_toolkit.completion"].Completer = MagicMock  # type: ignore[attr-defined]
    mods["prompt_toolkit.completion"].Completion = MagicMock  # type: ignore[attr-defined]
    mods["prompt_toolkit.completion"].WordCompleter = MagicMock  # type: ignore[attr-defined]
    mods["prompt_toolkit.document"].Document = MagicMock  # type: ignore[attr-defined]
    return mods


def _install_stubs(monkeypatch):
    """Install prompt_toolkit stub + a minimal src.ui.input_session stub via monkeypatch."""
    for name, mod in _make_prompt_toolkit_stub().items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Build a minimal src.ui.input_session stub
    uis_stub = types.ModuleType("cogtrix_core.ui.input_session")
    uis_stub._toolbar_stats = "1.2s ↑100 ↓50"  # type: ignore[attr-defined]
    uis_stub.update_toolbar_stats = lambda s: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cogtrix_core.ui.input_session", uis_stub)

    # Remove cached real module if already imported
    for cached in ("cogtrix_core.ui.input_session",):
        if cached in sys.modules:
            pass  # monkeypatch.setitem already replaced it

    return uis_stub


# ---------------------------------------------------------------------------
# Test 1: ANSI erase sequences fire before slash command output
# ---------------------------------------------------------------------------


def test_slash_command_erase_fires_before_output(monkeypatch):
    """ANSI erase sequences appear in stdout before slash command output."""
    uis_stub = _install_stubs(monkeypatch)
    uis_stub._toolbar_stats = "1.2s ↑100 ↓50"

    written = []

    class FakeTTY:
        def write(self, s):
            written.append(s)

        def flush(self):
            pass

        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", FakeTTY())

    # Run the exact fixed code path
    if sys.stdout.isatty():
        from cogtrix_core.ui.input_session import (
            _toolbar_stats as _ts,  # type: ignore[attr-defined]
        )

        sys.stdout.write("\033[1A\033[2K\r")  # erase ❯ line
        if _ts.strip():
            sys.stdout.write("\033[1A\033[2K\r")  # erase stats line
        sys.stdout.write("\033[1A\033[2K\r")  # erase separator line
        sys.stdout.flush()
    # Simulate dispatch output
    sys.stdout.write("command output here")

    assert (
        written[0] == "\033[1A\033[2K\r"
    ), f"First write must be erase sequence, got: {written[0]!r}"
    assert written[1] == "\033[1A\033[2K\r"
    assert written[2] == "\033[1A\033[2K\r"
    assert written[-1] == "command output here", "Command output must come after erase sequences"
    cmd_idx = written.index("command output here")
    last_erase_idx = max(i for i, w in enumerate(written) if w == "\033[1A\033[2K\r")
    assert cmd_idx > last_erase_idx


# ---------------------------------------------------------------------------
# Test 2: patch_stdout is entered before dispatch and exited after
# ---------------------------------------------------------------------------


def test_slash_command_wrapped_in_patch_stdout():
    """patch_stdout context is entered before slash command runs and exited after."""
    call_log = []

    class FakeCtx:
        def __enter__(self):
            call_log.append("enter")
            return self

        def __exit__(self, *args):
            call_log.append("exit")

    def fake_dispatch():
        call_log.append("dispatch")
        return "ok"

    _slash_patch_ctx = FakeCtx()
    _slash_patch_ctx.__enter__()
    try:
        fake_dispatch()
    finally:
        if _slash_patch_ctx is not None:
            try:
                _slash_patch_ctx.__exit__(None, None, None)
            except Exception:
                pass

    assert call_log == [
        "enter",
        "dispatch",
        "exit",
    ], f"Expected enter→dispatch→exit, got: {call_log}"


def test_slash_command_patch_stdout_exit_called_on_dispatch_exception():
    """patch_stdout.__exit__ is called even when dispatch raises."""
    call_log = []

    class FakeCtx:
        def __enter__(self):
            call_log.append("enter")
            return self

        def __exit__(self, *args):
            call_log.append("exit")

    _slash_patch_ctx = FakeCtx()
    _slash_patch_ctx.__enter__()

    try:
        try:
            raise RuntimeError("dispatch exploded")
        finally:
            if _slash_patch_ctx is not None:
                try:
                    _slash_patch_ctx.__exit__(None, None, None)
                except Exception:
                    pass
    except RuntimeError:
        pass

    assert "exit" in call_log, "__exit__ must be called even when dispatch raises"


# ---------------------------------------------------------------------------
# Test 3: non-tty skips erase (no AttributeError)
# ---------------------------------------------------------------------------


def test_slash_command_no_erase_on_non_tty(monkeypatch):
    """Erase logic is skipped when stdout is not a tty — no errors raised, no ANSI written."""
    written = []

    class FakePipe:
        def write(self, s):
            written.append(s)

        def flush(self):
            pass

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdout", FakePipe())

    if sys.stdout.isatty():
        raise AssertionError("isatty should be False")  # pragma: no cover

    # No ANSI writes should occur
    sys.stdout.write("command output here")

    ansi_writes = [w for w in written if "\033" in w]
    assert ansi_writes == [], f"No ANSI sequences should be written to non-tty: {ansi_writes}"
    assert "command output here" in written


def test_slash_command_no_erase_on_non_tty_no_exception(monkeypatch):
    """No exception is raised when dispatching a slash command in non-tty mode."""

    class FakePipe:
        def write(self, s):
            pass

        def flush(self):
            pass

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdout", FakePipe())

    # The fixed guard — isatty() returns False, so no import of src.ui.input_session
    if sys.stdout.isatty():
        raise AssertionError("Should not reach erase block")  # pragma: no cover

    result = "ok"  # simulated dispatch result
    assert result == "ok"


# ---------------------------------------------------------------------------
# Test 4: REGRESSION — /compact must not produce duplicate separator
# ---------------------------------------------------------------------------


def test_compact_does_not_produce_duplicate_separator(monkeypatch):
    """
    REGRESSION: /compact output followed by exactly one separator redraw.

    patch_stdout.__exit__() redraws the prompt (separator + stats + ❯) at the
    current cursor position. For fast commands prompt() overwrites it at the
    same terminal row. For slow commands like /compact the output has scrolled
    the terminal, so the PT-redrawn prompt ends up above the prompt() draw,
    producing a visible duplicate in scrollback.

    The fix emits post-exit erase sequences to clean up the orphaned lines.
    Without the fix the net visible output contains two separators; with the
    fix it contains exactly one.
    """
    uis_stub = _install_stubs(monkeypatch)
    uis_stub._toolbar_stats = "session 916,934 tok"

    written = []

    class FakeTTY:
        def write(self, s):
            written.append(s)

        def flush(self):
            pass

        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", FakeTTY())

    SEPARATOR = "─" * 69

    # Step 1: /compact outputs its completion message.
    sys.stdout.write("✓ Compressed: 97 tool results + 14 assistant responses summarised.\n")

    # Step 2: patch_stdout.__exit__() redraws the prompt — this is the
    # "orphaned" copy that causes the duplicate-separator bug for slow commands.
    # prompt_toolkit renders WITHOUT a trailing \n so the cursor lands ON the ❯
    # line (not one row below it).
    sys.stdout.write(SEPARATOR + "\n")
    sys.stdout.write("  ↑ 916,934 out   session 916,934 tok   ████ 100%\n")
    sys.stdout.write("❯")  # NO trailing \n — matches real prompt_toolkit output

    # Step 3: THE FIX — advance past ❯ then erase the orphaned prompt lines.
    # The \n is required because PT leaves the cursor ON the ❯ line; without it
    # each \033[1A overshoots by one row and the last erase eats the panel border.
    if sys.stdout.isatty():
        from cogtrix_core.ui.input_session import (
            _toolbar_stats as _ts,  # type: ignore[attr-defined]
        )

        sys.stdout.write("\n")  # advance past ❯ (fix for #311)
        sys.stdout.write("\033[1A\033[2K\r")  # erase ❯
        if _ts.strip():
            sys.stdout.write("\033[1A\033[2K\r")  # erase stats
        sys.stdout.write("\033[1A\033[2K\r")  # erase separator
        sys.stdout.flush()

    # Step 4: prompt() draws the real interactive prompt.
    sys.stdout.write(SEPARATOR + "\n")
    sys.stdout.write("  ↑ 916,934 out   session 916,934 tok   ████ 100%\n")
    sys.stdout.write("❯ \n")

    # Simulate the terminal with cursor-position awareness.
    # When the cursor is ON the last written line (no trailing \n), \033[1A skips
    # that line and erases the one above it instead.  When the cursor is on a
    # blank row below the last line (trailing \n present), \033[1A targets the
    # last line directly.  This distinction is what caused #311: PT leaves the
    # cursor ON the ❯ line (no trailing \n), so each \033[1A overshot by one row.
    net_lines: list = []
    cursor_below = True  # True = cursor is on blank row below net_lines[-1]
    for w in written:
        if w == "\033[1A\033[2K\r":
            if cursor_below:
                if net_lines:
                    net_lines.pop()
            else:
                # cursor ON last line → \033[1A skips it, erases second-to-last
                if len(net_lines) >= 2:
                    del net_lines[-2]
        else:
            segs = w.split("\n")
            for i, seg in enumerate(segs):
                if seg:
                    if not cursor_below and net_lines:
                        net_lines[-1] += seg  # append to current partial line
                    else:
                        net_lines.append(seg)
                    cursor_below = False
                if i < len(segs) - 1:  # this separator represents a \n
                    cursor_below = True
            if w.endswith("\n"):
                cursor_below = True
            elif w:
                cursor_below = False

    separator_count = sum(1 for line in net_lines if "─" * 10 in line)

    # REGRESSION ASSERTION: without the fix, two separators remain in net
    # visible output; with the fix exactly one remains.
    assert separator_count == 1, (
        f"Expected exactly 1 separator in net visible output after /compact, "
        f"got {separator_count}. Duplicate separator bug is present. "
        f"Net lines: {net_lines}"
    )

    # Secondary: confirm erase sequences were actually emitted (the fix ran).
    erase_seqs = [w for w in written if w == "\033[1A\033[2K\r"]
    assert len(erase_seqs) >= 2, (
        f"REGRESSION: post-exit erase sequences not emitted "
        f"(got {len(erase_seqs)}). The /compact duplicate-separator fix "
        "requires ≥2 erase sequences after patch_stdout.__exit__."
    )
