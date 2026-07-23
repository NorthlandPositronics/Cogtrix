"""Tests for conversation turn rendering."""

import sys
from io import StringIO

from rich.console import Console


def _make_console() -> Console:
    return Console(file=StringIO(), highlight=False, markup=False)


def test_print_user_turn_contains_you():
    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "Hello world")
    output = buf.getvalue()
    assert "you" in output
    assert "Hello world" in output


def test_print_user_turn_multiline():
    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "line one\nline two")
    output = buf.getvalue()
    assert "you" in output
    assert "line one" in output
    assert "line two" in output


def test_print_assistant_turn_header_contains_cogtrix():
    from cogtrix_core.ui.turns import print_assistant_turn_header

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_assistant_turn_header(console)
    output = buf.getvalue()
    assert "cogtrix" in output


def test_print_turn_divider_prints_something():
    from cogtrix_core.ui.turns import print_turn_divider

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_turn_divider(console)
    output = buf.getvalue()
    assert len(output.strip()) > 0


def test_print_user_turn_has_timestamp():
    import re

    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "test message")
    output = buf.getvalue()
    # HH:MM pattern
    assert re.search(r"\d{2}:\d{2}", output)


def test_theme_fallback_does_not_crash():
    """turn rendering works even if src.ui.theme is unavailable."""

    from cogtrix_core.ui import turns

    # Temporarily hide theme module
    orig = sys.modules.get("cogtrix_core.ui.theme")
    sys.modules["cogtrix_core.ui.theme"] = None  # type: ignore
    try:
        buf = StringIO()
        console = Console(file=buf, highlight=False, markup=False, width=80)
        turns.print_user_turn(console, "test")
    except ImportError:
        pass  # acceptable
    finally:
        if orig is not None:
            sys.modules["cogtrix_core.ui.theme"] = orig
        elif "cogtrix_core.ui.theme" in sys.modules:
            del sys.modules["cogtrix_core.ui.theme"]


def test_turns_importable_from_src_ui():
    from cogtrix_core.ui import print_assistant_turn_header, print_turn_divider, print_user_turn

    assert callable(print_user_turn)
    assert callable(print_assistant_turn_header)
    assert callable(print_turn_divider)


def test_print_user_turn_has_empty_top_rail():
    """An empty ╷ line appears before the you header."""
    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "hello")
    output = buf.getvalue()
    assert "╷" in output


def test_print_user_turn_header_uses_pipe():
    """Header uses │ you, not ╷ you."""
    import re

    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "hello")
    output = buf.getvalue()
    # The "you" label should follow │, not ╷
    assert re.search(r"│\s+you", output), f"Expected '│ you' in output: {repr(output)}"


def test_print_user_turn_formatting():
    """Verify print_user_turn includes expected formatting markers."""
    from io import StringIO

    from cogtrix_core.ui.turns import print_user_turn

    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=80)
    print_user_turn(console, "test message")
    output = buf.getvalue()

    # The output should contain the expected format with user label
    assert "│ " in output, "print_user_turn must include │ marker"
    assert "test message" in output, "print_user_turn must include the message"
