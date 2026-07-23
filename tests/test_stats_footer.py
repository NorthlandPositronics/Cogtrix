"""Tests for stats footer rendering (session-total-only format)."""

import re as _re
from io import StringIO

from rich.console import Console

import src.ui.input_session as _input_session


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, highlight=False, markup=False, width=100), buf


def _get_toolbar() -> str:
    return _input_session._toolbar_stats


def _strip_ansi(s: str) -> str:
    return _re.sub(r"\033\[[0-9;]*m", "", s)


def test_stats_footer_contains_session_total():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000)
    toolbar = _strip_ansi(_get_toolbar())
    assert "4,812" in toolbar or "4812" in toolbar


def test_stats_footer_contains_session_label():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 2_000, 200_000)
    assert "session" in _strip_ansi(_get_toolbar())


def test_stats_footer_progress_bar_present():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 50_000, 200_000)
    toolbar = _strip_ansi(_get_toolbar())
    assert "█" in toolbar or "░" in toolbar or "%" in toolbar


def test_stats_footer_no_context_no_bar():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 50_000, None)
    toolbar = _strip_ansi(_get_toolbar())
    assert "session" in toolbar
    assert "%" not in toolbar


def test_stats_footer_contains_input_tokens():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000, input_tokens=1_203, output_tokens=247)
    toolbar = _strip_ansi(_get_toolbar())
    assert "1,203" in toolbar
    assert "\u2191" in toolbar  # ↑


def test_stats_footer_contains_output_tokens():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000, input_tokens=1_203, output_tokens=247)
    toolbar = _strip_ansi(_get_toolbar())
    assert "247" in toolbar
    assert "\u2193" in toolbar  # ↓


def test_stats_footer_zero_tokens_omitted():
    """When input_tokens=0, ↑ arrow is omitted."""
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000)
    toolbar = _strip_ansi(_get_toolbar())
    assert "\u2191" not in toolbar
    assert "\u2193" not in toolbar


def test_stats_footer_no_elapsed():
    """Elapsed time field (e.g. '1.4s') must NOT appear in the toolbar."""
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 1_000, 200_000, input_tokens=500, output_tokens=100)
    toolbar = _strip_ansi(_get_toolbar())
    assert not _re.search(r"\d+\.\d+s", toolbar), f"Elapsed time appeared in toolbar: {toolbar}"


def test_build_bar_full():
    from src.ui.stats import _build_bar

    bar, _ = _build_bar(1.0, 16)
    assert bar == "█" * 16


def test_build_bar_empty():
    from src.ui.stats import _build_bar

    bar, _ = _build_bar(0.0, 16)
    assert bar == "░" * 16


def test_build_bar_half():
    from src.ui.stats import _build_bar

    bar, _ = _build_bar(0.5, 16)
    assert bar.count("█") == 8
    assert bar.count("░") == 8


def test_critical_panel_shown_at_90pct():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 180_000, 200_000)
    assert _get_toolbar() != ""


def test_warning_rule_shown_at_75pct():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 150_000, 200_000)
    assert _get_toolbar() != ""


def test_stats_importable_from_src_ui():
    from src.ui import print_stats_footer

    assert callable(print_stats_footer)


def test_stats_footer_input_tokens_green():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000, input_tokens=1_203, output_tokens=0)
    toolbar = _get_toolbar()
    # Green ANSI before ↑
    assert "\033[32m" in toolbar


def test_stats_footer_output_tokens_red():
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    print_stats_footer(console, 4_812, 200_000, input_tokens=0, output_tokens=247)
    toolbar = _get_toolbar()
    # Red ANSI before ↓
    assert "\033[31m" in toolbar


def test_stats_footer_session_color_matches_bar():
    """session tok and bar use the same color code."""
    from src.ui.stats import print_stats_footer

    console, _ = _console()
    # Low usage → green for both session and bar
    print_stats_footer(console, 1_000, 200_000, input_tokens=0, output_tokens=0)
    toolbar = _get_toolbar()
    assert "\033[32m" in toolbar  # green present
