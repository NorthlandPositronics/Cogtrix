"""Stats footer rendering — session token total and context progress bar."""

from __future__ import annotations

from rich.console import Console

_BAR_WIDTH = 16
_WARNING_THRESHOLD = 0.70  # yellow bar
_CRITICAL_THRESHOLD = 0.85  # red bar + panel


def _get_stats_colors() -> tuple[str, str, str]:
    """Return (normal, warning, critical) colors from active theme or fallback."""
    try:
        from src.ui.theme import get_theme

        t = get_theme()
        return t.stats, t.stats_warning, t.stats_critical
    except Exception:
        return "grey50", "yellow", "red"


def _build_bar(ratio: float, width: int = _BAR_WIDTH) -> tuple[str, str]:
    """Return (bar_str, color) for a given ratio 0–1."""
    normal_color, warning_color, critical_color = _get_stats_colors()
    filled = max(0, min(width, round(ratio * width)))
    bar = "█" * filled + "░" * (width - filled)
    if ratio >= _CRITICAL_THRESHOLD:
        color = critical_color
    elif ratio >= _WARNING_THRESHOLD:
        color = warning_color
    else:
        color = normal_color
    return bar, color


def print_stats_footer(
    console: Console,  # kept for API compat
    session_tokens: int,
    max_context_tokens: int | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Update the pinned toolbar with turn stats (no direct console output)."""
    try:
        from src.ui.input_session import update_toolbar_stats

        _RST = "\033[0m"
        _GREEN = "\033[32m"
        _RED = "\033[31m"

        # Determine bar color from context ratio
        if max_context_tokens and max_context_tokens > 0:
            ratio = min(1.0, session_tokens / max_context_tokens)
            if ratio >= _CRITICAL_THRESHOLD:
                bar_color = "\033[31m"
            elif ratio >= _WARNING_THRESHOLD:
                bar_color = "\033[33m"
            else:
                bar_color = "\033[32m"
        else:
            ratio = 0.0
            bar_color = "\033[32m"

        parts = []
        if input_tokens:
            parts.append(f"{_GREEN}\u2191 {input_tokens:,} out{_RST}")
        if output_tokens:
            parts.append(f"{_RED}\u2193 {output_tokens:,} in{_RST}")
        parts.append(f"{bar_color}session {session_tokens:,} tok{_RST}")
        if max_context_tokens and max_context_tokens > 0:
            bar_str, _ = _build_bar(ratio)
            pct = int(ratio * 100)
            parts.append(f"{bar_color}{bar_str}  {pct}%{_RST}")
        update_toolbar_stats("   ".join(parts))
    except Exception:
        pass
