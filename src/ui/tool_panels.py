"""Consistent Rich Panel rendering for tool results."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

_COLLAPSE_THRESHOLD = 2000  # chars — collapse output longer than this


def _get_theme_colors() -> tuple[str, str]:
    """Return (tool_name_color, border_color) from active theme or fallback."""
    try:
        from src.ui.theme import get_theme

        t = get_theme()
        return t.tool_name, t.tool_result_border
    except Exception:
        return "yellow", "grey50"


def render_tool_panel(
    console: Console,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    elapsed: float,
    *,
    collapse_threshold: int = _COLLAPSE_THRESHOLD,
) -> None:
    """Print a Rich Panel summarising a tool call and its result.

    Args:
        console: Rich console to print to.
        tool_name: Name of the tool (e.g. "search_web").
        args: Tool input arguments dict.
        result: Tool output string.
        elapsed: Wall-clock seconds the tool took.
        collapse_threshold: Collapse output longer than this many chars.
    """
    name_color, border_color = _get_theme_colors()

    # Title: tool name + elapsed
    title = Text()
    title.append(tool_name, style=f"bold {name_color}")
    title.append(f"  {elapsed:.1f}s", style="dim")

    # Build body
    lines: list[str] = []

    # Args summary (skip large or empty)
    if args:
        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            lines.append(f"{k}: {val_str}")
        lines.append("─" * 40)

    # Result
    if len(result) > collapse_threshold:
        lines.append(f"[{len(result):,} chars — truncated to {collapse_threshold:,}]")
        lines.append(result[:collapse_threshold])
    else:
        lines.append(result)

    body = "\n".join(lines)

    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_color,
            padding=(0, 1),
        )
    )


def render_diff_panel(
    console: Console,
    tool_name: str,
    path: str,
    diff: str,
    elapsed: float,
) -> None:
    """Print a Panel with a unified diff for write_file / patch_file results."""
    name_color, border_color = _get_theme_colors()

    title = Text()
    title.append(tool_name, style=f"bold {name_color}")
    title.append(f"  {path}", style="dim")
    title.append(f"  {elapsed:.1f}s", style="dim")

    syntax = Syntax(diff, "diff", theme="ansi_dark", line_numbers=False)
    console.print(
        Panel(
            syntax,
            title=title,
            title_align="left",
            border_style=border_color,
            padding=(0, 1),
        )
    )
