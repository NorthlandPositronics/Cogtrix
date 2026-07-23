"""Structured conversation turn rendering."""

from __future__ import annotations

import textwrap
from datetime import datetime

from rich.console import Console
from rich.rule import Rule
from rich.text import Text


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M")


def _get_colors() -> tuple[str, str, str, str]:
    """Return (user_label, assistant_label, dim, accent) from active theme or fallback."""
    try:
        from src.ui.theme import get_theme

        t = get_theme()
        return t.user_label, t.assistant_label, t.dim, t.accent
    except Exception:
        return "cyan", "blue", "grey50", "steel_blue1"


def print_user_turn(console: Console, message: str) -> None:
    """Print a user input turn with left rail."""
    user_color, _, dim, _ = _get_colors()
    ts = _timestamp()

    console.print(Text("╷", style=dim))

    header = Text()
    header.append("│ ", style=dim)
    header.append("you", style=f"bold {user_color}")
    header.append(f"  {ts}", style=dim)
    console.print(header)

    wrap_width = max(1, console.width - 2)
    for source_line in message.splitlines() or [""]:
        for seg in textwrap.wrap(source_line, width=wrap_width) or [""]:
            rail = Text()
            rail.append("│ ", style=dim)
            rail.append(seg)
            console.print(rail)

    console.print(Text("╵", style=dim))


def print_assistant_turn_header(console: Console) -> None:
    """Print the assistant turn header (call before streaming response)."""
    _, asst_color, dim, accent = _get_colors()
    ts = _timestamp()

    header = Text()
    header.append("◆ ", style=accent)
    header.append("cogtrix", style=f"bold {asst_color}")
    header.append(f"  {ts}", style=dim)
    console.print(header)


def print_turn_divider(console: Console) -> None:
    """Print a dim rule divider between turns."""
    _, _, dim, _ = _get_colors()
    console.print(Rule(style=dim))
