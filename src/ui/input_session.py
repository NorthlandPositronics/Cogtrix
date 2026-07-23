"""prompt_toolkit-based input session with persistent stats above the prompt."""

from __future__ import annotations

import os

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.styles import Style

_HISTORY_PATH = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "cogtrix",
    "input_history",
)


def _make_history() -> FileHistory | InMemoryHistory:
    try:
        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
        return FileHistory(_HISTORY_PATH)
    except Exception:
        return InMemoryHistory()


# Module-level toolbar state — updated after each agent turn
_toolbar_stats: str = ""

# Slash command list for tab completion
_slash_commands: list[str] = []


def set_slash_commands(commands: list[str]) -> None:
    """Provide the list of slash command names (with leading /) for tab completion."""
    global _slash_commands
    _slash_commands = sorted(commands)


class SlashCompleter(Completer):
    """Tab-complete slash commands in prompt_toolkit."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Only complete when the line starts with /
        if not text.startswith("/"):
            return
        for cmd in _slash_commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))


def update_toolbar_stats(stats_line: str) -> None:
    """Update the stats displayed above the prompt."""
    global _toolbar_stats
    _toolbar_stats = stats_line


def _get_prompt() -> ANSI:
    """Build the prompt: colored separator + right-aligned colored stats + cyan ❯."""
    import re
    import shutil

    width = shutil.get_terminal_size((80, 24)).columns
    sep = "\033[38;5;37m" + "─" * width + "\033[0m"
    raw_stats = _toolbar_stats
    if raw_stats.strip():
        visible = re.sub(r"\033\[[0-9;]*m", "", raw_stats).strip()
        padding = max(0, width - len(visible) - 1)
        stats_line = " " * padding + raw_stats.strip()
        return ANSI(f"{sep}\n{stats_line}\n\033[38;5;37m\u276f\033[0m ")
    return ANSI(f"{sep}\n\033[38;5;37m\u276f\033[0m ")


_COMPLETION_STYLE = Style.from_dict(
    {
        "completion-menu": "bg:#1a1a2e #e0e0e0",
        "completion-menu.completion": "bg:#1a1a2e #b0b0b0",
        "completion-menu.completion.current": "bg:#2d4f7c #ffffff bold",
        "scrollbar.background": "bg:#1a1a2e",
        "scrollbar.button": "bg:#2d4f7c",
    }
)


def create_session(
    history: FileHistory | InMemoryHistory | None = None,
    completer=None,
) -> PromptSession:
    """Create and return a configured PromptSession."""
    return PromptSession(
        message=_get_prompt,
        history=history if history is not None else _make_history(),
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer if completer is not None else SlashCompleter(),
        style=_COMPLETION_STYLE,
        reserve_space_for_menu=4,
        mouse_support=False,
    )
