"""Theme system — semantic color roles and built-in themes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThemeConfig:
    """Semantic color roles for the Cogtrix TUI."""

    # Structure / chrome
    accent: str = "steel_blue1"
    dim: str = "grey50"

    # Labels
    user_label: str = "cyan"
    assistant_label: str = "blue"

    # Tool panels
    tool_name: str = "yellow"
    tool_result_border: str = "grey50"

    # Status
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"

    # Stats footer
    stats: str = "grey50"
    stats_warning: str = "yellow"  # >= 70% context
    stats_critical: str = "red"  # >= 85% context


# Built-in themes

THEMES: dict[str, ThemeConfig] = {
    "default": ThemeConfig(),
    "minimal": ThemeConfig(
        accent="white",
        dim="grey37",
        user_label="bright_white",
        assistant_label="white",
        tool_name="white",
        tool_result_border="grey37",
        success="white",
        warning="white",
        error="bright_white",
        stats="grey37",
        stats_warning="white",
        stats_critical="bright_white",
    ),
    "dracula": ThemeConfig(
        accent="#bd93f9",
        dim="#6272a4",
        user_label="#8be9fd",
        assistant_label="#bd93f9",
        tool_name="#f1fa8c",
        tool_result_border="#6272a4",
        success="#50fa7b",
        warning="#ffb86c",
        error="#ff5555",
        stats="#6272a4",
        stats_warning="#ffb86c",
        stats_critical="#ff5555",
    ),
}

_DEFAULT_THEME_NAME = "default"
_active_theme: ThemeConfig = THEMES[_DEFAULT_THEME_NAME]


def set_theme(name: str) -> ThemeConfig:
    """Activate a built-in theme by name. Returns the active ThemeConfig."""
    global _active_theme
    theme = THEMES.get(name)
    if theme is None:
        raise ValueError(f"Unknown theme {name!r}. Available: {', '.join(THEMES)}")
    _active_theme = theme
    return _active_theme


def get_theme() -> ThemeConfig:
    """Return the currently active ThemeConfig."""
    return _active_theme
