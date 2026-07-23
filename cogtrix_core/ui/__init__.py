"""Cogtrix UI package."""

from .stats import print_stats_footer
from .theme import THEMES, ThemeConfig, get_theme, set_theme
from .tool_panels import render_diff_panel, render_tool_panel
from .turns import print_assistant_turn_header, print_turn_divider, print_user_turn

__all__ = [
    "ThemeConfig",
    "THEMES",
    "get_theme",
    "set_theme",
    "render_tool_panel",
    "render_diff_panel",
    "print_user_turn",
    "print_assistant_turn_header",
    "print_turn_divider",
    "print_stats_footer",
]
