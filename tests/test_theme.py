"""Tests for the theme system."""

import pytest


def test_default_theme_loads():
    from src.ui.theme import get_theme

    theme = get_theme()
    assert theme is not None
    assert theme.accent == "steel_blue1"


def test_all_builtin_themes_present():
    from src.ui.theme import THEMES

    assert "default" in THEMES
    assert "minimal" in THEMES
    assert "dracula" in THEMES


def test_set_theme_returns_config():
    from src.ui.theme import get_theme, set_theme

    t = set_theme("dracula")
    assert t.accent == "#bd93f9"
    assert get_theme().accent == "#bd93f9"
    set_theme("default")  # reset


def test_set_theme_unknown_raises():
    from src.ui.theme import set_theme

    with pytest.raises(ValueError, match="Unknown theme"):
        set_theme("nonexistent")


def test_minimal_theme_monochrome():
    from src.ui.theme import THEMES

    t = THEMES["minimal"]
    assert t.accent == "white"
    assert t.assistant_label == "white"


def test_dracula_theme_hex_colors():
    from src.ui.theme import THEMES

    t = THEMES["dracula"]
    assert t.accent.startswith("#")
    assert t.success.startswith("#")


def test_config_default_theme():
    from src.config import Config

    c = Config()
    assert c.theme == "default"


def test_config_custom_theme():
    from src.config import Config

    c = Config(theme="dracula")
    assert c.theme == "dracula"


def test_theme_config_has_required_roles():
    from src.ui.theme import ThemeConfig

    t = ThemeConfig()
    required = [
        "accent",
        "dim",
        "user_label",
        "assistant_label",
        "tool_name",
        "tool_result_border",
        "success",
        "warning",
        "error",
        "stats",
        "stats_warning",
        "stats_critical",
    ]
    for role in required:
        assert hasattr(t, role), f"Missing role: {role}"


def test_get_theme_after_set():
    from src.ui.theme import get_theme, set_theme

    set_theme("minimal")
    assert get_theme().accent == "white"
    set_theme("default")  # reset


def test_theme_config_is_dataclass():
    import dataclasses

    from src.ui.theme import ThemeConfig

    assert dataclasses.is_dataclass(ThemeConfig)
