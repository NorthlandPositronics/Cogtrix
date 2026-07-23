"""Tests for /session context bar fix (Issue #296)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cogtrix import _session_plain, _session_rich

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(session_id: str = "test-session", memory_mode: str = "conversation") -> MagicMock:
    cfg = MagicMock()
    cfg.session = session_id
    cfg.memory_mode = memory_mode
    return cfg


def _make_stats() -> dict:
    return {"total_messages": 5, "working_memory_size": 10}


# ---------------------------------------------------------------------------
# _session_rich tests
# ---------------------------------------------------------------------------


class TestSessionRich:
    def test_session_panel_shows_token_percentage(self):
        """Context bar shows token-based percentage, not message count."""
        cfg = _make_cfg()
        stats = _make_stats()

        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(
                cfg,
                stats,
                msg_count=5,
                session_tokens=65536,
                max_context_tokens=131072,
            )

        panel_body = mock_panel_cls.call_args[0][0]
        assert "50% of 131,072 tokens" in panel_body
        assert "of 10" not in panel_body

    def test_session_panel_no_context_bar_when_no_tokens(self):
        """No context bar shown if session_tokens=0 and max_context_tokens not set."""
        cfg = _make_cfg()
        stats = _make_stats()

        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, msg_count=5)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "Context" not in panel_body

    def test_context_bar_color_green_below_70(self):
        cfg = _make_cfg()
        stats = _make_stats()
        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, 5, session_tokens=65536, max_context_tokens=131072)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "[green]" in panel_body

    def test_context_bar_color_yellow_70_to_85(self):
        cfg = _make_cfg()
        stats = _make_stats()
        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        # 75% usage
        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, 5, session_tokens=98304, max_context_tokens=131072)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "[yellow]" in panel_body

    def test_context_bar_color_red_at_or_above_85(self):
        cfg = _make_cfg()
        stats = _make_stats()
        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        # 90% usage
        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, 5, session_tokens=117965, max_context_tokens=131072)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "[red]" in panel_body

    def test_context_bar_100_percent(self):
        cfg = _make_cfg()
        stats = _make_stats()
        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, 5, session_tokens=131072, max_context_tokens=131072)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "100% of 131,072 tokens" in panel_body
        assert "████████████████████" in panel_body

    def test_context_bar_clamped_at_100(self):
        """Tokens exceeding max_context_tokens are clamped to 100%."""
        cfg = _make_cfg()
        stats = _make_stats()
        mock_console = MagicMock()
        mock_panel_cls = MagicMock(side_effect=lambda body, **kw: body)

        with (
            patch("cogtrix.console", mock_console),
            patch("cogtrix.Panel", mock_panel_cls),
        ):
            _session_rich(cfg, stats, 5, session_tokens=200000, max_context_tokens=131072)

        panel_body = mock_panel_cls.call_args[0][0]
        assert "100% of 131,072 tokens" in panel_body


# ---------------------------------------------------------------------------
# _session_plain tests
# ---------------------------------------------------------------------------


class TestSessionPlain:
    def test_session_plain_shows_token_percentage(self, capsys):
        """Plain text path shows token-based context line."""
        cfg = _make_cfg()
        _session_plain(cfg, msg_count=5, session_tokens=65536, max_context_tokens=131072)
        out = capsys.readouterr().out
        assert "50% of 131,072 tokens" in out
        assert "of 10" not in out

    def test_session_plain_no_context_bar_when_no_tokens(self, capsys):
        """No context line when max_context_tokens not provided."""
        cfg = _make_cfg()
        _session_plain(cfg, msg_count=5)
        out = capsys.readouterr().out
        assert "Context" not in out
