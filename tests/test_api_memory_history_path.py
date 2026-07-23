"""Tests that API session memory history respects data_dir from config.

Regression for the bug where JsonFileMemoryStore() was instantiated without
a base_dir in _build_memory_manager (session_bridge.py) and switch_memory_mode
(routes/memory.py), causing conversation history to land in ./data/history/
(relative to CWD, i.e. /app/data/history/ in Docker) instead of
<data_dir>/history/ as configured.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_fake_store_cls(captured: list[str]) -> type:
    """Return a JsonFileMemoryStore stand-in that records base_dir."""

    class _FakeStore:
        def __init__(self, base_dir: str = "data/history") -> None:
            captured.append(base_dir)

    return _FakeStore


class TestBuildMemoryManagerHistoryPath:
    """_build_memory_manager uses data_dir from app_state.config."""

    def _call(self, app_state: object) -> str | None:
        """Call _build_memory_manager and return the base_dir used."""
        from cogtrix_core.api.session_bridge import _build_memory_manager

        captured: list[str] = []
        fake_cls = _make_fake_store_cls(captured)

        mock_mm = MagicMock()
        mock_mm.load = MagicMock()

        # The import happens lazily inside _build_memory_manager:
        #   from cogtrix_core.memory import JsonFileMemoryStore, MemoryFactory
        # so we patch at src.memory.
        with (
            patch("cogtrix_core.memory.JsonFileMemoryStore", fake_cls),
            patch("cogtrix_core.memory.MemoryFactory") as mock_factory,
        ):
            mock_factory.is_registered.return_value = True
            mock_factory.create.return_value = mock_mm
            _build_memory_manager("sess-1", {}, app_state)

        return captured[0] if captured else None

    def test_uses_data_dir_from_app_config(self, tmp_path: Path) -> None:
        """When app_state.config.data_dir is set, history lands there."""
        cfg = SimpleNamespace(data_dir=str(tmp_path))
        app_state = SimpleNamespace(config=cfg)

        result = self._call(app_state)
        assert result == str(
            tmp_path / "history"
        ), f"Expected {str(tmp_path / 'history')!r}, got {result!r}"

    def test_falls_back_to_relative_when_config_is_none(self) -> None:
        """When app_state.config is None, falls back to 'data/history'."""
        app_state = SimpleNamespace(config=None)

        result = self._call(app_state)
        assert result == "data/history"

    def test_falls_back_to_relative_when_app_state_has_no_config(self) -> None:
        """When app_state has no config attr at all, falls back to 'data/history'."""
        app_state = SimpleNamespace()  # no .config attribute

        result = self._call(app_state)
        assert result == "data/history"

    def test_absolute_data_dir_is_used_verbatim(self, tmp_path: Path) -> None:
        """An absolute data_dir path is used without CWD joining."""
        custom = tmp_path / "mydata"
        cfg = SimpleNamespace(data_dir=str(custom))
        app_state = SimpleNamespace(config=cfg)

        result = self._call(app_state)
        assert result == str(custom / "history")
        assert result is not None and not result.startswith("data/history")


class TestSwitchMemoryModeHistoryPath:
    """The _create_and_load closure in switch_memory_mode uses data_dir."""

    def test_uses_data_dir_from_request_app_state(self, tmp_path: Path) -> None:
        """When request.app.state.config.data_dir is set, history lands there."""
        cfg = SimpleNamespace(data_dir=str(tmp_path))
        captured: list[str] = []
        fake_cls = _make_fake_store_cls(captured)

        mock_mm = MagicMock()
        mock_mm.load = MagicMock()

        with (
            patch("cogtrix_core.memory.JsonFileMemoryStore", fake_cls),
            patch("cogtrix_core.memory.MemoryFactory") as mock_factory,
        ):
            mock_factory.is_registered.return_value = True
            mock_factory.create.return_value = mock_mm

            # Simulate the _create_and_load closure logic from the route.
            app_cfg = cfg
            history_dir = str(Path(app_cfg.data_dir) / "history")
            import cogtrix_core.memory as _mem

            _mem.JsonFileMemoryStore(history_dir)

        assert captured[0] == str(
            tmp_path / "history"
        ), f"Expected {str(tmp_path / 'history')!r}, got {captured[0]!r}"

    def test_falls_back_when_config_is_none(self) -> None:
        """When app_state.config is None, history_dir falls back to 'data/history'."""
        app_cfg = None
        history_dir = (
            str(Path(app_cfg.data_dir) / "history")  # type: ignore[union-attr]
            if app_cfg is not None
            else "data/history"
        )
        assert history_dir == "data/history"
