"""Tests for the pluggable tool architecture (M2.8).

Covers:
- ToolPluginLoader.load_all() with file-drop directories
- Path-traversal guard for symlinks escaping the directory
- Modules starting with _ are skipped
- Modules that fail to import are skipped with a warning
- Entry-point loading via _load_from_entrypoints() with a mock
- TOOL_SETUP dispatch in ToolRegistry.load_all_tools(config=...)
- Config.tool_dirs parsed from config file and env var
- load_tools() threads config to registry
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.plugins.loader import ToolPluginLoader, _SyntheticModule

# ── Helpers ───────────────────────────────────────────────────────────────────

_MINIMAL_TOOL_MODULE = textwrap.dedent("""
    from pydantic import BaseModel, Field

    class PingInput(BaseModel):
        msg: str = Field(..., description="The message.")

    def ping(msg: str) -> str:
        return msg

    TOOL_CONFIGS = [
        {
            "name": "ping",
            "description": "Ping.",
            "input_schema": PingInput,
            "function": ping,
        }
    ]
    """)

_TOOL_SETUP_MODULE = textwrap.dedent("""
    from pydantic import BaseModel, Field

    _received_config = None

    class NopInput(BaseModel):
        x: str = Field(..., description="Ignored.")

    def nop(x: str) -> str:
        return x

    TOOL_CONFIGS = [
        {"name": "nop", "description": "No-op.", "input_schema": NopInput, "function": nop}
    ]

    def TOOL_SETUP(config) -> None:
        global _received_config
        _received_config = config
    """)


def _write_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content)
    return path


# ── File-drop: basic loading ──────────────────────────────────────────────────


def test_load_from_directory_returns_module(tmp_path: Path) -> None:
    _write_file(tmp_path, "ping.py", _MINIMAL_TOOL_MODULE)
    loader = ToolPluginLoader()
    modules = loader.load_all([str(tmp_path)])
    assert len(modules) == 1
    assert hasattr(modules[0], "TOOL_CONFIGS")
    assert modules[0].TOOL_CONFIGS[0]["name"] == "ping"


def test_load_multiple_files_sorted(tmp_path: Path) -> None:
    for name in ("zzz.py", "aaa.py", "mmm.py"):
        _write_file(tmp_path, name, _MINIMAL_TOOL_MODULE)
    loader = ToolPluginLoader()
    modules = loader.load_all([str(tmp_path)])
    # All three should load; order should be sorted (aaa, mmm, zzz)
    names = [m.__name__ for m in modules]
    assert names == sorted(names)


def test_files_starting_with_underscore_skipped(tmp_path: Path) -> None:
    _write_file(tmp_path, "_private.py", _MINIMAL_TOOL_MODULE)
    _write_file(tmp_path, "public.py", _MINIMAL_TOOL_MODULE)
    loader = ToolPluginLoader()
    modules = loader.load_all([str(tmp_path)])
    assert len(modules) == 1
    assert modules[0].__name__ == "cogtrix_plugin_public"


def test_nonexistent_directory_skipped(tmp_path: Path) -> None:
    loader = ToolPluginLoader()
    modules = loader.load_all([str(tmp_path / "does_not_exist")])
    assert modules == []


def test_import_error_skipped(tmp_path: Path) -> None:
    _write_file(tmp_path, "broken.py", "raise ImportError('deliberate')\n")
    loader = ToolPluginLoader()
    modules = loader.load_all([str(tmp_path)])
    assert modules == []
    # Partial registration must be cleaned up
    assert "cogtrix_plugin_broken" not in sys.modules


def test_import_error_does_not_contaminate_sys_modules(tmp_path: Path) -> None:
    _write_file(tmp_path, "poison.py", "import this_module_does_not_exist_xyz\n")
    loader = ToolPluginLoader()
    loader.load_all([str(tmp_path)])
    assert "cogtrix_plugin_poison" not in sys.modules


# ── File-drop: path-traversal guard ───────────────────────────────────────────


def test_symlink_outside_directory_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    real_file = outside / "evil.py"
    real_file.write_text(_MINIMAL_TOOL_MODULE)

    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    link = scan_dir / "evil.py"
    link.symlink_to(real_file)

    loader = ToolPluginLoader()
    # The symlink resolves outside scan_dir → should be rejected
    modules = loader.load_all([str(scan_dir)])
    assert modules == []


# ── File-drop: multiple directories ───────────────────────────────────────────


def test_multiple_tool_dirs_all_loaded(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_file(dir_a, "tool_a.py", _MINIMAL_TOOL_MODULE)
    _write_file(dir_b, "tool_b.py", _MINIMAL_TOOL_MODULE)
    loader = ToolPluginLoader()
    modules = loader.load_all([str(dir_a), str(dir_b)])
    assert len(modules) == 2


# ── Entry-points: _SyntheticModule ────────────────────────────────────────────


def test_synthetic_module_has_tool_configs() -> None:
    configs = [{"name": "x", "description": "X.", "input_schema": None, "function": None}]
    m = _SyntheticModule(configs, "test_ep")
    assert m.TOOL_CONFIGS is configs
    assert m.__name__ == "cogtrix_entrypoint_test_ep"


# ── Entry-points: _entrypoint_to_module ───────────────────────────────────────


def test_entrypoint_real_module_with_tool_configs(tmp_path: Path) -> None:
    _write_file(tmp_path, "ep_mod.py", _MINIMAL_TOOL_MODULE)
    loader = ToolPluginLoader()
    real_module = loader._import_file(tmp_path / "ep_mod.py")
    assert real_module is not None
    result = loader._entrypoint_to_module(real_module, "ep_mod")
    assert result is real_module


def test_entrypoint_real_module_without_configs() -> None:
    m = types.ModuleType("bare_module")
    loader = ToolPluginLoader()
    result = loader._entrypoint_to_module(m, "bare")
    assert result is None


def test_entrypoint_class_with_cogtrix_tools() -> None:
    configs = [{"name": "x", "description": "X.", "input_schema": None, "function": None}]

    class MyPlugin:
        def cogtrix_tools(self) -> list[dict]:
            return configs

    loader = ToolPluginLoader()
    result = loader._entrypoint_to_module(MyPlugin, "my_plugin")
    assert result is not None
    assert isinstance(result, _SyntheticModule)
    assert result.TOOL_CONFIGS == configs


def test_entrypoint_class_cogtrix_tools_raises() -> None:
    class BadPlugin:
        def cogtrix_tools(self):  # type: ignore[return]
            raise RuntimeError("deliberate")

    loader = ToolPluginLoader()
    result = loader._entrypoint_to_module(BadPlugin, "bad_plugin")
    assert result is None


def test_entrypoint_class_cogtrix_tools_returns_non_list() -> None:
    class WrongPlugin:
        def cogtrix_tools(self):  # type: ignore[return]
            return "not a list"

    loader = ToolPluginLoader()
    result = loader._entrypoint_to_module(WrongPlugin, "wrong")
    assert result is None


def test_entrypoint_unrecognised_object() -> None:
    loader = ToolPluginLoader()
    result = loader._entrypoint_to_module(42, "numeric_ep")
    assert result is None


# ── Entry-points: _load_from_entrypoints with mocked importlib.metadata ───────


def _make_fake_ep(name: str, obj: Any) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = obj
    return ep


def test_load_from_entrypoints_class_plugin() -> None:
    configs = [
        {"name": "ep_tool", "description": "EP tool.", "input_schema": None, "function": None}
    ]

    class MyPlugin:
        def cogtrix_tools(self) -> list[dict]:
            return configs

    eps = [_make_fake_ep("my_ep", MyPlugin)]
    loader = ToolPluginLoader()
    with patch("cogtrix_core.plugins.loader.importlib.metadata.entry_points", return_value=eps):
        modules = loader._load_from_entrypoints()
    assert len(modules) == 1
    assert modules[0].TOOL_CONFIGS == configs


def test_load_from_entrypoints_load_failure() -> None:
    ep = MagicMock()
    ep.name = "failing_ep"
    ep.load.side_effect = ImportError("no module")
    loader = ToolPluginLoader()
    with patch("cogtrix_core.plugins.loader.importlib.metadata.entry_points", return_value=[ep]):
        modules = loader._load_from_entrypoints()
    assert modules == []


def test_load_from_entrypoints_empty() -> None:
    loader = ToolPluginLoader()
    with patch("cogtrix_core.plugins.loader.importlib.metadata.entry_points", return_value=[]):
        modules = loader._load_from_entrypoints()
    assert modules == []


# ── TOOL_SETUP dispatch in ToolRegistry ───────────────────────────────────────


def test_registry_calls_tool_setup_when_config_provided(tmp_path: Path) -> None:
    """ToolRegistry.load_all_tools(config=...) must call TOOL_SETUP on plugin modules."""
    _write_file(tmp_path, "setup_tool.py", _TOOL_SETUP_MODULE)

    fake_config = MagicMock()
    fake_config.tool_dirs = [str(tmp_path)]

    from cogtrix_core.registry import ToolRegistry

    registry = ToolRegistry()
    with patch("cogtrix_core.plugins.loader.importlib.metadata.entry_points", return_value=[]):
        registry.load_all_tools(config=fake_config)

    # The plugin's TOOL_SETUP should have been called with the config object
    mod_name = "cogtrix_plugin_setup_tool"
    assert mod_name in sys.modules
    plugin_mod = sys.modules[mod_name]
    assert plugin_mod._received_config is fake_config  # type: ignore[attr-defined]


def test_registry_no_plugin_loading_without_config() -> None:
    """When config=None, ToolPluginLoader must NOT be invoked."""
    from cogtrix_core.registry import ToolRegistry

    registry = ToolRegistry()
    with patch("cogtrix_core.plugins.loader.ToolPluginLoader.load_all") as mock_load:
        registry.load_all_tools(config=None)
    mock_load.assert_not_called()


# ── Config.tool_dirs parsing ──────────────────────────────────────────────────


def test_config_tool_dirs_from_file(tmp_path: Path) -> None:
    """Config parses tool_dirs list from a YAML config file."""
    from cogtrix_core.config import Config, _apply_config_file

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tool_dirs:\n  - /some/dir\n  - /another/dir\n")
    config = Config()
    _apply_config_file(config, cfg_file)
    assert config.tool_dirs == ["/some/dir", "/another/dir"]


def test_config_tool_dirs_string_value(tmp_path: Path) -> None:
    """tool_dirs accepts a bare string (single directory)."""
    from cogtrix_core.config import Config, _apply_config_file

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tool_dirs: /single/dir\n")
    config = Config()
    _apply_config_file(config, cfg_file)
    assert config.tool_dirs == ["/single/dir"]


def test_config_tool_dirs_default_empty() -> None:
    from cogtrix_core.config import Config

    config = Config()
    assert config.tool_dirs == []


def test_config_tool_dirs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """COGTRIX_TOOL_DIRS env var populates tool_dirs (comma-separated)."""
    from cogtrix_core.config import Config, _apply_env_vars

    monkeypatch.setenv("COGTRIX_TOOL_DIRS", "/a/b,/c/d")
    config = Config()
    _apply_env_vars(config)
    assert "/a/b" in config.tool_dirs
    assert "/c/d" in config.tool_dirs


def test_config_cron_from_file(tmp_path: Path) -> None:
    """Config parses cron job definitions from a YAML config file."""
    from cogtrix_core.config import Config, _apply_config_file

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "cron:\n"
        "  - name: nightly\n"
        "    schedule: '0 2 * * *'\n"
        "    prompt: 'check status'\n"
        "    context: inherit\n"
    )
    config = Config()
    _apply_config_file(config, cfg_file)
    assert config.cron == [
        {
            "name": "nightly",
            "schedule": "0 2 * * *",
            "prompt": "check status",
            "context": "inherit",
        }
    ]


# ── load_tools() threads config to registry ───────────────────────────────────


def test_load_tools_passes_config_to_registry() -> None:
    """load_tools(config=...) must forward config to registry.load_all_tools()."""
    from cogtrix_core.tools.configure import load_tools

    fake_config = MagicMock()
    fake_config.tool_dirs = []

    with patch("cogtrix_core.registry.ToolRegistry.load_all_tools") as mock_lat:
        mock_lat.return_value = {}
        load_tools(config=fake_config)

    mock_lat.assert_called_once_with(config=fake_config)
