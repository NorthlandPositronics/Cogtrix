"""Plugin tool loader — discovers and loads tools from external sources.

Three discovery tracks (run in this order):

1. **Built-ins** — ``cogtrix_core/tools/*.py`` modules handled by ``ToolRegistry`` itself;
   this loader does *not* touch them.

2. **File-drop** — any ``*.py`` file in a directory listed under ``tool_dirs``
   in the config file (or via ``COGTRIX_TOOL_DIRS`` env var).  Files whose
   names start with ``_`` are skipped.  Modules are imported with a synthetic
   ``cogtrix_plugin_<stem>`` module name so they never collide with built-ins.

3. **Entry-points** — installed packages that declare an entry-point in the
   ``cogtrix.tools`` group.  Each entry-point value must be one of:

   * A module path (``"my_pkg.my_tools"``) — the module is imported; it must
     expose ``TOOL_CONFIGS`` or ``TOOL_CONFIG`` in the same format as
     built-in tool modules.
   * A class reference (``"my_pkg.tools:MyPlugin"``) — the class is
     instantiated and its ``cogtrix_tools()`` method is called; the result
     must be a ``list[dict]`` of tool-config dicts.

   The pluggy ``hookimpl`` decorator is provided for plugin authors who want
   formal hook dispatch.  See ``cogtrix_core/plugins/spec.py`` and
   ``docs/TOOLS_AUTHORING.md``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

from cogtrix_core.logging_config import get_logger

# Entry-point group name used by installed plugin packages
ENTRY_POINT_GROUP = "cogtrix.tools"


class ToolPluginLoader:
    """Discover and return external tool modules from file-drop dirs and entry-points.

    ``load_all()`` returns a list of module-like objects, each of which exposes
    ``TOOL_CONFIGS`` or ``TOOL_CONFIG`` so that ``ToolRegistry.extract_tool_functions()``
    can process them with no special-casing.
    """

    def load_all(self, tool_dirs: list[str]) -> list[types.ModuleType]:
        """Return all external tool modules.

        Args:
            tool_dirs: Directories to scan for file-drop tools.

        Returns:
            List of module objects ready for ``ToolRegistry.extract_tool_functions()``.
        """
        modules: list[types.ModuleType] = []
        for dir_str in tool_dirs:
            modules.extend(self._load_from_directory(dir_str))
        modules.extend(self._load_from_entrypoints())
        return modules

    # ── Track 2: file-drop ────────────────────────────────────────────────────

    def _load_from_directory(self, dir_str: str) -> list[types.ModuleType]:
        log = get_logger()
        dir_path = Path(dir_str).expanduser().resolve()
        if not dir_path.is_dir():
            log.warning("tool_dirs entry %r is not an accessible directory", dir_str)
            return []

        modules: list[types.ModuleType] = []
        for py_file in sorted(dir_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            # Path-traversal guard: symlinks that escape the directory are rejected
            try:
                py_file.resolve().relative_to(dir_path)
            except ValueError:
                log.warning("Skipping plugin file outside declared directory: %s", py_file)
                continue
            module = self._import_file(py_file)
            if module is not None:
                modules.append(module)

        return modules

    def _import_file(self, py_file: Path) -> types.ModuleType | None:
        log = get_logger()
        # Prefix avoids collisions with built-in src.tools.* modules
        module_name = f"cogtrix_plugin_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                log.warning("Could not create module spec for %s", py_file)
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            log.debug("Loaded file-drop plugin: %s", py_file)
            return module
        except Exception as exc:
            log.warning("Failed to import plugin file %s: %s", py_file, exc, exc_info=True)
            # Clean up partial module registration on failure
            sys.modules.pop(module_name, None)
            return None

    # ── Track 3: entry-points ─────────────────────────────────────────────────

    def _load_from_entrypoints(self) -> list[types.ModuleType]:
        """Load tools from installed packages via the ``cogtrix.tools`` entry-point group."""
        log = get_logger()

        # Use importlib.metadata (stdlib 3.9+) for entry-point discovery.
        # pluggy's load_setuptools_entrypoints is used only when each plugin
        # implements the formal hookimpl interface.
        try:
            from importlib.metadata import entry_points as _entry_points

            eps = _entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:
            log.warning("Entry-point discovery failed: %s", exc, exc_info=True)
            return []

        modules: list[types.ModuleType] = []
        for ep in eps:
            try:
                obj = ep.load()
            except Exception as exc:
                log.warning("Failed to load entry-point %r: %s", ep.name, exc, exc_info=True)
                continue

            module = self._entrypoint_to_module(obj, ep.name)
            if module is not None:
                modules.append(module)
                log.debug("Loaded entry-point plugin: %s", ep.name)

        return modules

    def _entrypoint_to_module(self, obj: object, ep_name: str) -> types.ModuleType | None:
        """Convert a loaded entry-point object to a module-like namespace."""
        log = get_logger()

        # Case 1: already a module with TOOL_CONFIGS / TOOL_CONFIG
        if isinstance(obj, types.ModuleType):
            if hasattr(obj, "TOOL_CONFIGS") or hasattr(obj, "TOOL_CONFIG"):
                return obj
            log.warning(
                "Entry-point %r is a module but exposes no TOOL_CONFIGS/TOOL_CONFIG", ep_name
            )
            return None

        # Case 2: a class — instantiate and call cogtrix_tools() or hookimpl method
        if isinstance(obj, type):
            try:
                instance = obj()
            except Exception as exc:
                log.warning(
                    "Failed to instantiate entry-point class %r: %s", ep_name, exc, exc_info=True
                )
                return None
            return self._instance_to_module(instance, ep_name)

        # Case 3: already an instance
        if hasattr(obj, "cogtrix_tools") and not isinstance(obj, type):
            return self._instance_to_module(obj, ep_name)

        log.warning(
            "Entry-point %r loaded %r which is not a module, class, or plugin instance",
            ep_name,
            type(obj).__name__,
        )
        return None

    def _instance_to_module(self, instance: object, ep_name: str) -> types.ModuleType | None:
        """Call instance.cogtrix_tools() and wrap the result in a synthetic module."""
        log = get_logger()
        method = getattr(instance, "cogtrix_tools", None)
        if method is None:
            log.warning("Entry-point %r instance has no cogtrix_tools() method", ep_name)
            return None
        try:
            configs = method()
        except Exception as exc:
            log.warning("cogtrix_tools() on entry-point %r raised: %s", ep_name, exc)
            return None
        if not isinstance(configs, list):
            log.warning(
                "Entry-point %r cogtrix_tools() returned %r, expected list",
                ep_name,
                type(configs).__name__,
            )
            return None
        return _SyntheticModule(configs, ep_name)


# ── Helpers ───────────────────────────────────────────────────────────────────


class _SyntheticModule(types.ModuleType):
    """A minimal module-like object that holds ``TOOL_CONFIGS``.

    This lets ``ToolRegistry.extract_tool_functions()`` process entry-point
    plugins with no special-casing — it just sees a module with TOOL_CONFIGS.
    """

    def __init__(self, configs: list[dict], ep_name: str) -> None:
        super().__init__(f"cogtrix_entrypoint_{ep_name}")
        self.TOOL_CONFIGS = configs


__all__ = ["ToolPluginLoader", "ENTRY_POINT_GROUP"]
