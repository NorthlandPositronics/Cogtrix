"""
Tool Registry for Cogtrix Agent.
Dynamically loads and registers tools from the src/tools/ directory.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.logging_config import get_logger

# LangChain imports (support multiple namespaces)
try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    # Fallback for testing without LangChain installed
    StructuredTool = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool as _StructuredToolType

    from src.config import Config

from src.agent.safety import create_safe_tool


def _func_to_schema_name(func_name: str) -> str:
    """Convert 'foo_bar' to 'FooBarInput'."""
    return "".join(word.capitalize() for word in func_name.split("_")) + "Input"


def _scan_tool_metadata_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Extract tool name(s) and description(s) from a Python source file using AST.

    Reads ``TOOL_CONFIG`` and ``TOOL_CONFIGS`` module-level assignments without
    importing the module.  Only string-literal values are extracted; computed
    expressions (f-strings, concatenations, function calls) fall back to an
    empty string so the caller can use a placeholder description.

    Returns a list of dicts with at least ``"name"`` and ``"description"`` keys.
    Returns an empty list on parse failure.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return []

    def _str_value(node: ast.expr) -> str:
        """Return string value for a Constant(str) node, or '' otherwise."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # Handle implicit string concatenation (JoinedStr / BinOp)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _str_value(node.left) + _str_value(node.right)
        return ""

    def _extract_from_dict(dict_node: ast.Dict) -> dict[str, str | bool] | None:
        result: dict[str, str | bool] = {}
        requires_confirmation: bool | None = None
        for key_node, val_node in zip(dict_node.keys, dict_node.values, strict=False):
            if not isinstance(key_node, ast.Constant):
                continue
            key = key_node.value
            if key in ("name", "description"):
                result[key] = _str_value(val_node)
            elif key == "requires_confirmation":
                if isinstance(val_node, ast.Constant) and isinstance(val_node.value, bool):
                    requires_confirmation = val_node.value
        if "name" in result:
            if requires_confirmation is not None:
                result["requires_confirmation"] = requires_confirmation
            return result
        return None

    results: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "TOOL_CONFIG" and isinstance(node.value, ast.Dict):
                entry = _extract_from_dict(node.value)
                if entry:
                    results.append(entry)
            elif target.id == "TOOL_CONFIGS" and isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Dict):
                        entry = _extract_from_dict(elt)
                        if entry:
                            results.append(entry)

    return results


class LazyToolProxy:
    """Defers module import and StructuredTool creation until first use.

    Stored in ``available_tools`` during startup — the registry never imports
    the module until the agent calls ``request_tools(add=[...])`` and the
    tool is moved from ``available_tools`` to the active set.

    ``func`` and ``_run`` property access also triggers resolution so that
    ``apply_output_cap()`` (which reads these attrs) works transparently.
    """

    def __init__(
        self,
        name: str,
        description: str,
        module_name: str,
        registry: ToolRegistry,
    ) -> None:
        self.name = name
        self.description = description
        self._module_name = module_name
        self._registry = registry
        self._resolved: Any = None
        self._lock = threading.Lock()

    def _resolve(self) -> Any:
        """Import the module and register the real tool on first call."""
        if self._resolved is not None:
            return self._resolved
        with self._lock:
            if self._resolved is not None:
                return self._resolved
            log = get_logger()
            log.debug("Lazy-loading tool module: %s (for tool: %s)", self._module_name, self.name)
            module = self._registry.load_tool_module(self._module_name)
            if module is None:
                raise RuntimeError(
                    f"Failed to import tool module '{self._module_name}' " f"for tool '{self.name}'"
                )
            results = self._registry.extract_tool_functions(module)
            for func, cfg in results:
                self._registry.register_tool(func, cfg)
            tool = self._registry.tools.get(self.name)
            if tool is None:
                raise RuntimeError(
                    f"Tool '{self.name}' not found in module '{self._module_name}' "
                    f"after lazy load"
                )
            self._resolved = tool
        return self._resolved

    # ── Transparent delegation ────────────────────────────────────────

    @property
    def func(self) -> Any:
        return getattr(self._resolve(), "func", None)

    @property
    def _run(self) -> Any:
        return getattr(self._resolve(), "_run", None)

    @property
    def _uncapped_func(self) -> Any:
        return getattr(self._resolve(), "_uncapped_func", None)

    @_uncapped_func.setter
    def _uncapped_func(self, value: Any) -> None:
        self._resolve()._uncapped_func = value  # type: ignore[attr-defined]

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve().invoke(*args, **kwargs)

    def __copy__(self) -> Any:
        """Return the resolved real tool so copy.copy() gets a patchable object."""
        return self._resolve()

    def __repr__(self) -> str:
        return f"LazyToolProxy(name={self.name!r}, module={self._module_name!r})"


class ToolRegistry:
    """
    Registry for dynamically loading and managing LangChain tools.

    Scans the src/tools/ directory for Python modules containing tool functions
    and converts them to LangChain StructuredTool objects.
    """

    def __init__(self, tools_directory: str | None = None):
        """
        Initialize the tool registry.

        Args:
            tools_directory: Path to tools directory (defaults to src/tools/)
        """
        if tools_directory is None:
            # Default to src/tools relative to this file
            base_path = Path(__file__).parent
            tools_directory = str(base_path / "tools")

        self.tools_directory = Path(tools_directory)
        self.tools: dict[str, _StructuredToolType] = {}
        self.tool_metadata: dict[str, dict] = {}
        # Deferred tools: name → module_name.  Populated by load_all_tools()
        # for on-demand tools; the module is imported on first access.
        self._deferred: dict[str, str] = {}

    def scan_tools(self) -> list[str]:
        """
        Scan the tools directory for Python modules containing tool functions.

        Returns:
            List of module names that were discovered
        """
        discovered_modules: list[str] = []

        if not self.tools_directory.exists():
            log = get_logger()
            log.warning("Tools directory %s does not exist", self.tools_directory)
            return discovered_modules

        # Scan for Python files (excluding __init__.py)
        for file_path in sorted(self.tools_directory.glob("*.py")):
            if file_path.name == "__init__.py":
                continue

            module_name = file_path.stem
            discovered_modules.append(module_name)

        return discovered_modules

    def load_tool_module(self, module_name: str) -> object | None:
        """
        Dynamically import a tool module.

        Args:
            module_name: Name of the module (without .py extension)

        Returns:
            Imported module object, or None if import fails
        """
        try:
            # Import from src.tools package
            full_module_path = f"src.tools.{module_name}"
            module = importlib.import_module(full_module_path)
            return module
        except ImportError as e:
            log = get_logger()
            log.warning("Failed to import tool module %s: %s", module_name, e)
            return None
        except Exception as e:
            log = get_logger()
            log.error("Error loading tool module %s: %s", module_name, e)
            return None

    def extract_tool_functions(self, module: object) -> list[tuple]:
        """
        Extract tool function(s) and configuration(s) from a module.

        Supports both single TOOL_CONFIG and multiple TOOL_CONFIGS.

        Looks for TOOL_CONFIG or TOOL_CONFIGS dict(s) in the module that specify:
        - name: Function name
        - description: Tool description
        - input_schema: Pydantic BaseModel class
        - requires_confirmation: Boolean flag
        - function: (optional) Direct function reference

        Args:
            module: The imported module object

        Returns:
            List of (function, config_dict) tuples
        """
        results = []

        # Look for TOOL_CONFIGS (multiple tools) first
        if hasattr(module, "TOOL_CONFIGS"):
            configs = module.TOOL_CONFIGS
            for config in configs:
                config = config.copy()  # Don't modify original
                func_name = config.get("name")

                # Check for direct function reference
                if "function" in config:
                    func = config.pop("function")
                    results.append((func, config))
                elif func_name and hasattr(module, func_name):
                    func = getattr(module, func_name)
                    results.append((func, config))
                else:
                    log = get_logger()
                    log.warning(
                        f"TOOL_CONFIGS specifies function '{func_name}' "
                        "but it's not found in module"
                    )

        # Look for single TOOL_CONFIG
        elif hasattr(module, "TOOL_CONFIG"):
            config = module.TOOL_CONFIG.copy()  # Don't modify original
            func_name = config.get("name")

            # Check for direct function reference
            if "function" in config:
                func = config.pop("function")
                results.append((func, config))
            elif func_name and hasattr(module, func_name):
                func = getattr(module, func_name)
                results.append((func, config))
            else:
                log = get_logger()
                log.warning(
                    f"TOOL_CONFIG specifies function '{func_name}' " "but it's not found in module"
                )

        # Fallback: Look for functions with matching Pydantic schemas
        if not results:
            log = get_logger()
            log.warning(
                "Module %s: no tools resolved from TOOL_CONFIG/TOOL_CONFIGS — using fallback discovery",
                module.__name__,
            )
            input_schemas = {}
            for attr_name, attr_obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(attr_obj, BaseModel)
                    and attr_name.endswith("Input")
                    and attr_name != "BaseModel"
                ):
                    input_schemas[attr_name] = attr_obj

            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_") or not func.__doc__:
                    continue
                expected_schema = _func_to_schema_name(name)
                schema_class = input_schemas.get(expected_schema)
                if schema_class is None:
                    continue
                config = {
                    "name": name,
                    "description": (func.__doc__.split("\n\n")[0].strip() if func.__doc__ else ""),
                    "input_schema": schema_class,
                    "requires_confirmation": False,
                }
                results.append((func, config))

        return results

    def register_tool(self, func: Callable[..., Any], config: dict) -> Any | None:
        """
        Convert a Python function to a LangChain StructuredTool.

        Args:
            func: The Python function to convert
            config: Configuration dictionary with:
                - name: Tool name
                - description: Tool description
                - input_schema: Pydantic BaseModel class
                - requires_confirmation: Boolean flag

        Returns:
            StructuredTool object or None if creation fails
        """
        if StructuredTool is None:
            raise ImportError(
                "LangChain is not installed. Please run: pip install -r requirements.txt"
            )

        try:
            name = config.get("name", func.__name__)
            description = config.get("description", func.__doc__ or "")
            input_schema = config.get("input_schema")
            requires_confirmation = config.get("requires_confirmation", False)

            # Create StructuredTool (or SafeTool) from function
            if requires_confirmation:
                tool = create_safe_tool(
                    func=func,
                    name=name,
                    description=description,
                    confirm=requires_confirmation,
                    args_schema=input_schema,
                )
            else:
                if input_schema:
                    tool = StructuredTool.from_function(  # type: ignore[assignment]
                        func=func,
                        name=name,
                        description=description,
                        args_schema=input_schema,
                    )
                else:
                    # Fallback: create tool without explicit schema
                    tool = StructuredTool.from_function(  # type: ignore[assignment]
                        func=func,
                        name=name,
                        description=description,
                    )

            # Store metadata for safety layer (Milestone 3)
            self.tool_metadata[name] = {
                "requires_confirmation": requires_confirmation,
            }

            self.tools[name] = tool
            return tool

        except Exception as e:
            log = get_logger()
            log.error("Error registering tool %s: %s", config.get("name", "unknown"), e)
            return None

    def load_all_tools(self, config: Config | None = None) -> dict[str, _StructuredToolType]:
        """Scan tools directory and load all available tools.

        Modules that do NOT declare ``is_configured()`` or ``TOOL_SETUP()``
        are registered as ``LazyToolProxy`` stubs — the real module import is
        deferred until the tool is first activated via ``request_tools``.
        Modules that declare either hook are imported eagerly so the hook can
        run at startup (``TOOL_SETUP`` wires provider config; ``is_configured``
        gates availability on API-key presence).

        If *config* is provided, any built-in module that exposes a
        ``TOOL_SETUP(config)`` callable will have it invoked after import,
        and external tools from ``config.tool_dirs`` and installed
        ``cogtrix.tools`` entry-points will be loaded after the built-ins.

        Args:
            config: Optional Cogtrix configuration object.  When present,
                enables TOOL_SETUP dispatch and plugin loading.

        Returns:
            Dictionary mapping tool names to StructuredTool objects and
            ``LazyToolProxy`` stubs for deferred modules.
        """
        log = get_logger()
        module_names = self.scan_tools()

        log.debug("Discovered %d tool modules", len(module_names))

        for module_name in module_names:
            file_path = self.tools_directory / f"{module_name}.py"

            # Peek at the file with AST to decide whether to load eagerly.
            metadata = _scan_tool_metadata_from_file(file_path)

            # Determine if the module needs eager import.
            # We check for is_configured / TOOL_SETUP markers in the AST
            # (simple name-based heuristic — accurate for all built-in tools).
            needs_eager = self._module_needs_eager_import(file_path)

            if needs_eager or not metadata:
                # Eager path: import now, run TOOL_SETUP, check is_configured.
                module = self.load_tool_module(module_name)
                if module is None:
                    log.debug("Skipped module: %s (import failed)", module_name)
                    continue

                if (
                    config is not None
                    and hasattr(module, "TOOL_SETUP")
                    and callable(module.TOOL_SETUP)
                ):
                    try:
                        module.TOOL_SETUP(config)
                    except Exception as exc:
                        log.warning("TOOL_SETUP failed for %s: %s", module_name, exc)

                if hasattr(module, "is_configured") and callable(module.is_configured):
                    try:
                        if not module.is_configured():
                            log.debug("Skipped module: %s (not configured)", module_name)
                            continue
                    except Exception:
                        log.debug("Skipped module: %s (is_configured raised)", module_name)
                        continue

                results = self.extract_tool_functions(module)
                if not results:
                    log.debug("No tool function found in module: %s", module_name)
                    continue

                for func, tool_config in results:
                    tool = self.register_tool(func, tool_config)
                    if tool:
                        log.debug("Registered tool: %s", tool_config.get("name", func.__name__))
            else:
                # Lazy path: register proxy stubs without importing the module.
                for entry in metadata:
                    name = entry.get("name", "")
                    description = entry.get("description", "")
                    if not name:
                        continue
                    proxy: Any = LazyToolProxy(
                        name=name,
                        description=description,
                        module_name=module_name,
                        registry=self,
                    )
                    self.tools[name] = proxy  # type: ignore[assignment]
                    self._deferred[name] = module_name
                    # Use requires_confirmation from AST scan, default to False if not specified
                    requires_confirmation = entry.get("requires_confirmation", False)
                    self.tool_metadata[name] = {"requires_confirmation": requires_confirmation}
                    log.debug("Deferred tool stub registered: %s (module: %s)", name, module_name)

        # Load external plugins when config is available
        if config is not None:
            self._load_plugin_tools(config, log)

        log.info(
            "Loaded %d tools (%d deferred)",
            len(self.tools),
            len(self._deferred),
        )
        return self.tools

    @staticmethod
    def _module_needs_eager_import(file_path: Path) -> bool:
        """Return True only when the module MUST be imported at startup.

        A module needs eager import when it has ``is_configured`` that may
        return False (tool gated on API keys/config) OR ``TOOL_SETUP`` that
        must run at startup.  Modules whose ``is_configured`` always returns
        True (no external dependency) are safe to defer.

        Heuristic: if the source contains ``is_configured`` AND references
        an API key, env var, or config value, it's gated and needs eager
        import.  Otherwise it's always-available and can be deferred.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            return True  # Fail safe: import eagerly

        if "TOOL_SETUP" in source:
            return True

        if "is_configured" not in source:
            return False

        # Gated tools reference API keys, env vars, or config dicts
        gated_markers = (
            "api_key",
            "API_KEY",
            "os.environ",
            "os.getenv",
            "_config.get(",
            "_config[",
            "not configured",
        )
        return any(m in source for m in gated_markers)

    def _load_plugin_tools(self, config: Config, log: Any) -> None:
        """Load tools from file-drop directories and installed entry-points."""
        tool_dirs: list[str] = getattr(config, "tool_dirs", []) or []

        from src.plugins.loader import ToolPluginLoader

        loader = ToolPluginLoader()
        ext_modules = loader.load_all(tool_dirs)

        for module in ext_modules:
            module_label = getattr(module, "__name__", repr(module))

            if hasattr(module, "TOOL_SETUP") and callable(module.TOOL_SETUP):
                try:
                    module.TOOL_SETUP(config)
                except Exception as exc:
                    log.warning("TOOL_SETUP failed for plugin %s: %s", module_label, exc)

            results = self.extract_tool_functions(module)
            if not results:
                log.debug("No tool functions found in plugin module: %s", module_label)
                continue

            for func, tool_config in results:
                tool = self.register_tool(func, tool_config)
                if tool:
                    log.debug(
                        "Registered plugin tool: %s (from %s)",
                        tool_config.get("name", func.__name__),
                        module_label,
                    )

    def get_tool(self, name: str) -> _StructuredToolType | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation."""
        return self.tool_metadata.get(tool_name, {}).get("requires_confirmation", False)

    def list_tools(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self.tools.keys())

    def is_mcp_tool(self, name: str) -> bool:
        """Check if a tool came from an MCP server."""
        return self.tool_metadata.get(name, {}).get("source") == "mcp"

    def get_tool_server(self, name: str) -> str | None:
        """Return the MCP server name for an MCP tool."""
        meta = self.tool_metadata.get(name, {})
        return meta.get("server") if meta.get("source") == "mcp" else None
