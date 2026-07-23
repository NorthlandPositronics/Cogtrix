"""
Tool Registry for Cogtrix Agent.
Dynamically loads and registers tools from the src/tools/ directory.
"""

from __future__ import annotations

import importlib
import inspect
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

        If *config* is provided, any built-in module that exposes a
        ``TOOL_SETUP(config)`` callable will have it invoked after import,
        and external tools from ``config.tool_dirs`` and installed
        ``cogtrix.tools`` entry-points will be loaded after the built-ins.

        Args:
            config: Optional Cogtrix configuration object.  When present,
                enables TOOL_SETUP dispatch and plugin loading.

        Returns:
            Dictionary mapping tool names to StructuredTool objects
        """
        log = get_logger()
        module_names = self.scan_tools()

        log.debug("Discovered %d tool modules", len(module_names))

        for module_name in module_names:
            module = self.load_tool_module(module_name)
            if module is None:
                log.debug("Skipped module: %s (import failed)", module_name)
                continue

            # Call TOOL_SETUP if the module exposes one and config is available
            if config is not None and hasattr(module, "TOOL_SETUP") and callable(module.TOOL_SETUP):
                try:
                    module.TOOL_SETUP(config)
                except Exception as exc:
                    log.warning("TOOL_SETUP failed for %s: %s", module_name, exc)

            # Skip modules that declare is_configured() and return False
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

        # Load external plugins when config is available
        if config is not None:
            self._load_plugin_tools(config, log)

        log.info("Loaded %d tools", len(self.tools))
        return self.tools

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
