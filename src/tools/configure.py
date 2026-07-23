"""Tool configuration factories.

Centralizes the _configure_* functions that wire up external API keys,
provider settings, and runtime parameters for each tool module.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.logging_config import get_logger
from src.orchestration.compression import truncate_tool_output
from src.tools.resolver import resolve_tool_name as _resolve_tool_name

if TYPE_CHECKING:
    from src.config import Config
    from src.registry import ToolRegistry

try:
    from pydantic import BaseModel, Field

    class RequestToolsInput(BaseModel):
        """Input schema for managing the active tool set."""

        add: list[str] = Field(
            default_factory=list,
            description=(
                "Tool names to load from the available catalog. "
                "They become available immediately."
            ),
        )
        remove: list[str] = Field(
            default_factory=list,
            description=(
                "Tool names to release from the active set. "
                "They return to the catalog and can be re-added later."
            ),
        )
        query: str = Field(
            default="",
            description=(
                "Natural-language description of what you need. "
                "When provided (and add/remove are empty), returns semantically relevant tools."
            ),
        )

except ImportError:
    RequestToolsInput = None  # type: ignore[assignment,misc]


TOOL_OUTPUT_CAP_RATIO = 0.10
TOOL_OUTPUT_CAP_MIN_CHARS = 8_192


def compute_tool_output_cap(max_context_tokens: int) -> int:
    """Return the per-tool max output in *characters*."""
    chars_from_ratio = int(max_context_tokens * TOOL_OUTPUT_CAP_RATIO * 4)
    return max(chars_from_ratio, TOOL_OUTPUT_CAP_MIN_CHARS)


def build_tool_catalog(tools: dict[str, Any]) -> dict[str, str]:
    """Build a lightweight catalog: {tool_name: one-line description}.

    Args:
        tools: {name: tool_object} dict — can come from a ToolRegistry
            or any subset of tools.

    Returns:
        {name: short_description} dict.
    """
    catalog: dict[str, str] = {}
    for name, tool in tools.items():
        desc = getattr(tool, "description", "") or ""
        short = desc.split(". ")[0].split(".\n")[0]
        if len(short) > 120:
            short = short[:117] + "..."
        catalog[name] = short
    return catalog


def load_tools(
    tool_filter: str | None = None,
    config: Config | None = None,
) -> ToolRegistry:
    """Load tools from the tools directory.

    Args:
        tool_filter: Comma-separated list of tool names, or:
            - None/empty: load all tools
            - "none": load no tools
            - "minimal": load basic tools (file_ops, calculate)
        config: Optional Cogtrix configuration.  When provided, TOOL_SETUP is
            dispatched for each built-in module that declares it, and any
            external tools configured via ``tool_dirs`` (or the
            ``COGTRIX_TOOL_DIRS`` env var) and installed ``cogtrix.tools``
            entry-points are also loaded.

    Returns:
        ToolRegistry with loaded tools
    """
    from src.registry import ToolRegistry

    registry = ToolRegistry()

    if tool_filter == "none":
        return registry

    configure_agent_tools()
    registry.load_all_tools(config=config)

    if tool_filter is None or tool_filter == "":
        return registry

    if tool_filter == "minimal":
        allowed = {"read_file", "write_file", "list_directory", "calculate"}
        filtered_tools = {name: tool for name, tool in registry.tools.items() if name in allowed}
        registry.tools = filtered_tools
        return registry

    allowed_tools = {t.strip() for t in tool_filter.split(",") if t.strip()}
    if allowed_tools:
        filtered_tools = {
            name: tool for name, tool in registry.tools.items() if name in allowed_tools
        }
        registry.tools = filtered_tools

    return registry


def apply_output_cap(tool: Any, max_chars: int) -> Any:
    """Wrap *tool* so its output never exceeds *max_chars*.

    Returns a shallow copy of *tool* with a patched ``func`` (or ``_run``)
    so concurrent sessions that hold a reference to the original are not
    affected by the mutation.

    Idempotent: stores the unwrapped function as ``tool._uncapped_func``
    on the first call and always re-wraps from it, preventing nested
    cap wrappers when called multiple times (startup, expansion, delegate).
    """
    import copy
    import functools

    original_func = getattr(tool, "_uncapped_func", None)
    if original_func is None:
        original_func = getattr(tool, "func", None) or getattr(tool, "_run", None)
        if original_func is None:
            return tool

    @functools.wraps(original_func)
    def _capped(*args: Any, **kwargs: Any) -> Any:
        result = original_func(*args, **kwargs)
        if isinstance(result, str):
            return truncate_tool_output(result, max_chars)
        return result

    try:
        tool_copy = copy.copy(tool)
        tool_copy._uncapped_func = original_func
        if hasattr(tool_copy, "func"):
            tool_copy.func = _capped
        else:
            tool_copy._run = _capped
        return tool_copy
    except (AttributeError, TypeError):
        return tool


def configure_delegate_tool(
    config: Config,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Configure the delegate tool with runtime settings from config."""
    try:
        from src.tools.delegate import configure_delegate, set_status_callback

        providers_dict: dict[str, Any] = {}
        for name, prov_cfg in config.providers.items():
            providers_dict[name] = {
                "type": prov_cfg.type,
                "base_url": prov_cfg.base_url,
                "api_key": prov_cfg.api_key,
            }

        allowed = config.delegate_allowed_providers or config.list_providers()

        models_dict: dict[str, Any] = {}
        for mname, mcfg in config.models.items():
            models_dict[mname] = {
                "provider": mcfg.provider,
                "model": mcfg.model,
                "context_window": mcfg.context_window,
                "temperature": mcfg.temperature,
                "max_tokens": mcfg.max_tokens,
            }

        delegate_config = {
            "enabled": config.delegate_enabled,
            "default_timeout": config.delegate_default_timeout,
            "default_model_alias": config.active_model_alias,
            "allowed_providers": allowed,
            "allowed_models": config.delegate_allowed_models,
            "models": models_dict,
            "providers": providers_dict,
        }
        configure_delegate(delegate_config)

        if status_callback is not None:
            set_status_callback(status_callback)
    except ImportError:
        pass


def configure_delegate_tools(
    tools: list,
    available_tools: dict[str, Any] | None = None,
) -> None:
    """Pass all tools (active + on-demand) to the delegate module.

    Delegates receive the full toolset from the start so they can
    execute shell commands, read files, etc. without needing to
    request tools dynamically.  Delegation tools and ``deep_think``
    are automatically excluded to prevent recursion.
    """
    try:
        from src.tools.delegate import set_delegate_tools

        set_delegate_tools(tools, available_tools)
    except ImportError:
        pass


def configure_deep_think_tool(config: Config) -> None:
    """Configure the deep_think tool with provider settings."""
    try:
        from src.tools.deep_think import configure_deep_think

        providers_dict: dict[str, Any] = {}
        for name, prov_cfg in config.providers.items():
            providers_dict[name] = {
                "type": prov_cfg.type,
                "base_url": prov_cfg.base_url,
                "api_key": prov_cfg.api_key,
            }

        models_dict: dict[str, Any] = {}
        for mname, mcfg in config.models.items():
            models_dict[mname] = {
                "provider": mcfg.provider,
                "model": mcfg.model,
                "context_window": mcfg.context_window,
                "temperature": mcfg.temperature,
                "max_tokens": mcfg.max_tokens,
            }

        configure_deep_think(
            {
                "providers": providers_dict,
                "models": models_dict,
                "default_model_alias": config.active_model_alias,
            }
        )
    except ImportError:
        pass


def configure_tavily_tool(config: Config) -> None:
    """Configure the Tavily search tool with API key from config."""
    try:
        from src.tools.tavily_search import configure_tavily

        tavily_cfg: dict[str, Any] = {}
        if config.tavily_api_key:
            tavily_cfg["api_key"] = config.tavily_api_key
        configure_tavily(tavily_cfg)
    except ImportError:
        pass


def configure_exa_tool(config: Config) -> None:
    """Configure the Exa search tool with API key from config."""
    try:
        from src.tools.exa_search import configure_exa

        exa_cfg: dict[str, Any] = {}
        if config.exa_api_key:
            exa_cfg["api_key"] = config.exa_api_key
        configure_exa(exa_cfg)
    except ImportError:
        pass


def configure_brave_tool(config: Config) -> None:
    """Configure the Brave Search tool with API key from config."""
    try:
        from src.tools.brave_search import configure_brave

        brave_cfg: dict[str, Any] = {}
        if config.brave_api_key:
            brave_cfg["api_key"] = config.brave_api_key
        configure_brave(brave_cfg)
    except ImportError:
        pass


def configure_searxng_tool(config: Config) -> None:
    """Configure the SearXNG search tool with instance URL from config."""
    try:
        from src.tools.searxng_search import configure_searxng

        searxng_cfg: dict[str, Any] = {}
        if config.searxng_url:
            searxng_cfg["url"] = config.searxng_url
        configure_searxng(searxng_cfg)
    except ImportError:
        pass


def configure_serpapi_tool(config: Config) -> None:
    """Configure the SerpAPI search tool with API key from config."""
    try:
        from src.tools.serpapi_search import configure_serpapi

        serpapi_cfg: dict[str, Any] = {}
        if config.serpapi_api_key:
            serpapi_cfg["api_key"] = config.serpapi_api_key
        configure_serpapi(serpapi_cfg)
    except ImportError:
        pass


def configure_google_search_tool(config: Config) -> None:
    """Configure the Google Search tool with API key and CSE ID from config."""
    try:
        from src.tools.google_search import configure_google_search

        google_cfg: dict[str, Any] = {}
        if config.google_api_key:
            google_cfg["api_key"] = config.google_api_key
        if config.google_cse_id:
            google_cfg["cse_id"] = config.google_cse_id
        configure_google_search(google_cfg)
    except ImportError:
        pass


def configure_python_exec_tool(config: Config) -> None:
    """Configure the Python execution tool with session ID for persistent state."""
    try:
        from src.tools.python_exec import set_session

        set_session(config.session)
    except ImportError:
        pass


def configure_file_ops_tool(config: Config) -> None:
    """Configure file operations tool with allowed write directories."""
    from src.tools.file_ops import set_allowed_write_dirs

    set_allowed_write_dirs(config.allowed_write_paths)


def configure_cron_tool(
    config: Config,
    llm_factory: Callable[[], Any] | None = None,
) -> None:
    """Configure the cron scheduling tool.

    Args:
        config:      Active runtime config (used to resolve the data directory).
        llm_factory: Zero-argument callable returning the active ``BaseChatModel``.
                     Called fresh each time a job fires, so provider / model
                     changes are reflected automatically.
    """
    try:
        from src.tools.cron_tools import configure_cron

        data_dir = config.resolve_data_path("cron")
        configure_cron(data_dir=str(data_dir), llm_factory=llm_factory)
    except (ImportError, OSError):
        pass


def configure_agent_tools() -> None:
    """Initialise agent spawning and task management tools.

    Called by :func:`load_tools` so the module-level imports in
    ``src.tools.agent_tools`` are resolved at registry load time, surfacing
    any dependency issues early.
    """
    try:
        from src.tools.agent_tools import configure_agent_tools as _configure

        _configure()
    except (ImportError, OSError):
        pass


def configure_email_tool(config: Config) -> None:
    """Configure the email tools with IMAP/SMTP settings from config."""
    try:
        from src.tools.email_tools import configure_email

        email_cfg = config.services.get("email", {})
        configure_email(email_cfg)
    except ImportError:
        pass


def configure_rag_tool(config: Config) -> None:
    """Configure the RAG tool with runtime settings from config.

    After setting embedding parameters, updates the tool description with
    live index stats and sets ``_rag_auto_activate`` so callers can check
    whether to promote the tool from on-demand to active.
    """
    try:
        from src.tools.rag import TOOL_CONFIG as _rag_tool_config
        from src.tools.rag import (
            _build_description,
            configure_rag,
            knowledge_base_exists,
        )

        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
        rag_config: dict[str, str | None] = {
            "embedding_provider": emb_type,
            "embedding_model": emb_model,
            "base_url": emb_base_url,
            "api_key": emb_api_key,
            "vectordb_dir": str(config.resolve_data_path(config.rag.vectordb_dir) / "faiss_index"),
        }

        # Resolve the API uploads directory so the RAG tool can also search
        # per-document FAISS indexes created by the API ingestion pipeline.
        import os

        data_dir = os.environ.get("COGTRIX_DATA_DIR", config.data_dir)
        api_uploads = Path(data_dir, "api", "uploads").resolve()
        rag_config["api_uploads_dir"] = str(api_uploads)

        # Entity index (M4.3)
        entity_index_path = Path(data_dir, "rag", "entity_index.json").resolve()
        rag_config["entity_index_path"] = str(entity_index_path)

        # Score threshold (M4.3)
        rag_config["score_threshold"] = config.rag.score_threshold  # type: ignore[assignment]

        configure_rag(rag_config)

        # Update tool description with live index stats so the LLM
        # sees doc count / size in both the active tool schema and
        # the on-demand catalog.
        desc = _build_description()
        _rag_tool_config["description"] = desc

        # Set auto-activate flag so the tool is promoted from on-demand
        # to active when a knowledge base exists.
        global _rag_auto_activate
        _rag_auto_activate = knowledge_base_exists()
    except (ImportError, OSError):
        pass


def _update_rag_tool_description(tool: Any) -> None:
    """Update the description on an already-registered RAG StructuredTool.

    Called after :func:`configure_rag_tool` has set ``TOOL_CONFIG["description"]``
    with live index stats.
    """
    try:
        from src.tools.rag import TOOL_CONFIG as _rag_tool_config

        desc = _rag_tool_config.get("description")
        if desc and hasattr(tool, "description"):
            tool.description = desc
    except ImportError:
        pass


_rag_auto_activate: bool = False


def rag_should_auto_activate() -> bool:
    """Return True if the RAG tool should be in the active set.

    Set by :func:`configure_rag_tool` after checking for existing indexes.
    """
    return _rag_auto_activate


def filter_unconfigured_tools(registry: ToolRegistry) -> None:
    """Remove tools from the registry whose required API keys are missing.

    Each tool module can export an ``is_configured() -> bool`` function.
    If present and it returns False, all tools from that module are removed
    from the registry so the agent never sees them.

    ``LazyToolProxy`` stubs are skipped — they were registered precisely
    because their module has no ``is_configured`` guard, so they are
    implicitly always configured.
    """
    from src.registry import LazyToolProxy

    log = get_logger()

    to_remove: list[str] = []
    module_status: dict[str, bool] = {}

    for tool_name, tool_obj in registry.tools.items():
        # Lazy proxies have no is_configured() — skip without triggering import.
        if isinstance(tool_obj, LazyToolProxy):
            continue
        func = getattr(tool_obj, "_uncapped_func", None) or getattr(tool_obj, "func", None)
        if func is None:
            continue
        module_name = getattr(func, "__module__", "")
        if not module_name:
            continue

        if module_name in module_status:
            if not module_status[module_name]:
                to_remove.append(tool_name)
            continue

        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                module_status[module_name] = True
                continue

        checker = getattr(module, "is_configured", None)
        if checker is None:
            module_status[module_name] = True
            continue

        try:
            configured = checker()
        except Exception:  # noqa: BLE001
            configured = True

        module_status[module_name] = configured
        if not configured:
            to_remove.append(tool_name)

    for tool_name in to_remove:
        del registry.tools[tool_name]
        registry.tool_metadata.pop(tool_name, None)
        log.debug("Removed unconfigured tool: %s", tool_name)

    if to_remove:
        log.info(
            f"Filtered {len(to_remove)} unconfigured tool(s): " f"{', '.join(sorted(to_remove))}"
        )


TOOL_PRESETS: dict[str, set[str]] = {
    "reasoning": set(),
    "code": set(),
    "conversation": set(),
}


def apply_tool_preset(
    registry: ToolRegistry,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a full registry into *active* and *available* tools.

    Args:
        registry: Full tool registry with all tools loaded.
        mode: Memory mode name (e.g. 'reasoning', 'code', 'conversation').

    Returns:
        (active_tools, available_tools) — both are {name: tool} dicts.
    """
    preset = TOOL_PRESETS.get(mode, set())

    active: dict[str, Any] = {}
    available: dict[str, Any] = {}
    for name, tool in registry.tools.items():
        if name in preset:
            active[name] = tool
        else:
            available[name] = tool
    return active, available


def create_request_tools_tool(
    available_tools: dict[str, Any],
    catalog: dict[str, str],
    active_names: set[str] | None = None,
    protected_names: set[str] | None = None,
    tool_index: Any = None,
) -> Any:
    """
    Create the ``request_tools`` meta-tool.

    The model can **add** tools from the on-demand catalog and **remove**
    (release) tools it no longer needs.  Released tools go back to the
    catalog and can be re-requested later.

    Args:
        available_tools: {name: tool} of tools not currently in the agent.
        catalog: {name: short_description} for the on-demand catalog.
        active_names: Names of tools currently loaded in the agent.
            Used to build the "releasable" list in the description.
        protected_names: Tool names that cannot be released (mode
            presets + the meta-tool itself).
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return None

    if RequestToolsInput is None:
        return None

    _protected: set[str] = (protected_names or set()) | {"request_tools"}
    _active: set[str] = active_names or set()

    # ── Catalog text: tools available to add ──
    add_lines = []
    for name in sorted(available_tools):
        desc = catalog.get(name, "")
        add_lines.append(f"  - {name}: {desc}")
    add_catalog = "\n".join(add_lines) if add_lines else "  (none)"

    # ── Releasable list: active tools that are NOT protected ──
    releasable = sorted(_active - _protected - {"request_tools"})
    if releasable:
        remove_catalog = "\n".join(f"  - {n}" for n in releasable)
    else:
        remove_catalog = "  (none — all active tools are core to this mode)"

    def request_tools(
        add: list[str] | None = None,
        remove: list[str] | None = None,
        query: str = "",
    ) -> str:
        """Add or remove tools from the active agent toolkit."""
        add = add or []
        remove = remove or []

        # Semantic query: only used when add and remove are both empty.
        if query and not add and not remove:
            if tool_index is not None:
                hits = tool_index.search(query, k=8)
                if not hits:
                    return (
                        f"No tools matched '{query}'. "
                        "Try different keywords or call with no arguments to see the full catalog."
                    )
                lines = []
                for name in hits:
                    desc = catalog.get(name, "")
                    lines.append(f"  - {name}: {desc}")
                return f"Semantic search results for '{query}':\n" + "\n".join(lines)
            # No index — fall through to full catalog listing below.
            add = []
            remove = []

        # Resolve fuzzy/abbreviated names against both the available pool and the active set.
        # Track original → resolved mapping for name-correction guidance.
        resolved_add: list[str] = []
        fuzzy_renames: dict[str, str] = {}
        for name in add:
            resolved, _ = _resolve_tool_name(name, available_tools, _active)
            canonical = resolved if resolved is not None else name
            resolved_add.append(canonical)
            if canonical != name:
                fuzzy_renames[canonical] = name
        add = resolved_add

        # Deduplicate: if a name appears in both, add wins
        remove = [n for n in remove if n not in add]

        parts: list[str] = []

        # ── Additions ──
        valid_add = [n for n in add if n in available_tools]
        invalid_add = [n for n in add if n not in available_tools]
        if valid_add:
            loaded_parts: list[str] = []
            for n in valid_add:
                original = fuzzy_renames.get(n)
                if original is not None:
                    loaded_parts.append(f"{n} (resolved from '{original}')")
                else:
                    loaded_parts.append(n)
            parts.append(
                f"Tools loaded: {', '.join(loaded_parts)}. They are now active and ready to use."
            )
        if invalid_add:
            already_active = [n for n in invalid_add if n in _active]
            truly_unknown = [n for n in invalid_add if n not in _active]
            if already_active:
                parts.append(
                    f"Already active: {', '.join(already_active)}. You can use them directly."
                )
            if truly_unknown:
                parts.append(
                    f"Unknown tools: {', '.join(truly_unknown)}. "
                    "Use `add` with no arguments to see the available catalog."
                )

        # ── Removals ──
        blocked = [n for n in remove if n in _protected]
        valid_remove = [n for n in remove if n not in _protected and n in _active]
        unknown_remove = [n for n in remove if n not in _protected and n not in _active]
        if valid_remove:
            parts.append(
                f"Releasing: {', '.join(valid_remove)}. "
                "They will be removed from the active set."
            )
        if blocked:
            parts.append(f"Cannot release (core to this mode): {', '.join(blocked)}.")
        if unknown_remove:
            parts.append(f"Cannot release (not in active set): {', '.join(unknown_remove)}.")

        if not add and not remove:
            return (
                f"Tools you can ADD (on-demand catalog):\n{add_catalog}\n\n"
                f"Tools you can RELEASE (currently active, non-core):\n{remove_catalog}"
            )

        if not parts:
            parts.append("No changes made.")

        return " ".join(parts)

    tool = StructuredTool.from_function(
        func=request_tools,
        name="request_tools",
        description=(
            "Manage the active tool set. Call with no arguments to list all available tools "
            "with descriptions. Use `add` to load tools and `remove` to release tools you no "
            "longer need."
        ),
        args_schema=RequestToolsInput,
    )
    return tool
