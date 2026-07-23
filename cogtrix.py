#!/usr/bin/env python3
"""
Cogtrix Agent - CLI Entry Point
A modular LangChain agent with extensible tools and safety features.
Supports multiple LLM providers: OpenAI, Ollama.
"""

import atexit
import os
import sys
import time as _time_mod
import warnings
from pathlib import Path
from typing import Any

from src._version import __copyright__, __version__  # noqa: F401
from src.agent.core import (
    build_system_prompt,
    create_llm_from_provider_config,
)
from src.agent.safety import UserCancelledRun
from src.cli.args import color_enabled, parse_arguments
from src.cli.banner import print_startup
from src.cli.input import (
    load_input_history,
    prefill_next_input,
    read_multiline,
    run_inline_shell,
    save_input_history,
)
from src.config import Config, ConfigError, _resolve_model, load_config
from src.logging_config import (
    create_observability_handler,
    get_logger,
    log_agent_response,
    log_error,
    log_session_info,
    log_user_message,
    new_request_id,
    setup_logging,
)
from src.memory import JsonFileMemoryStore, MemoryFactory
from src.orchestration.compression import (
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    apply_message_compression,
    compress_tool_message,
    create_compression_llm,
    truncate_tool_output,
)
from src.orchestration.graph import (  # noqa: F401
    DEFAULT_RECURSION_LIMIT,
    EMPTY_RESPONSE_MSG,
    build_agent_graph,
)
from src.orchestration.intent import (  # noqa: F401
    ACTION_TARGETS,
    ACTION_VERBS,
    DEEP_THINK_TRIGGERS,
    DELEGATION_TRIGGERS,
    THINK_CATEGORIES,
    THINK_DEFAULT_CATEGORY,
    ThinkCategory,
    classify_think_task,
    prompt_requests_action,
    user_wants_deep_think,
    user_wants_delegation,
)
from src.orchestration.phases import (  # noqa: F401
    ACTION_TOOL_NAMES,
    MIN_GOOD_CONTEXT_LEN,
    RESEARCH_CAP_RATIO,
    STEP_LIMIT_PHRASES,
    WEB_TOOL_NAMES,
    WRITE_FAILURE_PREFIXES,
    agent_performed_writes,
    agent_used_web_tools,
    build_llm_for_decomposition,
    collect_tool_outputs,
    deep_think_had_good_context,
    extract_fetched_urls,
    extract_partial_results,
    extract_turn_messages,
    force_deep_think,
    force_delegation,
    is_step_limit_apology,
    preserve_tables_for_markdown,
    recover_from_step_limit,
    run_execution_phase,
    run_research_delegate,
    was_deep_think_called,
    was_delegation_called,
)
from src.orchestration.runner import (  # noqa: F401
    ToolCallLogger,
    advance_llm_generation,
    build_tool_results_response,
    extract_ai_content,
    extract_response,
    format_agent_error,
    has_phantom_tool_call,
    is_valid_response,
    log_tool_calls_from_result,
    run_agent,
)
from src.orchestration.session_orchestrator import SessionOrchestrator
from src.orchestration.session_state import SessionState
from src.prompt.optimizer import optimize_prompt
from src.registry import ToolRegistry
from src.tools.configure import (
    TOOL_OUTPUT_CAP_MIN_CHARS,
    TOOL_PRESETS,
    apply_output_cap,
    apply_tool_preset,
    build_tool_catalog,
    compute_tool_output_cap,
    configure_brave_tool,
    configure_deep_think_tool,
    configure_delegate_tool,
    configure_delegate_tools,
    configure_exa_tool,
    configure_google_search_tool,
    configure_python_exec_tool,
    configure_rag_tool,
    configure_serpapi_tool,
    configure_tavily_tool,
    create_request_tools_tool,
    filter_unconfigured_tools,
    load_tools,
)
from src.ui.spinner import _spinner

try:
    from src.cli.escape_monitor import EscapeMonitor

    _escape_monitor = EscapeMonitor()
    if _escape_monitor.available:
        _spinner.set_escape_monitor(_escape_monitor)
except Exception:  # noqa: BLE001
    _escape_monitor = None  # type: ignore[assignment]

# Optional Rich imports for pretty terminal output
try:
    from rich import box as rich_box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    rich_box = None  # type: ignore[misc, assignment]
    Align = None  # type: ignore[misc, assignment]
    Console = None  # type: ignore[misc, assignment]
    Group = None  # type: ignore[misc, assignment]
    Markdown = None  # type: ignore[misc, assignment]
    Panel = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

try:
    from src.mcp_client import MCP_AVAILABLE, MCPManager, MCPServerConfig
except ImportError:
    MCP_AVAILABLE = False
    MCPManager = None  # type: ignore[misc, assignment]
    MCPServerConfig = None  # type: ignore[misc, assignment]

# Initialize rich console if available
console = Console() if Console is not None else None

# Backward-compat aliases for compression functions (used by test imports)
_apply_message_compression = apply_message_compression
_compress_tool_message = compress_tool_message
_truncate_tool_output = truncate_tool_output


def _tool_expansion_ui(added: list[str], released: list[str], total: int) -> None:
    """Print a tool-expansion status line, pausing the spinner around it."""
    if not added and not released:
        return
    _spinner.pause()
    parts = []
    if added:
        parts.append(f"Added: {', '.join(added)}")
    if released:
        parts.append(f"Released: {', '.join(released)}")
    print(f"  [tools] {'; '.join(parts)} ({total} total)")
    _spinner.resume()


# ── Tool display categories ──────────────────────────────────────────
# Maps tool names to display categories for /tools output.
# Tools not listed here fall into "Other".
_TOOL_CATEGORIES: dict[str, list[str]] = {
    "Search": [
        "search_web",
        "search_news",
        "tavily_search",
        "tavily_extract",
        "google_search",
        "exa_search",
        "exa_find_similar",
        "exa_get_contents",
        "brave_search",
        "serpapi_search",
    ],
    "Web & HTTP": [
        "http_get",
        "http_post",
    ],
    "File Operations": [
        "read_file",
        "write_file",
        "append_file",
        "list_directory",
        "file_info",
    ],
    "Code Execution": [
        "execute_python",
        "execute_shell_command",
    ],
    "Reasoning & Delegation": [
        "deep_think",
        "delegate_task",
        "delegate_parallel",
    ],
    "Text & NLP": [
        "word_count",
        "find_replace",
        "extract_urls",
        "extract_emails",
        "text_compare",
        "split_text",
        "trim_text",
        "analyze_sentiment",
        "summarize_text",
        "extract_keywords",
    ],
    "Data & JSON": [
        "parse_json",
        "format_json",
        "query_json",
        "extract_json",
        "json_to_text",
        "calculate",
    ],
    "Date & Weather": [
        "get_current_datetime",
        "convert_timezone",
        "parse_date",
        "get_weather",
    ],
    "Knowledge Base": [
        "query_knowledge_base",
    ],
}

# Build reverse lookup: tool_name → category
_TOOL_TO_CATEGORY: dict[str, str] = {}
for _cat, _names in _TOOL_CATEGORIES.items():
    for _tname in _names:
        _TOOL_TO_CATEGORY[_tname] = _cat


_tool_logger = ToolCallLogger()
_is_valid_response = is_valid_response
_format_agent_error = format_agent_error
_extract_ai_content = extract_ai_content
_has_phantom_tool_call = has_phantom_tool_call
_extract_response = extract_response
_build_tool_results_response = build_tool_results_response
_log_tool_calls_from_result = log_tool_calls_from_result

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallback
except ImportError:
    _BaseCallback = object  # type: ignore[misc, assignment]


class _TokenAccumulator(_BaseCallback):  # type: ignore[misc]
    """Accumulates token usage across LLM calls within a single agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        llm_output = getattr(response, "llm_output", None)
        if llm_output:
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if usage:
                self.input_tokens += usage.get("prompt_tokens", 0)
                self.output_tokens += usage.get("completion_tokens", 0)
                return
        gens = getattr(response, "generations", None)
        if gens:
            for gen_list in gens:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            self.input_tokens += getattr(um, "input_tokens", 0)
                            self.output_tokens += getattr(um, "output_tokens", 0)


def _format_stats_line(elapsed: float, acc: _TokenAccumulator) -> str | None:
    """Build a compact stats string for display after agent responses."""
    parts: list[str] = []
    if elapsed > 0:
        parts.append(f"[dim yellow]{elapsed:.1f}s[/dim yellow]")
    if acc.input_tokens or acc.output_tokens:
        parts.append(
            f"[dim #8BC34A]{acc.input_tokens:,}\u2191[/dim #8BC34A]"
            f" [dim red]{acc.output_tokens:,}\u2193[/dim red]"
        )
    return "[dim] \u00b7 [/dim]".join(parts) if parts else None


_session = SessionState()


class _RichConfirmationUI:
    """ConfirmationUI implementation using Rich panels and stdin."""

    def render_prompt(
        self, tool_name: str, tool_input: dict, last_keys: frozenset[str], preview_limit: int
    ) -> None:
        def _preview(val: object) -> str:
            s = str(val)
            if len(s) <= preview_limit:
                return s
            return s[:preview_limit] + f"… ({len(s)} chars total)"

        if console:
            params_lines = []
            if isinstance(tool_input, dict) and tool_input:
                sorted_keys = sorted(
                    tool_input.keys(),
                    key=lambda k: (k in last_keys, len(str(tool_input[k]))),
                )
                for key in sorted_keys:
                    value = tool_input[key]
                    line = f"  [cyan]{key}:[/cyan] {_preview(value).replace('[', chr(92) + '[')}"
                    params_lines.append(line)
                params_text = "\n".join(params_lines)
            elif tool_input:
                params_text = f"  {_preview(tool_input).replace('[', chr(92) + '[')}"
            else:
                params_text = "  (none)"

            warn = "[bold bright_yellow]WARNING:[/bold bright_yellow] "
            exec_msg = f"Agent wants to execute: [bold]{tool_name}[/bold]\n\n"
            params_msg = f"[dim]Parameters:[/dim]\n{params_text}\n"
            hint_msg = (
                "\n[bright_white]"
                "[bold bright_red]Y[/bold bright_red]es  "
                "[bold bright_red]N[/bold bright_red]o  "
                "[bold bright_red]A[/bold bright_red]llow all  "
                "[bold bright_red]D[/bold bright_red]isable tool  "
                "[bold bright_red]F[/bold bright_red]orbid all  "
                "[bold bright_red]C[/bold bright_red]ancel"
                "[/bright_white]\n"
            )
            markup = f"{warn}{exec_msg}{params_msg}{hint_msg}"
            content = Text.from_markup(markup)
            console.print(Panel(content, title="Tool Execution Request", border_style="yellow"))
        else:
            print(f"\nWARNING: Agent wants to execute: {tool_name}")
            if isinstance(tool_input, dict):
                sorted_keys_p = sorted(
                    tool_input.keys(),
                    key=lambda k: (k in last_keys, len(str(tool_input[k]))),
                )
                for key in sorted_keys_p:
                    print(f"  {key}: {_preview(tool_input[key])}")
            else:
                print(f"Input: {_preview(tool_input)}")
            print("  Y=yes  N=no  A=allow all  D=disable tool  F=forbid all  C=cancel")

    def read_choice(self) -> str:
        return input("Allow? ")

    def show_message(self, message: str, style: str) -> None:
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(message)

    def pause_spinner(self) -> None:
        _spinner.pause()

    def resume_spinner(self) -> None:
        _spinner.resume()


_rich_ui = _RichConfirmationUI()

__license__ = "Cogtrix Source-Available License 1.0"

# Module-level flag: skip all tool safety confirmations (set by --no-confirm / -y)
# Accessed via _session.no_confirm

# Track resources for cleanup
_cleanup_resources: list = []


def _friendly_error(exc: Exception, provider: str = "", base_url: str = "") -> str:
    """Return a concise, user-friendly message for common exceptions.

    Falls back to ``str(exc)`` for truly unexpected errors.
    """
    msg = str(exc).lower()
    if "api_key" in msg or "api key" in msg or "authentication" in msg or "unauthorized" in msg:
        hint = (
            f"Provider '{provider}' requires an API key." if provider else "An API key is required."
        )
        return (
            f"{hint}\n"
            "   Set the appropriate environment variable or add 'api_key' to your config."
        )
    if "could not connect" in msg or "connection refused" in msg or "connection error" in msg:
        target = base_url or "(default endpoint)"
        return (
            f"Cannot reach provider '{provider}' at {target}.\n"
            "   Check that the service is running and the URL is correct."
        )
    if "not found" in msg and ("model" in msg or "deployment" in msg):
        return "Model not found. Verify the model name in your configuration."
    # Fallback: first line only, strip stack-trace noise
    first_line = str(exc).split("\n")[0].strip()
    return first_line


def _cleanup():
    """Clean up resources on exit."""
    # Suppress resource warnings during cleanup
    warnings.filterwarnings("ignore", category=ResourceWarning)

    for resource in _cleanup_resources:
        try:
            # Try to close httpx client if it's an LLM with a client
            if hasattr(resource, "client") and hasattr(resource.client, "close"):
                resource.client.close()
            # Try async client too
            if hasattr(resource, "async_client") and hasattr(resource.async_client, "aclose"):
                try:
                    import asyncio

                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(resource.async_client.aclose())
                    finally:
                        loop.close()
                except Exception:  # noqa: BLE001  # nosec B110
                    pass  # atexit handler; logging may be torn down
        except Exception:  # noqa: BLE001  # nosec B110
            pass  # atexit handler; best-effort cleanup only
    _cleanup_resources.clear()


# Register cleanup on exit
atexit.register(_cleanup)


def _try_configure_embeddings(
    memory_manager: Any,
    config: Any,
) -> None:
    """Best-effort embedding setup for hybrid memory vector recall.

    Delegates to the provider registry — same creation path as chat
    models.  If creation fails for any reason, vector recall is simply
    disabled; summary + sliding window still operate normally.

    No startup probe or fallback chain: failures surface naturally on
    first real use in ``SessionVectorStore.add_messages`` /
    ``SessionVectorStore.recall``, which already handle them gracefully.
    """
    _log = get_logger()
    emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()

    try:
        from src.providers import create_embeddings_from_config

        fn, tag = create_embeddings_from_config(
            emb_type, model=emb_model, base_url=emb_base_url, api_key=emb_api_key
        )
        memory_manager.set_embeddings(fn, tag)
        _log.debug("Memory vector recall: using %s", tag)
    except Exception as exc:
        _log.debug("Embedding provider '%s' unavailable: %s", emb_type, exc)
        _log.debug("Vector recall disabled — summary + sliding window still operate")


def _close_llm(llm_instance: Any) -> None:
    """Best-effort close of an LLM's HTTP client(s).

    Called when replacing an LLM during a live model/provider switch so
    we don't leak httpx connections until program exit.
    """
    try:
        if hasattr(llm_instance, "client") and hasattr(llm_instance.client, "close"):
            llm_instance.client.close()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        if hasattr(llm_instance, "async_client") and hasattr(llm_instance.async_client, "aclose"):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(llm_instance.async_client.aclose())
            finally:
                loop.close()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


# ---------------------------------------------------------------------------
# Slash command system
# ---------------------------------------------------------------------------


class SlashCommand:
    """Definition of a single slash command."""

    def __init__(
        self,
        name: str,
        handler,
        short_help: str,
        long_help: str = "",
        aliases: list | None = None,
    ):
        self.name = name
        self.handler = handler
        self.short_help = short_help
        self.long_help = long_help or short_help
        self.aliases = aliases or []


class SlashCommandRegistry:
    """
    Registry and dispatcher for interactive slash commands.

    Commands are prefixed with '/' and dispatched before any input reaches
    the LLM.  The registry also owns the '/help' command which auto-generates
    output from the registered metadata.
    """

    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}
        self._alias_map: dict[str, str] = {}  # alias -> canonical name
        # Context references set by main() before the loop starts
        self.config: Any | None = None
        self.memory_manager: Any | None = None
        self.registry: Any | None = None
        self.approvals: set[str] = set()  # tool auto-approval set
        self.available_tools: dict[str, Any] = {}
        self.system_prompt: str | None = None
        self.mcp_manager: Any = None

    # -- registration -------------------------------------------------------

    def register(self, cmd: SlashCommand) -> None:
        """Register a slash command."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._alias_map[alias] = cmd.name

    def _resolve(self, name: str) -> SlashCommand | None:
        """Resolve a command name or alias to its SlashCommand.

        Aliases are matched **case-sensitively** (so ``/m`` and ``/M`` can
        map to different commands).  Full command names use a
        case-insensitive fallback for convenience.
        """
        # 1. Exact alias match (case-sensitive — critical for /m vs /M)
        canonical = self._alias_map.get(name)
        if canonical:
            return self._commands.get(canonical)
        # 2. Exact command name match
        if name in self._commands:
            return self._commands[name]
        # 3. Case-insensitive fallback for full command names
        return self._commands.get(name.lower())

    # -- dispatch -----------------------------------------------------------

    def is_command(self, text: str) -> bool:
        """Return True if *text* looks like a slash command."""
        return text.startswith("/")

    def dispatch(self, text: str) -> str | None:
        """
        Execute a slash command.

        Returns:
            A string signal: "break" to exit the loop, "continue" to skip
            sending to the agent, or None if the command was not found.
        """
        parts = text.lstrip("/").split(None, 1)
        # Preserve original case for alias resolution (/m ≠ /M)
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1].strip() if len(parts) > 1 else ""

        cmd = self._resolve(cmd_name)
        if cmd is None:
            print(f"Unknown command: /{cmd_name}")
            print("Type /help for a list of commands.")
            return "continue"

        return cmd.handler(self, cmd_args)

    # -- built-in command handlers ------------------------------------------

    @staticmethod
    def _cmd_quit(_self, _args: str) -> str:
        """Handler for /quit."""
        from src.logging_config import get_logger as _get_log

        _get_log().info("Session ended by user (/quit)")
        print("\nGoodbye!")
        return "break"

    @staticmethod
    def _cmd_help(self, args: str) -> str:
        """Handler for /help [command]."""
        if args:
            # Detailed help for a specific command.
            # Try original case first (aliases are case-sensitive),
            # then fall back to lowercase for full command names.
            cmd = self._resolve(args)
            if cmd is None:
                cmd = self._resolve(args.lower())
            if cmd is None:
                print(f"Unknown command: {args}")
                print("Type /help for a list of commands.")
                return "continue"

            if console is not None:
                aliases = (
                    f"[dim]Aliases: {', '.join('/' + a for a in cmd.aliases)}[/dim]\n"
                    if cmd.aliases
                    else ""
                )
                body = f"[bold]/{cmd.name}[/bold]  {cmd.short_help}\n{aliases}\n{cmd.long_help}"
                console.print(Panel(body, border_style="cyan", padding=(1, 2)))
            else:
                aliases = (
                    f"  Aliases: {', '.join('/' + a for a in cmd.aliases)}\n" if cmd.aliases else ""
                )
                print(f"\n  /{cmd.name} — {cmd.short_help}")
                if aliases:
                    print(aliases)
                print(f"\n{cmd.long_help}")
            return "continue"

        # General help listing
        if console is not None:
            _help_rich(self)
        else:
            _help_plain(self)
        return "continue"

    @staticmethod
    def _cmd_info(self, _args: str) -> str:
        """Handler for /info."""
        cfg = self.config
        mm = self.memory_manager
        if not cfg or not mm:
            print("Session information not available.")
            return "continue"

        stats = mm.get_stats()
        msg_count = stats.get("total_messages", mm.get_message_count())

        try:
            provider_cfg = cfg.get_provider_config()
        except (ValueError, KeyError, AttributeError) as exc:
            if console is not None:
                console.print(f"[red]Provider configuration error:[/red] {exc}")
            else:
                print(f"Provider configuration error: {exc}")
            return "continue"
        model = cfg.model or provider_cfg.get_model()

        sp = self.system_prompt
        mm_mcp = self.mcp_manager
        if console is not None and Panel is not None:
            _info_rich(cfg, provider_cfg, model, stats, msg_count, sp, mm_mcp)
        else:
            _info_plain(cfg, provider_cfg, model, stats, msg_count, sp, mm_mcp)
        return "continue"

    @staticmethod
    def _cmd_tools(self, args: str) -> str:
        """Handler for /tools [search | enable <name> | disable <name>]."""
        reg = self.registry
        if not reg:
            print("Tool registry not available.")
            return "continue"

        available = self.available_tools

        if args.startswith("enable") and (len(args) == 6 or args[6] == " "):
            term = args[6:].strip()
            if not term:
                print("Usage: /tools enable <tool-name>")
                return "continue"
            if term in _session.denials:
                _session.denials.discard(term)
                print(f"Tool '{term}' re-enabled.")
            else:
                matches = [n for n in _session.denials if term in n]
                if len(matches) == 1:
                    _session.denials.discard(matches[0])
                    print(f"Tool '{matches[0]}' re-enabled.")
                elif len(matches) > 1:
                    print(f"Ambiguous: matches {matches}. Be more specific.")
                else:
                    print(f"Tool '{term}' is not disabled.")
            return "continue"

        if args.startswith("disable") and (len(args) == 7 or args[7] == " "):
            term = args[7:].strip()
            if not term:
                print("Usage: /tools disable <tool-name>")
                return "continue"
            all_known = (
                set(reg.tools.keys())
                | set(available.keys())
                | set(_session.all_tool_originals.keys())
            )
            if term in all_known:
                _session.denials.add(term)
                print(f"Tool '{term}' disabled for this session.")
            else:
                matches = [n for n in all_known if term in n]
                if len(matches) == 1:
                    _session.denials.add(matches[0])
                    print(f"Tool '{matches[0]}' disabled for this session.")
                elif len(matches) > 1:
                    print(f"Ambiguous: matches {matches}. Be more specific.")
                else:
                    print(f"Unknown tool '{term}'.")
            return "continue"

        if args.startswith("load") and (len(args) == 4 or args[4] == " "):
            term = args[4:].strip()
            if not term:
                print("Usage: /tools load <tool-name>")
                return "continue"
            if term in reg.tools:
                print(f"Tool '{term}' is already loaded.")
                return "continue"
            if term in _session.denials:
                print(f"Tool '{term}' is disabled. Use '/tools enable {term}' first.")
                return "continue"
            if term in available:
                return f"load_tool:{term}"
            matches = [n for n in available if term in n]
            if len(matches) == 1:
                return f"load_tool:{matches[0]}"
            if len(matches) > 1:
                print(f"Ambiguous: matches {matches}. Be more specific.")
            else:
                # Check if already active via substring
                active_matches = [n for n in reg.tools if term in n]
                if len(active_matches) == 1:
                    print(f"Tool '{active_matches[0]}' is already loaded.")
                else:
                    print(f"Unknown or unavailable tool '{term}'.")
            return "continue"

        active_names: set[str] = set(reg.tools.keys())
        all_names = sorted(active_names | set(available.keys()) | _session.denials)
        search_mode = False
        if args:
            search = args.lower()
            all_names = [n for n in all_names if search in n.lower()]
            if not all_names:
                print(f"No tools matching '{args}'.")
                return "continue"
            search_mode = True

        if console is not None and Table is not None:
            _tools_rich(reg, all_names, search_mode, args, available, active_names)
        else:
            _tools_plain(reg, all_names, search_mode, args, available, active_names)
        return "continue"

    @staticmethod
    def _cmd_clear(self, _args: str) -> str:
        """Handler for /clear."""
        mm = self.memory_manager
        if not mm:
            print("Memory manager not available.")
            return "continue"

        stats = mm.get_stats()
        count = stats.get("total_messages", mm.get_message_count())
        mm.clear()
        mm.save()
        if console is not None:
            console.print(f"[green]✓ Cleared [bold]{count}[/bold] messages from memory.[/green]")
        else:
            print(f"✓ Cleared {count} messages from memory.")
        return "continue"

    @staticmethod
    def _cmd_think(self, args: str) -> str:
        """Handler for /think <task>.

        Returns a ``deep_think:<task>`` signal so the main loop can
        execute the hybrid gather → analyze → synthesize pipeline
        (the handler itself doesn't have access to the agent/tools).
        """
        if not args:
            if console is not None:
                console.print(
                    "[dim]Usage:[/dim] [bold]/think[/bold] <task description>\n"
                    "[dim]  Example: /think Design a caching strategy for microservices[/dim]"
                )
            else:
                print("Usage: /think <task description>")
                print("  Example: /think Design a caching strategy for microservices")
            return "continue"

        try:
            from src.tools.deep_think import deep_think  # noqa: F401

            del deep_think  # only checking availability
        except ImportError:
            if console is not None:
                console.print("[red]Deep Think tool is not available.[/red]")
            else:
                print("Deep Think tool is not available.")
            return "continue"

        return f"deep_think:{args}"

    @staticmethod
    def _cmd_delegate(self, args: str) -> str:
        """Handler for /delegate <task>.

        Returns a ``delegate:<task>`` signal so the main loop can
        execute the forced delegation pipeline.
        """
        if not args:
            if console is not None:
                console.print(
                    "[dim]Usage:[/dim] [bold]/delegate[/bold] <task description>\n"
                    "[dim]  Example: /delegate Research top 10 AI companies and their market cap[/dim]"
                )
            else:
                print("Usage: /delegate <task description>")
                print("  Example: /delegate Research top 10 AI companies and their market cap")
            return "continue"

        try:
            from src.tools.delegate import delegate_task  # noqa: F401

            del delegate_task
        except ImportError:
            if console is not None:
                console.print("[red]Delegate tool is not available.[/red]")
            else:
                print("Delegate tool is not available.")
            return "continue"

        # Check if delegation is enabled in config
        cfg = self.config
        if cfg and not getattr(cfg, "delegate_enabled", True):
            if console is not None:
                console.print("[yellow]Delegation is disabled in configuration.[/yellow]")
            else:
                print("Delegation is disabled in configuration.")
            return "continue"

        return f"delegate:{args}"

    @staticmethod
    def _cmd_mode(self, args: str) -> str:
        """Handler for /mode [name]."""
        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        # Defaults from each memory mode class
        _DEFAULT_WM: dict[str, int] = {
            "conversation": 25,
            "code": 30,
            "reasoning": 40,
        }

        _VALID_MODES = {
            "conversation": "General chat, Q&A, research",
            "code": "Programming, debugging, file tracking",
            "reasoning": "Strategic planning, decision tracking",
        }

        # ── Switch mode if an argument was given ──────────────────
        if args:
            target = args.strip().lower()
            if target not in _VALID_MODES:
                valid = ", ".join(_VALID_MODES)
                if console is not None:
                    console.print(f"[red]Unknown mode:[/red] [bold]{target}[/bold]")
                    console.print(f"[dim]Available modes: {valid}[/dim]")
                else:
                    print(f"Unknown mode: {target}")
                    print(f"Available modes: {valid}")
                return "continue"
            if target == cfg.memory_mode:
                if console is not None:
                    console.print(f"[dim]Already in [bold]{target}[/bold] mode.[/dim]")
                else:
                    print(f"Already in {target} mode.")
                return "continue"
            # Signal the main loop to perform the actual switch
            return f"switch_mode:{target}"

        # ── Display modes (no argument) ───────────────────────────
        # Populate working memory sizes — use live value for the active mode
        wm_sizes: dict[str, int | None] = dict(_DEFAULT_WM)
        mm = self.memory_manager
        if mm:
            stats = mm.get_stats()
            live_wm = stats.get("working_memory_size")
            if live_wm is not None:
                wm_sizes[cfg.memory_mode] = live_wm

        if console is not None and Panel is not None:
            _mode_rich(cfg, _VALID_MODES, wm_sizes)
        else:
            _mode_plain(cfg, _VALID_MODES, wm_sizes)
        return "continue"

    @staticmethod
    def _cmd_model(self, args: str) -> str:
        """Handler for /model [name]."""
        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        if args:
            target = args.strip()
            # Signal the main loop to perform the model switch
            return f"switch_model:{target}"

        # ── No argument: show current model + available aliases ────
        try:
            provider_cfg = cfg.get_provider_config()
        except (ValueError, KeyError, AttributeError):
            provider_cfg = None

        current = cfg.model or (provider_cfg.get_model() if provider_cfg else "unknown")
        models = cfg.models or {}

        if console is not None:
            lines_out: list[str] = []
            lines_out.append(f"  [bold green]{current}[/bold green] [green]● active[/green]")
            if models:
                lines_out.append("")
                for mname, mcfg in models.items():
                    detail = f"{mcfg.provider}/{mcfg.model}"
                    is_current = mcfg.provider == cfg.provider and mcfg.model == cfg.model
                    if is_current:
                        name_fmt = f"[bold green]{mname:<16s}[/bold green]"
                        marker = " [green]● active[/green]"
                    else:
                        name_fmt = f"[bold]{mname:<16s}[/bold]"
                        marker = ""
                    lines_out.append(f"  {name_fmt} [dim]{detail}[/dim]{marker}")
            lines_out.append("")
            lines_out.append(
                "[dim]Switch: [bold]/model[/bold] <name>   " "(e.g. [bold]/model fast[/bold])[/dim]"
            )
            console.print(
                Panel(
                    "\n".join(lines_out),
                    title="Model",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n  Current model: {current}")
            if models:
                print()
                for mname, mcfg in models.items():
                    detail = f"{mcfg.provider}/{mcfg.model}"
                    marker = (
                        " ● active"
                        if (mcfg.provider == cfg.provider and mcfg.model == cfg.model)
                        else ""
                    )
                    print(f"    {mname:<16s} {detail}{marker}")
            print("\n  Switch: /model <name>   (e.g. /model fast)")
            print()
        return "continue"

    @staticmethod
    def _cmd_provider(self, args: str) -> str:
        """Handler for /provider [name]."""
        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        if args:
            target = args.strip()
            available = cfg.list_providers()
            if target not in available:
                if console is not None:
                    console.print(f"[red]Unknown provider:[/red] [bold]{target}[/bold]")
                    console.print(f"[dim]Available: {', '.join(available)}[/dim]")
                else:
                    print(f"Unknown provider: {target}")
                    print(f"Available: {', '.join(available)}")
                return "continue"
            if target == cfg.provider:
                if console is not None:
                    console.print(f"[dim]Already using provider [bold]{target}[/bold].[/dim]")
                else:
                    print(f"Already using provider {target}.")
                return "continue"
            return f"switch_provider:{target}"

        # ── No argument: show current provider + available ones ────
        available = cfg.list_providers()
        if console is not None:
            lines_out = []
            for pname in available:
                try:
                    pcfg = cfg.get_provider_config(pname)
                    ptype = pcfg.type
                    pmodel = pcfg.get_model()
                    detail = f"{ptype}, model: {pmodel}"
                except (ValueError, KeyError):
                    detail = "unconfigured"
                is_current = pname == cfg.provider
                if is_current:
                    name_fmt = f"[bold green]{pname:<20s}[/bold green]"
                    marker = " [green]● active[/green]"
                else:
                    name_fmt = f"[bold]{pname:<20s}[/bold]"
                    marker = ""
                lines_out.append(f"  {name_fmt} [dim]{detail}[/dim]{marker}")
            lines_out.append("")
            lines_out.append(
                "[dim]Switch: [bold]/provider[/bold] <name>   "
                "(e.g. [bold]/provider ollama[/bold])[/dim]"
            )
            body = "\n".join(lines_out)
            console.print()
            console.print(Panel(body, title="Providers", border_style="cyan", padding=(1, 2)))
            console.print()
        else:
            print("\n  Providers:")
            for pname in available:
                marker = " ● active" if pname == cfg.provider else ""
                try:
                    pcfg = cfg.get_provider_config(pname)
                    detail = f"{pcfg.type}, model: {pcfg.get_model()}"
                except (ValueError, KeyError):
                    detail = "unconfigured"
                print(f"    {pname:<20s} {detail}{marker}")
            print("\n  Switch: /provider <name>   (e.g. /provider ollama)")
            print()
        return "continue"

    @staticmethod
    def _cmd_session_switch(self, args: str) -> str:
        """Handler for /session [id]."""
        cfg = self.config
        mm = self.memory_manager
        if not cfg or not mm:
            print("Session info not available.")
            return "continue"

        if args:
            target = args.strip()
            if target == cfg.session:
                if console is not None:
                    console.print(f"[dim]Already in session [bold]{target}[/bold].[/dim]")
                else:
                    print(f"Already in session {target}.")
                return "continue"
            return f"switch_session:{target}"

        # No argument: show info (delegate to existing display logic)
        stats = mm.get_stats()
        msg_count = stats.get("total_messages", mm.get_message_count())

        if console is not None and Panel is not None:
            _session_rich(cfg, stats, msg_count)
        else:
            _session_plain(cfg, msg_count)
        return "continue"

    @staticmethod
    def _cmd_debug(self, _args: str) -> str:
        """Handler for /debug — toggle debug mode."""
        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        cfg.debug = not cfg.debug
        if cfg.debug:
            # Auto-enable logging and verbose when debug turns on
            if cfg.log_file is None:
                cfg.log_file = ""
            cfg.verbose = True
            setup_logging(log_file=cfg.log_file, debug=True, verbose=True)
            if console is not None:
                console.print(
                    "[green]Debug mode [bold]ON[/bold][/green] "
                    "[dim](logging + verbose enabled)[/dim]"
                )
            else:
                print("Debug mode ON (logging + verbose enabled)")
        else:
            setup_logging(log_file=cfg.log_file, debug=False, verbose=cfg.verbose)
            if console is not None:
                console.print("[yellow]Debug mode [bold]OFF[/bold][/yellow]")
            else:
                print("Debug mode OFF")
        # Signal main loop to rebuild callbacks
        return "rebuild_callbacks"

    @staticmethod
    def _cmd_verbose(self, _args: str) -> str:
        """Handler for /verbose — toggle verbose logging."""
        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        cfg.verbose = not cfg.verbose
        setup_logging(log_file=cfg.log_file, debug=cfg.debug, verbose=cfg.verbose)
        state = "ON" if cfg.verbose else "OFF"
        if console is not None:
            color = "green" if cfg.verbose else "yellow"
            console.print(f"[{color}]Verbose logging [bold]{state}[/bold][/{color}]")
        else:
            print(f"Verbose logging {state}")
        # Signal main loop to rebuild callbacks with new verbose setting
        return "rebuild_callbacks"

    @staticmethod
    def _cmd_approve(self, _args: str) -> str:
        """Handler for /approve — toggle auto-approval for tools."""
        _session.no_confirm = not _session.no_confirm
        state = "ON" if _session.no_confirm else "OFF"

        reg = self.registry
        if _session.no_confirm and reg:
            # Auto-approve all confirmation-requiring tools
            for name in reg.list_tools():
                if reg.requires_confirmation(name):
                    self.approvals.add(name)
        elif not _session.no_confirm:
            # Revoke auto-approvals (tools will prompt again)
            self.approvals.clear()
            _session.deny_all = False

        if console is not None:
            color = "yellow" if _session.no_confirm else "green"
            desc = (
                "all tools auto-approved"
                if _session.no_confirm
                else "tools will prompt for confirmation"
            )
            console.print(
                f"[{color}]Auto-approve [bold]{state}[/bold][/{color}] " f"[dim]({desc})[/dim]"
            )
        else:
            desc = (
                "all tools auto-approved"
                if _session.no_confirm
                else "tools will prompt for confirmation"
            )
            print(f"Auto-approve {state} ({desc})")
        return "continue"

    @staticmethod
    def _cmd_optimizer(self, args: str) -> str:
        """Handler for /optimizer — toggle or force-optimize a prompt."""
        if args:
            return f"optimize:{args}"

        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        cfg.prompt_optimizer = not cfg.prompt_optimizer
        state = "ON" if cfg.prompt_optimizer else "OFF"
        if console is not None:
            color = "green" if cfg.prompt_optimizer else "yellow"
            console.print(f"[{color}]Prompt optimizer [bold]{state}[/bold][/{color}]")
        else:
            print(f"Prompt optimizer {state}")
        return "continue"

    @staticmethod
    def _cmd_setup(self, _args: str) -> str:
        """Handler for /setup — launch the interactive setup wizard."""
        return "run_setup"

    @staticmethod
    def _cmd_system_prompt(self, _args: str) -> str:
        """Handler for /system_prompt — display the full system prompt."""
        sp = self.system_prompt
        if not sp:
            print("System prompt not available.")
            return "continue"

        sp_chars = len(sp)
        sp_tokens = sp_chars // 4  # rough estimate

        if console is not None and Panel is not None:
            escaped = sp.replace("[", "\\[")
            body = f"[dim]{escaped}[/dim]"
            title = f"System Prompt  ~{sp_tokens:,} tokens ({sp_chars:,} chars)"
            console.print()
            console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))
            console.print()
        else:
            print(f"\n  System Prompt  ~{sp_tokens:,} tokens ({sp_chars:,} chars)")
            print("  " + "-" * 50)
            for line in sp.split("\n"):
                print(f"  {line}")
            print()
        return "continue"

    @staticmethod
    def _cmd_mcp(self, args: str) -> str:
        """Handler for /mcp [restart [server-name]] — list or restart MCP servers."""
        mgr = self.mcp_manager
        if mgr is None:
            print("No MCP servers configured.")
            return "continue"

        if args.startswith("restart") and (len(args) == 7 or args[7] == " "):
            target = args[len("restart") :].strip() or None
            _reg = self.registry
            _builtin_names: set[str] = set()
            if _reg is not None:
                for _tname in list(_reg.tools):
                    if _reg.tool_metadata.get(_tname, {}).get("source") != "mcp":
                        _builtin_names.add(_tname)
                for _tname in list(self.available_tools):
                    if _reg.tool_metadata.get(_tname, {}).get("source") != "mcp":
                        _builtin_names.add(_tname)
            mgr.restart(target, builtin_tool_names=_builtin_names or None)

            reg = self.registry
            if reg is not None:
                # Track which tools were active vs on-demand before purge
                previously_active: set[str] = set()
                for tname in list(reg.tools):
                    meta = reg.tool_metadata.get(tname, {})
                    if meta.get("source") == "mcp":
                        srv = meta.get("server", "")
                        if target is None or srv == target:
                            previously_active.add(tname)
                            del reg.tools[tname]
                            reg.tool_metadata.pop(tname, None)
                            _session.all_tool_originals.pop(tname, None)
                            _session.all_tool_descriptions.pop(tname, None)
                for tname in list(self.available_tools):
                    meta = reg.tool_metadata.get(tname, {})
                    if meta.get("source") == "mcp":
                        srv = meta.get("server", "")
                        if target is None or srv == target:
                            self.available_tools.pop(tname, None)
                            reg.tool_metadata.pop(tname, None)
                            _session.all_tool_originals.pop(tname, None)
                            _session.all_tool_descriptions.pop(tname, None)

                new_tools = mgr.get_langchain_tools(server_name=target)
                for tname, tool_obj in new_tools.items():
                    meta = {
                        "requires_confirmation": (tool_obj.metadata or {}).get(
                            "requires_confirmation", True
                        ),
                        "source": "mcp",
                        "server": (tool_obj.metadata or {}).get("server", ""),
                    }
                    reg.tool_metadata[tname] = meta
                    _session.all_tool_originals[tname] = tool_obj
                    desc = getattr(tool_obj, "description", "") or ""
                    short = desc.split(". ")[0].split(".\n")[0]
                    if len(short) > 120:
                        short = short[:117] + "..."
                    _session.all_tool_descriptions[tname] = short
                    # Restore to same pool: active stays active, rest goes on-demand
                    if tname in previously_active:
                        reg.tools[tname] = tool_obj
                    else:
                        self.available_tools[tname] = tool_obj

            if target:
                print(f"MCP server '{target}' restarted.")
            else:
                print("All MCP servers restarted.")
            return "continue"

        server_info = mgr.get_server_info()
        if not server_info:
            print("No MCP servers configured.")
            return "continue"

        if console is not None and Panel is not None and Table is not None:
            tbl = Table(
                show_header=True,
                header_style="bold cyan",
                box=rich_box.SIMPLE if rich_box else None,
            )
            tbl.add_column("Server", style="bold")
            tbl.add_column("Status")
            tbl.add_column("Transport")
            tbl.add_column("Endpoint", overflow="fold")
            tbl.add_column("Tools", justify="right")
            for srv in server_info:
                status = (
                    "[bright_green]connected[/bright_green]"
                    if srv["connected"]
                    else "[red]disconnected[/red]"
                )
                tool_names = ", ".join(srv["tools"]) if srv["tools"] else "[dim]none[/dim]"
                tbl.add_row(
                    srv["name"],
                    status,
                    srv["transport"],
                    srv["endpoint"],
                    str(srv["tool_count"]),
                )
                if srv["tools"]:
                    tbl.add_row("", "", "", f"  [dim]{tool_names}[/dim]", "")
            console.print()
            console.print(Panel(tbl, title="MCP Servers", border_style="cyan", padding=(0, 1)))
            console.print()
        else:
            print("\n  MCP Servers")
            print("  " + "-" * 50)
            for srv in server_info:
                status = "connected" if srv["connected"] else "disconnected"
                print(f"  {srv['name']}  [{status}]  {srv['transport']}  {srv['endpoint']}")
                if srv["tools"]:
                    print(f"    Tools ({srv['tool_count']}): {', '.join(srv['tools'])}")
            print()
        return "continue"

    @staticmethod
    def _cmd_paste(_self, _args: str) -> str:
        """Handler for /paste (normally intercepted in the main loop)."""
        # /paste is intercepted before dispatch() so this handler is a
        # documentation-only fallback.  Print instructions just in case.
        if console is not None:
            console.print(
                "[dim]Multi-line paste mode:[/dim]\n"
                '  Type or paste text, then enter [yellow bold]"""[/yellow bold]'
                " on a new line to send.\n"
                '  Tip: you can also start any message with [yellow bold]"""'
                "[/yellow bold] to enter this mode."
            )
        else:
            print("\nMulti-line paste mode:")
            print('  Type or paste text, then enter """ on a new line to send.')
            print('  Tip: you can also start any message with """ to enter this mode.')
        return "continue"


def _help_rich(self_reg: SlashCommandRegistry) -> None:
    """Render the /help listing using Rich panels and tables."""
    # Group commands by category
    categories = [
        (
            "Session & Config",
            ["info", "session", "mode", "model", "provider", "setup"],
        ),
        (
            "Tools & Reasoning",
            ["tools", "think", "delegate", "approve", "optimizer"],
        ),
        (
            "Logging",
            ["debug", "verbose"],
        ),
        (
            "Input & Other",
            ["paste", "clear", "help", "quit"],
        ),
    ]

    lines: list[str] = []
    for cat_name, cmd_names in categories:
        lines.append(f"[bold cyan]{cat_name}[/bold cyan]")
        for name in cmd_names:
            cmd = self_reg._commands.get(name)
            if cmd is None:
                continue
            aliases = ""
            if cmd.aliases:
                aliases = f" [dim]({', '.join('/' + a for a in cmd.aliases)})[/dim]"
            lines.append(f"  [bold]/{cmd.name:<10s}[/bold] {cmd.short_help}{aliases}")
        lines.append("")

    lines.append("[dim]Type [bold]/help <command>[/bold] for detailed information.[/dim]")
    lines.append(
        '[dim]Use [bold]"""[/bold] or [bold]/paste[/bold] to enter multi-line input mode.[/dim]'
    )

    body = "\n".join(lines)
    console.print(Panel(body, title="Commands", border_style="cyan", padding=(1, 2)))  # type: ignore[union-attr]


def _help_plain(self_reg: SlashCommandRegistry) -> None:
    """Render the /help listing as plain text (no Rich)."""
    print("\nAvailable commands:\n")
    for cmd in self_reg._commands.values():
        alias_str = ""
        if cmd.aliases:
            alias_str = f" ({', '.join('/' + a for a in cmd.aliases)})"
        print(f"  /{cmd.name:<12s} {cmd.short_help}{alias_str}")
    print("\n  Type /help <command> for detailed information.")
    print('  Use """ or /paste to enter multi-line input mode.\n')


def _tool_desc(tool: Any, max_len: int = 60) -> str:
    """Extract a short description from a tool object."""
    desc = getattr(tool, "description", "") or ""
    # Take just the first sentence / line
    first = desc.split(". ")[0].split(".\n")[0].split("\n")[0]
    if len(first) > max_len:
        first = first[: max_len - 1] + "\u2026"
    return first


def _tool_status_tag(name: str, reg: Any, rich_mode: bool = False, on_demand: bool = False) -> str:
    """Return a status tag string for a tool.

    Possible tags: disabled, loaded, on-demand, confirm, auto-approved, or empty.

    Args:
        rich_mode: If True, wrap in Rich markup colours.
        on_demand: If True, tool is available but not yet loaded.
    """
    mcp_tag = ""
    if hasattr(reg, "is_mcp_tool") and reg.is_mcp_tool(name):
        mcp_tag = "[dim cyan]\\[mcp][/dim cyan] " if rich_mode else "[mcp] "

    if name in _session.denials:
        if rich_mode:
            return mcp_tag + "[red]\\[disabled] [/red]"
        return mcp_tag + "[disabled] "
    if name in _session.loaded_tools:
        if rich_mode:
            return mcp_tag + "[bright_green]\\[loaded]   [/bright_green]"
        return mcp_tag + "[loaded]   "
    if on_demand:
        if rich_mode:
            return mcp_tag + "[dim]\\[on-demand][/dim]"
        return mcp_tag + "[on-demand]"
    if not reg.requires_confirmation(name):
        return mcp_tag
    if rich_mode:
        if _session.no_confirm:
            return mcp_tag + "[green]\\[auto-approved][/green]"
        return mcp_tag + "[yellow]\\[confirm][/yellow]"
    return mcp_tag + ("[auto-approved]" if _session.no_confirm else "[confirm]")


def _categorize_tools(
    tool_names: list[str],
) -> list[tuple[str, list[str]]]:
    """Group tool names by category, preserving category order.

    Returns a list of (category_name, [tool_names]) tuples.
    Tools not in any known category go into "Other".
    """
    # Collect tools into their categories (preserve order of _TOOL_CATEGORIES)
    cat_tools: dict[str, list[str]] = {}
    uncategorized: list[str] = []
    tool_set = set(tool_names)

    for cat_name, members in _TOOL_CATEGORIES.items():
        matched = [n for n in members if n in tool_set]
        if matched:
            cat_tools[cat_name] = matched

    # Find tools not in any category
    known = {n for names in _TOOL_CATEGORIES.values() for n in names}
    uncategorized = sorted(n for n in tool_names if n not in known)

    result: list[tuple[str, list[str]]] = []
    for cat_name in _TOOL_CATEGORIES:
        if cat_name in cat_tools:
            result.append((cat_name, cat_tools[cat_name]))
    if uncategorized:
        result.append(("Other", uncategorized))
    return result


def _tools_rich(
    reg: Any,
    tool_names: list[str],
    search_mode: bool,
    search_term: str | None,
    available_tools: dict[str, Any] | None = None,
    active_names: set[str] | None = None,
) -> None:
    """Render /tools output using Rich tables inside a panel."""
    if console is None or Table is None:  # pragma: no cover – caller checks
        return

    groups = _categorize_tools(tool_names)
    total = len(tool_names)

    # Calculate available description width from terminal size.
    # Panel uses: 2 border + 2*2 padding = 6 chars on each side → 6+6 not quite;
    # Rich Panel: │ + 2 padding + content + 2 padding + │  = 1+2+…+2+1 = 6.
    # Each line: "  " indent (2) + name col (28) + "  " gap (2) = 32 fixed.
    # Tag (if present) adds ~10-16 chars but we size for the common (no-tag) case
    # and let tagged lines wrap naturally if needed.
    term_width = console.width or 100
    panel_overhead = 6  # │ + 2 padding each side + │
    line_indent = 2 + 28 + 2  # "  " + name column + "  " before desc
    desc_width = max(30, term_width - panel_overhead - line_indent)

    lines: list[str] = []
    for cat_name, names in groups:
        lines.append(f"[bold cyan]{cat_name}[/bold cyan]")
        for name in names:
            tool = reg.tools.get(name) or (available_tools.get(name) if available_tools else None)
            is_on_demand = name not in (active_names or set()) and name not in _session.denials
            tag = _tool_status_tag(name, reg, rich_mode=True, on_demand=is_on_demand)
            # Shrink description to fit when a status tag is present.
            # Visible widths: "[disabled] " = 11, "[loaded]" = 8, "[on-demand]" = 11,
            # "[confirm]" = 9, "[auto-approved]" = 15.
            # Add 1 (space before tag) + visible + 4 (gap to desc).
            tag_width = 0
            if tag:
                if "auto" in tag:
                    tag_width = 20
                elif "on-demand" in tag:
                    tag_width = 16
                elif "disabled" in tag:
                    tag_width = 16
                elif "loaded" in tag:
                    tag_width = 16
                else:
                    tag_width = 14
            avail = max(20, desc_width - tag_width)
            desc = _tool_desc(tool, max_len=avail) if tool else ""
            # Build formatted line
            parts = [f"  [bold]{name:<28s}[/bold]"]
            if tag:
                parts.append(f" {tag}")
            if desc:
                parts.append(f"    [dim]{desc}[/dim]")
            lines.append("".join(parts))
        lines.append("")  # blank line between categories

    if search_mode:
        title = f"Tools matching '{search_term}' ({total})"
    else:
        active_count = sum(1 for n in tool_names if active_names and n in active_names)
        demand_count = sum(
            1
            for n in tool_names
            if available_tools and n in available_tools and n not in _session.denials
        )
        disabled_count = sum(1 for n in tool_names if n in _session.denials)
        parts_title: list[str] = []
        if active_count:
            parts_title.append(f"{active_count} active")
        if demand_count:
            parts_title.append(f"{demand_count} on-demand")
        if disabled_count:
            parts_title.append(f"{disabled_count} disabled")
        title = f"Tools ({', '.join(parts_title)})" if parts_title else f"Tools ({total})"

    body = "\n".join(lines).rstrip()
    console.print()
    console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))
    console.print()


def _tools_plain(
    reg: Any,
    tool_names: list[str],
    search_mode: bool,
    search_term: str | None,
    available_tools: dict[str, Any] | None = None,
    active_names: set[str] | None = None,
) -> None:
    """Render /tools output as plain text (no Rich)."""
    import shutil

    groups = _categorize_tools(tool_names)
    total = len(tool_names)

    # Calculate description width from terminal size.
    # Each line: "    " (4) + name col (28) + "  " (2) = 34 fixed.
    term_width = shutil.get_terminal_size((100, 24)).columns
    desc_width = max(30, term_width - 34)

    if search_mode:
        print(f"\n  Tools matching '{search_term}' ({total}):\n")
    else:
        active_count = sum(1 for n in tool_names if active_names and n in active_names)
        demand_count = sum(
            1
            for n in tool_names
            if available_tools and n in available_tools and n not in _session.denials
        )
        disabled_count = sum(1 for n in tool_names if n in _session.denials)
        parts_title: list[str] = []
        if active_count:
            parts_title.append(f"{active_count} active")
        if demand_count:
            parts_title.append(f"{demand_count} on-demand")
        if disabled_count:
            parts_title.append(f"{disabled_count} disabled")
        summary = f"({', '.join(parts_title)})" if parts_title else f"({total})"
        print(f"\n  Tools {summary}:\n")

    for cat_name, names in groups:
        print(f"  [{cat_name}]")
        for name in names:
            tool = reg.tools.get(name) or (available_tools.get(name) if available_tools else None)
            is_on_demand = name not in (active_names or set()) and name not in _session.denials
            tag = _tool_status_tag(name, reg, on_demand=is_on_demand)
            tag_width = 0
            if tag:
                if "auto" in tag:
                    tag_width = 20
                elif "on-demand" in tag:
                    tag_width = 16
                elif "disabled" in tag:
                    tag_width = 16
                elif "loaded" in tag:
                    tag_width = 16
                else:
                    tag_width = 14
            avail = max(20, desc_width - tag_width)
            desc = _tool_desc(tool, max_len=avail) if tool else ""
            line = f"    {name:<28s}"
            if tag:
                line += f" {tag}"
            if desc:
                line += f"    {desc}"
            print(line)
        print()


def _mode_rich(cfg: Any, modes: dict[str, str], wm_sizes: dict[str, int | None]) -> None:
    """Render /mode output using Rich."""
    if console is None or Panel is None:  # pragma: no cover
        return

    lines: list[str] = []
    for name, desc in modes.items():
        is_current = name == cfg.memory_mode
        wm = wm_sizes.get(name)

        if is_current:
            marker = " [green]● active[/green]"
            name_fmt = f"[bold green]{name:<15s}[/bold green]"
        else:
            marker = ""
            name_fmt = f"[bold]{name:<15s}[/bold]"

        wm_info = f"  [dim]({wm} messages)[/dim]" if wm is not None else ""
        lines.append(f"  {name_fmt} [dim]{desc}[/dim]{wm_info}{marker}")

    lines.append("")
    lines.append("[dim]Switch: [bold]/mode[/bold] <name>   (e.g. [bold]/mode code[/bold])[/dim]")

    body = "\n".join(lines)
    console.print()
    console.print(Panel(body, title="Memory Modes", border_style="cyan", padding=(1, 2)))
    console.print()


def _mode_plain(cfg: Any, modes: dict[str, str], wm_sizes: dict[str, int | None]) -> None:
    """Render /mode output as plain text."""
    print("\n  Memory Modes\n")
    for name, desc in modes.items():
        is_current = name == cfg.memory_mode
        marker = " ● active" if is_current else ""
        wm = wm_sizes.get(name)
        wm_info = f"  ({wm} messages)" if wm is not None else ""
        print(f"    {name:<15s} {desc}{wm_info}{marker}")
    print("\n  Switch: /mode <name>   (e.g. /mode code)")
    print()


def _info_rich(
    cfg: Any,
    provider_cfg: Any,
    model: str,
    stats: dict,
    msg_count: int,
    system_prompt: str | None = None,
    mcp_manager: Any = None,
) -> None:
    """Render /info output using Rich."""
    if console is None or Panel is None:  # pragma: no cover
        return

    # ── Connection section ────────────────────────────────────
    lines: list[str] = []
    lines.append("[bold cyan]Connection[/bold cyan]")
    lines.append(f"  [bold]Provider[/bold]      {cfg.provider} [dim]({provider_cfg.type})[/dim]")
    lines.append(f"  [bold]Model[/bold]         {model}")
    if provider_cfg.num_ctx:
        lines.append(f"  [bold]Context size[/bold]  {provider_cfg.num_ctx:,} tokens")
    if system_prompt:
        sp_chars = len(system_prompt)
        sp_tokens = sp_chars // 4  # rough estimate
        lines.append(
            f"  [bold]System prompt[/bold] ~{sp_tokens:,} tokens [dim]({sp_chars:,} chars)[/dim]"
        )
    if mcp_manager is not None:
        server_info = mcp_manager.get_server_info()
        connected = sum(1 for s in server_info if s["connected"])
        total_tools = sum(s["tool_count"] for s in server_info)
        lines.append(f"  [bold]MCP servers[/bold]   {connected} connected ({total_tools} tools)")

    # ── Memory section ────────────────────────────────────────
    lines.append("")
    lines.append("[bold cyan]Memory[/bold cyan]")
    lines.append(f"  [bold]Mode[/bold]          {cfg.memory_mode}")
    lines.append(f"  [bold]Session[/bold]       {cfg.session}")
    lines.append(f"  [bold]Messages[/bold]      {msg_count}")

    wm_size = stats.get("working_memory_size")
    if wm_size is not None:
        lines.append(f"  [bold]Working mem[/bold]   {wm_size} messages")

    # Mode-specific extras
    extras: list[str] = []
    if stats.get("entity_count"):
        extras.append(f"Entities: {stats['entity_count']}")
    if stats.get("files_tracked"):
        extras.append(f"Files tracked: {stats['files_tracked']}")
    if stats.get("decision_count"):
        extras.append(f"Decisions: {stats['decision_count']}")
    if stats.get("has_summary"):
        extras.append("Summary: yes")
    if extras:
        lines.append(f"  [dim]{' · '.join(extras)}[/dim]")

    body = "\n".join(lines)
    console.print()
    console.print(Panel(body, title="Session Information", border_style="cyan", padding=(1, 2)))
    console.print()


def _info_plain(
    cfg: Any,
    provider_cfg: Any,
    model: str,
    stats: dict,
    msg_count: int,
    system_prompt: str | None = None,
    mcp_manager: Any = None,
) -> None:
    """Render /info output as plain text."""
    print("\n  Session Information")
    print("  " + "-" * 38)
    print(f"  Provider      {cfg.provider} ({provider_cfg.type})")
    print(f"  Model         {model}")
    if provider_cfg.num_ctx:
        print(f"  Context size  {provider_cfg.num_ctx:,} tokens")
    if system_prompt:
        sp_chars = len(system_prompt)
        sp_tokens = sp_chars // 4
        print(f"  System prompt ~{sp_tokens:,} tokens ({sp_chars:,} chars)")
    if mcp_manager is not None:
        server_info = mcp_manager.get_server_info()
        connected = sum(1 for s in server_info if s["connected"])
        total_tools = sum(s["tool_count"] for s in server_info)
        print(f"  MCP servers   {connected} connected ({total_tools} tools)")
    print()
    print(f"  Mode          {cfg.memory_mode}")
    print(f"  Session       {cfg.session}")
    print(f"  Messages      {msg_count}")
    wm_size = stats.get("working_memory_size")
    if wm_size is not None:
        print(f"  Working mem   {wm_size} messages")
    extras: list[str] = []
    if stats.get("entity_count"):
        extras.append(f"Entities: {stats['entity_count']}")
    if stats.get("files_tracked"):
        extras.append(f"Files tracked: {stats['files_tracked']}")
    if stats.get("decision_count"):
        extras.append(f"Decisions: {stats['decision_count']}")
    if extras:
        print(f"  {' · '.join(extras)}")
    print()


def _session_rich(cfg: Any, stats: dict, msg_count: int) -> None:
    """Render /session output using Rich."""
    if console is None or Panel is None:  # pragma: no cover
        return

    lines: list[str] = []
    lines.append(f"  [bold]Session[/bold]   {cfg.session}")
    lines.append(f"  [bold]Mode[/bold]      {cfg.memory_mode}")
    lines.append(f"  [bold]Messages[/bold]  {msg_count}")

    wm_size = stats.get("working_memory_size")
    if wm_size is not None:
        bar_max = wm_size
        bar_fill = min(msg_count, bar_max)
        pct = int(bar_fill / bar_max * 100) if bar_max else 0
        filled = int(bar_fill / bar_max * 20) if bar_max else 0
        bar = "█" * filled + "░" * (20 - filled)
        color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
        lines.append(f"  [bold]Context[/bold]   [{color}]{bar}[/{color}] {pct}% of {bar_max}")

    body = "\n".join(lines)
    console.print()
    console.print(Panel(body, title="Session", border_style="cyan", padding=(1, 2)))
    console.print()


def _session_plain(cfg: Any, msg_count: int) -> None:
    """Render /session output as plain text."""
    print(f"\n  Session   {cfg.session}")
    print(f"  Mode      {cfg.memory_mode}")
    print(f"  Messages  {msg_count}")
    print()


def _build_slash_commands() -> SlashCommandRegistry:
    """Create and populate the slash command registry."""
    reg = SlashCommandRegistry()

    reg.register(
        SlashCommand(
            name="help",
            handler=SlashCommandRegistry._cmd_help,
            short_help="Show available commands",
            long_help=(
                "Usage: /help [command]\n\n"
                "Without arguments, lists all available commands grouped\n"
                "by category. With a command name, shows detailed help.\n\n"
                "Examples:\n"
                "  /help          List all commands\n"
                "  /help info     Detailed help for /info\n"
                "  /help think    Detailed help for /think"
            ),
            aliases=["h", "?"],
        )
    )

    reg.register(
        SlashCommand(
            name="quit",
            handler=SlashCommandRegistry._cmd_quit,
            short_help="Exit the session",
            long_help=(
                "Usage: /quit\n\n"
                "Ends the current session and exits the program.\n"
                "Conversation history is preserved and will be restored\n"
                "when you resume the same session.\n\n"
                "The bare commands 'exit', 'quit', and 'q' (without /)\n"
                "also work for backward compatibility."
            ),
            aliases=["exit", "q"],
        )
    )

    reg.register(
        SlashCommand(
            name="info",
            handler=SlashCommandRegistry._cmd_info,
            short_help="Show session information",
            long_help=(
                "Usage: /info\n\n"
                "Displays current session information:\n"
                "  - Provider and model\n"
                "  - Context window size (num_ctx)\n"
                "  - Memory mode and working memory size\n"
                "  - Session ID and message count\n"
                "  - Mode-specific tracking (entities, files, decisions)"
            ),
            aliases=["i"],
        )
    )

    reg.register(
        SlashCommand(
            name="tools",
            handler=SlashCommandRegistry._cmd_tools,
            short_help="List / manage tools",
            long_help=(
                "Usage: /tools [search | load <name> | enable <name> | disable <name>]\n\n"
                "Without arguments, lists all tools grouped by category\n"
                "with status tags:\n"
                "  [confirm]        Requires user approval before running\n"
                "  [auto-approved]  Confirmation skipped (/approve active)\n"
                "  [loaded]         Dynamically loaded during the session\n"
                "  [on-demand]      Available but not yet loaded\n"
                "  [disabled]       Blocked — will not load or execute\n\n"
                "Subcommands:\n"
                "  /tools load <name>      Load an on-demand tool immediately\n"
                "  /tools enable <name>    Re-enable a disabled tool\n"
                "  /tools disable <name>   Disable a tool for this session\n\n"
                "With any other text, filters tools by name.\n\n"
                "Examples:\n"
                "  /tools                     List all tools by category\n"
                "  /tools search              Show search-related tools\n"
                "  /tools load exa_search     Load exa_search into active set\n"
                "  /tools disable shell       Disable execute_shell_command\n"
                "  /tools enable shell        Re-enable it"
            ),
            aliases=["t", "tool"],
        )
    )

    reg.register(
        SlashCommand(
            name="mcp",
            handler=SlashCommandRegistry._cmd_mcp,
            short_help="List or restart MCP server connections",
            long_help=(
                "Usage: /mcp [restart [server-name]]\n\n"
                "Without arguments, lists all configured MCP servers with\n"
                "connection status, transport type, endpoint, and tools.\n\n"
                "Subcommands:\n"
                "  /mcp restart               Restart all MCP server connections\n"
                "  /mcp restart <name>        Restart a specific server\n\n"
                "Examples:\n"
                "  /mcp                       Show all MCP server statuses\n"
                "  /mcp restart               Reconnect all servers\n"
                "  /mcp restart my-server     Reconnect 'my-server'"
            ),
        )
    )

    reg.register(
        SlashCommand(
            name="clear",
            handler=SlashCommandRegistry._cmd_clear,
            short_help="Clear conversation history",
            long_help=(
                "Usage: /clear\n\n"
                "Clears all messages from the current session's memory.\n"
                "The session ID is preserved, but conversation history\n"
                "and mode-specific tracking (entities, files, decisions)\n"
                "are reset.\n\n"
                "This action cannot be undone."
            ),
            aliases=["c"],
        )
    )

    reg.register(
        SlashCommand(
            name="think",
            handler=SlashCommandRegistry._cmd_think,
            short_help="Deep reasoning on a complex problem",
            long_help=(
                "Usage: /think <task description>\n\n"
                "Runs the Tree-of-Thought reasoning engine directly,\n"
                "bypassing the agent's tool selection.\n\n"
                "The engine:\n"
                "  1. Generates multiple solution approaches\n"
                "  2. Develops each in parallel with Chain-of-Thought\n"
                "  3. Evaluates, reflects, and cross-pollinates ideas\n"
                "  4. Iterates: Plan → Execute → Observe → Reflect\n"
                "  5. Synthesises the best elements into a final answer\n\n"
                "Typically takes 1-5 min (~15 LLM calls).\n"
                "Results are saved to session memory.\n\n"
                "Examples:\n"
                "  /think Design a caching strategy for microservices\n"
                "  /think Compare REST vs GraphQL for our API\n"
                "  /think How to migrate from monolith to services?"
            ),
            aliases=["T"],
        )
    )

    reg.register(
        SlashCommand(
            name="delegate",
            handler=SlashCommandRegistry._cmd_delegate,
            short_help="Force task delegation to other models",
            long_help=(
                "Usage: /delegate <task description>\n\n"
                "Forces the task to be broken into subtasks and delegated\n"
                "to other LLM models in parallel, bypassing the agent's\n"
                "own decision on whether to delegate.\n\n"
                "The system will:\n"
                "  1. Analyze the task and split it into subtasks\n"
                "  2. Assign each subtask to the best-fit model alias\n"
                "  3. Execute all subtasks in parallel\n"
                "  4. Synthesize results into a unified answer\n\n"
                "Results are saved to session memory.\n\n"
                "Examples:\n"
                "  /delegate Research top 10 AI companies and market cap\n"
                "  /delegate Compare Python, Rust, and Go for web backends\n"
                "  /delegate Translate this text into French, German, Spanish"
            ),
            aliases=["d"],
        )
    )

    reg.register(
        SlashCommand(
            name="mode",
            handler=SlashCommandRegistry._cmd_mode,
            short_help="Show / switch memory mode",
            long_help=(
                "Usage: /mode [name]\n\n"
                "Without arguments, lists available memory modes and highlights\n"
                "the active one.  With a mode name, switches immediately.\n\n"
                "Modes:\n"
                "  conversation  General chat, entity tracking    (25 msgs)\n"
                "  code          Programming, file/error tracking (30 msgs)\n"
                "  reasoning     Planning, decision tracking      (40 msgs)\n\n"
                "Examples:\n"
                "  /mode           Show all modes\n"
                "  /mode code      Switch to code mode\n"
                "  /M reasoning    Switch to reasoning mode\n\n"
                "Switching preserves the current session but rebuilds the\n"
                "system prompt and memory context for the new mode."
            ),
            aliases=["M"],
        )
    )

    reg.register(
        SlashCommand(
            name="model",
            handler=SlashCommandRegistry._cmd_model,
            short_help="Show / switch model",
            long_help=(
                "Usage: /model [name]\n\n"
                "Without arguments, shows the current model and available\n"
                "model aliases (from config).  With a name, switches the\n"
                "active model immediately.\n\n"
                "The name can be:\n"
                "  - A model alias (e.g. 'fast', 'reasoning')\n"
                "  - A literal model name (e.g. 'gpt-4.1-mini', 'qwen3:8b')\n\n"
                "Aliases may also change the provider (see config file).\n\n"
                "Examples:\n"
                "  /model             Show current model + aliases\n"
                "  /model fast        Switch to the 'fast' alias\n"
                "  /m gpt-4.1         Switch to gpt-4.1"
            ),
            aliases=["m"],
        )
    )

    reg.register(
        SlashCommand(
            name="provider",
            handler=SlashCommandRegistry._cmd_provider,
            short_help="Show / switch LLM provider",
            long_help=(
                "Usage: /provider [name]\n\n"
                "Without arguments, lists all configured providers and\n"
                "highlights the active one.  With a name, switches to\n"
                "that provider immediately.\n\n"
                "The LLM is rebuilt with the new provider's settings\n"
                "(model, base_url, temperature, etc.).\n\n"
                "Examples:\n"
                "  /provider                 List providers\n"
                "  /provider spark-cluster   Switch to spark-cluster\n"
                "  /p ollama                 Switch to ollama"
            ),
            aliases=["p"],
        )
    )

    reg.register(
        SlashCommand(
            name="session",
            handler=SlashCommandRegistry._cmd_session_switch,
            short_help="Show / switch session",
            long_help=(
                "Usage: /session [id]\n\n"
                "Without arguments, shows the current session ID, message\n"
                "count, and memory mode.\n\n"
                "With a session ID, saves the current session and switches\n"
                "to the new one.  If the session already has history it\n"
                "will be loaded; otherwise a fresh session starts.\n\n"
                "Examples:\n"
                "  /session               Show session info\n"
                "  /session project-x     Switch to 'project-x'\n"
                "  /s debug-issue         Start a new 'debug-issue' session"
            ),
            aliases=["s"],
        )
    )

    reg.register(
        SlashCommand(
            name="debug",
            handler=SlashCommandRegistry._cmd_debug,
            short_help="Toggle debug mode",
            long_help=(
                "Usage: /debug\n\n"
                "Toggles debug mode on/off.  When enabled:\n"
                "  - Log level is set to DEBUG\n"
                "  - Verbose logging is auto-enabled\n"
                "  - File logging starts (default: cogtrix.log)\n\n"
                "Equivalent to the --debug CLI flag."
            ),
            aliases=["D"],
        )
    )

    reg.register(
        SlashCommand(
            name="verbose",
            handler=SlashCommandRegistry._cmd_verbose,
            short_help="Toggle verbose logging",
            long_help=(
                "Usage: /verbose\n\n"
                "Toggles verbose logging on/off.  When enabled, full\n"
                "LLM interactions (tokens, thinking, tool calls) are\n"
                "logged without truncation.\n\n"
                "Equivalent to the -v / --verbose CLI flag."
            ),
            aliases=["v"],
        )
    )

    reg.register(
        SlashCommand(
            name="approve",
            handler=SlashCommandRegistry._cmd_approve,
            short_help="Auto-approve all tool confirmations",
            long_help=(
                "Usage: /approve\n\n"
                "Toggles automatic approval for tools that normally\n"
                "require confirmation (file writes, shell commands, etc.).\n\n"
                "When ON, all tools run without prompting.\n"
                "When OFF, tools will prompt for confirmation again\n"
                "and all disabled tools are re-enabled.\n\n"
                "Equivalent to the -y / --no-confirm CLI flag."
            ),
            aliases=["a"],
        )
    )

    reg.register(
        SlashCommand(
            name="optimizer",
            handler=SlashCommandRegistry._cmd_optimizer,
            short_help="Toggle optimizer or force-optimize a prompt",
            long_help=(
                "Usage: /optimizer [prompt]\n\n"
                "Without arguments: toggles the prompt optimizer on/off.\n"
                "With a prompt: forces optimization on the given text\n"
                "(bypassing the length gate) and sends the result to the\n"
                "agent.\n\n"
                "Examples:\n"
                "  /optimizer              Toggle on/off\n"
                "  /optimizer Search for MCP server docs\n"
                "                          Optimize and run this prompt"
            ),
            aliases=["o"],
        )
    )

    reg.register(
        SlashCommand(
            name="paste",
            handler=SlashCommandRegistry._cmd_paste,
            short_help="Enter multi-line paste mode",
            long_help=(
                "Usage: /paste [optional first line]\n\n"
                "Enter multi-line input mode for pasting text with\n"
                "newlines (logs, code, data, web pages).\n\n"
                'Finish input: type """ on a new line\n'
                "Cancel:       press Ctrl+C\n\n"
                "Alternative — start any message with triple quotes:\n\n"
                '  """your pasted text here\n'
                "  more lines...\n"
                '  """\n\n'
                "Single-line shortcut:\n"
                '  """some text in one shot"""'
            ),
            aliases=["P"],
        )
    )

    reg.register(
        SlashCommand(
            name="setup",
            handler=SlashCommandRegistry._cmd_setup,
            short_help="Launch the setup wizard",
            long_help=(
                "Usage: /setup\n\n"
                "Launches the interactive setup wizard to create or edit\n"
                "your configuration file (~/.cogtrix.yaml).\n\n"
                "The wizard walks you through:\n"
                "  1. Connecting to an LLM provider\n"
                "  2. Configuring features via LLM-guided Q&A\n"
                "  3. Validating and saving the config\n\n"
                "After the wizard completes, Cogtrix reloads the config\n"
                "and reconnects to the new provider without restarting."
            ),
        )
    )

    # Hidden commands (not listed in /help categories)
    reg.register(
        SlashCommand(
            name="system_prompt",
            handler=SlashCommandRegistry._cmd_system_prompt,
            short_help="Display the full system prompt",
            aliases=["sp"],
        )
    )

    return reg


# ── Tool presets per memory mode ─────────────────────────────────────────
# Tools listed here are loaded into the agent at startup; all others are
# available on demand via the ``request_tools`` meta-tool.
# Currently every mode starts lean (empty preset) so the LLM requests
# only the tools it needs for the task at hand.

# Short one-liner descriptions for the request_tools catalog.
# Populated at startup from the full registry.
# Accessed via _session.all_tool_descriptions

# Original (unwrapped) tool objects keyed by name.
# Populated once at startup before wrapping/splitting so released
# tools can be returned to the available pool without double-wrapping.
# Accessed via _session.all_tool_originals

_apply_tool_preset = apply_tool_preset


_create_request_tools_tool = create_request_tools_tool


def check_config(config: Config) -> int:
    """
    Validate and display configuration details.

    Returns:
        0 on success, 1 on error
    """
    if not console:
        print("\nConfiguration Check\n")
        try:
            pc = config.resolve_provider_config()
            print(f"  Provider: {config.provider} ({pc.type})")
            actual_model = config.model or pc.get_model()
            print(f"  Model: {actual_model}")
            if pc.base_url:
                print(f"  Base URL: {pc.base_url}")
            if pc.api_key:
                print("  API Key: ***configured***")
        except ValueError as e:
            print(f"  Error: {e}")
            return 1
        for name in config.list_providers():
            current = " (current)" if name == config.provider else ""
            try:
                prov = config.get_provider_config(name)
                print(f"  - {name}: {prov.type}/{prov.get_model()}{current}")
            except ValueError:
                print(f"  - {name}: built-in{current}")
        print("\n  Configuration valid\n")
        return 0

    console.print("\n[bold cyan]Configuration Check[/bold cyan]\n")

    # Config file location
    if config.config_file_path:
        console.print(f"[green]✓[/green] Config file: {config.config_file_path}")
    else:
        console.print("[yellow]ℹ[/yellow] No config file found (using defaults)")

    # Provider check
    console.print(f"\n[bold]Provider:[/bold] {config.provider}")

    try:
        provider_config = config.resolve_provider_config()
        console.print(f"  Type: {provider_config.type}")
        # Show the resolved model (from -m or alias), not provider default
        actual_model = config.model or provider_config.get_model()
        console.print(f"  Model: {actual_model}")
        if provider_config.base_url:
            console.print(f"  Base URL: {provider_config.base_url}")
        if provider_config.api_key:
            console.print("  API Key: [dim]***configured***[/dim]")
        elif provider_config.type == "openai":
            import os

            if os.getenv("OPENAI_API_KEY"):
                console.print("  API Key: [dim]***from environment***[/dim]")
            else:
                console.print(
                    "  [yellow]⚠ API Key: not configured "
                    "(set in config or OPENAI_API_KEY env var)[/yellow]"
                )
        if provider_config.num_ctx:
            console.print(f"  Context Size: {provider_config.num_ctx}")
        if provider_config.temperature is not None:
            console.print(f"  Temperature: {provider_config.temperature}")
    except ValueError as e:
        console.print(f"  [red]✗ Error: {e}[/red]")
        return 1

    # List all providers
    console.print("\n[bold]Available Providers:[/bold]")
    for name in config.list_providers():
        current = " [cyan](current)[/cyan]" if name == config.provider else ""
        try:
            prov = config.get_provider_config(name)
            console.print(f"  • {name}: {prov.type}/{prov.get_model()}{current}")
        except ValueError:
            console.print(f"  • {name}: [dim]built-in[/dim]{current}")

    # Memory mode
    console.print(f"\n[bold]Memory Mode:[/bold] {config.memory_mode}")

    # RAG settings
    console.print("\n[bold]RAG Settings:[/bold]")
    console.print(f"  Docs Dir: {config.rag.docs_dir}")
    console.print(f"  VectorDB Dir: {config.rag.vectordb_dir}")
    if config.rag.model:
        console.print(f"  Embedding Model: {config.rag.model}")
    else:
        emb_type, _, _, _ = config.resolve_embedding_config()
        console.print(f"  Embedding Provider: {emb_type} (default)")

    # Services
    if config.services:
        console.print(f"\n[bold]Services:[/bold] ({len(config.services)} configured)")
        for svc_name, svc_cfg in sorted(config.services.items()):
            has_key = bool(svc_cfg.get("api_key"))
            status = "[dim]***configured***[/dim]" if has_key else "[yellow]no key[/yellow]"
            console.print(f"  • {svc_name}: {status}")

    # Delegate settings
    if config.delegate_enabled:
        console.print("\n[bold]Delegate Tool:[/bold] enabled")
        if config.delegate_allowed_providers:
            console.print(f"  Allowed Providers: {', '.join(config.delegate_allowed_providers)}")
        if config.models:
            console.print(f"  Models: {', '.join(config.models.keys())}")
    else:
        console.print("\n[bold]Delegate Tool:[/bold] disabled")

    console.print("\n[green]✓ Configuration valid[/green]\n")
    return 0


def run_ingest(args, config: Config) -> int:
    """
    Run document ingestion and exit.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    from src.rag import IngestConfig, ingest_documents

    # Build ingest configuration from args and config
    docs_dir = Path(args.docs_dir if args.docs_dir else config.rag.docs_dir)
    vectordb_dir = Path(args.vectordb_dir if args.vectordb_dir else config.rag.vectordb_dir)

    # Resolve embedding config from models registry
    # Priority: CLI args > env vars > rag.model > active provider fallback
    emb_provider_arg = (
        getattr(args, "embedding_provider", None) or config.embedding_provider_override
    )
    emb_model_arg = getattr(args, "embedding_model", None) or config.embedding_model_override
    if emb_provider_arg:
        emb_type = emb_provider_arg
        emb_model = emb_model_arg or None
        emb_base_url: str | None = None
        emb_api_key: str | None = None
        if emb_type in config.providers:
            pc = config.providers[emb_type]
            emb_type = pc.type
            emb_base_url = pc.get_base_url()
            emb_api_key = pc.api_key
    else:
        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
        if emb_model_arg:
            emb_model = emb_model_arg

    ingest_config = IngestConfig(
        docs_dir=docs_dir,
        vectordb_dir=vectordb_dir,
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
        embedding_provider=emb_type,
        embedding_model=emb_model,
        base_url=emb_base_url,
        api_key=emb_api_key,
    )

    if console:
        console.print("[bold]📚 RAG Document Ingestion[/bold]\n")
        console.print(f"  Documents directory: [cyan]{ingest_config.docs_dir}[/cyan]")
        console.print(f"  Vector DB output:    [cyan]{ingest_config.vectordb_dir}[/cyan]")
        console.print(f"  Embedding provider:  [cyan]{emb_type}[/cyan]")
        if ingest_config.embedding_model:
            console.print(f"  Embedding model:     [cyan]{ingest_config.embedding_model}[/cyan]")
        if emb_base_url:
            console.print(f"  Base URL:            [cyan]{emb_base_url}[/cyan]")
        console.print()
    else:
        print("📚 RAG Document Ingestion\n")
        print(f"  Documents directory: {ingest_config.docs_dir}")
        print(f"  Vector DB output:    {ingest_config.vectordb_dir}")
        print(f"  Embedding provider:  {emb_type}")
        if ingest_config.embedding_model:
            print(f"  Embedding model:     {ingest_config.embedding_model}")
        if emb_base_url:
            print(f"  Base URL:            {emb_base_url}")
        print()

    # Run ingestion
    result = ingest_documents(ingest_config)

    # Report results
    if result.success:
        if console:
            console.print(f"[green]✓ Loaded {result.documents_loaded} document(s)[/green]")
            console.print(f"[green]✓ Created {result.chunks_created} chunk(s)[/green]")
            console.print(f"[green]✓ Saved to {result.vector_store_path}[/green]")
        else:
            print(f"✓ Loaded {result.documents_loaded} document(s)")
            print(f"✓ Created {result.chunks_created} chunk(s)")
            print(f"✓ Saved to {result.vector_store_path}")
        return 0
    else:
        if console:
            console.print("[red]✗ Ingestion failed[/red]")
            for error in result.errors:
                console.print(f"[red]  - {error}[/red]")
        else:
            print("✗ Ingestion failed")
            for error in result.errors:
                print(f"  - {error}")
        return 1


# ── Tool output capping ──────────────────────────────────────────────
# Backward-compat aliases: tool output cap moved to src/tools/configure.py
_TOOL_OUTPUT_CAP_RATIO = 0.10
_TOOL_OUTPUT_CAP_MIN_CHARS = TOOL_OUTPUT_CAP_MIN_CHARS
_compute_tool_output_cap = compute_tool_output_cap


def create_safe_tool_wrapper(
    tool,
    tool_name: str,
    registry: ToolRegistry,
    approvals: set,
    session_state: SessionState | None = None,
):
    """Wrap a tool with confirmation gate. Delegates to src.agent.safety."""
    from src.agent.safety import create_safe_tool_wrapper as _safety_wrapper

    return _safety_wrapper(
        tool,
        tool_name,
        registry,
        approvals,
        session_state=session_state or _session,
        ui=_rich_ui,
    )


# Backward-compat aliases: graph builder and constants moved to src/orchestration/graph.py
_build_agent_graph = build_agent_graph
_EMPTY_RESPONSE_MSG = EMPTY_RESPONSE_MSG


def run_single_prompt(
    prompt_text: str,
    memory_manager: Any,
    registry: ToolRegistry,
    approvals: set,
    output_file: str | None = None,
    no_stream: bool = False,
    log: Any = None,
    callbacks: list | None = None,
    # Tool-expansion support
    llm: Any = None,
    system_prompt: str | None = None,
    available_tools: dict | None = None,
    active_tools_list: list | None = None,
    config: Any = None,
    max_context_tokens: int | None = None,
) -> int:
    """
    Process a single prompt in non-interactive mode.

    Args:
        prompt_text: The prompt to send to the agent
        memory_manager: Memory manager instance
        registry: Tool registry
        approvals: Set of approved tools
        output_file: Optional file to write response to
        no_stream: If True, suppress streaming output
        log: Logger instance
        callbacks: Optional list of callback handlers for LLM observability
        llm: LLM instance (for tool-expansion rebuild)
        system_prompt: System prompt (for tool-expansion rebuild)
        available_tools: Tools available on request (for tool-expansion)
        active_tools_list: List of currently active tool objects

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Start new request tracking
        new_request_id()

        _session.reset_for_new_prompt()

        # Log user message
        log_user_message(prompt_text)

        # Preserve the user's original phrasing before the optimizer
        # rewrites it — memory, classification, delegation must all
        # see what the user actually typed.
        original_input = prompt_text

        # Run prepare_context and optimize_prompt concurrently when optimizer is enabled.
        # The two operations are independent: the optimizer rewrites the prompt text
        # while prepare_context reads conversation history — no data dependency.
        if config and config.prompt_optimizer:
            import concurrent.futures as _cf

            with _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep") as _pool:
                _ctx_future = _pool.submit(memory_manager.prepare_context, prompt_text)
                _opt_future = _pool.submit(optimize_prompt, prompt_text, llm)
                context = _ctx_future.result()
                prompt_text = _opt_future.result()
        else:
            context = memory_manager.prepare_context(prompt_text)

        if log:
            log.debug(
                f"Non-interactive prompt: {len(prompt_text)} chars, "
                f"context: {context.context_messages_count} messages"
            )

        # Run agent
        wants_deep = user_wants_deep_think(original_input)

        agent_msgs: list = []

        compression_llm = None
        if config and config.context_compression_model:
            compression_llm = create_compression_llm(config.context_compression_model, config)

        _acc = _TokenAccumulator()
        _agent_cbs = (callbacks or []) + [_acc]
        _agent_t0 = _time_mod.monotonic()
        _spinner.start()
        try:
            output = run_agent(
                prompt_text,
                context.messages,
                registry,
                approvals,
                context_prefix=context.context_prefix,
                callbacks=_agent_cbs,
                result_messages=agent_msgs,
                llm=llm,
                system_prompt=system_prompt,
                available_tools=dict(available_tools) if available_tools else available_tools,
                active_tools_list=active_tools_list,
                max_context_tokens=max_context_tokens,
                preset_tools=(TOOL_PRESETS.get(config.memory_mode, set()) if config else set()),
                context_compression=config.context_compression if config else True,
                compression_min_age=(
                    config.context_compression_min_age if config else COMPRESSION_MIN_AGE_CYCLES
                ),
                compression_min_chars=(
                    config.context_compression_min_chars if config else COMPRESSION_MIN_CHARS
                ),
                compression_llm=compression_llm,
                session_state=_session,
                confirmation_ui=_rich_ui,
                on_tool_expansion=_tool_expansion_ui,
                parallel_tool_execution=config.parallel_tool_execution if config else True,
            )
        finally:
            _spinner.stop()

        # ── Enforce deep_think when the user requested it ────────
        # Force-call if: (a) agent skipped deep_think entirely, OR
        # (b) agent called it but with inadequate context (references
        # instead of actual data — fewer than MIN_GOOD_CONTEXT_LEN chars).
        # However, for tool-intensive tasks (bug hunting, sysadmin, etc.)
        # "think deeply" is treated as a quality hint — the agent's
        # actual tool work is more valuable than isolated reasoning.
        _research_output: str = ""
        if wants_deep and output:
            _task_cat = classify_think_task(original_input, llm) if llm else None
            if _task_cat and _task_cat.tool_intensive:
                log.info(
                    "Skipping force deep_think: task classified as '%s' "
                    "(tool-intensive — agent's tool work is the primary output)",
                    _task_cat.name,
                )
            else:
                called = was_deep_think_called(agent_msgs)
                if not called or not deep_think_had_good_context(agent_msgs):
                    if called:
                        log.info(
                            "deep_think was called but with inadequate context "
                            "(<%d chars) — forcing re-call with full data",
                            MIN_GOOD_CONTEXT_LEN,
                        )
                    tool_data = collect_tool_outputs(agent_msgs)

                    # Run research delegate if web tools were used to get
                    # high-fidelity content for deep_think.
                    _rd_enabled = (
                        getattr(config, "research_delegate_enabled", True) if config else True
                    )
                    if _rd_enabled and agent_used_web_tools(agent_msgs):
                        fetched_urls = extract_fetched_urls(agent_msgs)
                        if fetched_urls:
                            _rd_timeout = (
                                getattr(config, "research_delegate_timeout", 300) if config else 300
                            )
                            _rd_cap = (
                                getattr(config, "research_delegate_cap_ratio", RESEARCH_CAP_RATIO)
                                if config
                                else RESEARCH_CAP_RATIO
                            )
                            _spinner.start()
                            try:
                                _research_output = run_research_delegate(
                                    fetched_urls,
                                    original_input,
                                    max_context_tokens=max_context_tokens,
                                    timeout=_rd_timeout,
                                    cap_ratio=_rd_cap,
                                )
                            finally:
                                _spinner.stop()

                    _spinner.start()
                    try:
                        output = force_deep_think(
                            original_input,
                            output,
                            tool_data,
                            log,
                            research_context=_research_output or None,
                        )
                    finally:
                        _spinner.stop()

        # ── Enforce delegation when the query warrants it ────────
        if (
            not wants_deep
            and output
            and config is not None
            and getattr(config, "delegate_enabled", False)
            and user_wants_delegation(original_input)
            and not was_delegation_called(agent_msgs)
        ):
            log.info(
                "Auto-detected delegation-worthy query but agent "
                "did not delegate — forcing parallel delegation"
            )
            tool_data = collect_tool_outputs(agent_msgs)
            _spinner.start()
            try:
                forced = force_delegation(original_input, output, tool_data, config, log)
                if forced and forced != output:
                    output = forced
            finally:
                _spinner.stop()

        # Snapshot turn messages before the execution phase can
        # append its own HumanMessage to agent_msgs.
        turn_msgs = extract_turn_messages(agent_msgs)

        # ── Execution phase: act on the analysis ─────────────────
        # If the prompt asks for file creation/changes but the agent
        # only produced text (no write_file calls), feed the analysis
        # back to the agent and let it actually execute.
        if (
            output
            and prompt_requests_action(original_input)
            and not agent_performed_writes(agent_msgs)
        ):
            log.info(
                "Prompt requests file actions but none were performed " "— running execution phase"
            )
            _spinner.start()
            try:
                exec_output, exec_msgs = run_execution_phase(
                    output,
                    prompt_text,
                    context.messages,
                    registry,
                    approvals,
                    context_prefix=context.context_prefix,
                    callbacks=_agent_cbs,
                    llm=llm,
                    system_prompt=system_prompt,
                    available_tools=dict(available_tools) if available_tools else available_tools,
                    active_tools_list=active_tools_list,
                    max_context_tokens=max_context_tokens,
                    preset_tools=(TOOL_PRESETS.get(config.memory_mode, set()) if config else set()),
                    session_state=_session,
                    on_tool_expansion=_tool_expansion_ui,
                    parallel_tool_execution=config.parallel_tool_execution if config else True,
                )
            finally:
                _spinner.stop()
            if exec_output:
                # Combine: analysis + execution summary
                output = output + "\n\n---\n\n" + exec_output
                turn_msgs.extend(exec_msgs)

        _agent_elapsed = _time_mod.monotonic() - _agent_t0

        # Guard: never produce an empty response
        if not output or not output.strip():
            output = _EMPTY_RESPONSE_MSG
            if log:
                log.error("Empty output after run_agent")

        # Log agent response
        log_agent_response(output)

        # Only save valid responses to history (skip empty/error responses).
        # Pass the full agent chain (tool calls + results) so the agent
        # can continue iterating on complex tasks across restarts.
        if _is_valid_response(output):
            memory_manager.update(original_input, output, agent_messages=turn_msgs or None)
            memory_manager.save()
        else:
            if log:
                log.warning("Skipping history save: empty or error response")

        # Output result
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output, encoding="utf-8")
                if not no_stream:
                    print(f"Response written to: {output_file}")
            except Exception as e:
                print(f"Error writing to output file: {e}")
                print(output)  # Still print to stdout
                return 1
        else:
            # Print to stdout
            if not no_stream and console is not None and Markdown is not None:
                console.rule("Agent", style="blue")
                console.print(
                    Padding(
                        Markdown(preserve_tables_for_markdown(output)),
                        (1, 0, 1, 2),
                    )
                )
                stats_text = _format_stats_line(_agent_elapsed, _acc)
                if stats_text:
                    console.print(Align.right(Text.from_markup(stats_text)))
                console.rule(style="dim blue")
            else:
                print(output)

        return 0

    except KeyboardInterrupt:
        if log:
            log.info("Non-interactive session interrupted")
        print("\nInterrupted.")
        return 130

    except Exception as e:
        log_error(e, context="Non-interactive prompt error", include_trace=True)
        print(f"Error: {e}")
        return 1


def main():
    """Main CLI loop."""
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration (CLI > env > config file > defaults)
    try:
        config = load_config(args)
    except ConfigError as e:
        if console:
            console.print("\n[bold red]Configuration Error[/bold red]\n")
            console.print(f"[yellow]{e}[/yellow]\n")
        else:
            print(f"\nConfiguration Error:\n{e}\n")
        sys.exit(1)

    # Handle --check-config mode (early exit)
    if args.check_config:
        sys.exit(check_config(config))

    # Handle --ingest mode (early exit)
    if args.ingest:
        sys.exit(run_ingest(args, config))

    # Handle --setup mode (early exit)
    if getattr(args, "setup", False):
        from pathlib import Path as _Path

        from src.setup_wizard import run_setup_wizard

        _setup_output = _Path(args.setup_output) if args.setup_output else None
        run_setup_wizard(
            setup_docs_url=args.setup_docs,
            output_path=_setup_output,
        )
        sys.exit(0)

    # Auto-launch setup wizard when no config and no provider env vars detected
    if (
        config.config_file_path is None
        and not any(
            os.environ.get(k)
            for k in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "XAI_API_KEY",
                "COGTRIX_OLLAMA",
                "OLLAMA_BASE_URL",
            )
        )
        and not getattr(args, "prompt", None)
        and not getattr(args, "prompt_file", None)
        and not getattr(args, "assistant", False)
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    ):
        from src.setup_wizard import run_setup_wizard

        if console is not None:
            console.print("\n  [bold]No configuration found.[/bold] " "Starting setup wizard...\n")
        else:
            print("\n  No configuration found. Starting setup wizard...\n")

        run_setup_wizard()

        # Reload config after wizard creates the file
        try:
            config = load_config(args)
        except ConfigError as exc:
            print(f"\n  Could not load config after setup: {exc}")
            sys.exit(1)

    # Initialize logging (must be done early, after config is loaded)
    setup_logging(log_file=config.log_file, debug=config.debug, verbose=config.verbose)
    log = get_logger()

    if config.log_file is not None:
        log_file_display = config.log_file or "cogtrix.log"
        debug_str = " (debug)" if config.debug else ""
        if console is not None:
            console.print(f"  [dim]Logging to: {log_file_display}{debug_str}[/dim]")
        else:
            print(f"  Logging to: {log_file_display}{debug_str}")

    # Memory manager setup
    memory_store = JsonFileMemoryStore()

    # --no-confirm / -y: skip all tool safety confirmations
    _session.no_confirm = getattr(args, "no_confirm", False)
    approvals = _session.approvals

    try:
        memory_manager = MemoryFactory.create(
            mode=config.memory_mode,
            store=memory_store,
            session_id=config.session,
            config=config.memory_config,
        )
        memory_manager.load()
    except ValueError as e:
        print(f"⚠ Invalid memory mode: {e}")
        print(f"Available modes: {MemoryFactory.available_modes()}")
        sys.exit(1)

    # ── Inject API keys BEFORE loading tools ────────────────────
    # Tool modules check is_configured() during registry loading.
    # Keys must be in place so the check succeeds for config-file keys
    # (env-var keys are read directly and don't need this step).
    configure_tavily_tool(config)
    configure_exa_tool(config)
    configure_brave_tool(config)
    configure_serpapi_tool(config)
    configure_google_search_tool(config)

    # Load tools on startup
    tool_filter = getattr(args, "tools", None)
    if tool_filter == "none":
        registry = ToolRegistry()
    else:
        registry = load_tools(tool_filter)
    try:
        tool_names = registry.list_tools()
        if not tool_names:
            log.warning("No tools loaded")
    except Exception as e:
        log.error(f"Error loading tools: {e}")
        registry = ToolRegistry()

    # Configure remaining tools that need runtime settings
    def _delegation_status(message: str) -> None:
        _spinner.pause()
        if console is not None:
            console.print(f"  [dim]{message}[/dim]")
        else:
            print(f"  {message}")
        _spinner.resume()

    configure_delegate_tool(config, status_callback=_delegation_status)
    configure_rag_tool(config)
    configure_python_exec_tool(config)
    configure_deep_think_tool(config)

    try:
        from src.tools.deep_think import set_progress_callback

        def _deep_think_progress(msg: str) -> None:
            """Spinner-aware progress callback for deep think."""
            try:
                _spinner.pause()
            except Exception:  # noqa: BLE001
                pass
            import sys

            sys.stdout.write("\033[2K\r")
            sys.stdout.flush()
            print(f"  [think] {msg}")
            try:
                _spinner.resume()
            except Exception:  # noqa: BLE001
                pass

        set_progress_callback(_deep_think_progress)
    except ImportError:
        pass

    # ── Remove tools whose required API keys are missing ─────────
    total_registered = len(registry.list_tools())
    filter_unconfigured_tools(registry)

    # ── Connect to MCP servers ───────────────────────────────────────────────
    _mcp_manager: MCPManager | None = None  # type: ignore[assignment]
    if MCP_AVAILABLE and config.mcp_servers:
        _mcp_manager = MCPManager()
        _KNOWN_MCP_FIELDS = {
            "command",
            "args",
            "env",
            "url",
            "headers",
            "requires_confirmation",
            "timeout",
        }
        _mcp_configs = []
        for _mcp_name, _srv_cfg in config.mcp_servers.items():
            _unknown = set(_srv_cfg) - _KNOWN_MCP_FIELDS
            if _unknown:
                log.warning(
                    "MCP server '%s': ignoring unknown config keys: %s",
                    _mcp_name,
                    ", ".join(sorted(_unknown)),
                )
            _filtered = {k: v for k, v in _srv_cfg.items() if k in _KNOWN_MCP_FIELDS}
            _mcp_configs.append(MCPServerConfig(name=_mcp_name, **_filtered))
        mcp_tools = _mcp_manager.connect_all(_mcp_configs, builtin_tool_names=set(registry.tools))
        for tool_name, tool_obj in mcp_tools.items():
            registry.tools[tool_name] = tool_obj
            registry.tool_metadata[tool_name] = {
                "requires_confirmation": (tool_obj.metadata or {}).get(
                    "requires_confirmation", True
                ),
                "source": "mcp",
                "server": (tool_obj.metadata or {}).get("server", ""),
            }
        if mcp_tools:
            log.info(
                "Loaded %d MCP tool(s) from %d server(s)", len(mcp_tools), len(config.mcp_servers)
            )
        atexit.register(lambda: _mcp_manager.close_all() if _mcp_manager else None)
    elif not MCP_AVAILABLE and config.mcp_servers:
        log.warning(
            "mcp_servers configured but 'mcp' package not installed; " "run: uv pip install mcp"
        )

    # ── Apply tool presets ───────────────────────────────────────
    # Build the full catalog before splitting (for request_tools description).
    _session.all_tool_descriptions = build_tool_catalog(registry.tools)
    _session.all_tool_originals = dict(registry.tools)

    # Split into active (full schemas in agent) and available (on-demand)
    available_tools: dict[str, Any] = {}
    if tool_filter is None and registry.list_tools():
        active_dict, available_tools = _apply_tool_preset(registry, config.memory_mode)
        if available_tools:
            # Apply preset: only active tools stay in registry
            registry.tools = active_dict
    # else: all tools remain active (custom filter or no preset)

    # ── Startup banner with full summary ─────────────────────────
    active_count = len(registry.list_tools())
    on_demand = len(available_tools)
    configured_count = active_count + on_demand
    unavailable_count = total_registered - configured_count
    if tool_filter == "none":
        tools_text = "disabled"
    elif on_demand and active_count == 0:
        if unavailable_count:
            tools_text = f"{on_demand} on demand ({unavailable_count} unavailable)"
        else:
            tools_text = f"{on_demand} on demand"
    elif on_demand:
        tools_text = f"{active_count} active (+{on_demand} on request)"
    else:
        tools_text = f"{active_count} loaded"

    if _mcp_manager is not None:
        mcp_info = _mcp_manager.get_server_info()
        mcp_tool_count = sum(s["tool_count"] for s in mcp_info)
        if mcp_tool_count:
            tools_text += f" (+{mcp_tool_count} via MCP)"

    _startup_stats = memory_manager.get_stats()
    _startup_msg_count = _startup_stats.get("total_messages", memory_manager.get_message_count())
    _confirm_count = (
        sum(1 for n in registry.list_tools() if registry.requires_confirmation(n))
        if _session.no_confirm
        else 0
    )

    print_startup(
        config,
        tools_text=tools_text,
        configured_count=configured_count,
        total_registered=total_registered,
        session_id=config.session,
        msg_count=_startup_msg_count,
        no_confirm=_session.no_confirm,
        confirm_count=_confirm_count,
    )

    # Check if LangGraph is available (modern agent API)
    try:
        from langgraph.prebuilt import create_react_agent  # noqa: F401
    except ImportError as import_err:
        print("\n⚠️  Agent requires LangGraph to be installed.")
        print(f"   Error: {import_err}")
        print("\n   Please install dependencies:")
        print("   pip install langgraph")
        sys.exit(1)

    # Pre-approve all tools if --no-confirm / -y was passed
    if _session.no_confirm:
        for name in registry.list_tools():
            if registry.requires_confirmation(name):
                approvals.add(name)
        if approvals:
            log.info(f"Auto-approved {len(approvals)} tool(s) (--no-confirm)")

    # Wrap tools with safety interceptors
    tools = []
    for tool_name, tool in registry.tools.items():
        if registry.requires_confirmation(tool_name):
            safe_tool = create_safe_tool_wrapper(tool, tool_name, registry, approvals)
            tools.append(safe_tool)
        else:
            tools.append(tool)

    # Add request_tools meta-tool if there are on-demand tools
    _preset_names = TOOL_PRESETS.get(config.memory_mode, set())
    if available_tools:
        _active_tool_names = set(registry.tools.keys())
        rt_tool = _create_request_tools_tool(
            available_tools,
            _session.all_tool_descriptions,
            active_names=_active_tool_names,
            protected_names=_preset_names,
        )
        if rt_tool:
            tools.append(rt_tool)

    # Give delegate agents access to ALL tools (active + on-demand)
    configure_delegate_tools(tools, available_tools)

    # Get provider configuration
    try:
        provider_config = config.resolve_provider_config()
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print(f"   Available providers: {', '.join(config.list_providers())}")
        sys.exit(1)

    # Verify the provider has the credentials / endpoint needed to work
    _ptype = provider_config.type
    _has_key = bool(provider_config.api_key)
    _has_url = bool(provider_config.get_base_url())
    # OpenAI-compatible providers need an API key or a custom base URL
    # (self-hosted endpoints often work without a key).
    # Ollama only needs a reachable base URL (no key required).
    if _ptype in ("openai", "anthropic", "google") and not _has_key and not _has_url:
        _pname = provider_config.name
        print("\n  Cogtrix requires access to a language model to function.")
        print(f"  Provider '{_pname}' is not configured — no API key or endpoint found.\n")
        print("  To get started, either:")
        print("    1. Set the appropriate environment variable (e.g. OPENAI_API_KEY)")
        print("    2. Add provider credentials to your configuration file")
        print("    3. Use a local provider like Ollama that doesn't require an API key\n")
        print("  See the documentation for setup instructions.")
        sys.exit(1)

    # Build agent with configured provider and model
    # Include mode-specific system prompt additions
    try:
        mode_adds = memory_manager.get_system_prompt_additions()
        system_prompt = build_system_prompt(
            mode_additions=mode_adds,
            models=config.models,
            delegation_models=config.delegate_allowed_models,
            tool_instructions=provider_config.tool_instructions,
        )
        log.debug(f"System prompt length: {len(system_prompt)} chars")
        log.debug(f"Mode additions: {mode_adds if mode_adds else 'None'}")

        # Create LLM from provider config.
        # Apply a default max_tokens cap for the main agent to prevent
        # runaway generations (e.g. 12K+ token phantom tool calls).
        # Deep think and delegate are uncapped — they set their own limits.
        _DEFAULT_MAX_TOKENS = 4096
        if provider_config.max_tokens is None:
            provider_config.max_tokens = _DEFAULT_MAX_TOKENS
        llm = create_llm_from_provider_config(provider_config)

        # Token budget for context trimming (from provider num_ctx or default)
        from src.agent.core import _DEFAULT_CONTEXT_WINDOW

        max_context_tokens = provider_config.num_ctx or _DEFAULT_CONTEXT_WINDOW

        # Cap individual tool outputs to prevent context overflow.
        # Applied to active tools; on-demand tools get capped when
        # they are activated inside the expansion loop.
        _tool_output_cap = _compute_tool_output_cap(max_context_tokens)
        log.debug(
            "Tool output cap: %d chars (context window: %d tokens)",
            _tool_output_cap,
            max_context_tokens,
        )
        for t in tools:
            apply_output_cap(t, _tool_output_cap)

        # Wire LLM into memory manager for hybrid summarization
        memory_manager.set_llm(llm)
        _try_configure_embeddings(memory_manager, config)

        # Register LLM for cleanup on exit
        _cleanup_resources.append(llm)

        # Use actual model name (resolved from provider config), not CLI alias
        actual_model = provider_config.get_model()
        mode_info = f", mode: {config.memory_mode}"
        prov_model = f"{config.provider}: {actual_model}"
        if console:
            console.print(f"[green]✓ Agent ready[/green] " f"[dim]({prov_model}{mode_info})[/dim]")
        else:
            print(f"✓ Agent ready ({prov_model}{mode_info})")

        # Log session info with actual model
        log_session_info(
            session_id=config.session,
            message_count=_startup_msg_count,
            memory_mode=config.memory_mode,
            provider=config.provider,
            model=actual_model,
        )

    except ImportError as e:
        prov = config.provider
        log_error(e, context=f"Provider '{prov}' not available", include_trace=True)
        print(f"\n⚠️  Provider '{prov}' not available: {e}")
        print("   Please install the required package.")
        sys.exit(1)
    except Exception as e:
        log_error(e, context="Failed to initialize agent", include_trace=True)
        friendly = _friendly_error(
            e,
            provider=config.provider,
            base_url=provider_config.get_base_url() if provider_config else "",
        )
        print(f"\n⚠️  {friendly}")

        if config.debug:
            import traceback

            print("\n   Debug traceback:")
            traceback.print_exc()

        sys.exit(1)

    # Handle --assistant mode (headless messaging daemon)
    if getattr(args, "assistant", False):
        from src.assistant.service import AssistantService

        _asst_compression_llm = None
        if config.context_compression_model:
            _asst_compression_llm = create_compression_llm(config.context_compression_model, config)

        _asst_system_prompt: str | None = None
        if getattr(args, "system_prompt", None):
            _asst_system_prompt = args.system_prompt
        elif getattr(args, "system_prompt_file", None):
            _sp_path = Path(args.system_prompt_file)
            if not _sp_path.exists():
                print(f"\n⚠️  System prompt file not found: {args.system_prompt_file}")
                sys.exit(1)
            _asst_system_prompt = _sp_path.read_text(encoding="utf-8").strip()
            if not _asst_system_prompt:
                print(f"\n⚠️  System prompt file is empty: {args.system_prompt_file}")
                sys.exit(1)

        service = AssistantService(
            config=config,
            llm=llm,
            registry=registry,
            system_prompt=system_prompt,
            available_tools=available_tools or {},
            active_tools=tools,
            max_context_tokens=max_context_tokens,
            compression_llm=_asst_compression_llm,
            cli_system_prompt=_asst_system_prompt,
            agent_runner=run_agent,
        )
        service.run()
        sys.exit(0)

    # Handle non-interactive mode (--prompt or --prompt-file)
    prompt_text = None
    if hasattr(args, "prompt") and args.prompt:
        prompt_text = args.prompt
    elif hasattr(args, "prompt_file") and args.prompt_file:
        try:
            prompt_file = Path(args.prompt_file)
            if not prompt_file.exists():
                print(f"\n⚠️  Prompt file not found: {args.prompt_file}")
                sys.exit(1)
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            if not prompt_text:
                print(f"\n⚠️  Prompt file is empty: {args.prompt_file}")
                sys.exit(1)
        except Exception as e:
            print(f"\n⚠️  Error reading prompt file: {e}")
            sys.exit(1)

    if prompt_text:
        # Non-interactive mode: process single prompt and exit
        # Create observability callbacks if debug mode is enabled
        callbacks = []
        if config.debug:
            obs_handler = create_observability_handler(verbose=config.verbose)
            if obs_handler:
                callbacks.append(obs_handler)
                log.debug("LLM observability handler enabled")

        exit_code = run_single_prompt(
            prompt_text=prompt_text,
            memory_manager=memory_manager,
            registry=registry,
            approvals=approvals,
            output_file=getattr(args, "output", None),
            no_stream=getattr(args, "no_stream", False),
            log=log,
            callbacks=callbacks if callbacks else None,
            llm=llm,
            system_prompt=system_prompt,
            available_tools=(
                available_tools
                if (available_tools or TOOL_PRESETS.get(config.memory_mode))
                else None
            ),
            active_tools_list=tools,
            config=config,
            max_context_tokens=max_context_tokens,
        )
        sys.exit(exit_code)

    # Set up slash commands
    slash_cmds = _build_slash_commands()
    slash_cmds.config = config
    slash_cmds.memory_manager = memory_manager
    slash_cmds.registry = registry
    slash_cmds.approvals = approvals
    slash_cmds.available_tools = available_tools
    slash_cmds.system_prompt = system_prompt
    slash_cmds.mcp_manager = _mcp_manager

    # Session orchestrator: snapshot/rollback helper for switch handlers
    session_orch = SessionOrchestrator(config, slash_cmds)

    # Output file for interactive mode (append each response)
    output_file: str | None = getattr(args, "output", None)
    if output_file:
        print(f"📄 Responses will be appended to: {output_file}")

    if console is not None:
        console.print(
            "[dim]Type your message, [bold]/help[/bold] for commands, "
            '[yellow bold]"""[/yellow bold] or [bold]/paste[/bold] for multi-line input. '
            "[bold]/quit[/bold] or Ctrl+D to exit.[/dim]"
        )
    else:
        print(
            'Type your message, /help for commands, """ or /paste for multi-line input.'
            " /quit or Ctrl+D to exit."
        )

    # Load input history from previous sessions
    load_input_history()
    atexit.register(save_input_history)
    if _escape_monitor is not None and _escape_monitor.available:
        atexit.register(_escape_monitor._restore_terminal)

    # Create observability callbacks if debug mode is enabled
    callbacks = []
    if config.debug:
        obs_handler = create_observability_handler(verbose=config.verbose)
        if obs_handler:
            callbacks.append(obs_handler)
            log.debug("LLM observability handler enabled for interactive mode")

    compression_llm = None
    if config.context_compression_model:
        compression_llm = create_compression_llm(config.context_compression_model, config)

    # Main input/output loop
    while True:
        try:
            if color_enabled():
                # \001/\002 are readline markers for non-printing chars
                # so readline calculates prompt width correctly.
                _prompt = "\n\001\033[36m\002\u276f\001\033[0m\002 "
            else:
                _prompt = "\n> "
            user_input = input(_prompt).strip()

            if not user_input:
                continue

            already_optimized = False

            # ── Multi-line paste mode (triple-quote or /paste) ─────
            if user_input.startswith('"""'):
                # Single-line shortcut: """content here"""
                if user_input.endswith('"""') and len(user_input) > 6:
                    user_input = user_input[3:-3].strip()
                else:
                    first = user_input[3:].strip()
                    user_input = read_multiline(first)
                if not user_input:
                    continue
            elif user_input.startswith("!"):
                # ── Inline shell command ───────────────────────────
                run_inline_shell(user_input[1:].strip())
                continue
            elif user_input.startswith("/"):
                _cmd_parts = user_input.lstrip("/").split(None, 1)
                cmd_word = _cmd_parts[0].lower() if _cmd_parts else ""
                if cmd_word == "paste":
                    parts = user_input.split(None, 1)
                    first = parts[1].strip() if len(parts) > 1 else ""
                    user_input = read_multiline(first)
                    if not user_input:
                        continue
                else:
                    # Regular slash commands (e.g. /help, /quit, /info)
                    result = slash_cmds.dispatch(user_input)
                    if result == "break":
                        break
                    if isinstance(result, str) and result.startswith("switch_mode:"):
                        new_mode = result.split(":", 1)[1]
                        _snap = session_orch.snapshot(
                            memory_manager=memory_manager,
                            system_prompt=system_prompt,
                            registry_tools=registry.tools,
                            available_tools=available_tools,
                            tools=tools,
                        )
                        try:
                            # Save current memory state before switching
                            memory_manager.save()
                            # Look up the correct mode config (not the old mode's)
                            new_mode_config = config.memory_modes.get(new_mode)
                            # Create new memory manager for the target mode
                            memory_manager = MemoryFactory.create(
                                mode=new_mode,
                                store=memory_store,
                                session_id=config.session,
                                config=new_mode_config,
                            )
                            memory_manager.load()
                            config.memory_mode = new_mode
                            config.memory_config = new_mode_config
                            # Wire LLM and embeddings into the new manager
                            memory_manager.set_llm(llm)
                            _try_configure_embeddings(memory_manager, config)
                            # Rebuild system prompt with new mode additions
                            mode_adds = memory_manager.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                            )

                            # Re-apply tool presets for the new mode:
                            # rebuild from _session.all_tool_originals (which has
                            # every tool before splitting) so dynamically
                            # activated tools aren't lost.
                            if tool_filter is None:
                                registry.tools = dict(_session.all_tool_originals)
                                active_dict, available_tools = _apply_tool_preset(
                                    registry, new_mode
                                )
                                if available_tools:
                                    registry.tools = active_dict
                                _session.loaded_tools.clear()
                                # Re-wrap tools with safety interceptors
                                tools.clear()
                                for tn, tl in registry.tools.items():
                                    if registry.requires_confirmation(tn):
                                        tools.append(
                                            create_safe_tool_wrapper(tl, tn, registry, approvals)
                                        )
                                    else:
                                        tools.append(tl)
                                # Add request_tools meta-tool if needed
                                _preset_names = TOOL_PRESETS.get(new_mode, set())
                                if available_tools:
                                    rt = _create_request_tools_tool(
                                        available_tools,
                                        _session.all_tool_descriptions,
                                        active_names=set(registry.tools.keys()),
                                        protected_names=_preset_names,
                                    )
                                    if rt:
                                        tools.append(rt)

                            # Update slash command references
                            slash_cmds.memory_manager = memory_manager
                            slash_cmds.system_prompt = system_prompt
                            slash_cmds.available_tools = available_tools
                            if console is not None:
                                console.print(
                                    f"[green]Switched to [bold]{new_mode}[/bold] mode.[/green]"
                                )
                            else:
                                print(f"Switched to {new_mode} mode.")
                            log.info(f"Live mode switch: {new_mode}")
                        except Exception as exc:
                            restored = session_orch.rollback(_snap, tools_list=tools)
                            memory_manager = restored["memory_manager"]
                            system_prompt = restored["system_prompt"]
                            available_tools = restored["available_tools"]
                            registry.tools = _snap.registry_tools
                            log.error(f"Mode switch failed: {exc}")
                            if console is not None:
                                console.print(f"[red]Mode switch failed:[/red] {exc}")
                            else:
                                print(f"Mode switch failed: {exc}")

                    elif isinstance(result, str) and result.startswith("switch_model:"):
                        new_model = result.split(":", 1)[1]
                        _snap = session_orch.snapshot(
                            system_prompt=system_prompt,
                        )
                        try:
                            # Cross-provider model resolution
                            alias, mc = config.find_model_entry(new_model)
                            if mc is not None and new_model not in config.models:
                                if console is not None:
                                    console.print(
                                        f"[dim]Resolved [bold]{new_model}[/bold]"
                                        f" \u2192 provider=[bold]{mc.provider}[/bold]"
                                        + (f", alias=[bold]{alias}[/bold]" if alias else "")
                                        + "[/dim]"
                                    )
                                else:
                                    resolved_msg = (
                                        f"Resolved {new_model} \u2192 provider={mc.provider}"
                                    )
                                    if alias:
                                        resolved_msg += f", alias={alias}"
                                    print(resolved_msg)
                            config.model = alias if alias else new_model
                            _resolve_model(config)

                            # Get updated provider config (clone with model params merged)
                            provider_config = config.resolve_provider_config()
                            actual_model = config.model or provider_config.get_model()

                            # Create new LLM
                            new_llm = create_llm_from_provider_config(provider_config)

                            mode_adds = memory_manager.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                            )
                            slash_cmds.system_prompt = system_prompt

                            # All potential failures are past — now atomically swap
                            old_llm = llm
                            llm = new_llm
                            max_context_tokens = provider_config.num_ctx or _DEFAULT_CONTEXT_WINDOW
                            _cleanup_resources.append(llm)

                            # Update hybrid memory LLM reference
                            memory_manager.set_llm(llm)

                            # Reconfigure tools that depend on provider/model
                            configure_delegate_tool(config, status_callback=_delegation_status)
                            configure_deep_think_tool(config)

                            # Rebuild compression LLM for new provider/model
                            if config.context_compression:
                                try:
                                    compression_llm = create_compression_llm(
                                        config.context_compression_model, config
                                    )
                                except Exception:
                                    compression_llm = None
                            else:
                                compression_llm = None

                            # Close old LLM last — it's no longer referenced
                            _close_llm(old_llm)
                            advance_llm_generation()
                            if old_llm in _cleanup_resources:
                                _cleanup_resources.remove(old_llm)

                            if console is not None:
                                prov = config.provider
                                console.print(
                                    f"[green]Switched to model "
                                    f"[bold]{actual_model}[/bold] "
                                    f"[dim]({prov})[/dim][/green]"
                                )
                            else:
                                print(f"Switched to model {actual_model} " f"({config.provider})")
                            log.info(
                                f"Live model switch: {actual_model} "
                                f"(provider: {config.provider})"
                            )
                        except Exception as exc:
                            restored = session_orch.rollback(_snap)
                            system_prompt = restored["system_prompt"]
                            provider_config = config.resolve_provider_config()
                            log.error(f"Model switch failed: {exc}")
                            friendly = _friendly_error(exc, provider=config.provider)
                            if console is not None:
                                console.print(f"[red]Model switch failed:[/red] {friendly}")
                            else:
                                print(f"Model switch failed: {friendly}")

                    elif isinstance(result, str) and result.startswith("switch_provider:"):
                        new_provider = result.split(":", 1)[1]
                        _snap = session_orch.snapshot(
                            system_prompt=system_prompt,
                        )
                        try:
                            config.provider = new_provider
                            config._active_model = None
                            provider_config = config.resolve_provider_config()

                            # Update model to the new provider's default
                            config.model = provider_config.get_model()

                            # Create new LLM
                            new_llm = create_llm_from_provider_config(provider_config)

                            mode_adds = memory_manager.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                            )
                            slash_cmds.system_prompt = system_prompt

                            # All potential failures are past — now atomically swap
                            old_llm = llm
                            llm = new_llm
                            max_context_tokens = provider_config.num_ctx or _DEFAULT_CONTEXT_WINDOW
                            _cleanup_resources.append(llm)

                            # Update hybrid memory LLM reference
                            memory_manager.set_llm(llm)

                            # Reconfigure tools that depend on provider/model
                            configure_delegate_tool(config, status_callback=_delegation_status)
                            configure_deep_think_tool(config)

                            # Rebuild compression LLM for new provider/model
                            if config.context_compression:
                                try:
                                    compression_llm = create_compression_llm(
                                        config.context_compression_model, config
                                    )
                                except Exception:
                                    compression_llm = None
                            else:
                                compression_llm = None

                            # Close old LLM last — it's no longer referenced
                            _close_llm(old_llm)
                            advance_llm_generation()
                            if old_llm in _cleanup_resources:
                                _cleanup_resources.remove(old_llm)

                            actual_model = provider_config.get_model()
                            if console is not None:
                                console.print(
                                    f"[green]Switched to provider "
                                    f"[bold]{new_provider}[/bold] "
                                    f"[dim](model: {actual_model})[/dim][/green]"
                                )
                            else:
                                print(
                                    f"Switched to provider {new_provider} "
                                    f"(model: {actual_model})"
                                )
                            log.info(
                                f"Live provider switch: {new_provider} " f"(model: {actual_model})"
                            )
                        except Exception as exc:
                            restored = session_orch.rollback(_snap)
                            system_prompt = restored["system_prompt"]
                            provider_config = config.resolve_provider_config()
                            log.error(f"Provider switch failed: {exc}")
                            friendly = _friendly_error(exc, provider=new_provider)
                            if console is not None:
                                console.print(f"[red]Provider switch failed:[/red] {friendly}")
                            else:
                                print(f"Provider switch failed: {friendly}")

                    elif isinstance(result, str) and result.startswith("switch_session:"):
                        new_session = result.split(":", 1)[1]
                        _snap = session_orch.snapshot(
                            memory_manager=memory_manager,
                            system_prompt=system_prompt,
                        )
                        try:
                            # Save current session
                            memory_manager.save()

                            # Switch session ID
                            config.session = new_session

                            # Create new memory manager for the new session
                            new_mm = MemoryFactory.create(
                                mode=config.memory_mode,
                                store=memory_store,
                                session_id=new_session,
                                config=config.memory_config,
                            )
                            new_mm.load()

                            # Rebuild system prompt (mode additions may differ)
                            mode_adds = new_mm.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                            )
                            # Success — commit the new memory manager
                            memory_manager = new_mm

                            # Reset per-session tool state so disabled/loaded
                            # tools from the previous session don't leak over.
                            _session.reset_for_new_session()

                            # Wire LLM and embeddings into the new manager
                            memory_manager.set_llm(llm)
                            _try_configure_embeddings(memory_manager, config)

                            # Update slash command references
                            slash_cmds.memory_manager = memory_manager
                            slash_cmds.system_prompt = system_prompt

                            # Update Python exec tool session
                            configure_python_exec_tool(config)

                            msg_count = memory_manager.get_message_count()
                            if console is not None:
                                console.print(
                                    f"[green]Switched to session "
                                    f"[bold]{new_session}[/bold] "
                                    f"[dim]({msg_count} messages)[/dim][/green]"
                                )
                            else:
                                print(
                                    f"Switched to session {new_session} " f"({msg_count} messages)"
                                )
                            log.info(
                                f"Live session switch: {new_session} " f"({msg_count} messages)"
                            )
                        except Exception as exc:
                            restored = session_orch.rollback(_snap)
                            memory_manager = restored["memory_manager"]
                            system_prompt = restored["system_prompt"]
                            log.error(f"Session switch failed: {exc}")
                            if console is not None:
                                console.print(f"[red]Session switch failed:[/red] {exc}")
                            else:
                                print(f"Session switch failed: {exc}")

                    elif isinstance(result, str) and result.startswith("load_tool:"):
                        load_name = result.split(":", 1)[1]
                        if load_name in available_tools:
                            tool_obj = available_tools.pop(load_name)
                            apply_output_cap(tool_obj, _tool_output_cap)
                            if registry.requires_confirmation(load_name):
                                if _session.no_confirm:
                                    approvals.add(load_name)
                                tool_obj = create_safe_tool_wrapper(
                                    tool_obj, load_name, registry, approvals
                                )
                            tools.append(tool_obj)
                            registry.tools[load_name] = _session.all_tool_originals.get(
                                load_name, tool_obj
                            )
                            _session.loaded_tools.add(load_name)
                            if console is not None:
                                console.print(
                                    f"[green]Tool [bold]{load_name}[/bold] loaded.[/green]"
                                )
                            else:
                                print(f"Tool '{load_name}' loaded.")
                        else:
                            if console is not None:
                                console.print(
                                    f"[yellow]Tool '{load_name}' is not available "
                                    f"to load.[/yellow]"
                                )
                            else:
                                print(f"Tool '{load_name}' is not available to load.")

                    elif isinstance(result, str) and result.startswith("deep_think:"):
                        # ── Hybrid /think: gather → analyze → synthesize ──
                        think_task = result.split(":", 1)[1]
                        try:
                            from datetime import date as _date

                            _today = _date.today().strftime("%B %d, %Y")

                            # Classify the task to pick specialised prompts
                            think_cat = classify_think_task(think_task, llm=llm)

                            # Stage 1: Gather data via the agent
                            if console is not None:
                                console.print(
                                    f"[dim]Stage 1/2:[/dim] Gathering data "
                                    f"[dim](strategy: {think_cat.name})[/dim]…"
                                )
                            else:
                                print(
                                    f"Stage 1/2: Gathering data " f"(strategy: {think_cat.name})…"
                                )

                            gather_prompt = think_cat.gather_template.replace(
                                "{today}", _today
                            ).replace("{task}", think_task)
                            gather_context = memory_manager.prepare_context(gather_prompt)
                            gather_msgs: list = []

                            _think_compression_llm = None
                            if config.context_compression_model:
                                _think_compression_llm = create_compression_llm(
                                    config.context_compression_model, config
                                )

                            _spinner.start()
                            try:
                                gather_output = run_agent(
                                    gather_prompt,
                                    gather_context.messages,
                                    registry,
                                    approvals,
                                    context_prefix=gather_context.context_prefix,
                                    callbacks=callbacks if callbacks else None,
                                    result_messages=gather_msgs,
                                    llm=llm,
                                    system_prompt=system_prompt,
                                    available_tools=(
                                        dict(available_tools)
                                        if (available_tools or TOOL_PRESETS.get(config.memory_mode))
                                        else None
                                    ),
                                    active_tools_list=tools,
                                    max_context_tokens=max_context_tokens,
                                    preset_tools=TOOL_PRESETS.get(config.memory_mode, set()),
                                    context_compression=config.context_compression,
                                    compression_min_age=config.context_compression_min_age,
                                    compression_min_chars=config.context_compression_min_chars,
                                    compression_llm=_think_compression_llm,
                                    session_state=_session,
                                    confirmation_ui=_rich_ui,
                                    on_tool_expansion=_tool_expansion_ui,
                                    parallel_tool_execution=config.parallel_tool_execution,
                                )
                            finally:
                                _spinner.stop()

                            # Stage 2: Deep analysis with gathered data
                            if console is not None:
                                console.print("[dim]Stage 2/2:[/dim] Deep analysis…")
                            else:
                                print("Stage 2/2: Deep analysis…")

                            from src.tools.deep_think import deep_think

                            tool_data = collect_tool_outputs(gather_msgs)
                            analysis_preamble = think_cat.analysis_preamble.replace(
                                "{today}", _today
                            )
                            context_parts: list[str] = [f"## Instructions\n\n{analysis_preamble}"]
                            if tool_data.strip():
                                context_parts.append(
                                    "## Gathered data (from tool calls)\n\n" + tool_data
                                )
                            if gather_output and gather_output.strip():
                                context_parts.append(
                                    "## Agent's research findings\n\n" + gather_output
                                )
                            full_context = "\n\n---\n\n".join(context_parts)

                            # Reframe the task for Stage 2 using the
                            # category-specific framing.
                            analysis_task = think_cat.stage2_task_framing.replace(
                                "{task}", think_task
                            )

                            _spinner.start()
                            try:
                                think_result = deep_think(
                                    task=analysis_task,
                                    context=full_context or "",
                                    max_iterations=3,
                                    num_branches=3,
                                    beam_width=2,
                                )
                            finally:
                                _spinner.stop()

                            # Display result
                            if console is not None and Markdown is not None:
                                console.print()
                                console.print(
                                    Panel(
                                        Markdown(preserve_tables_for_markdown(think_result)),
                                        title="Deep Think",
                                        border_style="magenta",
                                        padding=(1, 2),
                                    )
                                )
                            else:
                                print()
                                print(think_result)

                            # Save to memory
                            memory_manager.update(f"/think {think_task}", think_result)
                            memory_manager.save()

                        except KeyboardInterrupt:
                            _spinner.stop()
                            if console is not None:
                                console.print(
                                    "\n[yellow]Deep Think interrupted.[/yellow]"
                                    " [dim]Edit your prompt or press Enter to re-send.[/dim]"
                                )
                            else:
                                print(
                                    "\nDeep Think interrupted. Edit your prompt or press Enter to re-send."
                                )
                            prefill_next_input(f"/think {think_task}")
                        except Exception as exc:
                            _spinner.stop()
                            friendly = _friendly_error(exc, provider=config.provider)
                            if console is not None:
                                console.print(f"[red]Deep Think failed:[/red] {friendly}")
                            else:
                                print(f"Deep Think failed: {friendly}")
                            if config.debug:
                                import traceback

                                traceback.print_exc()

                    elif isinstance(result, str) and result.startswith("delegate:"):
                        # ── Forced /delegate: decompose → parallel delegate ──
                        delegate_task_text = result.split(":", 1)[1]
                        try:
                            if console is not None:
                                console.print(
                                    "[dim]Decomposing task for parallel delegation…[/dim]"
                                )
                            else:
                                print("Decomposing task for parallel delegation…")

                            _spinner.start()
                            try:
                                delegate_output = force_delegation(
                                    user_input=delegate_task_text,
                                    agent_response="",
                                    tool_outputs="",
                                    config=config,
                                    log=log,
                                )
                            finally:
                                _spinner.stop()

                            # Display result
                            if console is not None and Markdown is not None:
                                console.print()
                                console.print(
                                    Panel(
                                        Markdown(preserve_tables_for_markdown(delegate_output)),
                                        title="Delegation Results",
                                        border_style="cyan",
                                        padding=(1, 2),
                                    )
                                )
                            else:
                                print()
                                print(delegate_output)

                            # Save to memory
                            memory_manager.update(
                                f"/delegate {delegate_task_text}", delegate_output
                            )
                            memory_manager.save()

                        except KeyboardInterrupt:
                            _spinner.stop()
                            if console is not None:
                                console.print(
                                    "\n[yellow]Delegation interrupted.[/yellow]"
                                    " [dim]Edit your prompt or press Enter to re-send.[/dim]"
                                )
                            else:
                                print(
                                    "\nDelegation interrupted. Edit your prompt or press Enter to re-send."
                                )
                            prefill_next_input(f"/delegate {delegate_task_text}")
                        except Exception as exc:
                            _spinner.stop()
                            friendly = _friendly_error(exc, provider=config.provider)
                            if console is not None:
                                console.print(f"[red]Delegation failed:[/red] {friendly}")
                            else:
                                print(f"Delegation failed: {friendly}")
                            if config.debug:
                                import traceback

                                traceback.print_exc()

                    elif result == "run_setup":
                        from src.setup_wizard import run_setup_wizard

                        try:
                            run_setup_wizard()
                        except SystemExit:
                            continue

                        # Reload config and rebuild LLM connection
                        try:
                            config = load_config(args)
                            provider_config = config.resolve_provider_config()
                            new_llm = create_llm_from_provider_config(provider_config)
                            _close_llm(llm)
                            advance_llm_generation()
                            if llm in _cleanup_resources:
                                _cleanup_resources.remove(llm)
                            llm = new_llm
                            max_context_tokens = provider_config.num_ctx or _DEFAULT_CONTEXT_WINDOW
                            _cleanup_resources.append(llm)

                            memory_manager.set_llm(llm)
                            configure_delegate_tool(config, status_callback=_delegation_status)
                            configure_deep_think_tool(config)

                            slash_cmds.config = config
                            actual_model = provider_config.get_model()
                            if console is not None:
                                console.print(
                                    f"\n[green]Config reloaded — using "
                                    f"[bold]{config.provider}[/bold] "
                                    f"[dim](model: {actual_model})[/dim][/green]"
                                )
                            else:
                                print(
                                    f"\nConfig reloaded — using {config.provider} "
                                    f"(model: {actual_model})"
                                )
                        except Exception as exc:
                            friendly = _friendly_error(exc, provider=config.provider)
                            if console is not None:
                                console.print(
                                    f"[yellow]Config reload failed:[/yellow] {friendly}\n"
                                    f"[dim]Continuing with previous settings.[/dim]"
                                )
                            else:
                                print(f"Config reload failed: {friendly}")
                                print("Continuing with previous settings.")

                    elif result == "rebuild_callbacks":
                        # Rebuild observability callbacks (e.g. after /debug toggle)
                        callbacks.clear()
                        if config.debug:
                            obs_handler = create_observability_handler(verbose=config.verbose)
                            if obs_handler:
                                callbacks.append(obs_handler)
                                log.debug("LLM observability handler rebuilt")

                    elif isinstance(result, str) and result.startswith("optimize:"):
                        # Force-optimize the prompt, then fall through to agent
                        user_input = result.split(":", 1)[1]
                        user_input = optimize_prompt(user_input, llm, force=True)
                        already_optimized = True
                    else:
                        continue

                    # All slash-command handlers return to the prompt
                    # except "optimize:" which falls through to agent execution.
                    if not (isinstance(result, str) and result.startswith("optimize:")):
                        continue

            # Backward compat: bare exit/quit/q still works
            if user_input.lower() in ["exit", "quit", "q"]:
                log.info("Session ended by user")
                print("\nGoodbye!")
                break

            # Start new request tracking
            new_request_id()

            # Reset blanket-forbid flag at the start of each prompt
            # cycle, before prepare_context() — if the previous cycle
            # raised inside prepare_context(), the flag would otherwise
            # stick and block all tools on the next cycle.
            _session.reset_for_new_prompt()

            # Log user message
            log_user_message(user_input)

            # Preserve the user's original phrasing before the optimizer
            # rewrites it — memory, classification, delegation, and the
            # output file must all record what the user actually typed.
            # Set before try so KeyboardInterrupt handlers can use it.
            original_input = user_input

            try:
                # Run prepare_context and optimize_prompt concurrently when optimizer is enabled.
                # The two operations are independent: the optimizer rewrites the prompt text
                # while prepare_context reads conversation history — no data dependency.
                if config.prompt_optimizer and not already_optimized:
                    import concurrent.futures as _cf

                    with _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep") as _pool:
                        _ctx_future = _pool.submit(memory_manager.prepare_context, user_input)
                        _opt_future = _pool.submit(optimize_prompt, user_input, llm)
                        context = _ctx_future.result()
                        user_input = _opt_future.result()
                    already_optimized = True
                else:
                    context = memory_manager.prepare_context(user_input)

                # Debug: log context details
                log.debug(
                    f"Context: mode={context.mode}, "
                    f"{context.context_messages_count} messages"
                    + (f", ~{context.token_estimate} tokens" if context.token_estimate else "")
                )

                wants_deep = user_wants_deep_think(original_input)

                agent_msgs: list = []

                _acc = _TokenAccumulator()
                _agent_cbs = (callbacks or []) + [_acc]
                _agent_t0 = _time_mod.monotonic()
                _spinner.start()
                try:
                    output = run_agent(
                        user_input,
                        context.messages,
                        registry,
                        approvals,
                        context_prefix=context.context_prefix,
                        callbacks=_agent_cbs,
                        result_messages=agent_msgs,
                        llm=llm,
                        system_prompt=system_prompt,
                        available_tools=(
                            dict(available_tools)
                            if (available_tools or TOOL_PRESETS.get(config.memory_mode))
                            else None
                        ),
                        active_tools_list=tools,
                        max_context_tokens=max_context_tokens,
                        preset_tools=TOOL_PRESETS.get(config.memory_mode, set()),
                        context_compression=config.context_compression,
                        compression_min_age=config.context_compression_min_age,
                        compression_min_chars=config.context_compression_min_chars,
                        compression_llm=compression_llm,
                        session_state=_session,
                        confirmation_ui=_rich_ui,
                        on_tool_expansion=_tool_expansion_ui,
                        parallel_tool_execution=config.parallel_tool_execution,
                    )
                finally:
                    _spinner.stop()

                # ── Enforce deep_think when the user requested it ──
                # Force-call if agent skipped it OR called with bad context.
                # Skip for tool-intensive tasks where the agent's tool
                # work is the primary deliverable.
                _research_output: str = ""
                if wants_deep and output:
                    _task_cat = classify_think_task(original_input, llm)
                    if _task_cat.tool_intensive:
                        log.info(
                            "Skipping force deep_think: task classified "
                            "as '%s' (tool-intensive — agent's tool work "
                            "is the primary output)",
                            _task_cat.name,
                        )
                    else:
                        called = was_deep_think_called(agent_msgs)
                        if not called or not deep_think_had_good_context(agent_msgs):
                            if called:
                                log.info(
                                    "deep_think was called but with "
                                    "inadequate context — forcing "
                                    "re-call with full data"
                                )
                            tool_data = collect_tool_outputs(agent_msgs)

                            # Run research delegate for web-sourced data
                            _rd_enabled = getattr(config, "research_delegate_enabled", True)
                            if _rd_enabled and agent_used_web_tools(agent_msgs):
                                fetched_urls = extract_fetched_urls(agent_msgs)
                                if fetched_urls:
                                    _rd_timeout = getattr(config, "research_delegate_timeout", 300)
                                    _rd_cap = getattr(
                                        config,
                                        "research_delegate_cap_ratio",
                                        RESEARCH_CAP_RATIO,
                                    )
                                    _spinner.start()
                                    try:
                                        _research_output = run_research_delegate(
                                            fetched_urls,
                                            original_input,
                                            max_context_tokens=max_context_tokens,
                                            timeout=_rd_timeout,
                                            cap_ratio=_rd_cap,
                                        )
                                    finally:
                                        _spinner.stop()

                            _spinner.start()
                            try:
                                output = force_deep_think(
                                    original_input,
                                    output,
                                    tool_data,
                                    log,
                                    research_context=_research_output or None,
                                )
                            finally:
                                _spinner.stop()

                # ── Enforce delegation when the query warrants it ──
                if (
                    not wants_deep
                    and output
                    and user_wants_delegation(original_input)
                    and not was_delegation_called(agent_msgs)
                    and config.delegate_enabled
                ):
                    log.info(
                        "Auto-detected delegation-worthy query but agent "
                        "did not delegate — forcing parallel delegation"
                    )
                    tool_data = collect_tool_outputs(agent_msgs)
                    _spinner.start()
                    try:
                        forced = force_delegation(original_input, output, tool_data, config, log)
                        if forced and forced != output:
                            output = forced
                    finally:
                        _spinner.stop()

                # Snapshot turn messages before the execution phase can
                # append its own HumanMessage to agent_msgs, which would
                # cause extract_turn_messages to return only exec messages.
                turn_msgs = extract_turn_messages(agent_msgs)

                # ── Execution phase: act on analysis ──────────────
                if (
                    output
                    and prompt_requests_action(original_input)
                    and not agent_performed_writes(agent_msgs)
                ):
                    log.info(
                        "Prompt requests file actions but none were "
                        "performed — running execution phase"
                    )
                    _spinner.start()
                    try:
                        exec_output, exec_msgs = run_execution_phase(
                            output,
                            user_input,
                            context.messages,
                            registry,
                            approvals,
                            context_prefix=context.context_prefix,
                            callbacks=_agent_cbs,
                            llm=llm,
                            system_prompt=system_prompt,
                            available_tools=(
                                dict(available_tools) if available_tools else available_tools
                            ),
                            active_tools_list=tools,
                            max_context_tokens=max_context_tokens,
                            preset_tools=TOOL_PRESETS.get(config.memory_mode, set()),
                            session_state=_session,
                            on_tool_expansion=_tool_expansion_ui,
                            parallel_tool_execution=config.parallel_tool_execution,
                        )
                    finally:
                        _spinner.stop()
                    if exec_output:
                        output = output + "\n\n---\n\n" + exec_output
                        turn_msgs.extend(exec_msgs)

                _agent_elapsed = _time_mod.monotonic() - _agent_t0

                # Guard: never display an empty response
                if not output or not output.strip():
                    output = _EMPTY_RESPONSE_MSG
                    log.error("Empty output after run_agent")

                # Log agent response
                log_agent_response(output)

                # Display output with rich formatting if available
                if console is not None and Markdown is not None:
                    console.rule("Agent", style="blue")
                    console.print(
                        Padding(
                            Markdown(preserve_tables_for_markdown(output)),
                            (1, 0, 1, 2),
                        )
                    )
                    stats_text = _format_stats_line(_agent_elapsed, _acc)
                    if stats_text:
                        console.print(Align.right(Text.from_markup(stats_text)))
                    console.rule(style="dim blue")
                else:
                    print(f"\nAgent: {output}")

                # Append to output file if -o was specified in interactive mode
                if output_file and _is_valid_response(output):
                    try:
                        out_path = Path(output_file)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(f"## You\n\n{original_input}\n\n")
                            f.write(f"## Agent\n\n{output}\n\n---\n\n")
                    except Exception as e:
                        log.error(f"Error appending to output file: {e}")

                # Only save valid responses to history (skip empty/error).
                # Pass the full agent chain so the agent can continue
                # iterating on complex tasks across restarts (Ralph Loop).
                if _is_valid_response(output):
                    memory_manager.update(original_input, output, agent_messages=turn_msgs or None)
                    memory_manager.save()
                else:
                    log.warning("Skipping history save: empty or error response")

            except UserCancelledRun:
                if console:
                    console.print("[yellow]Workflow cancelled.[/yellow]")
                else:
                    print("Workflow cancelled.")
                prefill_next_input(original_input)
                continue

            except KeyboardInterrupt:
                _spinner.stop()
                if console:
                    console.print(
                        "\n[yellow]Interrupted.[/yellow]"
                        " [dim]Edit your prompt or press Enter to re-send.[/dim]"
                    )
                else:
                    print("\nInterrupted. Edit your prompt or press Enter to re-send.")
                prefill_next_input(original_input)
                continue

            except Exception as e:
                log_error(e, context="Agent execution error", include_trace=True)
                friendly = _friendly_error(e, provider=config.provider)
                if console:
                    console.print(f"[red]Error:[/red] {friendly}")
                else:
                    print(f"Error: {friendly}")
                if config.debug:
                    import traceback

                    traceback.print_exc()
                continue

        except KeyboardInterrupt:
            # Ctrl+C at the prompt clears the line and returns to a fresh
            # prompt (same as Bash).  Use Ctrl+D or /quit to exit.
            _spinner.stop()
            print()
            continue
        except EOFError:
            log.info("Session ended (EOF)")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            log_error(e, context="Unexpected error", include_trace=True)
            print(f"\nError: {e}")
            sys.exit(1)

    # Persist memory on exit — ensure any background summarization
    # completes and the full history (including the last turn) is saved.
    try:
        memory_manager.save()
    except Exception:  # noqa: BLE001
        pass  # best-effort; process is exiting


if __name__ == "__main__":
    main()
