#!/usr/bin/env python3
"""
Cogtrix Agent - CLI Entry Point
A modular LangChain agent with extensible tools and safety features.
Supports multiple LLM providers: OpenAI, Ollama.
"""

import difflib
import sys
from pathlib import Path
from typing import Any

from src.config import ConfigError
from src.logging_config import (
    setup_logging,
)
from src.orchestration.compression import (
    _CHARS_PER_TOKEN,
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    apply_message_compression,
)
from src.orchestration.runner import (  # noqa: F401
    ToolCallLogger,
    build_tool_results_response,
    extract_ai_content,
    extract_response,
    format_agent_error,
    invalidate_llm_caches,
    is_valid_response,
    log_tool_calls_from_result,
    run_agent,
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

# Module-level globals for runtime injection
_console: Any | None = None
_session: Any | None = None


def configure(console: Any, session: Any) -> None:
    """Configure the module with runtime dependencies."""
    global _console, _session
    _console = console
    _session = session


try:
    from src.mcp_client import MCP_AVAILABLE, MCPManager, MCPServerConfig
except ImportError:
    MCP_AVAILABLE = False
    MCPManager = None  # type: ignore[misc, assignment]
    MCPServerConfig = None  # type: ignore[misc, assignment]


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
        "save_to_knowledge_base",
    ],
}

# Secondary category map for tools not covered by _TOOL_CATEGORIES.
# These override the "Other" fallback and are displayed after core categories.
_TOOL_CATEGORY_MAP: dict[str, str] = {
    # Git
    "git_add": "Git",
    "git_checkout": "Git",
    "git_commit": "Git",
    "git_create_branch": "Git",
    "git_diff": "Git",
    "git_log": "Git",
    "git_status": "Git",
    # GitHub
    "gh_comment_issue": "GitHub",
    "gh_create_issue": "GitHub",
    "gh_get_file": "GitHub",
    "gh_list_prs": "GitHub",
    # WhatsApp
    "whatsapp_check": "WhatsApp",
    "whatsapp_contacts": "WhatsApp",
    "whatsapp_send": "WhatsApp",
    "whatsapp_send_image": "WhatsApp",
    # Cron
    "cron_add": "Cron",
    "cron_list": "Cron",
    "cron_remove": "Cron",
    # Goals
    "abandon_goal": "Goals",
    "add_subgoal": "Goals",
    "complete_goal": "Goals",
    "list_goals": "Goals",
    "set_goal": "Goals",
    # Tasks & Agents
    "cancel_task": "Tasks & Agents",
    "get_task_result": "Tasks & Agents",
    "get_task_status": "Tasks & Agents",
    "list_tasks": "Tasks & Agents",
    "spawn_agent": "Tasks & Agents",
    "send_to_agent": "Tasks & Agents",
    "read_agent_inbox": "Tasks & Agents",
    "report_progress": "Tasks & Agents",
    # Development
    "generate_tests": "Development",
    "patch_file": "Development",
    "self_improve": "Development",
}

# Ordered list of secondary categories for display (after core categories).
_SECONDARY_CATEGORY_ORDER: list[str] = [
    "Git",
    "GitHub",
    "WhatsApp",
    "Cron",
    "Goals",
    "Tasks & Agents",
    "Development",
]

# Build reverse lookup: tool_name → category
_TOOL_TO_CATEGORY: dict[str, str] = {}
for _cat, _names in _TOOL_CATEGORIES.items():
    for _tname in _names:
        _TOOL_TO_CATEGORY[_tname] = _cat
for _tname, _cat in _TOOL_CATEGORY_MAP.items():
    _TOOL_TO_CATEGORY.setdefault(_tname, _cat)


_tool_logger = ToolCallLogger()
_is_valid_response = is_valid_response
_format_agent_error = format_agent_error
_extract_ai_content = extract_ai_content


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
        self.project_context_path: str | None = None
        self.last_input: str | None = None  # last user prompt, for /retry
        self.compression_llm: Any = None
        self.max_context_tokens: int | None = None
        self.last_input_tokens: int = 0

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
            # Suggest closest known command using difflib
            known_commands = list(self._commands.keys()) + [
                a for aliases in [c.aliases for c in self._commands.values()] for a in aliases
            ]
            suggestions = difflib.get_close_matches(
                cmd_name.lower(), [c.lower() for c in known_commands], n=3, cutoff=0.3
            )
            if suggestions:
                sug_list = ", ".join("/" + s for s in suggestions)
                print(f"Did you mean one of: {sug_list}?")
            else:
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
                # Suggest closest known command using difflib
                known_commands = list(self._commands.keys()) + [
                    a for aliases in [c.aliases for c in self._commands.values()] for a in aliases
                ]
                suggestions = difflib.get_close_matches(
                    args.lower(), [c.lower() for c in known_commands], n=3, cutoff=0.3
                )
                if suggestions:
                    sug_list = ", ".join("/" + s for s in suggestions)
                    print(f"Did you mean one of: {sug_list}?")
                else:
                    print("Type /help for a list of commands.")
                return "continue"

            if _console is not None:
                aliases = (
                    f"[dim]Aliases: {', '.join('/' + a for a in cmd.aliases)}[/dim]\n"
                    if cmd.aliases
                    else ""
                )
                body = f"[bold]/{cmd.name}[/bold]  {cmd.short_help}\n{aliases}\n{cmd.long_help}"
                _console.print(Panel(body, border_style="cyan", padding=(1, 2)))
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
        if _console is not None:
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
            provider_cfg, model_cfg = cfg.resolve_llm_config()
        except (ValueError, KeyError, AttributeError, ConfigError) as exc:
            if _console is not None:
                _console.print(f"[red]Provider configuration error:[/red] {exc}")
            else:
                print(f"Provider configuration error: {exc}")
            return "continue"

        sp = self.system_prompt
        mm_mcp = self.mcp_manager
        pcp = self.project_context_path
        if _console is not None and Panel is not None:
            _info_rich(cfg, provider_cfg, model_cfg, stats, msg_count, sp, mm_mcp, pcp)
        else:
            _info_plain(cfg, provider_cfg, model_cfg, stats, msg_count, sp, mm_mcp, pcp)
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
                if _console is not None:
                    _console.print("[dim]Usage: /tools enable <tool-name>[/dim]")
                else:
                    print("Usage: /tools enable <tool-name>")
                return "continue"
            enabled_name: str | None = None
            if term in _session.denials:
                _session.denials.discard(term)
                enabled_name = term
            else:
                matches = [n for n in _session.denials if term in n]
                if len(matches) == 1:
                    _session.denials.discard(matches[0])
                    enabled_name = matches[0]
                elif len(matches) > 1:
                    msg = f"Ambiguous: '{term}' matches {', '.join(matches)}. Be more specific."
                    if _console is not None:
                        _console.print(f"[yellow]{msg}[/yellow]")
                    else:
                        print(msg)
                else:
                    if _console is not None:
                        _console.print(f"[dim]Tool '{term}' is not disabled.[/dim]")
                    else:
                        print(f"Tool '{term}' is not disabled.")
            if enabled_name is not None:
                if _console is not None:
                    _console.print(f"[green]✓ Tool '{enabled_name}' re-enabled.[/green]")
                else:
                    print(f"Tool '{enabled_name}' re-enabled.")
                tool_obj = (
                    _session.all_tool_originals.get(enabled_name)
                    or available.get(enabled_name)
                    or reg.tools.get(enabled_name)
                )
                if tool_obj is not None:
                    desc = getattr(tool_obj, "description", "") or ""
                    short_desc = desc.split(". ")[0].split(".\n")[0]
                    if len(short_desc) > 120:
                        short_desc = short_desc[:117] + "..."
                    if short_desc:
                        if _console is not None:
                            _console.print(f"  [dim]{short_desc}[/dim]")
                        else:
                            print(f"  {short_desc}")
                    func = getattr(tool_obj, "_uncapped_func", None) or getattr(
                        tool_obj, "func", None
                    )
                    module_name = getattr(func, "__module__", "") if func is not None else ""
                    module = sys.modules.get(module_name) if module_name else None
                    checker = getattr(module, "is_configured", None) if module is not None else None
                    if checker is not None:
                        try:
                            configured = checker()
                        except Exception:  # noqa: BLE001
                            configured = True
                        if not configured:
                            _not_cfg = "  ⚠ Not configured — check environment variables or config."
                            if _console is not None:
                                _console.print(f"  [yellow]{_not_cfg.strip()}[/yellow]")
                            else:
                                print(_not_cfg)
            return "continue"

        if args.startswith("disable") and (len(args) == 7 or args[7] == " "):
            term = args[7:].strip()
            if not term:
                if _console is not None:
                    _console.print("[dim]Usage: /tools disable <tool-name>[/dim]")
                else:
                    print("Usage: /tools disable <tool-name>")
                return "continue"
            all_known = (
                set(reg.tools.keys())
                | set(available.keys())
                | set(_session.all_tool_originals.keys())
            )
            if term in all_known:
                _session.denials.add(term)
                _session.pinned_tools.discard(term)
                _session.loaded_tools.discard(term)
                if _console is not None:
                    _console.print(f"[yellow]Tool '{term}' disabled for this session.[/yellow]")
                else:
                    print(f"Tool '{term}' disabled for this session.")
            else:
                matches = [n for n in all_known if term in n]
                if len(matches) == 1:
                    _session.denials.add(matches[0])
                    _session.pinned_tools.discard(matches[0])
                    _session.loaded_tools.discard(matches[0])
                    if _console is not None:
                        _console.print(
                            f"[yellow]Tool '{matches[0]}' disabled for this session.[/yellow]"
                        )
                    else:
                        print(f"Tool '{matches[0]}' disabled for this session.")
                elif len(matches) > 1:
                    msg = f"Ambiguous: '{term}' matches {', '.join(matches)}. Be more specific."
                    if _console is not None:
                        _console.print(f"[yellow]{msg}[/yellow]")
                    else:
                        print(msg)
                else:
                    if _console is not None:
                        _console.print(f"[red]Unknown tool '{term}'.[/red]")
                    else:
                        print(f"Unknown tool '{term}'.")
            return "continue"

        if args.startswith("load") and (len(args) == 4 or args[4] == " "):
            term = args[4:].strip()
            if not term:
                if _console is not None:
                    _console.print("[dim]Usage: /tools load <tool-name>[/dim]")
                else:
                    print("Usage: /tools load <tool-name>")
                return "continue"
            if term in reg.tools:
                if term not in _session.pinned_tools:
                    # Promote agent-loaded tool to pinned
                    _session.pinned_tools.add(term)
                    _session.loaded_tools.add(term)
                    if _console is not None:
                        _console.print(
                            f"[green]✓ Tool '{term}' pinned"
                            " (was agent-loaded, now persists).[/green]"
                        )
                    else:
                        print(f"Tool '{term}' pinned (was agent-loaded, now persists).")
                else:
                    if _console is not None:
                        _console.print(f"[dim]Tool '{term}' is already loaded and pinned.[/dim]")
                    else:
                        print(f"Tool '{term}' is already loaded and pinned.")
                return "continue"
            if term in _session.denials:
                msg = f"Tool '{term}' is disabled. Use '/tools enable {term}' first."
                if _console is not None:
                    _console.print(f"[yellow]{msg}[/yellow]")
                else:
                    print(msg)
                return "continue"
            if term in available:
                return f"load_tool:{term}"
            matches = [n for n in available if term in n]
            if len(matches) == 1:
                return f"load_tool:{matches[0]}"
            if len(matches) > 1:
                msg = f"Ambiguous: '{term}' matches {', '.join(matches)}. Be more specific."
                if _console is not None:
                    _console.print(f"[yellow]{msg}[/yellow]")
                else:
                    print(msg)
            else:
                # Check if already active via substring
                active_matches = [n for n in reg.tools if term in n]
                if len(active_matches) == 1:
                    if _console is not None:
                        _console.print(f"[dim]Tool '{active_matches[0]}' is already loaded.[/dim]")
                    else:
                        print(f"Tool '{active_matches[0]}' is already loaded.")
                else:
                    if _console is not None:
                        _console.print(f"[red]Unknown or unavailable tool '{term}'.[/red]")
                    else:
                        print(f"Unknown or unavailable tool '{term}'.")
            return "continue"

        if args.startswith("unload") and (len(args) == 6 or args[6] == " "):
            term = args[6:].strip()
            if not term:
                if _console is not None:
                    _console.print("[dim]Usage: /tools unload <tool-name>[/dim]")
                else:
                    print("Usage: /tools unload <tool-name>")
                return "continue"
            if term in _session.pinned_tools:
                return f"unload_tool:{term}"
            # Fuzzy match against pinned tools
            matches = [n for n in _session.pinned_tools if term in n]
            if len(matches) == 1:
                return f"unload_tool:{matches[0]}"
            if len(matches) > 1:
                msg = f"Ambiguous: '{term}' matches {', '.join(matches)}. Be more specific."
                if _console is not None:
                    _console.print(f"[yellow]{msg}[/yellow]")
                else:
                    print(msg)
            elif term in _session.loaded_tools:
                msg = f"Tool '{term}' was loaded by the agent and will be auto-unloaded next turn."
                if _console is not None:
                    _console.print(f"[dim]{msg}[/dim]")
                else:
                    print(msg)
            else:
                if _console is not None:
                    _console.print(f"[dim]Tool '{term}' is not currently loaded.[/dim]")
                else:
                    print(f"Tool '{term}' is not currently loaded.")
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

        if _console is not None and Table is not None:
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
        if _console is not None:
            _console.print(f"[green]✓ Cleared [bold]{count}[/bold] messages from memory.[/green]")
        else:
            print(f"✓ Cleared {count} messages from memory.")
        return "continue"

    def _cmd_undo(self, _args: str) -> str:
        """Handler for /undo — remove the last exchange from memory."""
        mm = self.memory_manager
        if not mm:
            print("Memory manager not available.")
            return "continue"
        removed = mm.pop_last_turn()
        if removed == 0:
            if _console is not None:
                _console.print("[dim]Nothing to undo — no previous exchange in memory.[/dim]")
            else:
                print("Nothing to undo — no previous exchange in memory.")
        else:
            if _console is not None:
                _console.print(
                    f"[green]✓ Undone:[/green] removed last exchange "
                    f"([dim]{removed} message(s) removed[/dim])"
                )
            else:
                print(f"✓ Undone: removed last exchange ({removed} message(s) removed)")
        return "continue"

    @staticmethod
    def _cmd_compact(self, args: str) -> str:
        """Handler for /compact — manually trigger context compression."""
        mm = self.memory_manager
        if not mm:
            print("Memory manager not available.")
            return "continue"

        messages = getattr(mm, "_messages", None)
        if not messages:
            if _console is not None:
                _console.print("[dim]Nothing to compress — no messages in memory.[/dim]")
            else:
                print("Nothing to compress — no messages in memory.")
            return "continue"

        aggressive = args.strip().lower() == "aggressive"

        llm = self.compression_llm or getattr(mm, "_llm", None)
        max_ctx = self.max_context_tokens or 16_384

        # For aggressive mode force the threshold trigger by computing effective max_context_tokens
        # from the actual message content so that total_chars >= threshold_chars always holds.
        if aggressive:
            total_chars = sum(
                len(c) for m in messages if isinstance((c := getattr(m, "content", "")), str)
            )
            if total_chars > 0:
                # threshold_chars = int(max_ctx * 4 * 0.72); we want total_chars >= threshold_chars
                # Setting max_ctx = total_chars // 3 gives threshold_chars ≈ total_chars * 0.96
                max_ctx = max(16_384, total_chars // 3)

        before_contents = [getattr(m, "content", None) for m in messages]

        _timeout_info: dict = {}

        # Show a percentage progress bar while compression LLM calls complete.
        def _run_compression(_on_progress=None):
            if aggressive:
                return apply_message_compression(
                    messages,
                    call_count=999,
                    compression_cache={},
                    llm=llm,
                    max_context_tokens=max_ctx,
                    min_age_cycles=0,
                    min_chars=0,
                    emergency_threshold=0.0,
                    timeout_info=_timeout_info,
                    progress_callback=_on_progress,
                )
            else:
                return apply_message_compression(
                    messages,
                    call_count=999,
                    compression_cache={},
                    llm=llm,
                    max_context_tokens=max_ctx,
                    min_age_cycles=COMPRESSION_MIN_AGE_CYCLES,
                    min_chars=COMPRESSION_MIN_CHARS,
                    timeout_info=_timeout_info,
                    progress_callback=_on_progress,
                )

        if _console is not None:
            try:
                from rich.align import Align
                from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
                from rich.rule import Rule

                class _RightProgress(Progress):
                    """Progress bar with a teal rule above, right-aligned."""

                    def get_renderables(self):
                        yield Rule(style="dim blue")
                        yield Align.right(self.make_tasks_table(self.tasks))

                with _RightProgress(
                    TextColumn("[cyan]Compressing context:[/cyan]"),
                    BarColumn(bar_width=20),
                    TaskProgressColumn(),
                    console=_console,
                    transient=True,
                ) as progress:
                    task_id = progress.add_task("compress", total=None)

                    def _on_progress(completed: int, total: int) -> None:
                        progress.update(task_id, total=total, completed=completed)

                    compressed = _run_compression(_on_progress)
            except Exception:
                compressed = _run_compression()
        else:
            compressed = _run_compression()

        changed_indices = [
            i
            for i, (before, after) in enumerate(zip(before_contents, compressed, strict=False))
            if before != getattr(after, "content", before)
        ]
        changed = len(changed_indices)

        if changed == 0:
            last_tokens = self.last_input_tokens
            pct = int(last_tokens / max_ctx * 100) if max_ctx and last_tokens else 0
            # Print directly to stdout so prompt_toolkit's post-dispatch
            # redraw doesn't erase the message.
            print(
                f"Nothing to compress. (Messages are too recent or too short. Context at {pct}%.)"
            )
            return "continue"

        mm._messages = compressed
        mm.save()

        tool_changed = sum(
            1 for i in changed_indices if type(messages[i]).__name__ == "ToolMessage"
        )
        ai_changed = sum(1 for i in changed_indices if type(messages[i]).__name__ == "AIMessage")

        before_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
        after_chars = sum(len(getattr(m, "content", "") or "") for m in compressed)
        reduction = int((1 - after_chars / before_chars) * 100) if before_chars else 0

        prefix = "Aggressive compression:" if aggressive else "Compressed:"
        if tool_changed > 0 or ai_changed > 0:
            parts = []
            if tool_changed > 0:
                parts.append(f"{tool_changed} tool result{'s' if tool_changed != 1 else ''}")
            if ai_changed > 0:
                parts.append(f"{ai_changed} assistant response{'s' if ai_changed != 1 else ''}")
            summary_str = " + ".join(parts) + " summarised."
        else:
            summary_str = f"{changed} message(s) summarised."

        # Timeout note computed later in the print() output path

        # Proportional scaling from the last real token count is far more
        # accurate than chars/_CHARS_PER_TOKEN (which assumes a fixed ratio
        # that varies wildly between prose ~4 and URLs ~1.5 chars/token).
        if self.last_input_tokens > 0 and before_chars > 0:
            estimated_tokens = int(self.last_input_tokens * (after_chars / before_chars))
        else:
            estimated_tokens = after_chars // _CHARS_PER_TOKEN
        self.last_input_tokens = estimated_tokens

        # Update toolbar stats so the next prompt shows updated context %
        try:
            from src.ui.stats import print_stats_footer as _print_stats_footer_compact

            _print_stats_footer_compact(
                console=_console,
                session_tokens=estimated_tokens,
                max_context_tokens=self.max_context_tokens or 16_384,
            )
        except Exception:
            pass

        # Use print() not _console.print() — prompt_toolkit's post-dispatch
        # redraw erases _console.print() output but leaves print() visible.
        _timeout_note_plain = ""
        if _timeout_info.get("timed_out"):
            _done = _timeout_info.get("completed", 0)
            _ttotal = _timeout_info.get("total", 0)
            _timeout_note_plain = f" ⚠ Timed out — {_done}/{_ttotal} items."
        print(f"✓ {prefix} {summary_str}{_timeout_note_plain}")
        print(f"  Context reduced by ~{reduction}% ({before_chars:,} → {after_chars:,} chars)")
        return "continue"

    def _cmd_retry(self, _args: str) -> str:
        """Handler for /retry — re-run the last prompt.

        Returns a special signal that is handled in the main loop,
        which sets user_input = last_input and falls through to processing.
        """
        if not self.last_input:
            if _console is not None:
                _console.print("[dim]Nothing to retry — no previous prompt in this session.[/dim]")
            else:
                print("Nothing to retry — no previous prompt in this session.")
            return "continue"
        # Signal to main loop (handled specially alongside /paste)
        return f"retry:{self.last_input}"

    @staticmethod
    def _cmd_think(self, args: str) -> str:
        """Handler for /think <task>.

        Returns a ``deep_think:<task>`` signal so the main loop can
        execute the hybrid gather → analyze → synthesize pipeline
        (the handler itself doesn't have access to the agent/tools).
        """
        if not args:
            if _console is not None:
                _console.print(
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
            if _console is not None:
                _console.print("[red]Deep Think tool is not available.[/red]")
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
            if _console is not None:
                _console.print(
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
            if _console is not None:
                _console.print("[red]Delegate tool is not available.[/red]")
            else:
                print("Delegate tool is not available.")
            return "continue"

        # Check if delegation is enabled in config
        cfg = self.config
        if cfg and not getattr(cfg, "delegate_enabled", True):
            if _console is not None:
                _console.print("[yellow]Delegation is disabled in configuration.[/yellow]")
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
            "reasoning": 30,
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
                if _console is not None:
                    _console.print(f"[red]Unknown mode:[/red] [bold]{target}[/bold]")
                    _console.print(f"[dim]Available modes: {valid}[/dim]")
                else:
                    print(f"Unknown mode: {target}")
                    print(f"Available modes: {valid}")
                return "continue"
            if target == cfg.memory_mode:
                if _console is not None:
                    _console.print(f"[dim]Already in [bold]{target}[/bold] mode.[/dim]")
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

        if _console is not None and Panel is not None:
            _mode_rich(cfg, _VALID_MODES, wm_sizes)
        else:
            _mode_plain(cfg, _VALID_MODES, wm_sizes)
        return "continue"

    @staticmethod
    def _cmd_memory(self, _args: str) -> str:
        """Handler for /memory."""
        mm = self.memory_manager
        cfg = self.config
        if not mm or not cfg:
            print("Memory not available.")
            return "continue"

        stats = mm.get_stats()
        msg_count = stats.get("total_messages", mm.get_message_count())
        session_id = stats.get("session_id", cfg.session)

        if msg_count == 0 and not stats.get("has_summary"):
            if _console is not None:
                _console.print(
                    "\n  [bold]Memory[/bold] — fresh session [dim](no history loaded)[/dim]\n"
                )
            else:
                print("\nMemory — fresh session (no history loaded)\n")
            return "continue"

        has_summary = stats.get("has_summary", False)
        vector_ready = stats.get("vector_recall_ready", False)
        summary_text: str | None = getattr(mm, "_summary", None)

        if _console is not None and Panel is not None:
            lines: list[str] = []
            lines.append(f"[bold cyan]Memory[/bold cyan] — session: {session_id}")
            lines.append(f"  [bold]{'Mode':<16s}[/bold]  {stats.get('mode', cfg.memory_mode)}")
            lines.append(f"  [bold]{'Messages':<16s}[/bold]  {msg_count} in history")
            summary_status = "✓ Rolling summary active" if has_summary else "✗ No summary yet"
            summary_style = "green" if has_summary else "dim"
            lines.append(
                f"  [bold]{'Summary':<16s}[/bold]  [{summary_style}]{summary_status}[/{summary_style}]"
            )
            if vector_ready:
                vec_count = stats.get("vector_count", "")
                vec_detail = f" ({vec_count} embeddings)" if vec_count else ""
                lines.append(
                    f"  [bold]{'Semantic recall':<16s}[/bold]  [green]✓ Active{vec_detail}[/green]"
                )
            else:
                lines.append(
                    f"  [bold]{'Semantic recall':<16s}[/bold]  [dim]✗ Not configured[/dim]"
                )
            if summary_text:
                truncated = len(summary_text) > 200
                if truncated:
                    tail = summary_text[-200:]
                    # Snap to the next word boundary so we don't cut mid-word
                    space_idx = tail.find(" ")
                    if 0 < space_idx < 40:
                        tail = tail[space_idx + 1 :]
                    preview = f"…{tail.strip()}"
                else:
                    preview = summary_text.strip()
                lines.append("")
                lines.append("[bold]Summary preview[/bold]:")
                lines.append(f'  [dim]"{preview}"[/dim]')
            body = "\n".join(lines)
            _console.print()
            _console.print(Panel(body, border_style="cyan", padding=(1, 2)))
            _console.print()
        else:
            print(f"\nMemory — session: {session_id}")
            print(f"  {'Mode':<16s}{stats.get('mode', cfg.memory_mode)}")
            print(f"  {'Messages':<16s}{msg_count} in history")
            summary_status = "✓ Rolling summary active" if has_summary else "✗ No summary yet"
            print(f"  {'Summary':<16s}{summary_status}")
            if vector_ready:
                vec_count = stats.get("vector_count", "")
                vec_detail = f" ({vec_count} embeddings)" if vec_count else ""
                print(f"  {'Semantic recall':<16s}✓ Active{vec_detail}")
            else:
                print(f"  {'Semantic recall':<16s}✗ Not configured")
            if summary_text:
                if len(summary_text) > 200:
                    _tail = summary_text[-200:]
                    _si = _tail.find(" ")
                    if 0 < _si < 40:
                        _tail = _tail[_si + 1 :]
                    preview = f"…{_tail.strip()}"
                else:
                    preview = summary_text.strip()
                print(f'\nSummary preview:\n  "{preview}"')
            print()
        return "continue"

    def _cmd_export(self, args: str) -> str:
        """Handler for /export."""
        from datetime import datetime as _dt

        mm = self.memory_manager
        cfg = self.config
        if not mm or not cfg:
            if _console is not None:
                _console.print("[red]Memory not available.[/red]")
            else:
                print("Memory not available.")
            return "continue"

        # Collect (human, ai) turn pairs from _messages
        raw_messages = getattr(mm, "_messages", [])
        turns: list[tuple[str, str]] = []
        pending_human: str | None = None
        for msg in raw_messages:
            content = ""
            if hasattr(msg, "content") and msg.content:
                c = msg.content
                content = c if isinstance(c, str) else str(c)
            cls_name = type(msg).__name__
            if cls_name == "HumanMessage":
                pending_human = content
            elif cls_name == "AIMessage" and pending_human is not None:
                turns.append((pending_human, content))
                pending_human = None

        if not turns:
            if _console is not None:
                _console.print("[dim]No conversation to export yet.[/dim]")
            else:
                print("No conversation to export yet.")
            return "continue"

        # Parse args: optional format keyword and/or path
        args = args.strip()
        fmt = "md"
        out_path: str | None = None

        parts = args.split() if args else []
        if parts and parts[0].lower() in ("html", "md", "markdown"):
            fmt = "html" if parts[0].lower() == "html" else "md"
            parts = parts[1:]
        if parts:
            out_path = " ".join(parts)

        session_id = cfg.session or "default"
        timestamp = _dt.now().strftime("%Y%m%d-%H%M%S")
        ext = ".html" if fmt == "html" else ".md"
        default_name = f"conversation-{session_id}-{timestamp}{ext}"

        if out_path:
            path = Path(out_path).expanduser()
            if path.is_dir():
                path = path / default_name
        else:
            path = Path.cwd() / default_name

        content_str = (
            _export_html(turns, session_id, timestamp)
            if fmt == "html"
            else _export_markdown(turns, session_id, timestamp)
        )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content_str, encoding="utf-8")
            n = len(turns)
            label = f"{n} turn{'s' if n != 1 else ''}"
            if _console is not None:
                _console.print(
                    f"  [green]\u2713[/green] Exported to [cyan]{path}[/cyan] [dim]({label})[/dim]"
                )
            else:
                print(f"  \u2713 Exported to {path} ({label})")
        except OSError as e:
            if _console is not None:
                _console.print(f"[red]Export failed: {e}[/red]")
            else:
                print(f"Export failed: {e}")

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
        alias = cfg.active_model_alias
        try:
            _active_mc = cfg.get_active_model()
            current = alias or _active_mc.model
        except (ValueError, KeyError, AttributeError, ConfigError):
            current = alias or "unknown"
        models = cfg.models or {}

        if _console is not None:
            lines_out: list[str] = []
            lines_out.append(f"  [bold green]{current}[/bold green]  [green]● active[/green]")
            if models:
                lines_out.append("")
                for mname, mcfg in models.items():
                    detail = f"{mcfg.provider}/{mcfg.model}"
                    is_current = mname == alias
                    if is_current:
                        name_fmt = f"[bold green]{mname:<16s}[/bold green]"
                        marker = "  [green]● active[/green]"
                    else:
                        name_fmt = f"[bold]{mname:<16s}[/bold]"
                        marker = ""
                    lines_out.append(f"  {name_fmt} [dim]{detail}[/dim]{marker}")
            lines_out.append("")
            lines_out.append(
                "[dim]Switch: [bold]/model[/bold] <name>   (e.g. [bold]/model fast[/bold])[/dim]"
            )
            _console.print(
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
                    marker = " ● active" if mname == alias else ""
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
            if _console is not None:
                _console.print(
                    "[dim]Provider switching is no longer supported. "
                    "Use [bold]/model[/bold] <alias> to switch models — "
                    "the provider is derived from the model configuration.[/dim]"
                )
            else:
                print("Provider switching is no longer supported.")
                print(
                    "Use /model <alias> to switch models — "
                    "the provider is derived from the model configuration."
                )
            return "continue"

        # ── No argument: show available providers (read-only) ────
        available = cfg.list_providers()
        try:
            active_provider = cfg.get_active_model().provider
        except (ValueError, KeyError, AttributeError, ConfigError):
            active_provider = None
        if _console is not None:
            lines_out = []
            for pname in available:
                try:
                    pcfg = cfg.get_provider_config(pname)
                    ptype = pcfg.type
                    base_url = getattr(pcfg, "base_url", None)
                    detail = f"type: {ptype}"
                    if base_url:
                        detail += f", url: {base_url}"
                except (ValueError, KeyError):
                    detail = "unconfigured"
                is_current = pname == active_provider
                if is_current:
                    name_fmt = f"[bold green]{pname:<20s}[/bold green]"
                    marker = " [green]● active model's provider[/green]"
                else:
                    name_fmt = f"[bold]{pname:<20s}[/bold]"
                    marker = ""
                lines_out.append(f"  {name_fmt} [dim]{detail}[/dim]{marker}")
            lines_out.append("")
            lines_out.append("[dim]Use [bold]/model[/bold] <alias> to switch models.[/dim]")
            body = "\n".join(lines_out)
            _console.print()
            _console.print(Panel(body, title="Providers", border_style="cyan", padding=(1, 2)))
            _console.print()
        else:
            print("\n  Providers:")
            for pname in available:
                marker = " ● active model's provider" if pname == active_provider else ""
                try:
                    pcfg = cfg.get_provider_config(pname)
                    base_url = getattr(pcfg, "base_url", None)
                    detail = f"type: {pcfg.type}"
                    if base_url:
                        detail += f", url: {base_url}"
                except (ValueError, KeyError):
                    detail = "unconfigured"
                print(f"    {pname:<20s} {detail}{marker}")
            print("\n  Use /model <alias> to switch models.")
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
                if _console is not None:
                    _console.print(f"[dim]Already in session [bold]{target}[/bold].[/dim]")
                else:
                    print(f"Already in session {target}.")
                return "continue"
            return f"switch_session:{target}"

        # No argument: show info (delegate to existing display logic)
        stats = mm.get_stats()
        msg_count = stats.get("total_messages", mm.get_message_count())

        # Use last real token count if available; otherwise call
        # prepare_context() for a calibrated estimate (tier cache when warm,
        # chars-based heuristic on cold start).
        _ctx_tokens = self.last_input_tokens
        _tier_counts: dict | None = None
        if not _ctx_tokens and msg_count > 0:
            try:
                ctx = mm.prepare_context("")
                _ctx_tokens = getattr(ctx, "token_estimate", 0) or 0
                _tier_counts = getattr(ctx, "tier_token_counts", None) or None
                if not _ctx_tokens:
                    _total_chars = sum(len(getattr(m, "content", "") or "") for m in ctx.messages)
                    _ctx_tokens = _total_chars // _CHARS_PER_TOKEN
            except Exception:
                pass
        elif msg_count > 0:
            try:
                ctx = mm.prepare_context("")
                _tier_counts = getattr(ctx, "tier_token_counts", None) or None
            except Exception:
                pass

        if _console is not None and Panel is not None:
            _session_rich(
                cfg,
                stats,
                msg_count,
                session_tokens=_ctx_tokens,
                max_context_tokens=self.max_context_tokens,
                tier_token_counts=_tier_counts,
            )
        else:
            _session_plain(
                cfg,
                msg_count,
                session_tokens=_ctx_tokens,
                max_context_tokens=self.max_context_tokens,
            )
        return "continue"

    @staticmethod
    def _cmd_debug(self, _args: str) -> str:
        """Handler for /debug [0-3] — cycle or set verbosity level."""
        from src.logging_config import get_verbosity, set_verbosity

        cfg = self.config
        if not cfg:
            print("Config not available.")
            return "continue"

        arg = _args.strip()
        if arg in ("0", "1", "2", "3"):
            new_level = int(arg)
        else:
            # Cycle: 0→1→2→3→0
            new_level = (get_verbosity() + 1) % 4

        cfg.verbosity = new_level
        cfg.debug = new_level >= 1
        cfg.verbose = new_level >= 2
        if cfg.debug and cfg.log_file is None:
            cfg.log_file = ""

        setup_logging(
            log_file=cfg.log_file,
            debug=cfg.debug,
            verbose=cfg.verbose,
            verbosity=new_level,
        )
        set_verbosity(new_level)

        _LEVEL_NAMES = {0: "normal", 1: "debug", 2: "verbose", 3: "trace"}
        label = _LEVEL_NAMES[new_level]
        msg = f"Verbosity: {label} ({new_level})"
        if _console is not None:
            color = "green" if new_level > 0 else "yellow"
            _console.print(f"[{color}]{msg}[/{color}]")
        else:
            print(msg)
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
        if _console is not None:
            color = "green" if cfg.verbose else "yellow"
            _console.print(f"[{color}]Verbose logging [bold]{state}[/bold][/{color}]")
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

        if _console is not None:
            color = "yellow" if _session.no_confirm else "green"
            desc = (
                "all tools auto-approved"
                if _session.no_confirm
                else "tools will prompt for confirmation"
            )
            _console.print(
                f"[{color}]Auto-approve [bold]{state}[/bold][/{color}] [dim]({desc})[/dim]"
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
        if _console is not None:
            color = "green" if cfg.prompt_optimizer else "yellow"
            _console.print(f"[{color}]Prompt optimizer [bold]{state}[/bold][/{color}]")
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

        if _console is not None and Panel is not None:
            try:
                from rich.markdown import Markdown as _Md

                body = _Md(sp)
            except ImportError:
                body = sp.replace("[", "\\[")
            title = f"System Prompt  ~{sp_tokens:,} tokens ({sp_chars:,} chars)"
            _console.print()
            _console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))
            _console.print()
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
                        "pin": True,  # conservative default: matches startup behavior
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

        if _console is not None and Panel is not None and Table is not None:
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
            _console.print()
            _console.print(Panel(tbl, title="MCP Servers", border_style="cyan", padding=(0, 1)))
            _console.print()
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
    def _cmd_agents(_self, args: str) -> str:
        """Handler for /agents [name | reload]."""
        from src.agent.agents_md import (  # codeql[py/clear-text-logging-sensitive-data] no sensitive data logged here; agent metadata contains no credentials
            get_agent,
            list_agents,
            load_default_agents,
        )

        args = args.strip()

        if args == "reload":
            agents = load_default_agents()
            count = len(agents)
            if _console is not None:
                _console.print(f"[dim]Reloaded AGENTS.md — {count} agent(s) loaded.[/dim]")
            else:
                print(f"Reloaded AGENTS.md — {count} agent(s) loaded.")
            return "continue"

        if args:
            # Show details of one agent
            agent = get_agent(args)
            if agent is None:
                if _console is not None:
                    _console.print(f"[red]Unknown agent:[/red] [bold]{args}[/bold]")
                    _console.print("[dim]Use /agents to list available agents.[/dim]")
                else:
                    print(f"Unknown agent: {args}")
                    print("Use /agents to list available agents.")
                return "continue"
            if _console is not None and Panel is not None and Table is not None:
                tbl = Table(show_header=False, box=None, padding=(0, 1))
                tbl.add_column("Field", style="bold cyan", min_width=14)
                tbl.add_column("Value")
                tbl.add_row("Name", agent.name)
                tbl.add_row("Model alias", agent.model_alias or "[dim]default[/dim]")
                tbl.add_row("Memory mode", agent.memory_mode or "[dim]default[/dim]")
                if agent.tools_include:
                    tbl.add_row("Tools include", ", ".join(agent.tools_include))
                if agent.tools_exclude:
                    tbl.add_row("Tools exclude", ", ".join(agent.tools_exclude))
                if agent.prompt_file:
                    tbl.add_row("Prompt file", agent.prompt_file)
                if agent.description:
                    tbl.add_row("Description", agent.description)
                if agent.system_prompt:
                    preview = agent.system_prompt[:200]
                    if len(agent.system_prompt) > 200:
                        preview += "…"
                    tbl.add_row("System prompt", preview)
                _console.print()
                _console.print(
                    Panel(tbl, title=f"Agent: {agent.name}", border_style="cyan", padding=(0, 1))
                )
                _console.print()
            else:
                print(f"\n  Agent: {agent.name}")
                print(f"  Model alias  : {agent.model_alias or '(default)'}")
                print(f"  Memory mode  : {agent.memory_mode or '(default)'}")
                if agent.tools_include:
                    print(f"  Tools include: {', '.join(agent.tools_include)}")
                if agent.tools_exclude:
                    print(f"  Tools exclude: {', '.join(agent.tools_exclude)}")
                if agent.prompt_file:
                    print(f"  Prompt file  : {agent.prompt_file}")
                if agent.description:
                    print(f"  Description  : {agent.description}")
                if agent.system_prompt:
                    preview = agent.system_prompt[:200]
                    if len(agent.system_prompt) > 200:
                        preview += "..."
                    print(f"  System prompt: {preview}")
                print()
            return "continue"

        # List all agents
        agents = list_agents()
        if not agents:
            if _console is not None:
                _console.print(
                    "[dim]No agents loaded. Create an AGENTS.md in your project directory.[/dim]"
                )
            else:
                print("No agents loaded. Create an AGENTS.md in your project directory.")
            return "continue"

        if _console is not None and Panel is not None and Table is not None:
            tbl = Table(show_header=True, box=None, padding=(0, 1))
            tbl.add_column("Name", style="bold")
            tbl.add_column("Model", style="cyan")
            tbl.add_column("Mode", style="green")
            tbl.add_column("Description", style="dim")
            for a in agents:
                desc_line = a.description.splitlines()[0] if a.description else ""
                tbl.add_row(
                    a.name,
                    a.model_alias or "-",
                    a.memory_mode or "-",
                    desc_line[:60],
                )
            _console.print()
            _console.print(Panel(tbl, title="Agents", border_style="cyan", padding=(0, 1)))
            _console.print("[dim]  /agents <name>    — show agent details[/dim]")
            _console.print("[dim]  /agents reload    — reload from AGENTS.md[/dim]")
            _console.print()
        else:
            print("\n  Agents")
            print("  " + "-" * 60)
            for a in agents:
                desc_line = a.description.splitlines()[0] if a.description else ""
                model = a.model_alias or "-"
                mode = a.memory_mode or "-"
                print(f"  {a.name:<20}  {model:<10}  {mode:<14}  {desc_line[:40]}")
            print()
            print("  /agents <name>  — show agent details")
            print("  /agents reload  — reload from AGENTS.md")
            print()
        return "continue"

    @staticmethod
    def _cmd_tasks(self, args: str) -> str:
        """Handler for /tasks [status | task_id]."""
        try:
            from src.tasks.queue import TaskStatus, get_task_queue
        except ImportError:
            if _console is not None:
                _console.print("[red]Task queue module not available.[/red]")
            else:
                print("Task queue module not available.")
            return "continue"

        args = args.strip()

        try:
            queue = get_task_queue()
        except RuntimeError as exc:
            if _console is not None:
                _console.print(f"[dim]{exc}[/dim]")
            else:
                print(str(exc))
            return "continue"

        _STATUS_STYLE: dict[str, str] = {
            "PENDING": "yellow",
            "RUNNING": "blue",
            "COMPLETED": "green",
            "FAILED": "red",
            "CANCELLED": "dim",
        }

        # Check if args looks like a task ID (8+ hex chars, not a known status word)
        _args_lower = args.lower()
        _is_id_lookup = (
            args
            and len(args) >= 8
            and all(c in "0123456789abcdefABCDEF-" for c in args)
            and _args_lower not in {s.value.lower() for s in TaskStatus}
        )
        if _is_id_lookup:
            task = queue.get(args)
            if task is None:
                # Try prefix match
                all_tasks = queue.list(limit=200)
                matches = [t for t in all_tasks if t.task_id.startswith(args)]
                task = matches[0] if len(matches) == 1 else None
            if task is None:
                if _console is not None:
                    _console.print(f"[red]Task not found:[/red] {args}")
                else:
                    print(f"Task not found: {args}")
                return "continue"

            sty = _STATUS_STYLE.get(task.status.value, "")
            status_str = f"[{sty}]{task.status.value}[/{sty}]" if sty else task.status.value
            duration_str = ""
            if task.started_at is not None and task.finished_at is not None:
                duration_str = f"  duration: {task.finished_at - task.started_at:.1f}s"
            elif task.started_at is not None:
                import time as _time

                duration_str = f"  running for {_time.time() - task.started_at:.1f}s"

            if _console is not None and Panel is not None:
                import datetime

                def _fmt_ts(ts: float | None) -> str:
                    if ts is None:
                        return "—"
                    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

                lines = [
                    f"[bold]ID[/bold]      {task.task_id}",
                    f"[bold]Agent[/bold]   [cyan]{task.agent_name}[/cyan]",
                    f"[bold]Status[/bold]  {status_str}",
                    f"[bold]Created[/bold] {_fmt_ts(task.created_at)}",
                    f"[bold]Started[/bold] {_fmt_ts(task.started_at)}",
                    f"[bold]Finished[/bold] {_fmt_ts(task.finished_at)}"
                    + (
                        f"  ({task.finished_at - task.started_at:.1f}s)"
                        if task.started_at and task.finished_at
                        else ""
                    ),
                    "",
                    "[bold]Prompt[/bold]",
                    task.prompt,
                ]
                if task.error:
                    lines += ["", f"[bold][red]Error[/red][/bold]  {task.error}"]

                from rich.console import Group

                parts: list = ["\n".join(lines)]
                if task.result:
                    try:
                        from rich.markdown import Markdown as _TaskMd

                        parts.append("")
                        parts.append("[bold]Result[/bold]")
                        parts.append(_TaskMd(task.result))
                    except ImportError:
                        parts.append(f"\n[bold]Result[/bold]\n{task.result}")

                _console.print()
                _console.print(
                    Panel(
                        Group(*parts),
                        title=f"Task {task.task_id[:8]}",
                        border_style="cyan",
                        padding=(0, 1),
                    )
                )
                _console.print()
            else:
                print(f"\n  Task {task.task_id}")
                print(f"  Agent:   {task.agent_name}")
                print(f"  Status:  {task.status.value}{duration_str}")
                print(f"  Prompt:  {task.prompt}")
                if task.result:
                    print(f"  Result:  {task.result}")
                if task.error:
                    print(f"  Error:   {task.error}")
                print()
            return "continue"

        # Otherwise treat args as a status filter
        status_filter: TaskStatus | None = None
        if args:
            try:
                status_filter = TaskStatus(_args_lower.upper())
            except ValueError:
                valid = ", ".join(s.value.lower() for s in TaskStatus)
                if _console is not None:
                    _console.print(f"[red]Unknown status or task ID:[/red] {args}")
                    _console.print(f"[dim]Valid statuses: {valid}[/dim]")
                else:
                    print(f"Unknown status: {args}")
                    print(f"Valid statuses: {valid}")
                return "continue"

        _sid = getattr(self.config, "session", "default") if self.config else "default"
        tasks = queue.list(status=status_filter, limit=50, session_id=_sid)
        if not tasks:
            msg = "No tasks found." if not args else f"No {_args_lower.upper()} tasks found."
            if _console is not None:
                _console.print(f"[dim]{msg}[/dim]")
            else:
                print(msg)
            return "continue"

        if _console is not None and Panel is not None and Table is not None:
            tbl = Table(show_header=True, box=None, padding=(0, 1))
            tbl.add_column("ID", style="bold", min_width=10)
            tbl.add_column("Agent", style="cyan")
            tbl.add_column("Status", min_width=12)
            tbl.add_column("Prompt", style="dim")
            for t in tasks:
                sty = _STATUS_STYLE.get(t.status.value, "")
                status_cell = f"[{sty}]{t.status.value}[/{sty}]" if sty else t.status.value
                tbl.add_row(t.task_id[:8], t.agent_name, status_cell, t.prompt)
            title = f"Tasks ({len(tasks)})" + (f" — {_args_lower.upper()}" if args else "")
            _console.print()
            _console.print(Panel(tbl, title=title, border_style="cyan", padding=(0, 1)))
            _console.print("[dim]  /spawn <agent> <desc>   — submit a new task[/dim]")
            _console.print("[dim]  /tasks <id>             — view task details and result[/dim]")
            _console.print()
        else:
            print(f"\n  Tasks ({len(tasks)})")
            print("  " + "-" * 70)
            for t in tasks:
                print(f"  {t.task_id[:8]}  {t.agent_name:<20}  {t.status.value:<12}  {t.prompt}")
            print()
        return "continue"

    @staticmethod
    def _cmd_spawn(self, args: str) -> str:
        """Handler for /spawn <agent_name> <task_description>."""
        try:
            from src.tasks.queue import get_task_queue, submit_task
        except ImportError:
            if _console is not None:
                _console.print("[red]Task queue module not available.[/red]")
            else:
                print("Task queue module not available.")
            return "continue"

        args = args.strip()
        if not args:
            if _console is not None:
                _console.print("[red]Usage:[/red] /spawn <agent_name> <task_description>")
            else:
                print("Usage: /spawn <agent_name> <task_description>")
            return "continue"

        parts = args.split(None, 1)
        if len(parts) < 2:
            if _console is not None:
                _console.print("[red]Usage:[/red] /spawn <agent_name> <task_description>")
            else:
                print("Usage: /spawn <agent_name> <task_description>")
            return "continue"

        agent_name, prompt = parts[0], parts[1]
        try:
            get_task_queue()
        except RuntimeError as exc:
            if _console is not None:
                _console.print(f"[dim]{exc}[/dim]")
            else:
                print(str(exc))
            return "continue"

        _sid = getattr(self.config, "session", "default") if self.config else "default"
        task_id = submit_task(agent_name, prompt, session_id=_sid)
        if _console is not None:
            _console.print(
                f"[green]Task submitted:[/green] [bold]{task_id[:8]}[/bold]"
                f"  agent=[cyan]{agent_name}[/cyan]"
            )
            _console.print(f"[dim]  {prompt}[/dim]")
        else:
            print(f"Task submitted: {task_id[:8]}  agent={agent_name}")
            print(f"  {prompt}")
        return "continue"

    def _cmd_goal(self, args: str) -> str:
        """Handler for /goal [set <desc> | complete <id> | abandon <id> | list]."""
        try:
            from src.tasks.goal_tracker import get_goal_stack
        except ImportError:
            if _console is not None:
                _console.print("[red]Goal tracker module not available.[/red]")
            else:
                print("Goal tracker module not available.")
            return "continue"

        cfg = self.config
        session_id = cfg.session if cfg else "default"
        data_dir = cfg.data_dir if cfg else "data"

        stack = get_goal_stack(session_id, data_dir)

        args_stripped = args.strip()
        subcmd, _, rest = args_stripped.partition(" ")
        subcmd = subcmd.lower()
        rest = rest.strip()

        if not args_stripped or subcmd == "list":
            goals = stack.list_active()
            if not goals:
                if _console is not None:
                    _console.print(
                        "[dim]No active goals. Use /goal set <description> to add one.[/dim]"
                    )
                else:
                    print("No active goals. Use /goal set <description> to add one.")
                return "continue"
            if _console is not None and Panel is not None and Table is not None:
                tbl = Table(show_header=True, box=None, padding=(0, 1))
                tbl.add_column("ID", style="bold", min_width=8)
                tbl.add_column("Status", style="green", min_width=10)
                tbl.add_column("Description")
                for g in goals:
                    indent = "  " if g.parent_id else ""
                    tbl.add_row(g.goal_id, g.status.value, f"{indent}{g.description}")
                _console.print()
                _console.print(
                    Panel(
                        tbl,
                        title=f"Active Goals ({len(goals)})",
                        border_style="green",
                        padding=(0, 1),
                    )
                )
                _console.print()
            else:
                print(f"\n  Active Goals ({len(goals)})")
                print("  " + "-" * 60)
                for g in goals:
                    indent = "    " if g.parent_id else "  "
                    print(f"{indent}{g.goal_id}  {g.status.value:<12}  {g.description}")
                print()
            return "continue"

        if subcmd == "set":
            if not rest:
                if _console is not None:
                    _console.print("[red]Usage:[/red] /goal set <description>")
                else:
                    print("Usage: /goal set <description>")
                return "continue"
            goal_id = stack.push(rest)
            if _console is not None:
                _console.print(f"[green]Goal set:[/green] [bold]{goal_id}[/bold]  {rest[:80]}")
            else:
                print(f"Goal set: {goal_id}  {rest[:80]}")
            return "continue"

        if subcmd == "complete":
            if not rest:
                if _console is not None:
                    _console.print("[red]Usage:[/red] /goal complete <goal_id>")
                else:
                    print("Usage: /goal complete <goal_id>")
                return "continue"
            if stack.complete(rest):
                if _console is not None:
                    _console.print(f"[green]Goal completed:[/green] {rest}")
                else:
                    print(f"Goal completed: {rest}")
            else:
                if _console is not None:
                    _console.print(f"[red]Unknown goal:[/red] {rest}")
                else:
                    print(f"Unknown goal: {rest}")
            return "continue"

        if subcmd == "abandon":
            if not rest:
                if _console is not None:
                    _console.print("[red]Usage:[/red] /goal abandon <goal_id>")
                else:
                    print("Usage: /goal abandon <goal_id>")
                return "continue"
            if stack.abandon(rest):
                if _console is not None:
                    _console.print(f"[dim]Goal abandoned: {rest}[/dim]")
                else:
                    print(f"Goal abandoned: {rest}")
            else:
                if _console is not None:
                    _console.print(f"[red]Unknown goal:[/red] {rest}")
                else:
                    print(f"Unknown goal: {rest}")
            return "continue"

        # Unknown subcommand
        if _console is not None:
            _console.print(f"[red]Unknown subcommand:[/red] {subcmd}")
            _console.print(
                "[dim]Usage: /goal [set <desc> | complete <id> | abandon <id> | list][/dim]"
            )
        else:
            print(f"Unknown subcommand: {subcmd}")
            print("Usage: /goal [set <desc> | complete <id> | abandon <id> | list]")
        return "continue"

    @staticmethod
    def _cmd_paste(_self, _args: str) -> str:
        """Handler for /paste (normally intercepted in the main loop)."""
        # /paste is intercepted before dispatch() so this handler is a
        # documentation-only fallback.  Print instructions just in case.
        if _console is not None:
            _console.print(
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
    from rich.table import Table

    categories = [
        ("Session & Config", ["info", "session", "mode", "model", "provider", "setup", "compact"]),
        ("Tools & Reasoning", ["tools", "think", "delegate", "approve", "optimizer"]),
        ("Logging", ["debug", "verbose"]),
        ("Input & Other", ["paste", "clear", "undo", "retry", "help", "quit"]),
    ]

    renderables: list[Any] = []
    for cat_name, cmd_names in categories:
        renderables.append(Text.from_markup(f"  [bold cyan]{cat_name}[/bold cyan]"))
        tbl = Table(box=None, padding=(0, 2, 0, 0), show_header=False, expand=False)
        tbl.add_column(style="bold", no_wrap=True, min_width=16)
        tbl.add_column()
        tbl.add_column(style="dim")
        for name in cmd_names:
            cmd = self_reg._commands.get(name)
            if cmd is None:
                continue
            aliases_str = ", ".join("/" + a for a in cmd.aliases) if cmd.aliases else ""
            tbl.add_row(f"  /{cmd.name}", cmd.short_help, aliases_str)
        renderables.append(tbl)
        renderables.append(Text(""))
    renderables.append(
        Text.from_markup(
            "  [dim]Type [bold white]/help <command>[/bold white] for detailed information.[/dim]"
        )
    )
    _console.print()  # type: ignore[union-attr]
    _console.print(  # type: ignore[union-attr]
        Panel(Group(*renderables), title="Commands", border_style="cyan", padding=(1, 2))
    )
    _console.print()  # type: ignore[union-attr]


def _help_plain(self_reg: SlashCommandRegistry) -> None:
    """Render the /help listing as plain text (no Rich)."""
    categories = [
        (
            "Session & Config",
            ["info", "session", "mode", "model", "provider", "setup", "compact"],
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
            ["paste", "clear", "undo", "retry", "help", "quit"],
        ),
    ]

    print("\nAvailable commands:\n")
    for cat_name, cmd_names in categories:
        print(f"  {cat_name}")
        for name in cmd_names:
            cmd = self_reg._commands.get(name)
            if cmd is None:
                continue
            alias_str = ""
            if cmd.aliases:
                alias_str = f"  ({', '.join('/' + a for a in cmd.aliases)})"
            desc = cmd.short_help.ljust(34)
            print(f"  /{cmd.name:<14s} {desc}{alias_str}")
        print()
    print("  Type /help <command> for detailed information.")
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
    if name in _session.pinned_tools:
        if rich_mode:
            return mcp_tag + "[cyan]\\[pinned]   [/cyan]"
        return mcp_tag + "[pinned]   "
    if name in _session.loaded_tools:
        if rich_mode:
            return mcp_tag + "[green]\\[loaded]   [/green]"
        return mcp_tag + "[loaded]   "
    if on_demand:
        if rich_mode:
            return mcp_tag + "[bright_magenta]\\[on-demand][/bright_magenta]"
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
    tool_set = set(tool_names)

    for cat_name, members in _TOOL_CATEGORIES.items():
        matched = [n for n in members if n in tool_set]
        if matched:
            cat_tools[cat_name] = matched

    # Find tools not in any primary category
    known_primary = {n for names in _TOOL_CATEGORIES.values() for n in names}
    not_primary = [n for n in tool_names if n not in known_primary]

    # Apply secondary category map; anything left is truly "Other"
    secondary_tools: dict[str, list[str]] = {}
    truly_other: list[str] = []
    for name in sorted(not_primary):
        sec_cat = _TOOL_CATEGORY_MAP.get(name)
        if sec_cat:
            secondary_tools.setdefault(sec_cat, []).append(name)
        else:
            truly_other.append(name)

    result: list[tuple[str, list[str]]] = []
    for cat_name in _TOOL_CATEGORIES:
        if cat_name in cat_tools:
            result.append((cat_name, cat_tools[cat_name]))
    for cat_name in _SECONDARY_CATEGORY_ORDER:
        if cat_name in secondary_tools:
            result.append((cat_name, secondary_tools[cat_name]))
    if truly_other:
        result.append(("Other", truly_other))
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
    if _console is None or Table is None:  # pragma: no cover – caller checks
        return

    groups = _categorize_tools(tool_names)
    total = len(tool_names)

    # Calculate available description width from terminal size.
    # Panel uses: 2 border + 2*2 padding = 6 chars on each side → 6+6 not quite;
    # Rich Panel: │ + 2 padding + content + 2 padding + │  = 1+2+…+2+1 = 6.
    # Each line: "  " indent (2) + name col (28) + "  " gap (2) = 32 fixed.
    # Tag (if present) adds ~10-16 chars but we size for the common (no-tag) case
    # and let tagged lines wrap naturally if needed.
    term_width = _console.width or 100
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
    _console.print()
    _console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))
    _console.print()


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
    if _console is None or Panel is None:  # pragma: no cover
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
    _console.print()
    _console.print(Panel(body, title="Memory Modes", border_style="cyan", padding=(1, 2)))
    _console.print()


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
    model_cfg: Any,
    stats: dict,
    msg_count: int,
    system_prompt: str | None = None,
    mcp_manager: Any = None,
    project_context_path: str | None = None,
) -> None:
    """Render /info output using Rich."""
    if _console is None or Panel is None:  # pragma: no cover
        return

    alias = cfg.active_model_alias or model_cfg.model
    # ── Connection section ────────────────────────────────────
    lines: list[str] = []
    lines.append("[bold cyan]Connection[/bold cyan]")
    lines.append(
        f"  [bold]{'Model':<13s}[/bold]  {alias} "
        f"[dim]({model_cfg.provider}/{model_cfg.model})[/dim]"
    )
    lines.append(
        f"  [bold]{'Provider':<13s}[/bold]  {model_cfg.provider} [dim]({provider_cfg.type})[/dim]"
    )
    if model_cfg.context_window:
        lines.append(f"  [bold]{'Context size':<13s}[/bold]  {model_cfg.context_window:,} tokens")
    if system_prompt:
        sp_chars = len(system_prompt)
        sp_tokens = sp_chars // 4  # rough estimate
        lines.append(
            f"  [bold]{'System prompt':<13s}[/bold]  ~{sp_tokens:,} tokens"
            f" [dim]({sp_chars:,} chars)[/dim]"
        )
    if project_context_path:
        lines.append(f"  [bold]{'COGTRIX.md':<13s}[/bold]  [dim]{project_context_path}[/dim]")
    if mcp_manager is not None:
        server_info = mcp_manager.get_server_info()
        connected = sum(1 for s in server_info if s["connected"])
        total_tools = sum(s["tool_count"] for s in server_info)
        lines.append(
            f"  [bold]{'MCP servers':<13s}[/bold]  {connected} connected ({total_tools} tools)"
        )

    # ── Memory section ────────────────────────────────────────
    lines.append("")
    lines.append("[bold cyan]Memory[/bold cyan]")
    lines.append(f"  [bold]{'Mode':<13s}[/bold]  {cfg.memory_mode}")
    lines.append(f"  [bold]{'Session':<13s}[/bold]  {cfg.session}")
    lines.append(f"  [bold]{'Messages':<13s}[/bold]  {msg_count}")

    wm_size = stats.get("working_memory_size")
    if wm_size is not None:
        lines.append(f"  [bold]{'Working mem':<13s}[/bold]  {wm_size} messages")

    # Mode-specific extras
    if stats.get("has_summary"):
        lines.append(f"  [bold]{'Summary':<13s}[/bold]  Active")
    if stats.get("entity_count"):
        lines.append(f"  [bold]{'Entities':<13s}[/bold]  {stats['entity_count']}")
    if stats.get("files_tracked"):
        lines.append(f"  [bold]{'Files':<13s}[/bold]  {stats['files_tracked']} tracked")
    if stats.get("decision_count"):
        lines.append(f"  [bold]{'Decisions':<13s}[/bold]  {stats['decision_count']}")

    # TCC status — emitted when mode subclass includes tier_cache_ready in get_stats()
    _tcc_ready = stats.get("tier_cache_ready")
    if _tcc_ready is not None:
        tcc_status = "active" if _tcc_ready else "warming"
        lines.append(f"  [bold]{'TCC':<13s}[/bold]  {tcc_status}")

    body = "\n".join(lines)
    _console.print()
    _console.print(Panel(body, title="Session Information", border_style="cyan", padding=(1, 2)))
    _console.print()


def _info_plain(
    cfg: Any,
    provider_cfg: Any,
    model_cfg: Any,
    stats: dict,
    msg_count: int,
    system_prompt: str | None = None,
    mcp_manager: Any = None,
    project_context_path: str | None = None,
) -> None:
    """Render /info output as plain text."""
    alias = cfg.active_model_alias or model_cfg.model
    print("\n  Session Information")
    print("  " + "-" * 38)
    print(f"  Model         {alias} ({model_cfg.provider}/{model_cfg.model})")
    print(f"  Provider      {model_cfg.provider} ({provider_cfg.type})")
    if model_cfg.context_window:
        print(f"  Context size  {model_cfg.context_window:,} tokens")
    if system_prompt:
        sp_chars = len(system_prompt)
        sp_tokens = sp_chars // 4
        print(f"  System prompt ~{sp_tokens:,} tokens ({sp_chars:,} chars)")
    if project_context_path:
        print(f"  COGTRIX.md    {project_context_path}")
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


def _session_rich(
    cfg: Any,
    stats: dict,
    msg_count: int,
    session_tokens: int = 0,
    max_context_tokens: int | None = None,
    tier_token_counts: dict | None = None,
) -> None:
    """Render /session output using Rich."""
    if _console is None or Panel is None:  # pragma: no cover
        return

    lines: list[str] = []
    lines.append(f"  [bold]Session[/bold]   {cfg.session}")
    lines.append(f"  [bold]Mode[/bold]      {cfg.memory_mode}")
    lines.append(f"  [bold]Messages[/bold]  {msg_count}")

    if max_context_tokens and max_context_tokens > 0:
        pct = int(min(1.0, session_tokens / max_context_tokens) * 100) if session_tokens else 0
        filled = int(pct * 20 // 100)
        bar = "█" * filled + "░" * (20 - filled)
        color = "green" if pct < 70 else ("yellow" if pct < 85 else "red")
        lines.append(
            f"  [bold]Context[/bold]   [{color}]{bar}[/{color}] "
            f"{pct}% of {max_context_tokens:,} tokens"
        )

    if tier_token_counts:
        tier_parts = [
            f"T{t}: {tier_token_counts[t]:,}"
            for t in sorted(tier_token_counts)
            if tier_token_counts[t] > 0
        ]
        if tier_parts:
            lines.append(f"  [dim]Tiers     {' · '.join(tier_parts)}[/dim]")

    body = "\n".join(lines)
    _console.print()
    _console.print(Panel(body, title="Session", border_style="cyan", padding=(1, 2)))
    _console.print()


def _session_plain(
    cfg: Any,
    msg_count: int,
    session_tokens: int = 0,
    max_context_tokens: int | None = None,
) -> None:
    """Render /session output as plain text."""
    print(f"\n  Session   {cfg.session}")
    print(f"  Mode      {cfg.memory_mode}")
    print(f"  Messages  {msg_count}")
    if max_context_tokens and max_context_tokens > 0:
        pct = int(min(1.0, session_tokens / max_context_tokens) * 100) if session_tokens else 0
        filled = int(pct * 20 // 100)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  Context   {bar} {pct}% of {max_context_tokens:,} tokens")
    print()


def _export_markdown(turns: list[tuple[str, str]], session_id: str, timestamp: str) -> str:
    """Render conversation turns as Markdown."""
    lines = [
        f"# Cogtrix Session: {session_id}",
        "",
        f"_Exported: {timestamp.replace('-', ' ', 1).replace('-', ':')}_",
        "",
        "---",
        "",
    ]
    for i, (human, ai) in enumerate(turns, 1):
        lines.append(f"## Turn {i}")
        lines.append("")
        lines.append(f"**You:** {human}")
        lines.append("")
        lines.append("**Agent:**")
        lines.append("")
        lines.append(ai)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _export_html(turns: list[tuple[str, str]], session_id: str, timestamp: str) -> str:
    """Render conversation turns as a clean HTML page."""
    import html as _html_mod
    import re as _re

    def esc(s: str) -> str:
        return _html_mod.escape(s)

    def md_to_html(s: str) -> str:
        """Minimal markdown: code blocks, inline code, bold, newlines."""
        # Fenced code blocks
        s = _re.sub(
            r"```(\w*)\n(.*?)```",
            lambda m: f'<pre><code class="language-{m.group(1)}">{esc(m.group(2))}</code></pre>',
            s,
            flags=_re.DOTALL,
        )
        # Inline code
        s = _re.sub(r"`([^`]+)`", lambda m: f"<code>{esc(m.group(1))}</code>", s)
        # Bold
        s = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        # Newlines to <br>
        s = s.replace("\n", "<br>\n")
        return s

    ts_display = timestamp.replace("-", " ", 1).replace("-", ":")
    turns_html = []
    for i, (human, ai) in enumerate(turns, 1):
        turns_html.append(f"""
    <div class="turn">
      <div class="turn-num">Turn {i}</div>
      <div class="bubble user"><span class="label">You</span><div class="body">{esc(human)}</div></div>
      <div class="bubble agent"><span class="label">Agent</span><div class="body">{md_to_html(ai)}</div></div>
    </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cogtrix \u2014 {esc(session_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 860px; margin: 40px auto; padding: 0 20px;
          background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 12px; }}
  .meta {{ color: #8b949e; font-size: 0.85em; margin-bottom: 32px; }}
  .turn {{ margin-bottom: 28px; }}
  .turn-num {{ font-size: 0.75em; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }}
  .bubble {{ border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }}
  .user {{ background: #161b22; border: 1px solid #30363d; }}
  .agent {{ background: #0d2137; border: 1px solid #1c4a6e; }}
  .label {{ font-size: 0.75em; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.06em; display: block; margin-bottom: 6px; }}
  .user .label {{ color: #58a6ff; }}
  .agent .label {{ color: #3fb950; }}
  .body {{ font-size: 0.95em; }}
  code {{ background: #161b22; border: 1px solid #30363d; padding: 1px 5px;
          border-radius: 4px; font-size: 0.88em; }}
  pre {{ background: #161b22; border: 1px solid #30363d; padding: 12px;
         border-radius: 6px; overflow-x: auto; }}
  pre code {{ border: none; padding: 0; background: none; }}
  strong {{ color: #e6edf3; }}
</style>
</head>
<body>
<h1>Cogtrix Session: {esc(session_id)}</h1>
<div class="meta">Exported: {esc(ts_display)} &nbsp;&middot;&nbsp; {len(turns)} turn{"s" if len(turns) != 1 else ""}</div>
{"".join(turns_html)}
</body>
</html>
"""


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
            aliases=[],
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
            aliases=[],
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
                "  - Context window size (context_window)\n"
                "  - Memory mode and working memory size\n"
                "  - Session ID and message count\n"
                "  - Mode-specific tracking (entities, files, decisions)"
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="tools",
            handler=SlashCommandRegistry._cmd_tools,
            short_help="List / manage tools",
            long_help=(
                "Usage: /tools [search | load | unload | enable | disable]\n\n"
                "Without arguments, lists all tools grouped by category\n"
                "with status tags:\n"
                "  [confirm]        Requires user approval before running\n"
                "  [auto-approved]  Confirmation skipped (/approve active)\n"
                "  [pinned]         Manually loaded — persists across turns\n"
                "  [loaded]         Loaded by the agent — auto-unloaded next turn\n"
                "  [on-demand]      Available but not yet loaded\n"
                "  [disabled]       Blocked — will not load or execute\n\n"
                "Subcommands:\n"
                "  /tools load <name>      Load and pin a tool (persists across turns)\n"
                "  /tools unload <name>    Unload a pinned tool\n"
                "  /tools enable <name>    Re-enable a disabled tool\n"
                "  /tools disable <name>   Disable a tool for this session\n\n"
                "Tools loaded by the agent via request_tools are automatically\n"
                "unloaded at the start of each new prompt. Manually loaded tools\n"
                "stay active until you /tools unload them.\n\n"
                "With any other text, filters tools by name.\n\n"
                "Examples:\n"
                "  /tools                     List all tools by category\n"
                "  /tools search              Show search-related tools\n"
                "  /tools load exa_search     Pin exa_search into active set\n"
                "  /tools unload exa_search   Unpin and unload exa_search\n"
                "  /tools disable shell       Disable execute_shell_command\n"
                "  /tools enable shell        Re-enable it"
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="mcp",
            handler=SlashCommandRegistry._cmd_mcp,
            short_help="List / restart MCP servers",
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="think",
            handler=SlashCommandRegistry._cmd_think,
            short_help="Force deep-reasoning pipeline",
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="delegate",
            handler=SlashCommandRegistry._cmd_delegate,
            short_help="Force task delegation",
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
            aliases=[],
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
                "  reasoning     Planning, decision tracking      (30 msgs)\n\n"
                "Examples:\n"
                "  /mode           Show all modes\n"
                "  /mode code      Switch to code mode\n"
                "  /M reasoning    Switch to reasoning mode\n\n"
                "Switching preserves the current session but rebuilds the\n"
                "system prompt and memory context for the new mode."
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="memory",
            handler=SlashCommandRegistry._cmd_memory,
            short_help="Show memory state for the current session",
            long_help=(
                "Usage: /memory\n\n"
                "Displays the current memory state including message count,\n"
                "rolling summary status, and semantic recall availability.\n\n"
                "Also shows a preview of the rolling summary when one exists."
            ),
            aliases=["mem"],
        )
    )

    reg.register(
        SlashCommand(
            name="agents",
            handler=SlashCommandRegistry._cmd_agents,
            short_help="List / inspect named agents from AGENTS.md",
            long_help=(
                "Usage: /agents [name | reload]\n\n"
                "Without arguments, lists all agents loaded from AGENTS.md.\n\n"
                "Subcommands:\n"
                "  /agents <name>    Show full details of a named agent\n"
                "  /agents reload    Reload AGENTS.md from disk\n\n"
                "AGENTS.md is searched for in the current directory, then\n"
                "the home directory.  Each level-2 heading (## Name) defines\n"
                "an agent block with an optional yaml config fence:\n\n"
                "  ## My Agent\n"
                "  Description text.\n"
                "  ```yaml\n"
                "  system_prompt: You are helpful.\n"
                "  model_alias: fast\n"
                "  memory_mode: conversation\n"
                "  tools_include: [web_search]\n"
                "  tools_exclude: [shell]\n"
                "  ```\n\n"
                "Examples:\n"
                "  /agents              List all loaded agents\n"
                "  /agents researcher   Show details of 'researcher'\n"
                "  /agents reload       Reload from AGENTS.md"
            ),
        )
    )

    reg.register(
        SlashCommand(
            name="tasks",
            handler=SlashCommandRegistry._cmd_tasks,
            short_help="List background tasks or view task details",
            long_help=(
                "Usage: /tasks [status | task_id]\n\n"
                "Without arguments, lists the 50 most recent background tasks.\n\n"
                "Pass a task ID (or 8-char prefix) to view full details and result:\n"
                "  /tasks 869e4a2f\n\n"
                "Optional status filter:\n"
                "  pending, running, completed, failed, cancelled\n\n"
                "Examples:\n"
                "  /tasks               List all recent tasks\n"
                "  /tasks running       Show only running tasks\n"
                "  /tasks 869e4a2f     View full result for task 869e4a2f\n"
                "  /task failed         Show failed tasks (short alias)"
            ),
            aliases=["task"],
        )
    )

    reg.register(
        SlashCommand(
            name="spawn",
            handler=SlashCommandRegistry._cmd_spawn,
            short_help="Submit a background task to the task queue",
            long_help=(
                "Usage: /spawn <agent_name> <task_description>\n\n"
                "Submits a background task to the queue for the named agent.\n"
                "The task runs asynchronously; track progress with /tasks.\n\n"
                "Examples:\n"
                "  /spawn researcher Summarise the latest arXiv ML papers\n"
                "  /spawn coder Refactor src/tools/shell.py for better error handling"
            ),
        )
    )

    reg.register(
        SlashCommand(
            name="goal",
            handler=SlashCommandRegistry._cmd_goal,
            short_help="Manage session goals",
            long_help=(
                "Usage: /goal [set <desc> | complete <id> | abandon <id> | list]\n\n"
                "Track objectives for the current session.  Goals are persisted\n"
                "to disk and survive session restarts.\n\n"
                "Subcommands:\n"
                "  /goal                  List active goals\n"
                "  /goal list             List active goals\n"
                "  /goal set <desc>       Create a new goal\n"
                "  /goal complete <id>    Mark a goal as completed\n"
                "  /goal abandon <id>     Mark a goal as abandoned\n\n"
                "Examples:\n"
                "  /goal set Migrate auth module to JWT\n"
                "  /goal complete a1b2c3d4\n"
                "  /goals                 (short alias)"
            ),
            aliases=["goals"],
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="provider",
            handler=SlashCommandRegistry._cmd_provider,
            short_help="List configured LLM providers",
            long_help=(
                "Usage: /provider\n\n"
                "Lists all configured providers with their type and\n"
                "base URL.  The provider used by the active model is\n"
                "highlighted.\n\n"
                "Providers define connection endpoints only (type,\n"
                "base_url, api_key).  To change which provider is\n"
                "used, switch to a model that references it:\n"
                "  /model <alias>\n\n"
                "Example:\n"
                "  /provider                 List providers\n"
                "  /p                        Same (short alias)"
            ),
            aliases=[],
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="debug",
            handler=SlashCommandRegistry._cmd_debug,
            short_help="Cycle or set verbosity level (0–3)",
            long_help=(
                "Usage: /debug [0|1|2|3]\n\n"
                "Without an argument, cycles through verbosity levels 0→1→2→3→0.\n"
                "With a numeric argument, jumps directly to that level:\n\n"
                "  0 = normal  (no debug logging)\n"
                "  1 = debug   (DEBUG log level; equivalent to --debug)\n"
                "  2 = verbose (DEBUG + full LLM interactions)\n"
                "  3 = trace   (verbose + fine-grained internal tracing)\n\n"
                "Levels 1–3 auto-enable file logging (cogtrix.log by default)."
            ),
            aliases=[],
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="approve",
            handler=SlashCommandRegistry._cmd_approve,
            short_help="Toggle tool auto-approval",
            long_help=(
                "Usage: /approve\n\n"
                "Toggles automatic approval for tools that normally\n"
                "require confirmation (file writes, shell commands, etc.).\n\n"
                "When ON, all tools run without prompting.\n"
                "When OFF, tools will prompt for confirmation again\n"
                "and all disabled tools are re-enabled.\n\n"
                "Equivalent to the -y / --no-confirm CLI flag."
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="optimizer",
            handler=SlashCommandRegistry._cmd_optimizer,
            short_help="Toggle prompt optimizer",
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
            aliases=[],
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
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="undo",
            handler=SlashCommandRegistry._cmd_undo,
            short_help="Remove the last exchange from memory",
            long_help=(
                "Usage: /undo\n\n"
                "Removes the last user message and assistant response\n"
                "from conversation memory.\n\n"
                "Useful when you want to rephrase or retry a prompt\n"
                "without the failed or unwanted exchange in history.\n\n"
                "This action cannot itself be undone."
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="compact",
            handler=SlashCommandRegistry._cmd_compact,
            short_help="Compress context in place — summarise old messages",
            long_help=(
                "Usage: /compact [aggressive]\n\n"
                "Summarises old messages in place to reduce context usage.\n"
                "History is preserved — messages are condensed, not deleted.\n\n"
                "  /compact             Standard: messages older than 3 turns, >2000 chars\n"
                "  /compact aggressive  Emergency: compress all messages regardless of age or size\n\n"
                "Use when the context bar shows high usage but you want to continue\n"
                "the current session rather than starting a new one.\n\n"
                "See also: /clear (delete history), /session new (fresh session)"
            ),
            aliases=[],
        )
    )

    reg.register(
        SlashCommand(
            name="retry",
            handler=SlashCommandRegistry._cmd_retry,
            short_help="Re-run the last prompt through the agent",
            long_help=(
                "Usage: /retry\n\n"
                "Re-sends your last prompt to the agent without\n"
                "re-typing it.\n\n"
                "Useful when a response was incomplete, incorrect,\n"
                "or interrupted. Combine with /undo first to remove\n"
                "the failed exchange before retrying:\n\n"
                "  /undo   → remove last exchange\n"
                "  /retry  → re-run the same prompt fresh"
            ),
            aliases=[],
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

    reg.register(
        SlashCommand(
            name="export",
            handler=SlashCommandRegistry._cmd_export,
            short_help="Export conversation to markdown or HTML",
            long_help=(
                "Usage: /export [html|md] [path]\n\n"
                "Saves the current session conversation to a file.\n\n"
                "Format (optional, default: md):\n"
                "  md / markdown   Plain Markdown file\n"
                "  html            Styled HTML page (dark theme)\n\n"
                "Path (optional):\n"
                "  If omitted, saves to current directory with auto name.\n"
                "  conversation-<session>-<timestamp>.md\n\n"
                "Examples:\n"
                "  /export                    Save as markdown\n"
                "  /export html               Save as HTML\n"
                "  /export ~/notes/chat.md    Save to explicit path\n"
                "  /export html ~/chat.html   HTML at explicit path\n"
                "  /save                      Alias for /export"
            ),
            aliases=["save"],
        )
    )

    # Hidden commands (not listed in /help categories)
    reg.register(
        SlashCommand(
            name="system_prompt",
            handler=SlashCommandRegistry._cmd_system_prompt,
            short_help="Display the full system prompt",
            aliases=[],
        )
    )

    return reg
