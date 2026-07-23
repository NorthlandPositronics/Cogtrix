#!/usr/bin/env python3
"""
Cogtrix Agent - CLI Entry Point
A modular LangChain agent with extensible tools and safety features.
Supports multiple LLM providers: OpenAI, Ollama.
"""

import atexit
import concurrent.futures as _cf
import os
import re
import signal
import sys
import time as _time_mod
import warnings
from pathlib import Path
from typing import Any

import cogtrix_core.cli.commands as commands
import cogtrix_core.ui.confirmation as confirmation
from cogtrix_core._version import __copyright__, __version__  # noqa: F401
from cogtrix_core.agent.core import (
    build_system_prompt,
    format_milestone_instructions,
)
from cogtrix_core.agent.safety import AgentExecutionError, UserCancelledRun
from cogtrix_core.analysis.session_metrics import write_session_metrics
from cogtrix_core.cli.args import color_enabled, parse_arguments
from cogtrix_core.cli.banner import print_startup

# Slash command system - extracted to cogtrix_core/cli/commands.py
from cogtrix_core.cli.commands import (  # noqa: F401
    SlashCommand,
    SlashCommandRegistry,
    configure,
)
from cogtrix_core.cli.input import (
    load_input_history,
    prefill_next_input,
    read_multiline,
    run_inline_shell,
    save_input_history,
    set_slash_commands,
    setup_readline_completion,
)
from cogtrix_core.config import Config, ConfigError, _resolve_model, load_config
from cogtrix_core.logging_config import (
    create_observability_handler,
    get_logger,
    log_agent_response,
    log_error,
    log_session_info,
    log_user_message,
    new_request_id,
    setup_logging,
)
from cogtrix_core.memory import JsonFileMemoryStore, MemoryFactory
from cogtrix_core.memory.context import MemoryContext
from cogtrix_core.memory.mode_selector import classify_memory_mode, should_switch_mode
from cogtrix_core.orchestration.compression import (
    apply_message_compression,
    compress_tool_message,
    create_compression_llm,
    truncate_tool_output,
)
from cogtrix_core.orchestration.graph import (  # noqa: F401
    DEFAULT_RECURSION_LIMIT,
    EMPTY_RESPONSE_MSG,
    build_agent_graph,
)
from cogtrix_core.orchestration.intent import (  # noqa: F401
    _COMPLEX_QUERY_MARKERS,
    _SIMPLE_QUERY_KEYWORDS,
    ACTION_TARGETS,
    ACTION_VERBS,
    DEEP_THINK_TRIGGERS,
    DELEGATION_TRIGGERS,
    THINK_CATEGORIES,
    THINK_DEFAULT_CATEGORY,
    TaskComplexity,
    ThinkCategory,
    classify_task_complexity,
    classify_think_task,
    prompt_requests_action,
    user_wants_deep_think,
    user_wants_delegation,
)
from cogtrix_core.orchestration.phases import (  # noqa: F401
    ACTION_TOOL_NAMES,
    MIN_GOOD_CONTEXT_LEN,
    RESEARCH_CAP_RATIO,
    STEP_LIMIT_PHRASES,
    WEB_TOOL_NAMES,
    WRITE_FAILURE_PREFIXES,
    _looks_like_research_query,
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
from cogtrix_core.orchestration.reflection_delegate import (
    ACCOUNTABILITY_PROMPT,
    PRE_ACTION_CONFIRMATION_PROMPT,
)
from cogtrix_core.orchestration.run_config import AgentRunConfig
from cogtrix_core.orchestration.runner import (  # noqa: F401
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
from cogtrix_core.orchestration.session_orchestrator import SessionOrchestrator
from cogtrix_core.orchestration.session_state import SessionState
from cogtrix_core.prompt.optimizer import (
    PromptPlan,
    optimize_prompt,
)
from cogtrix_core.prompt.optimizer import (
    set_progress_callback as set_optimizer_callback,
)
from cogtrix_core.providers import create_chat_model_from_configs
from cogtrix_core.registry import ToolRegistry
from cogtrix_core.tools.configure import (
    TOOL_OUTPUT_CAP_MIN_CHARS,
    TOOL_PRESETS,
    _update_rag_tool_description,
    apply_output_cap,
    apply_tool_preset,
    build_tool_catalog,
    compute_tool_output_cap,
    configure_brave_tool,
    configure_cron_tool,
    configure_deep_think_tool,
    configure_delegate_tool,
    configure_delegate_tools,
    configure_email_tool,
    configure_exa_tool,
    configure_file_ops_tool,
    configure_file_read_dirs,
    configure_google_search_tool,
    configure_python_exec_tool,
    configure_rag_tool,
    configure_searxng_tool,
    configure_serpapi_tool,
    configure_tavily_tool,
    create_request_tools_tool,
    filter_unconfigured_tools,
    load_tools,
    rag_should_auto_activate,
)
from cogtrix_core.tools.report_progress import (
    create_report_progress_tool,
)
from cogtrix_core.tools.report_progress import (
    set_progress_callback as set_milestone_callback,
)
from cogtrix_core.ui.confirmation import _RichConfirmationUI, _TokenAccumulator
from cogtrix_core.ui.input_session import create_session as _create_input_session
from cogtrix_core.ui.spinner import _spinner

try:
    from cogtrix_core.cli.escape_monitor import EscapeMonitor

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
    from cogtrix_core.mcp_client import MCP_AVAILABLE, MCPManager, MCPServerConfig
except ImportError:
    MCP_AVAILABLE = False
    MCPManager = None  # type: ignore[misc, assignment]
    MCPServerConfig = None  # type: ignore[misc, assignment]


# Global flag to track if shutdown has been initiated
_shutdown_initiated: bool = False


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Handle SIGTERM signal for graceful shutdown.

    Raises KeyboardInterrupt to trigger the existing cleanup path.
    This ensures SIGTERM and Ctrl+C share the same shutdown logic.
    """
    global _shutdown_initiated
    if _shutdown_initiated:
        # Second SIGTERM - force exit immediately
        log = get_logger()
        log.info("Received second SIGTERM, force exiting...")
        import os as _os_exit

        _os_exit._exit(1)

    _shutdown_initiated = True
    log = get_logger()
    log.info("Received SIGTERM, initiating graceful shutdown...")
    # Raise KeyboardInterrupt to trigger the existing cleanup path
    raise KeyboardInterrupt("SIGTERM received")


# Initialize rich console if available
if Console is not None:

    class _Console(Console):  # type: ignore[misc, valid-type]
        """Console variant that defaults crop=False so panel closing borders are never clipped."""

        def print(self, *objects: Any, crop: bool = False, **kwargs: Any) -> None:
            super().print(*objects, crop=crop, **kwargs)

    console: _Console | None = _Console()
else:
    _Console = None  # type: ignore[assignment, misc]
    console = None

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
    get_logger().debug("[tools] %s (%d total)", "; ".join(parts), total)
    _spinner.resume()


def _load_cli_system_prompt(args: Any) -> str | None:
    """Return a CLI system prompt override from ``args`` if one was supplied.

    ``--system-prompt`` wins over ``--system-prompt-file`` because the parser
    already enforces mutual exclusion, and inline text avoids a file read.
    """
    inline_prompt = getattr(args, "system_prompt", None)
    if inline_prompt:
        return inline_prompt

    prompt_file = getattr(args, "system_prompt_file", None)
    if not prompt_file:
        return None

    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {prompt_file} "
            "(provide a valid file path with --system-prompt-file)"
        )

    prompt_text = path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(
            f"System prompt file is empty: {prompt_file} "
            "(provide a file with non-empty content or omit --system-prompt-file)"
        )

    return prompt_text


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
_extract_response = extract_response
_build_tool_results_response = build_tool_results_response
_log_tool_calls_from_result = log_tool_calls_from_result


def _run_agent_cli(*args: Any, **kwargs: Any) -> str:
    """CLI wrapper around ``run_agent`` (#2124).

    ``run_agent`` now raises :class:`AgentExecutionError` on an irrecoverable
    turn failure instead of returning the error text as a normal answer. The
    interactive CLI deliberately *displays* that message (its existing panels
    already render ``run_agent``'s string return), so we convert the typed
    failure back into its display message here. Other callers (API turn
    runner, assistant handler) catch the exception themselves to surface a
    proper error frame / reply.
    """
    try:
        return run_agent(*args, **kwargs)
    except AgentExecutionError as exc:
        return exc.user_message


def _format_stats_line(
    elapsed: float,
    acc: _TokenAccumulator,
    session_acc: "_TokenAccumulator | None" = None,
) -> str | None:
    """Build a compact stats string for display after agent responses."""
    parts: list[str] = []
    if elapsed > 0:
        parts.append(f"[dim yellow]{elapsed:.1f}s[/dim yellow]")
    if acc.input_tokens or acc.output_tokens:
        parts.append(f"[dim]↑ {acc.input_tokens:,}  ↓ {acc.output_tokens:,}[/dim]")
    if session_acc is not None and (session_acc.input_tokens or session_acc.output_tokens):
        total = session_acc.input_tokens + session_acc.output_tokens
        parts.append(f"[dim]session: {total:,} tok[/dim]")
    return "[dim] \u00b7 [/dim]".join(parts) if parts else None


def _apply_profile(config: "Config", profile_name: str) -> None:
    """Apply a named profile's settings to config.

    Profile keys map directly to Config field names.
    Unrecognised keys are silently skipped.
    """
    profile = config.profiles.get(profile_name)
    if profile is None:
        _log = get_logger()
        _log.warning(
            "Profile '%s' not found in config. Available: %s",
            profile_name,
            list(config.profiles.keys()),
        )
        if console:
            console.print(f"[yellow]Profile '{profile_name}' not found.[/yellow]")
        return
    _PROFILE_FIELDS = {
        "model",
        "memory_mode",
        "prompt_optimizer",
        "quick_mode",
        "auto_route",
        "auto_route_fast_model",
        "git_native",
        "banner",
        "no_confirm",
    }
    for key, value in profile.items():
        if key not in _PROFILE_FIELDS:
            continue
        if key == "model":
            config.active_model_alias = str(value)
        elif key == "no_confirm":
            pass  # applied separately via _session.no_confirm
        elif key == "banner":
            if value is None or (isinstance(value, str) and value.strip() == ""):
                config.banner = "off"
            else:
                _banner_val = str(value).lower().strip()
                if _banner_val in ("full", "compact", "off", "none", "false", "0"):
                    config.banner = (
                        "off" if _banner_val in ("off", "none", "false", "0") else _banner_val
                    )
        elif hasattr(config, key):
            try:
                setattr(config, key, type(getattr(config, key))(value))
            except Exception:
                setattr(config, key, value)
    get_logger().info("Applied profile '%s'", profile_name)


def _context_advisory(
    used_tokens: int,
    max_tokens: int | None,
    console: "object | None" = None,
) -> None:
    """Print a context-usage advisory when session token usage crosses warning thresholds.

    Thresholds:
      >= 70% and < 85%: dim yellow warning
      >= 85%:           dim red warning (approaching hard limit)
    """
    if not max_tokens or max_tokens <= 0 or used_tokens <= 0:
        return
    pct = int(used_tokens * 100 / max_tokens)
    if pct < 70:
        return
    if pct >= 85:
        msg = f"\u26a0 Context at {pct}% \u2014 approaching limit; start /session new"
        style = "dim red"
    else:
        msg = f"\u26a0 Context at {pct}% \u2014 consider /session new or --quick"
        style = "dim yellow"
    if console is not None:
        try:
            console.print(f"  [{style}]{msg}[/{style}]")  # type: ignore[union-attr]
        except Exception:
            print(f"  {msg}")
    else:
        print(f"  {msg}")


_session = SessionState()
commands.configure(console, _session)
confirmation.configure(console)
_session_tokens = _TokenAccumulator()
_active_milestones: list = []


_rich_ui = _RichConfirmationUI()

__license__ = "Cogtrix Source-Available License 1.0"

# Module-level flag: skip all tool safety confirmations (set by --no-confirm / -y)
# Accessed via _session.no_confirm

# Track resources for cleanup
_cleanup_resources: list = []


_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
}


def _friendly_error(exc: Exception, provider: str = "", base_url: str = "") -> str:
    """Return a concise, user-friendly message for common exceptions.

    Falls back to ``str(exc)`` for truly unexpected errors.
    """
    msg = str(exc).lower()
    if any(
        x in msg for x in ("rate limit", "rate_limit", "429", "quota exceeded", "too many requests")
    ):
        return (
            f"Rate limit reached for provider '{provider}'.\n"
            "   Wait a moment and try again, or switch to a different model with /model."
        )
    if any(
        x in msg
        for x in (
            "context_length_exceeded",
            "context window",
            "maximum context length",
            "too many tokens",
            "reduce the length",
            "context length",
        )
    ):
        return (
            "Message history is too long for this model's context window.\n"
            "   Options: /session <name> to start fresh, /mode to change memory mode,\n"
            "   or switch to a larger-context model with /model."
        )
    if "api_key" in msg or "api key" in msg or "authentication" in msg or "unauthorized" in msg:
        if provider:
            env_var = _PROVIDER_ENV_VARS.get(provider.lower(), "")
            env_hint = f"Set {env_var}" if env_var else "Set the appropriate environment variable"
            return (
                f"Provider '{provider}' requires an API key.\n"
                f"   {env_hint} or add 'api_key' under '{provider}' in your config file."
            )
        return (
            "An API key is required.\n"
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


def _load_project_context() -> tuple[str, str | None]:
    """Load COGTRIX.md from cwd, home, or XDG config dir.

    Returns (content, path) where content is truncated to 4000 chars.
    Returns ("", None) if no file found.
    """
    search_paths = [
        Path.cwd() / "COGTRIX.md",
        Path.home() / "COGTRIX.md",
        Path.home() / ".config" / "cogtrix" / "COGTRIX.md",
    ]
    for path in search_paths:
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content) > 4000:
                    content = content[:4000] + "\n[... truncated at 4000 chars ...]"
                return content, str(path)
            except OSError:
                continue
    return "", None


_AT_FILE_RE = re.compile(r"@([\w./\-]+)")
_AT_MAX_FILE_CHARS = 50_000
_AT_MAX_FILES = 5


def _build_completion_script(shell: str, data_dir: str | None = None) -> str:
    """Build a bash or zsh completion script for the cogtrix CLI."""
    flags = [
        "--help",
        "--setup",
        "--check-config",
        "--model",
        "--session",
        "--memory-mode",
        "--config-file",
        "--data-dir",
        "--prompt",
        "--prompt-file",
        "--no-stream",
        "--silent",
        "--system-prompt",
        "--system-prompt-file",
        "--output",
        "--no-confirm",
        "--auto-route",
        "--quick",
        "--git-native",
        "--log",
        "--verbose",
        "--debug",
        "--verbosity",
        "--tools",
        "--activate-tools",
        "--allow-write-path",
        "--ingest",
        "--docs-dir",
        "--vectordb-dir",
        "--embedding-provider",
        "--embedding-model",
        "--assistant",
        "--setup-docs",
        "--setup-output",
        "--install-completion",
    ]
    short_flags = [
        "-h",
        "-m",
        "-s",
        "-M",
        "-c",
        "-o",
        "-y",
        "-R",
        "-Q",
        "-G",
        "-S",
        "-v",
    ]
    memory_modes = "conversation code reasoning"

    # Discover session names from data_dir
    sessions = ""
    if data_dir:
        try:
            sessions_path = Path(data_dir) / "history"
            if sessions_path.is_dir():
                names = [p.stem for p in sessions_path.glob("*.json") if p.is_file()]
                sessions = " ".join(sorted(names))
        except Exception:
            pass

    all_flags = " ".join(flags + short_flags)

    if shell == "zsh":
        return f"""#compdef cogtrix

_cogtrix() {{
  local context state state_descr line
  typeset -A opt_args

  _arguments \\
    "(--help -h)"{{"(--help -h)",-h,--help}}"[show help]" \\
    "(--session -s)"{{"(--session -s)",-s,--session}}"[session ID]: :->sessions" \\
    "(--memory-mode -M)"{{"(--memory-mode -M)",-M,--memory-mode}}"[memory mode]: :(conversation code reasoning)" \\
    "(--model -m)"{{"(--model -m)",-m,--model}}"[model alias]:model alias:" \\
    "(--quick -Q)"{{"(--quick -Q)",-Q,--quick}}"[quick mode]" \\
    "(--auto-route -R)"{{"(--auto-route -R)",-R,--auto-route}}"[auto model routing]" \\
    "(--git-native -G)"{{"(--git-native -G)",-G,--git-native}}"[git auto-commit mode]" \\
    "(--no-confirm -y)"{{"(--no-confirm -y)",-y,--no-confirm}}"[auto-approve tools]" \\
    "(--prompt)--prompt[single prompt]: :_message prompt" \\
    "*: :_message args"

  case $state in
    sessions)
      local sessions=({sessions})
      _describe "session" sessions
      ;;
  esac
}}

_cogtrix "$@"

# To install: add this to ~/.zshrc
#   source <(cogtrix --install-completion zsh)
"""
    else:
        # bash
        return f"""# Bash completion for cogtrix
_cogtrix_completion() {{
  local cur prev opts
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  prev="${{COMP_WORDS[COMP_CWORD-1]}}"
  opts="{all_flags}"

  case "$prev" in
    --memory-mode|-M)
      COMPREPLY=( $(compgen -W "{memory_modes}" -- "$cur") )
      return 0
      ;;
    --session|-s)
      COMPREPLY=( $(compgen -W "{sessions}" -- "$cur") )
      return 0
      ;;
    --model|-m|--config-file|-c|--prompt-file|--system-prompt-file|--output|-o|--data-dir|--log|--docs-dir|--vectordb-dir|--setup-output|--setup-docs|--allow-write-path)
      COMPREPLY=( $(compgen -f -- "$cur") )
      return 0
      ;;
    --embedding-provider)
      COMPREPLY=( $(compgen -W "openai ollama" -- "$cur") )
      return 0
      ;;
    --tools)
      COMPREPLY=( $(compgen -W "all none minimal" -- "$cur") )
      return 0
      ;;
  esac

  if [[ "$cur" == -* ]]; then
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
    return 0
  fi
}}

complete -F _cogtrix_completion cogtrix cogtrix.py

# To install: add this to ~/.bashrc
#   source <(cogtrix --install-completion)
"""


def _classify_query_complexity(text: str) -> str:
    """Return 'simple' or 'complex' based on a fast local heuristic.

    Heuristic:
    - Any complex marker → complex
    - Prompts that already look like code/reasoning or non-simple task work
      → complex
    - Word count > 80 → complex (long query likely needs deep reasoning)
    - First word is a simple keyword AND word count <= 30 → simple
    - Otherwise → complex (conservative default)
    """
    words = text.lower().split()
    if not words:
        return "simple"

    # Cross-validate against the other prompt classifiers so this router does
    # not contradict established code / reasoning / task-complexity signals.
    if classify_memory_mode(text) != "conversation":
        return "complex"
    if classify_task_complexity(text) != TaskComplexity.SIMPLE:
        return "complex"

    if any(w in _COMPLEX_QUERY_MARKERS for w in words):
        return "complex"
    if len(words) > 80:
        return "complex"
    if words[0] in _SIMPLE_QUERY_KEYWORDS and len(words) <= 30:
        return "simple"
    return "complex"


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


def _expand_at_references(text: str) -> tuple[str, list[str]]:
    """Expand @file and @folder references in user input.

    Returns (expanded_text, list_of_injected_paths).
    Binary files and missing paths are left as-is with an inline note.
    """
    injected: list[str] = []
    count = [0]  # mutable for closure

    def _replace(m: re.Match[str]) -> str:
        ref = m.group(1)
        if count[0] >= _AT_MAX_FILES:
            return m.group(0)
        path = Path(ref)
        if not path.is_absolute():
            path = Path.cwd() / ref
        if path.is_dir():
            count[0] += 1
            lines = [f"[Directory: {ref}]"]
            try:
                all_entries = sorted(path.rglob("*", recurse_symlinks=False))
                for entry in all_entries[:60]:
                    if _should_include_entry(entry, path):
                        rel = entry.relative_to(path)
                        if entry.is_file():
                            size = entry.stat().st_size
                            lines.append(f"  {rel}  ({size:,} bytes)")
                        elif entry.is_dir():
                            lines.append(f"  {rel}/")
                if len(all_entries) > 60:
                    lines.append("  ... (truncated)")
            except OSError:
                pass
            injected.append(str(path))
            return "\n".join(lines)
        elif path.is_file():
            count[0] += 1
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                truncated = len(content) > _AT_MAX_FILE_CHARS
                if truncated:
                    content = content[:_AT_MAX_FILE_CHARS]
                header = f"[Content of {ref}{'  (truncated)' if truncated else ''}]\n"
                injected.append(str(path))
                return header + content
            except OSError as exc:
                return f"[@{ref}: error reading — {exc}]"
        else:
            return m.group(0)  # not found, leave as-is

    expanded = _AT_FILE_RE.sub(_replace, text)
    return expanded, injected


def _should_include_entry(entry: Path, root: Path) -> bool:
    """Check if an entry should be included in directory listing.

    Applies bounded traversal rules:
    - Hidden files (dotfiles) are excluded
    - Directory depth limited to 3 levels
    - Symlinks to directories are excluded to prevent infinite loops
    """
    # Skip hidden files (dotfiles)
    if entry.name.startswith("."):
        return False

    # Depth check: relative path must have at most 3 directory components
    # e.g., level1/file.txt has 1 dir component, level1/level2/file.txt has 2
    rel = entry.relative_to(root)
    dir_parts = rel.parts[:-1] if rel.parts else ()
    if len(dir_parts) > 3:
        return False

    # Skip symlinks to directories to prevent infinite loops
    if entry.is_symlink() and entry.is_dir():
        return False

    return True


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
    """Store embedding config for lazy initialisation.

    The embedding provider SDK is NOT created here — this avoids the
    ~280 ms provider-SDK init cost at session startup.  The provider is
    created on first actual use (first ``prepare_context()`` that needs
    vector recall, or first background ``_run_slow_path()`` call).

    Falls back to calling ``set_embeddings()`` directly when the manager
    does not expose ``set_embedding_config()`` (e.g. third-party subclasses).
    """
    _log = get_logger()
    try:
        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
    except Exception as exc:
        _log.debug("Could not resolve embedding config: %s", exc)
        return

    vector_dir = str(config.resolve_data_path("vectordb/sessions"))
    if hasattr(memory_manager, "set_embedding_config"):
        memory_manager.set_embedding_config(
            emb_type,
            emb_model,
            emb_base_url,
            emb_api_key,
            vector_store_dir=vector_dir,
        )
        _log.debug("Embedding config stored for lazy init (provider: %s)", emb_type)
    else:
        try:
            from cogtrix_core.providers import create_embeddings_from_config

            fn, tag = create_embeddings_from_config(
                emb_type, model=emb_model, base_url=emb_base_url, api_key=emb_api_key
            )
            memory_manager.set_embeddings(fn, tag, vector_store_dir=vector_dir)
            _log.debug("Memory vector recall: using %s", tag)
        except Exception as exc2:
            _log.debug("Embedding provider '%s' unavailable: %s", emb_type, exc2)


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
    console.print()  # type: ignore[union-attr]
    console.print(  # type: ignore[union-attr]
        Panel(Group(*renderables), title="Commands", border_style="cyan", padding=(1, 2))
    )
    console.print()  # type: ignore[union-attr]


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
    model_cfg: Any,
    stats: dict,
    msg_count: int,
    system_prompt: str | None = None,
    mcp_manager: Any = None,
    project_context_path: str | None = None,
) -> None:
    """Render /info output using Rich."""
    if console is None or Panel is None:  # pragma: no cover
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
    console.print()
    console.print(Panel(body, title="Session Information", border_style="cyan", padding=(1, 2)))
    console.print()


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
    if console is None or Panel is None:  # pragma: no cover
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
    console.print()
    console.print(Panel(body, title="Session", border_style="cyan", padding=(1, 2)))
    console.print()


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


def _build_slash_commands() -> SlashCommandRegistry:
    """Create and populate the slash command registry."""
    reg = commands.SlashCommandRegistry()

    reg.register(
        SlashCommand(
            name="help",
            handler=commands.SlashCommandRegistry._cmd_help,
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
            handler=commands.SlashCommandRegistry._cmd_quit,
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
            handler=commands.SlashCommandRegistry._cmd_info,
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
            handler=commands.SlashCommandRegistry._cmd_tools,
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
            handler=commands.SlashCommandRegistry._cmd_mcp,
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
            handler=commands.SlashCommandRegistry._cmd_clear,
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
            handler=commands.SlashCommandRegistry._cmd_think,
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
            handler=commands.SlashCommandRegistry._cmd_delegate,
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
            handler=commands.SlashCommandRegistry._cmd_mode,
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
            handler=commands.SlashCommandRegistry._cmd_memory,
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
            handler=commands.SlashCommandRegistry._cmd_agents,
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
            handler=commands.SlashCommandRegistry._cmd_tasks,
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
            handler=commands.SlashCommandRegistry._cmd_spawn,
            short_help="Submit a background task to the task queue",
            long_help=(
                "Usage: /spawn <agent_name> <task_description>\n\n"
                "Submits a background task to the queue for the named agent.\n"
                "The task runs asynchronously; track progress with /tasks.\n\n"
                "Examples:\n"
                "  /spawn researcher Summarise the latest arXiv ML papers\n"
                "  /spawn coder Refactor cogtrix_core/tools/shell.py for better error handling"
            ),
        )
    )

    reg.register(
        SlashCommand(
            name="goal",
            handler=commands.SlashCommandRegistry._cmd_goal,
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
            handler=commands.SlashCommandRegistry._cmd_model,
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
            handler=commands.SlashCommandRegistry._cmd_provider,
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
            handler=commands.SlashCommandRegistry._cmd_session_switch,
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
            handler=commands.SlashCommandRegistry._cmd_debug,
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
            handler=commands.SlashCommandRegistry._cmd_verbose,
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
            handler=commands.SlashCommandRegistry._cmd_approve,
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
            handler=commands.SlashCommandRegistry._cmd_optimizer,
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
            handler=commands.SlashCommandRegistry._cmd_paste,
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
            handler=commands.SlashCommandRegistry._cmd_undo,
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
            handler=commands.SlashCommandRegistry._cmd_compact,
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
            handler=commands.SlashCommandRegistry._cmd_retry,
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
            handler=commands.SlashCommandRegistry._cmd_setup,
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
            handler=commands.SlashCommandRegistry._cmd_export,
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
            handler=commands.SlashCommandRegistry._cmd_system_prompt,
            short_help="Display the full system prompt",
            aliases=[],
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
            pc, mc = config.resolve_llm_config()
            print(f"  Provider: {mc.provider} ({pc.type})")
            actual_model = mc.model
            print(f"  Model: {actual_model}")
            if pc.base_url:
                print(f"  Base URL: {pc.base_url}")
            if pc.api_key:
                print("  API Key: ***configured***")
        except (ValueError, Exception) as e:
            print(f"  Error: {e}")
            return 1
        try:
            _active_provider = config.get_active_model().provider
        except Exception:
            _active_provider = None
        for name in config.list_providers():
            current = " (current)" if name == _active_provider else ""
            try:
                prov = config.get_provider_config(name)
                print(f"  - {name}: {prov.type}{current}")
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

    try:
        _check_pc, _check_mc = config.resolve_llm_config()
        _check_provider_name = _check_mc.provider
    except Exception:
        _check_pc, _check_mc, _check_provider_name = None, None, "unknown"

    # Provider check
    console.print(f"\n[bold]Provider:[/bold] {_check_provider_name}")

    try:
        if _check_pc is None or _check_mc is None:
            raise ValueError("Could not resolve LLM configuration")
        provider_config = _check_pc
        model_config = _check_mc
        console.print(f"  Type: {provider_config.type}")
        actual_model = model_config.model
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
        if model_config.context_window:
            console.print(f"  Context Size: {model_config.context_window}")
        if model_config.temperature is not None:
            console.print(f"  Temperature: {model_config.temperature}")
    except ValueError as e:
        console.print(f"  [red]✗ Error: {e}[/red]")
        return 1

    # List all providers
    console.print("\n[bold]Available Providers:[/bold]")
    for name in config.list_providers():
        current = " [cyan](current)[/cyan]" if name == _check_provider_name else ""
        try:
            prov = config.get_provider_config(name)
            console.print(f"  • {name}: {prov.type}{current}")
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
    from cogtrix_core.rag import IngestConfig, ingest_documents

    # Build ingest configuration from args and config
    docs_dir = Path(args.docs_dir if args.docs_dir else config.rag.docs_dir)
    # #2216: write the index to the SAME directory the query side reads
    # (configure_rag_tool) — i.e. ``<vectordb_dir>/faiss_index`` — via the
    # shared helper. Previously this wrote straight to ``vectordb_dir`` (no
    # ``faiss_index`` segment), so query_knowledge_base never found a
    # CLI-ingested index.
    vectordb_dir = config.resolve_rag_index_dir(args.vectordb_dir)

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
        # #1981: pass through the BM25 sidecar build flag from operator
        # config — when set, ``ingest_documents`` writes a ``bm25.pkl``
        # alongside the FAISS index so the query path can run hybrid.
        build_bm25_sidecar=config.rag.build_bm25_sidecar,
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
        print(  # codeql[py/clear-text-logging-sensitive-data]
            f"  Vector DB output:    {ingest_config.vectordb_dir}"
        )
        print(f"  Embedding provider:  {emb_type}")  # codeql[py/clear-text-logging-sensitive-data]
        if ingest_config.embedding_model:
            print(  # codeql[py/clear-text-logging-sensitive-data]
                f"  Embedding model:     {ingest_config.embedding_model}"
            )
        if emb_base_url:
            print(  # codeql[py/clear-text-logging-sensitive-data]
                f"  Base URL:            {emb_base_url}"
            )
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
# Backward-compat aliases: tool output cap moved to cogtrix_core/tools/configure.py
_TOOL_OUTPUT_CAP_RATIO = 0.10
_TOOL_OUTPUT_CAP_MIN_CHARS = TOOL_OUTPUT_CAP_MIN_CHARS
_compute_tool_output_cap = compute_tool_output_cap


def create_safe_tool_wrapper(
    tool,
    tool_name: str,
    registry: ToolRegistry,
    approvals: set,
    session_state: SessionState | None = None,
    git_native: bool = False,
    tool_trust: dict | None = None,
):
    """Wrap a tool with confirmation gate. Delegates to src.agent.safety."""
    from cogtrix_core.agent.safety import create_safe_tool_wrapper as _safety_wrapper

    return _safety_wrapper(
        tool,
        tool_name,
        registry,
        approvals,
        session_state=session_state or _session,
        ui=_rich_ui,
        git_native=git_native,
        tool_trust=tool_trust,
    )


def _maybe_skip_force_deep_think_for_tool_intensive_task(
    wants_deep: bool,
    original_input: str,
    llm: Any,
    log: Any = None,
) -> bool:
    """Disable force-deep-think for tool-intensive tasks so delegation can still run."""
    if not wants_deep or llm is None:
        return wants_deep

    task_cat = classify_think_task(original_input, llm)
    if task_cat and task_cat.tool_intensive:
        if log:
            log.info(
                "Skipping force deep_think: task classified as '%s' "
                "(tool-intensive — agent's tool work is the primary output)",
                task_cat.name,
            )
        return False

    return wants_deep


# Backward-compat aliases: graph builder and constants moved to cogtrix_core/orchestration/graph.py
_build_agent_graph = build_agent_graph
_EMPTY_RESPONSE_MSG = EMPTY_RESPONSE_MSG
_MCP_TOOLS_READY_EVENT: Any | None = None


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
    deny_all_tools: bool = False,
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
        if deny_all_tools:
            _session.deny_all = True

        # Log user message
        log_user_message(prompt_text)

        # Preserve the user's original phrasing before the optimizer
        # rewrites it — memory, classification, delegation must all
        # see what the user actually typed.
        original_input = prompt_text

        # Run prepare_context and optimize_prompt concurrently when optimizer is enabled.
        # The two operations are independent: the optimizer rewrites the prompt text
        # while prepare_context reads conversation history — no data dependency.
        _progress_tool: object = None
        _run_sys_prompt = system_prompt
        _al = active_tools_list if active_tools_list is not None else []
        _agent_t0 = _time_mod.monotonic()
        _spinner.start()
        try:
            if config and config.prompt_optimizer:
                # Use explicit ThreadPoolExecutor (not `with`) so shutdown(wait=False)
                # can be used on timeout — `__exit__` calls shutdown(wait=True) which
                # blocks on hung threads.
                _pool = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep")
                try:
                    _ctx_future = _pool.submit(memory_manager.prepare_context, prompt_text)
                    _opt_future = _pool.submit(
                        optimize_prompt, prompt_text, llm, plan_milestones=True
                    )
                    try:
                        context = _ctx_future.result(timeout=60)
                        plan = _opt_future.result(timeout=60)
                    except _cf.TimeoutError:
                        _ctx_future.cancel()
                        _opt_future.cancel()
                        _pool.shutdown(wait=False)
                        log.warning(
                            "Prompt prep timed out after 60s — falling back to sequential path"
                        )
                        context = memory_manager.prepare_context(prompt_text)
                    else:
                        prompt_text = plan.text
                        _progress_tool, _run_sys_prompt = _inject_milestones(
                            plan, _al, _active_milestones, system_prompt or ""
                        )
                finally:
                    _pool.shutdown(wait=False)
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

            # ── Research delegate pre-flight ─────────────────────────────────────
            # When research_delegate_auto is enabled and session context is above the
            # threshold, research queries are routed to a subagent instead of the
            # main agent to prevent raw web content from filling the main context.
            _preflight_output: str | None = None
            _rd_auto = getattr(config, "research_delegate_auto", False) if config else False
            if _rd_auto and max_context_tokens and max_context_tokens > 0:
                _rd_threshold = (
                    getattr(config, "research_delegate_auto_threshold", 0.50) if config else 0.50
                )
                _ctx_est = getattr(context, "token_estimate", 0) or 0
                _session_ratio = _ctx_est / max_context_tokens
                if _session_ratio >= _rd_threshold and _looks_like_research_query(original_input):
                    from cogtrix_core.tools.delegate import delegate_task, get_delegate_tools

                    if get_delegate_tools():
                        log.info(
                            "research_delegate_auto: pre-flighting research query "
                            "(session context %.0f%%, threshold %.0f%%)",
                            _session_ratio * 100,
                            _rd_threshold * 100,
                        )
                        _spinner.start()
                        try:
                            _preflight_output = delegate_task(
                                task=original_input,
                                use_tools=True,
                                timeout=(
                                    getattr(config, "research_delegate_timeout", 300)
                                    if config
                                    else 300
                                ),
                            )
                        except Exception as _e:
                            log.warning(
                                "research_delegate_auto pre-flight failed: %s — falling back", _e
                            )
                            _preflight_output = None
                        finally:
                            _spinner.stop()

            _acc = _TokenAccumulator()
            _agent_cbs = (callbacks or []) + [_acc]
            if _preflight_output:
                output = _preflight_output
            else:
                _run_config = AgentRunConfig.from_app_config(config)
                _run_config.llm = llm
                _run_config.system_prompt = _run_sys_prompt
                _run_config.available_tools = (
                    dict(available_tools) if available_tools else available_tools
                )
                _run_config.active_tools_list = active_tools_list
                _run_config.max_context_tokens = max_context_tokens
                _run_config.preset_tools = (
                    TOOL_PRESETS.get(config.memory_mode, set()) if config else set()
                )
                _run_config.compression_llm = compression_llm
                _run_config.session_state = _session
                _run_config.memory_manager = memory_manager
                _run_config.confirmation_ui = _rich_ui
                _run_config.on_tool_expansion = _tool_expansion_ui
                _run_config.tools_ready = _MCP_TOOLS_READY_EVENT
                output = _run_agent_cli(
                    prompt_text,
                    context.messages,
                    registry,
                    approvals,
                    context_prefix=context.context_prefix,
                    callbacks=_agent_cbs,
                    result_messages=agent_msgs,
                    config=_run_config,
                )
        finally:
            _spinner.stop()
            _cleanup_milestones(_progress_tool, _al, _active_milestones)

        _maybe_write_session_metrics(getattr(config, "log_file", None))

        # ── Enforce deep_think when the user requested it ────────
        # Force-call if: (a) agent skipped deep_think entirely, OR
        # (b) agent called it but with inadequate context (references
        # instead of actual data — fewer than MIN_GOOD_CONTEXT_LEN chars).
        # However, for tool-intensive tasks (bug hunting, sysadmin, etc.)
        # "think deeply" is treated as a quality hint — the agent's
        # actual tool work is more valuable than isolated reasoning.
        _research_output: str = ""
        wants_deep = _maybe_skip_force_deep_think_for_tool_intensive_task(
            wants_deep, original_input, llm, log
        )
        if wants_deep and output:
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
                _rd_enabled = getattr(config, "research_delegate_enabled", True) if config else True
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
        # Skip if the model already produced a substantial response.
        _resp_substantial = len(output or "") > 500
        if (
            not wants_deep
            and output
            and not _resp_substantial
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
                "Prompt requests file actions but none were performed — running execution phase"
            )
            _spinner.start()
            try:
                _exec_run_config = AgentRunConfig.from_app_config(config)
                _exec_run_config.llm = llm
                _exec_run_config.system_prompt = system_prompt
                _exec_run_config.available_tools = (
                    dict(available_tools) if available_tools else available_tools
                )
                _exec_run_config.active_tools_list = active_tools_list
                _exec_run_config.max_context_tokens = max_context_tokens
                _exec_run_config.preset_tools = (
                    TOOL_PRESETS.get(config.memory_mode, set()) if config else set()
                )
                _exec_run_config.session_state = _session
                _exec_run_config.on_tool_expansion = _tool_expansion_ui
                exec_output, exec_msgs = run_execution_phase(
                    output,
                    prompt_text,
                    context.messages,
                    registry,
                    approvals,
                    context_prefix=context.context_prefix,
                    callbacks=_agent_cbs,
                    config=_exec_run_config,
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
                console.rule(style="dim blue")
                console.print(
                    Padding(
                        Markdown(preserve_tables_for_markdown(output)),
                        (1, 0, 1, 2),
                    )
                )
                _session_tokens.input_tokens += _acc.input_tokens
                _session_tokens.output_tokens += _acc.output_tokens
                # print_stats_footer emits the progress bar and context warnings
                # (replaces _format_stats_line + _context_advisory)
                from cogtrix_core.ui.stats import print_stats_footer as _print_stats_footer

                _print_stats_footer(
                    console=console,
                    session_tokens=_acc.last_input_tokens or _acc.input_tokens,
                    max_context_tokens=max_context_tokens,
                    input_tokens=_acc.input_tokens,
                    output_tokens=_acc.output_tokens,
                )
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


def _milestone_progress(milestone_index: int, status: str) -> None:
    milestones = list(_active_milestones)  # atomic snapshot
    if not milestones:
        return
    total = len(milestones)
    idx = max(1, min(milestone_index, total))
    title = milestones[idx - 1].title
    ctx = f"[{idx}/{total}] {title}"
    if status:
        ctx += f" \u2014 {status}"
    _spinner.set_context(ctx)


def _inject_milestones(
    plan: PromptPlan,
    tools_list: list,
    milestones_store: list,
    sys_prompt: str,
) -> tuple[object, str]:
    """Inject report_progress tool and augment system prompt when plan has milestones.

    Returns (progress_tool, augmented_sys_prompt). When plan has no milestones,
    returns (None, sys_prompt) unchanged.
    """
    if not plan.has_milestones:
        return None, sys_prompt
    tool = create_report_progress_tool(plan.milestones)  # can raise — do first
    milestones_store[:] = plan.milestones  # only on success
    tools_list.append(tool)
    instr = format_milestone_instructions(plan.milestones)
    augmented = sys_prompt + "\n\n" + instr
    _spinner.set_context(f"[1/{len(plan.milestones)}] {plan.milestones[0].title}")
    return tool, augmented


def _cleanup_milestones(
    progress_tool: object,
    tools_list: list,
    milestones_store: list,
) -> None:
    if progress_tool is None:
        return
    tools_list[:] = [t for t in tools_list if getattr(t, "name", "") != "report_progress"]
    milestones_store.clear()
    _spinner.clear_context()


def _maybe_write_session_metrics(log_file: str | None) -> None:
    """Write session metrics after a run if a log file is configured.

    This is a best-effort operation: failures are silently ignored so
    metrics never crash the CLI.
    """
    if log_file is None:
        return
    resolved = log_file if log_file else "cogtrix.log"
    try:
        p = Path(resolved)
        if p.exists():
            write_session_metrics(str(p))
    except Exception:
        pass


def main():
    """Main CLI loop."""
    # Parse command line arguments
    args = parse_arguments()

    # Shell completion — handled before config loading (no config needed)
    if getattr(args, "install_completion", None) is not None:
        import os as _os

        _shell = args.install_completion
        if _shell == "auto" or not _shell:
            _shell = "zsh" if "zsh" in _os.environ.get("SHELL", "") else "bash"
        print(_build_completion_script(_shell, data_dir=getattr(args, "data_dir", None)))
        raise SystemExit(0)

    # Silent mode: disable all ANSI/spinner output before any Rich or spinner
    # initialization so NO_COLOR is effective from the very start.
    if getattr(args, "silent", False):
        import os as _os

        _os.environ.setdefault("NO_COLOR", "1")

    # Load configuration (CLI > env > config file > defaults)
    try:
        config = load_config(args)
    except ConfigError as e:
        if console:
            console.print("\n[bold red]Configuration Error[/bold red]\n")
            console.print(f"[yellow]{e}[/yellow]\n")
            console.print(
                "[dim]  → Run [bold]cogtrix.py --setup[/bold] to generate a new config file.[/dim]"
            )
            console.print(
                "[dim]  → Run [bold]cogtrix.py --check-config[/bold] to validate an existing one.[/dim]\n"  # noqa: E501
            )
        else:
            print(f"\nConfiguration Error:\n{e}\n")
            print("  → Run `cogtrix.py --setup` to generate a new config file.")
            print("  → Run `cogtrix.py --check-config` to validate an existing one.\n")
        sys.exit(1)

    # Load COGTRIX.md project context (cwd → home → XDG config dir)
    _project_context, _project_context_path = _load_project_context()

    # Handle --check-config mode (early exit)
    if args.check_config:
        sys.exit(check_config(config))

    # Handle --ingest mode (early exit)
    if args.ingest:
        sys.exit(run_ingest(args, config))

    # Handle --setup mode (early exit)
    if getattr(args, "setup", False):
        from pathlib import Path as _Path

        from cogtrix_core.setup_wizard import run_setup_wizard

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
                "DEEPSEEK_API_KEY",
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
        from cogtrix_core.setup_wizard import run_setup_wizard

        if console is not None:
            console.print("\n  [bold]No configuration found.[/bold] Starting setup wizard...\n")
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
    setup_logging(
        log_file=config.log_file,
        debug=config.debug,
        verbose=config.verbose,
        verbosity=config.verbosity,
    )
    log = get_logger()

    # Register SIGTERM handler for graceful shutdown
    # SIGTERM should behave like Ctrl+C (KeyboardInterrupt)
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        log.debug("SIGTERM handler registered")
    except (OSError, ValueError) as exc:
        # SIGTERM not available on some platforms (e.g., Windows)
        log.debug(f"Could not register SIGTERM handler: {exc}")

    if config.log_file is not None:
        log_file_display = config.log_file or "cogtrix.log"
        debug_str = " (debug)" if config.debug else ""
        if console is not None:
            console.print(f"[dim]Logging to: {log_file_display}{debug_str}[/dim]")
        else:
            print(f"Logging to: {log_file_display}{debug_str}")

    # Dump full resolved config at DEBUG level
    config.dump_debug(log)

    # Memory manager setup
    memory_store = JsonFileMemoryStore(str(config.resolve_data_path("history")))

    # --profile / -P: apply a named config profile (before individual flag overrides)
    _profile_name = getattr(args, "profile", None)
    if _profile_name:
        _apply_profile(config, _profile_name)
        _profile_data = config.profiles.get(_profile_name, {})
        if _profile_data.get("no_confirm"):
            _session.no_confirm = True

    # --no-confirm / -y: skip all tool safety confirmations (overrides profile)
    if getattr(args, "no_confirm", False):
        _session.no_confirm = True

    # --auto-route / -R: enable auto model routing from CLI
    if getattr(args, "auto_route", False):
        config.auto_route = True

    # --quick / -Q: skip optimizer, memory, and compression
    if getattr(args, "quick", False):
        config.quick_mode = True

    # --git-native / -G: auto stage+commit after each file write
    if getattr(args, "git_native", False):
        config.git_native = True
    _git_native = config.git_native

    # --no-banner: suppress the startup banner
    if getattr(args, "no_banner", False):
        config.banner = "off"
    if getattr(args, "pipe", False) and not sys.stdout.isatty():
        config.banner = "off"

    # Apply theme
    if config:
        from cogtrix_core.ui.theme import set_theme

        try:
            set_theme(config.theme)
        except ValueError:
            pass  # unknown theme name — keep default

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
    configure_searxng_tool(config)
    configure_serpapi_tool(config)
    configure_google_search_tool(config)

    # Load tools on startup
    tool_filter = getattr(args, "tools", None)
    if tool_filter == "none":
        registry = ToolRegistry()
    else:
        registry = load_tools(tool_filter, config=config)
    try:
        tool_names = registry.list_tools()
        if not tool_names:
            log.warning("No tools loaded")
    except Exception as e:
        log.error("Error loading tools: %s", e)
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

    configure_file_ops_tool(config)
    configure_file_read_dirs(config)

    # Wire cron tools: LLM factory always reflects the current llm variable
    # (which is a mutable reference updated on model/provider switches).
    _cron_llm_ref: list = []  # [llm] — updated after each model switch

    def _clone_session_state(source: SessionState) -> SessionState:
        clone = SessionState(no_confirm=source.no_confirm)
        clone.denials = set(source.denials)
        clone.deny_all = source.deny_all
        clone.approvals = set(source.approvals)
        clone.loaded_tools = set(source.loaded_tools)
        clone.pinned_tools = set(source.pinned_tools)
        clone.all_tool_descriptions = dict(source.all_tool_descriptions)
        clone.all_tool_originals = dict(source.all_tool_originals)
        clone.checkpoint_store = source.checkpoint_store
        return clone

    def _run_inherited_cron_job(job: Any) -> str:
        try:
            cron_context = memory_manager.prepare_context(job.prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cron job %s context preparation failed: %s", job.id, exc)
            cron_context = None

        inherited_session = _clone_session_state(_session)
        inherited_system_prompt = system_prompt
        if cron_context is not None and getattr(cron_context, "system_additions", None):
            inherited_system_prompt = (
                f"{inherited_system_prompt}\n{cron_context.system_additions}"
                if inherited_system_prompt
                else str(cron_context.system_additions)
            )

        _cron_run_config = AgentRunConfig.from_app_config(config)
        _cron_run_config.llm = _cron_llm_ref[0] if _cron_llm_ref else llm
        _cron_run_config.system_prompt = inherited_system_prompt
        _cron_run_config.available_tools = dict(available_tools) if available_tools else None
        _cron_run_config.active_tools_list = list(tools) if tools else None
        _cron_run_config.max_context_tokens = max_context_tokens
        _cron_run_config.preset_tools = TOOL_PRESETS.get(config.memory_mode, set())
        _cron_run_config.compression_llm = compression_llm
        _cron_run_config.session_state = inherited_session
        _cron_run_config.tools_ready = _MCP_TOOLS_READY_EVENT
        _cron_run_config.checkpoint_store = _session.checkpoint_store
        return _run_agent_cli(
            job.prompt,
            cron_context.messages if cron_context is not None else [],
            registry,
            set(approvals),
            context_prefix=(cron_context.context_prefix if cron_context is not None else None),
            config=_cron_run_config,
        )

    configure_cron_tool(
        config,
        llm_factory=lambda: _cron_llm_ref[0] if _cron_llm_ref else None,
        job_runner=_run_inherited_cron_job,
    )
    configure_email_tool(config)

    # Load named agent definitions from AGENTS.md (cwd or home dir)
    from cogtrix_core.agent import registry as _agent_registry
    from cogtrix_core.agent.agents_md import load_default_agents as _load_agents_md

    _agents_md = _load_agents_md()
    _agent_registry.load_from_config(config)
    _agent_registry.merge_from_agents_md(_agents_md)

    # Initialise background task queue for /spawn and /tasks commands
    try:
        from cogtrix_core.tasks.queue import init_task_queue

        _data_dir = Path(getattr(config, "data_dir", "data"))
        _tasks_db = _data_dir / "tasks" / "tasks.db"
        _tasks_log = _data_dir / "tasks"
        _tasks_db.parent.mkdir(parents=True, exist_ok=True)
        _task_queue = init_task_queue(_tasks_db, _tasks_log)
        # start() is deferred until after set_runner() — see below
    except Exception as _tq_exc:
        log.debug("Task queue init skipped: %s", _tq_exc)

    def _reconfigure_all_tools(cfg: Any, ctx_tokens: int, tool_list: list) -> None:
        """Call every configure_* function and recompute the tool output cap."""
        configure_delegate_tool(cfg, status_callback=_delegation_status)
        configure_deep_think_tool(cfg)
        configure_tavily_tool(cfg)
        configure_exa_tool(cfg)
        configure_brave_tool(cfg)
        configure_searxng_tool(cfg)
        configure_serpapi_tool(cfg)
        configure_google_search_tool(cfg)
        configure_python_exec_tool(cfg)
        configure_file_ops_tool(cfg)
        configure_file_read_dirs(cfg)
        configure_rag_tool(cfg)
        configure_email_tool(cfg)
        cap = _compute_tool_output_cap(ctx_tokens)
        for t in tool_list:
            apply_output_cap(t, cap)

    try:
        from cogtrix_core.tools.deep_think import set_progress_callback

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

    def _optimizer_progress(msg: str) -> None:
        """Spinner-aware progress callback for the prompt optimizer."""
        try:
            _spinner.pause()
        except Exception:  # noqa: BLE001
            pass
        print(msg)
        try:
            _spinner.resume()
        except Exception:  # noqa: BLE001
            pass

    set_optimizer_callback(_optimizer_progress)

    set_milestone_callback(_milestone_progress)

    # ── Remove tools whose required API keys are missing ─────────
    total_registered = len(registry.list_tools())
    filter_unconfigured_tools(registry)

    # ── Connect to MCP servers ───────────────────────────────────────────────
    _mcp_manager: MCPManager | None = None  # type: ignore[assignment]
    global _MCP_TOOLS_READY_EVENT
    _MCP_TOOLS_READY_EVENT = None
    if MCP_AVAILABLE and config.mcp_servers and tool_filter != "none":
        _mcp_manager = MCPManager()
        # ``KNOWN`` = fields forwarded to MCPServerConfig as kwargs.
        # ``DOC_ONLY`` = fields the user keeps in their YAML for human
        # reference (typically to link this config to a docker-compose
        # YAML or external server's own settings) but that Cogtrix does
        # not consume programmatically. Accepting them silently avoids
        # false-positive "ignoring unknown config keys" warnings while
        # still surfacing genuine typos.
        from cogtrix_core.mcp_client import DOC_ONLY_MCP_FIELDS, KNOWN_MCP_FIELDS

        _RECOGNISED_MCP_FIELDS = KNOWN_MCP_FIELDS | DOC_ONLY_MCP_FIELDS
        _mcp_configs = []
        for _mcp_name, _srv_cfg in config.mcp_servers.items():
            _unknown = set(_srv_cfg) - _RECOGNISED_MCP_FIELDS
            if _unknown:
                log.warning(
                    "MCP server '%s': ignoring unknown config keys: %s",
                    _mcp_name,
                    ", ".join(sorted(_unknown)),
                )
            _filtered = {k: v for k, v in _srv_cfg.items() if k in KNOWN_MCP_FIELDS}
            _mcp_configs.append(MCPServerConfig(name=_mcp_name, **_filtered))
        # Build a map of server_name -> pin so we can tag each tool's metadata.
        _mcp_pin_map = {cfg.name: cfg.pin for cfg in _mcp_configs}
        mcp_tools = _mcp_manager.connect_all(_mcp_configs, builtin_tool_names=set(registry.tools))
        for tool_name, tool_obj in mcp_tools.items():
            registry.tools[tool_name] = tool_obj
            _srv_name = (tool_obj.metadata or {}).get("server", "")
            registry.tool_metadata[tool_name] = {
                "requires_confirmation": (tool_obj.metadata or {}).get(
                    "requires_confirmation", True
                ),
                "source": "mcp",
                "server": _srv_name,
                "pin": _mcp_pin_map.get(_srv_name, True),
            }
        if mcp_tools:
            log.info(
                "Loaded %d MCP tool(s) from %d server(s)", len(mcp_tools), len(config.mcp_servers)
            )
        _MCP_TOOLS_READY_EVENT = _mcp_manager.tools_ready
        atexit.register(lambda: _mcp_manager.close_all() if _mcp_manager else None)
    elif not MCP_AVAILABLE and config.mcp_servers:
        log.warning(
            "mcp_servers configured but 'mcp' package not installed; run: uv pip install mcp"
        )

    # ── Apply tool presets ───────────────────────────────────────
    # Build the full catalog before splitting (for request_tools description).
    _session.all_tool_descriptions = build_tool_catalog(registry.tools)
    _session.all_tool_originals = dict(registry.tools)

    # Create and register checkpoint tool (must be done before wrapping with safety)
    from cogtrix_core.tools.checkpoint import CheckpointStore, create_checkpoint_tool

    _checkpoint_store = CheckpointStore()
    _checkpoint_tool = create_checkpoint_tool(_checkpoint_store)
    if _checkpoint_tool is not None:
        registry.tools["checkpoint"] = _checkpoint_tool
        _session.all_tool_descriptions["checkpoint"] = getattr(_checkpoint_tool, "description", "")
        _session.all_tool_originals["checkpoint"] = _checkpoint_tool
        registry.tool_metadata["checkpoint"] = {"requires_confirmation": False}
        _session.loaded_tools.add("checkpoint")
        _session.pinned_tools.add("checkpoint")
        _session.checkpoint_store = _checkpoint_store
        log.debug("Registered checkpoint tool in registry")

    # Split into active (full schemas in agent) and available (on-demand)
    available_tools: dict[str, Any] = {}
    if tool_filter is None and registry.list_tools():
        active_dict, available_tools = _apply_tool_preset(registry, config.memory_mode)
        if available_tools:
            # Apply preset: only active tools stay in registry
            registry.tools = active_dict

        # Auto-pin MCP tools whose server config has pin=True (the default).
        # apply_tool_preset() moves every tool into available_tools because all
        # mode presets are empty. The LLM never discovers on-demand tools that
        # are not in its training data, so MCP tools would be permanently invisible.
        # This pass promotes them back into the active set so the LLM sees them
        # in its bound function list from the very first turn.
        for _mcp_tool_name in list(available_tools):
            if registry.tool_metadata.get(_mcp_tool_name, {}).get("pin", False):
                _mcp_tool = available_tools.pop(_mcp_tool_name)
                registry.tools[_mcp_tool_name] = _mcp_tool
                _session.loaded_tools.add(_mcp_tool_name)
                _session.pinned_tools.add(_mcp_tool_name)
                log.debug(
                    "Auto-pinned MCP tool '%s' from server '%s'",
                    _mcp_tool_name,
                    registry.tool_metadata[_mcp_tool_name].get("server", "?"),
                )

        # Warn when too many MCP tools are pinned — each schema consumes ~300 tokens
        _pinned_mcp_count = sum(
            1
            for n in _session.pinned_tools
            if registry.tool_metadata.get(n, {}).get("source") == "mcp"
        )
        if _pinned_mcp_count > 50:
            log.warning(
                "MCP: %d tools are pinned into the active set (pin=True). "
                "This adds ~%d tokens of tool schema overhead per turn. "
                "Consider setting pin: false for large MCP servers.",
                _pinned_mcp_count,
                _pinned_mcp_count * 300,
            )

        # Auto-activate query_knowledge_base when a knowledge base exists
        if rag_should_auto_activate() and "query_knowledge_base" in available_tools:
            rag_tool = available_tools.pop("query_knowledge_base")
            _update_rag_tool_description(rag_tool)
            registry.tools["query_knowledge_base"] = rag_tool
            _session.loaded_tools.add("query_knowledge_base")
            _session.pinned_tools.add("query_knowledge_base")

    # Pin tools requested via --activate-tools
    _activate_tools_arg = getattr(args, "activate_tools", None)
    if _activate_tools_arg:
        for _aname in (n.strip() for n in _activate_tools_arg.split(",")):
            if not _aname:
                continue
            if _aname in available_tools:
                _atool = available_tools.pop(_aname)
                registry.tools[_aname] = _atool
                _session.loaded_tools.add(_aname)
                _session.pinned_tools.add(_aname)
                log.debug("Pinned tool via --activate-tools: %s", _aname)
            elif _aname in registry.tools:
                _session.loaded_tools.add(_aname)
                _session.pinned_tools.add(_aname)
            elif _aname in _session.all_tool_originals:
                # Tool exists but was excluded (e.g. --tools none); promote it
                _atool = _session.all_tool_originals[_aname]
                registry.tools[_aname] = _atool
                _session.loaded_tools.add(_aname)
                _session.pinned_tools.add(_aname)
                log.debug("Pinned tool via --activate-tools (from originals): %s", _aname)
            else:
                log.warning("--activate-tools: unknown tool '%s' (skipped)", _aname)

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

    # Store embedding config for lazy init; report has_embeddings=True when
    # the config is present (provider is expected to be available).
    _try_configure_embeddings(memory_manager, config)
    _has_embeddings = (
        getattr(memory_manager, "_lazy_emb_type", None) is not None
        or getattr(memory_manager, "_vector_store", None) is not None
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
        project_context_path=_project_context_path,
        has_embeddings=_has_embeddings,
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
            log.info("Auto-approved %d tool(s) (--no-confirm)", len(approvals))

    # Wrap tools with safety interceptors
    tools = []
    for tool_name, tool in registry.tools.items():
        if registry.requires_confirmation(tool_name):
            safe_tool = create_safe_tool_wrapper(
                tool,
                tool_name,
                registry,
                approvals,
                session_state=_session,
                git_native=_git_native,
                tool_trust=config.tool_trust,
            )
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

    # Get provider and model configuration
    try:
        provider_config, model_config = config.resolve_llm_config()
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
            active_tool_names={getattr(t, "name", "") for t in tools} | set(available_tools),
            decision_accountability_prompt=(
                ACCOUNTABILITY_PROMPT if config.decision_accountability_enabled else None
            ),
            pre_action_confirmation_prompt=(
                PRE_ACTION_CONFIRMATION_PROMPT if config.pre_action_confirmation_enabled else None
            ),
        )
        if _project_context:
            system_prompt = f"## Project Context (from COGTRIX.md)\n\n{_project_context}\n\n---\n\n{system_prompt}"
        log.debug("System prompt length: %d chars", len(system_prompt))
        log.debug("Mode additions: %s", mode_adds if mode_adds else "None")
        log.debug("=== System prompt ===\n%s\n=== End system prompt ===", system_prompt)

        _cli_system_prompt: str | None = None
        try:
            _cli_system_prompt = _load_cli_system_prompt(args)
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"\n⚠️  {e}")
            sys.exit(1)

        if _cli_system_prompt:
            system_prompt = _cli_system_prompt

        # Create LLM from provider and model configs.
        # Apply a default max_tokens cap for the main agent to prevent
        # runaway generations (e.g. 12K+ token phantom tool calls).
        # Deep think and delegate are uncapped — they set their own limits.
        _DEFAULT_MAX_TOKENS = 4096
        if model_config.max_tokens is None:
            from copy import copy as _copy

            model_config = _copy(model_config)
            model_config.max_tokens = _DEFAULT_MAX_TOKENS
        llm = create_chat_model_from_configs(provider_config, model_config)

        # Auto model routing: build a fast LLM for simple queries
        _fast_llm = None
        _fast_model_name: str | None = None
        _auto_route_enabled = getattr(args, "auto_route", False) or config.auto_route
        _quick_mode = config.quick_mode
        if _auto_route_enabled and config.auto_route_fast_model:
            try:
                _fast_pc, _fast_mc = config.resolve_llm_config_for(config.auto_route_fast_model)
                if _fast_mc.max_tokens is None:
                    from copy import copy as _copy_fast

                    _fast_mc = _copy_fast(_fast_mc)
                    _fast_mc.max_tokens = _DEFAULT_MAX_TOKENS
                _fast_llm = create_chat_model_from_configs(_fast_pc, _fast_mc)
                _fast_model_name = _fast_mc.model
                log.info("Auto-route fast model ready: %s", _fast_model_name)
            except Exception as _e:
                log.warning("Auto-route fast model failed to load: %s — routing disabled", _e)
                _auto_route_enabled = False

        # Token budget for context trimming (from model context_window or default)
        # Using the local constant since _DEFAULT_CONTEXT_WINDOW is not defined in agent/core.py
        DEFAULT_CONTEXT_WINDOW = 32_768

        max_context_tokens = model_config.context_window or DEFAULT_CONTEXT_WINDOW

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
        memory_manager._max_context_tokens = max_context_tokens  # type: ignore[attr-defined]

        # Register LLM for cleanup on exit
        _cleanup_resources.append(llm)
        _cron_llm_ref[:] = [llm]  # make current LLM available to cron jobs

        # Wire the background task runner now that llm / tools are ready
        if "_task_queue" in dir():
            try:
                _tq_llm = llm
                _tq_registry = registry
                _tq_available_tools = available_tools
                _tq_tools = tools
                _tq_max_context_tokens = max_context_tokens
                _tq_config = config
                _tq_system_prompt = system_prompt

                def _task_runner(record: Any) -> str:
                    from cogtrix_core.agent import registry as _ar
                    from cogtrix_core.agent.registry import filter_tools_for_agent
                    from cogtrix_core.orchestration.session_state import SessionState

                    agent_cfg = _ar.get(record.agent_name)
                    _sp = (
                        getattr(agent_cfg, "system_prompt", None) or _tq_system_prompt
                        if agent_cfg is not None
                        else _tq_system_prompt
                    )
                    try:
                        # Start with just request_tools — the agent loads
                        # what it needs on demand (same as a fresh session).
                        _tq_run_config = AgentRunConfig.from_app_config(config)
                        _tq_run_config.llm = _tq_llm
                        _tq_run_config.system_prompt = _sp

                        # Filter tools based on the agent's include/exclude config
                        _tq_avail = dict(_tq_available_tools) if _tq_available_tools else {}
                        if agent_cfg is not None and (
                            agent_cfg.tools_include or agent_cfg.tools_exclude
                        ):
                            _tq_avail, _tq_filtered_list = filter_tools_for_agent(
                                record.agent_name, _tq_avail
                            )
                            _tq_run_config.available_tools = _tq_avail if _tq_avail else None
                            _tq_run_config.active_tools_list = (
                                _tq_filtered_list if _tq_filtered_list else None
                            )
                            log.info(
                                "Task agent %r tool filtering: %d tools after filtering "
                                "(include: %s, exclude: %s)",
                                record.agent_name,
                                len(_tq_avail),
                                agent_cfg.tools_include,
                                agent_cfg.tools_exclude,
                            )
                        else:
                            _tq_run_config.available_tools = _tq_avail if _tq_avail else None
                            _tq_run_config.active_tools_list = (
                                list(_tq_tools) if _tq_tools else None
                            )
                        _tq_run_config.max_context_tokens = _tq_max_context_tokens
                        _tq_run_config.session_state = SessionState(no_confirm=True)
                        _tq_run_config.memory_manager = memory_manager
                        _tq_run_config.parallel_tool_execution = (
                            _tq_config.parallel_tool_execution if _tq_config else True
                        )
                        _tq_run_config.checkpoint_store = _session.checkpoint_store
                        _tq_run_config.tools_ready = _MCP_TOOLS_READY_EVENT
                        result = _run_agent_cli(
                            record.prompt,
                            [],
                            _tq_registry,
                            set(),
                            config=_tq_run_config,
                        )
                        return result or "[no output]"
                    except Exception as _exc:  # noqa: BLE001
                        return f"[error] {type(_exc).__name__}: {_exc}"

                _task_queue.set_runner(_task_runner)
                _task_queue.start()  # deferred from init — runner must be set first
            except Exception as _tr_exc:
                log.debug("Task runner wiring skipped: %s", _tr_exc)

        # Use actual model name (resolved from model config), not CLI alias
        actual_model = model_config.model
        mode_info = f", mode: {config.memory_mode}"
        prov_model = f"{provider_config.name}: {actual_model}"
        if console:
            console.print(f"[green]✓ Agent ready[/green] [dim]({prov_model}{mode_info})[/dim]")
        else:
            print(f"✓ Agent ready ({prov_model}{mode_info})")

        # Log session info with actual model
        log_session_info(
            session_id=config.session,
            message_count=_startup_msg_count,
            memory_mode=config.memory_mode,
            provider=provider_config.name,
            model=actual_model,
        )

    except ImportError as e:
        prov = (
            model_config.provider
            if model_config
            else (provider_config.name if provider_config else "unknown")
        )
        log_error(e, context=f"Provider '{prov}' not available", include_trace=True)
        print(f"\n⚠️  Provider '{prov}' not available: {e}")
        print("   Please install the required package.")
        sys.exit(1)
    except Exception as e:
        log_error(e, context="Failed to initialize agent", include_trace=True)
        friendly = _friendly_error(
            e,
            provider=provider_config.name if provider_config else "unknown",
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
        from cogtrix_core.assistant.service import AssistantService

        _asst_compression_llm = None
        if config.context_compression_model:
            _asst_compression_llm = create_compression_llm(config.context_compression_model, config)

        service = AssistantService(
            config=config,
            llm=llm,
            registry=registry,
            system_prompt=system_prompt,
            available_tools=available_tools or {},
            active_tools=tools,
            max_context_tokens=max_context_tokens,
            compression_llm=_asst_compression_llm,
            cli_system_prompt=_cli_system_prompt,
            agent_runner=run_agent,
        )
        service.run()
        sys.exit(0)

    # Handle non-interactive mode (--prompt, --prompt-file, --silent, or positional PROMPT)
    _silent_mode = getattr(args, "silent", False)
    prompt_text = None
    if hasattr(args, "prompt") and args.prompt:
        prompt_text = args.prompt
    elif hasattr(args, "prompt_file") and args.prompt_file:
        try:
            prompt_file = Path(args.prompt_file)
            if not prompt_file.exists():
                print(f"\n⚠️  Prompt file not found: {args.prompt_file}", file=sys.stderr)
                sys.exit(1)
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            if not prompt_text:
                print(f"\n⚠️  Prompt file is empty: {args.prompt_file}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"\n⚠️  Error reading prompt file: {e}", file=sys.stderr)
            sys.exit(1)
    elif getattr(args, "pipe", False):
        # --pipe: read prompt from stdin (--prompt takes precedence)
        if not (hasattr(args, "prompt") and args.prompt):
            if sys.stdin.isatty():
                print("cogtrix --pipe: enter prompt (Ctrl-D to submit):", file=sys.stderr)
            prompt_text = sys.stdin.read().strip()
        if not prompt_text:
            print(
                "Error: --pipe requires a prompt (pipe it via stdin or use --prompt).",
                file=sys.stderr,
            )
            sys.exit(1)
    elif _silent_mode:
        # In silent mode, accept the prompt from the positional arg or stdin
        inline = getattr(args, "inline_prompt", None)
        if inline:
            prompt_text = inline.strip()
        elif not sys.stdin.isatty():
            prompt_text = sys.stdin.read().strip()
        if not prompt_text:
            print(
                "Error: --silent requires a prompt "
                "(pass it as a positional argument, via --prompt, or pipe it via stdin).",
                file=sys.stderr,
            )
            sys.exit(1)
    elif getattr(args, "inline_prompt", None):
        # Positional prompt without --silent was already flagged as silent in parse_arguments
        prompt_text = args.inline_prompt.strip()

    if prompt_text:
        # Non-interactive mode: process single prompt and exit
        # Create observability callbacks if debug mode is enabled
        callbacks = []
        if config.debug:
            obs_handler = create_observability_handler(verbose=config.verbose)
            if obs_handler:
                callbacks.append(obs_handler)
                log.debug("LLM observability handler enabled")

        # --silent / --pipe: auto-deny tool confirmations unless -y was given
        _deny_all = (_silent_mode or getattr(args, "pipe", False)) and not getattr(
            args, "no_confirm", False
        )

        exit_code = run_single_prompt(
            prompt_text=prompt_text,
            memory_manager=memory_manager,
            registry=registry,
            approvals=approvals,
            output_file=getattr(args, "output", None),
            no_stream=_silent_mode or getattr(args, "no_stream", False),
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
            deny_all_tools=_deny_all,
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
    slash_cmds.project_context_path = _project_context_path
    slash_cmds.max_context_tokens = max_context_tokens

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

    # Load input history and set up tab completion for slash commands.
    # Only full command names — short aliases (/h, /t, etc.) are excluded
    # from autocompletion since tab-complete makes them redundant.
    load_input_history()
    _all_cmds = [f"/{name}" for name in slash_cmds._commands]
    set_slash_commands(_all_cmds)
    try:
        from cogtrix_core.ui.input_session import set_slash_commands as _pt_set_cmds

        _pt_set_cmds(_all_cmds)
    except Exception:
        pass
    setup_readline_completion()
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
    slash_cmds.compression_llm = compression_llm
    memory_manager._compression_llm = compression_llm  # type: ignore[attr-defined]

    # Adaptive memory tracking
    _user_manually_set_mode: bool = False
    _prompt_count: int = 0
    _recent_prompts: list[str] = []

    def _do_mode_switch(new_mode: str) -> bool:
        """Perform a live memory mode switch. Returns True on success."""
        nonlocal memory_manager, system_prompt, available_tools, tools
        _snap = session_orch.snapshot(
            memory_manager=memory_manager,
            system_prompt=system_prompt,
            registry_tools=registry.tools,
            available_tools=available_tools,
            tools=tools,
        )
        try:
            memory_manager.save()
            new_mode_config = config.memory_modes.get(new_mode)
            memory_manager = MemoryFactory.create(
                mode=new_mode,
                store=memory_store,
                session_id=config.session,
                config=new_mode_config,
            )
            memory_manager.load()
            config.memory_mode = new_mode
            config.memory_config = new_mode_config
            memory_manager.set_llm(llm)
            _try_configure_embeddings(memory_manager, config)
            mode_adds = memory_manager.get_system_prompt_additions()
            system_prompt = build_system_prompt(
                mode_additions=mode_adds,
                models=config.models,
                delegation_models=config.delegate_allowed_models,
                tool_instructions=provider_config.tool_instructions,
                active_tool_names={getattr(t, "name", "") for t in tools} | set(available_tools),
                decision_accountability_prompt=(
                    ACCOUNTABILITY_PROMPT if config.decision_accountability_enabled else None
                ),
                pre_action_confirmation_prompt=(
                    PRE_ACTION_CONFIRMATION_PROMPT
                    if config.pre_action_confirmation_enabled
                    else None
                ),
            )
            if _project_context:
                system_prompt = f"## Project Context (from COGTRIX.md)\n\n{_project_context}\n\n---\n\n{system_prompt}"
            if tool_filter is None:
                registry.tools = dict(_session.all_tool_originals)
                active_dict, available_tools = _apply_tool_preset(registry, new_mode)
                if available_tools:
                    registry.tools = active_dict
                if rag_should_auto_activate() and "query_knowledge_base" in available_tools:
                    _rag = available_tools.pop("query_knowledge_base")
                    _update_rag_tool_description(_rag)
                    registry.tools["query_knowledge_base"] = _rag
                    _session.loaded_tools.add("query_knowledge_base")
                    _session.pinned_tools.add("query_knowledge_base")
                for _pname in list(_session.pinned_tools):
                    if _pname in available_tools:
                        registry.tools[_pname] = available_tools.pop(_pname)
                _session.loaded_tools &= _session.pinned_tools
                tools.clear()
                for tn, tl in registry.tools.items():
                    if registry.requires_confirmation(tn):
                        tools.append(
                            create_safe_tool_wrapper(
                                tl,
                                tn,
                                registry,
                                approvals,
                                session_state=_session,
                                git_native=_git_native,
                                tool_trust=config.tool_trust,
                            )
                        )
                    else:
                        tools.append(tl)
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
            slash_cmds.memory_manager = memory_manager
            slash_cmds.system_prompt = system_prompt
            slash_cmds.available_tools = available_tools
            return True
        except Exception as exc:
            restored = session_orch.rollback(_snap, tools_list=tools)
            memory_manager = restored["memory_manager"]
            system_prompt = restored["system_prompt"]
            available_tools = restored["available_tools"]
            registry.tools = _snap.registry_tools
            log.error("Mode switch failed: %s", exc)
            return False

    # Create prompt_toolkit session for pinned input prompt
    _pt_session: object = None
    if sys.stdout.isatty() and not getattr(args, "pipe", False):
        try:
            _pt_session = _create_input_session()
        except Exception:
            _pt_session = None

    # Main input/output loop
    while True:
        try:
            if color_enabled():
                # \001/\002 are readline markers for non-printing chars
                # so readline calculates prompt width correctly.
                _mode = getattr(config, "memory_mode", "conversation")
                _mode_prefix = (
                    f"\001\033[35m\002{_mode}\001\033[0m\002 " if _mode != "conversation" else ""
                )
                _prompt = f"\n{_mode_prefix}\001\033[36m\002\u276f\001\033[0m\002 "
            else:
                _mode = getattr(config, "memory_mode", "conversation")
                _mode_prefix = f"{_mode} " if _mode != "conversation" else ""
                _prompt = f"\n{_mode_prefix}> "
            if _pt_session is not None:
                try:
                    _raw = _pt_session.prompt()
                    user_input = _raw.strip() if _raw is not None else ""
                    # Erase separator + stats + ❯ lines from scrollback immediately
                    # after Enter so the Rich "you" panel is the only echo visible.
                    if sys.stdout.isatty() and console is not None:
                        from cogtrix_core.ui.input_session import _toolbar_stats as _ts

                        sys.stdout.write("\033[1A\033[2K\r")  # erase ❯ input line
                        if _ts.strip():
                            sys.stdout.write("\033[1A\033[2K\r")  # erase stats line
                        sys.stdout.write("\033[1A\033[2K\r")  # erase separator line
                        sys.stdout.flush()
                except (EOFError, KeyboardInterrupt):
                    raise
            else:
                user_input = input(_prompt).strip()

            if not user_input:
                # Guard against a tight busy-loop when stdin is an orphaned
                # PTY slave (e.g. `docker compose run` with the terminal
                # subsequently disconnected).  In that state, input() returns
                # "" immediately instead of blocking, which would peg one CPU
                # core at ~100%.  A short sleep keeps the loop alive for cron
                # jobs while consuming negligible CPU.  See issue #100.
                import time as _time

                _time.sleep(0.05)
                continue

            already_optimized = False
            _pending_plan: PromptPlan | None = None

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
                elif cmd_word == "retry":
                    if not slash_cmds.last_input:
                        if console is not None:
                            console.print(
                                "[dim]Nothing to retry — no previous prompt in this session.[/dim]"
                            )
                        else:
                            print("Nothing to retry — no previous prompt in this session.")
                        continue
                    user_input = slash_cmds.last_input
                    # fall through to normal prompt processing (no continue)
                else:
                    # Regular slash commands (e.g. /help, /quit, /info)
                    # ── Wrap output in patch_stdout to suppress prompt redraws ──────────
                    result = slash_cmds.dispatch(user_input)
                    if result == "break":
                        break
                    if isinstance(result, str) and result.startswith("switch_mode:"):
                        _user_manually_set_mode = True
                        new_mode = result.split(":", 1)[1]
                        if _do_mode_switch(new_mode):
                            if console is not None:
                                console.print(
                                    f"[green]Switched to [bold]{new_mode}[/bold] mode.[/green]"
                                )
                            else:
                                print(f"Switched to {new_mode} mode.")
                            log.info("Live mode switch: %s", new_mode)
                        else:
                            if console is not None:
                                console.print("[red]Mode switch failed.[/red]")
                            else:
                                print("Mode switch failed.")

                    elif isinstance(result, str) and result.startswith("switch_model:"):
                        new_model = result.split(":", 1)[1]
                        _snap = session_orch.snapshot(
                            system_prompt=system_prompt,
                            available_tools=available_tools,
                        )
                        _prev_alias = config.active_model_alias
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
                            config.active_model_alias = alias if alias else new_model
                            _resolve_model(config)

                            # Get updated provider and model config
                            provider_config, model_config = config.resolve_llm_config()
                            actual_model = model_config.model

                            # Create new LLM
                            new_llm = create_chat_model_from_configs(provider_config, model_config)

                            mode_adds = memory_manager.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                                active_tool_names={getattr(t, "name", "") for t in tools}
                                | set(available_tools),
                                decision_accountability_prompt=(
                                    ACCOUNTABILITY_PROMPT
                                    if config.decision_accountability_enabled
                                    else None
                                ),
                                pre_action_confirmation_prompt=(
                                    PRE_ACTION_CONFIRMATION_PROMPT
                                    if config.pre_action_confirmation_enabled
                                    else None
                                ),
                            )
                            if _project_context:
                                system_prompt = f"## Project Context (from COGTRIX.md)\n\n{_project_context}\n\n---\n\n{system_prompt}"
                            slash_cmds.system_prompt = system_prompt

                            # All potential failures are past — now atomically swap
                            old_llm = llm
                            llm = new_llm
                            max_context_tokens = (
                                model_config.context_window or DEFAULT_CONTEXT_WINDOW
                            )
                            slash_cmds.max_context_tokens = max_context_tokens
                            _cleanup_resources.append(llm)
                            _cron_llm_ref[:] = [llm]

                            # Update hybrid memory LLM reference
                            memory_manager.set_llm(llm)

                            # Reconfigure all tools for new provider/model
                            _reconfigure_all_tools(config, max_context_tokens, tools)

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
                            slash_cmds.compression_llm = compression_llm
                            memory_manager._compression_llm = compression_llm  # type: ignore[attr-defined]
                            memory_manager._max_context_tokens = max_context_tokens  # type: ignore[attr-defined]

                            # Close old LLM last — it's no longer referenced
                            _close_llm(old_llm)
                            invalidate_llm_caches()
                            if old_llm in _cleanup_resources:
                                _cleanup_resources.remove(old_llm)

                            if console is not None:
                                console.print(
                                    f"[green]Switched to model "
                                    f"[bold]{actual_model}[/bold] "
                                    f"[dim]({model_config.provider})[/dim][/green]"
                                )
                            else:
                                print(f"Switched to model {actual_model} ({model_config.provider})")
                            log.info(
                                f"Live model switch: {actual_model} "
                                f"(provider: {model_config.provider})"
                            )
                        except Exception as exc:
                            config.active_model_alias = _prev_alias
                            restored = session_orch.rollback(_snap)
                            system_prompt = restored["system_prompt"]
                            available_tools = restored["available_tools"]
                            log.error("Model switch failed: %s", exc)
                            try:
                                provider_config, _ = config.resolve_llm_config()
                                friendly = _friendly_error(exc, provider=provider_config.name)
                            except Exception:
                                friendly = str(exc)
                            if console is not None:
                                console.print(f"[red]Model switch failed:[/red] {friendly}")
                            else:
                                print(f"Model switch failed: {friendly}")

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
                                active_tool_names={getattr(t, "name", "") for t in tools}
                                | set(available_tools),
                                decision_accountability_prompt=(
                                    ACCOUNTABILITY_PROMPT
                                    if config.decision_accountability_enabled
                                    else None
                                ),
                                pre_action_confirmation_prompt=(
                                    PRE_ACTION_CONFIRMATION_PROMPT
                                    if config.pre_action_confirmation_enabled
                                    else None
                                ),
                            )
                            if _project_context:
                                system_prompt = f"## Project Context (from COGTRIX.md)\n\n{_project_context}\n\n---\n\n{system_prompt}"
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
                                print(f"Switched to session {new_session} ({msg_count} messages)")
                            log.info(f"Live session switch: {new_session} ({msg_count} messages)")
                        except Exception as exc:
                            restored = session_orch.rollback(_snap)
                            memory_manager = restored["memory_manager"]
                            system_prompt = restored["system_prompt"]
                            log.error("Session switch failed: %s", exc)
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
                                    tool_obj,
                                    load_name,
                                    registry,
                                    approvals,
                                    session_state=_session,
                                    git_native=_git_native,
                                    tool_trust=config.tool_trust,
                                )
                            tools.append(tool_obj)
                            registry.tools[load_name] = _session.all_tool_originals.get(
                                load_name, tool_obj
                            )
                            _session.loaded_tools.add(load_name)
                            _session.pinned_tools.add(load_name)
                            if console is not None:
                                console.print(
                                    f"[green]Tool [bold]{load_name}[/bold] loaded (pinned).[/green]"
                                )
                            else:
                                print(f"Tool '{load_name}' loaded (pinned).")
                        else:
                            if console is not None:
                                console.print(
                                    f"[yellow]Tool '{load_name}' is not available to load.[/yellow]"
                                )
                            else:
                                print(f"Tool '{load_name}' is not available to load.")

                    elif isinstance(result, str) and result.startswith("unload_tool:"):
                        unload_name = result.split(":", 1)[1]
                        if unload_name in _session.pinned_tools:
                            _session.pinned_tools.discard(unload_name)
                            _session.loaded_tools.discard(unload_name)
                            # Return tool to on-demand pool
                            _orig = _session.all_tool_originals.get(unload_name)
                            if _orig is not None:
                                available_tools[unload_name] = _orig
                            registry.tools.pop(unload_name, None)
                            tools[:] = [t for t in tools if getattr(t, "name", None) != unload_name]
                            if console is not None:
                                console.print(
                                    f"[green]Tool [bold]{unload_name}[/bold] unloaded.[/green]"
                                )
                            else:
                                print(f"Tool '{unload_name}' unloaded.")
                        elif unload_name in _session.loaded_tools:
                            if console is not None:
                                console.print(
                                    f"[yellow]Tool '{unload_name}' was loaded by the "
                                    f"agent and will be auto-unloaded next turn.[/yellow]"
                                )
                            else:
                                print(
                                    f"Tool '{unload_name}' was loaded by the agent "
                                    "and will be auto-unloaded next turn."
                                )
                        else:
                            if console is not None:
                                console.print(
                                    f"[yellow]Tool '{unload_name}' is not "
                                    f"currently loaded.[/yellow]"
                                )
                            else:
                                print(f"Tool '{unload_name}' is not currently loaded.")

                    elif isinstance(result, str) and result.startswith("deep_think:"):
                        # ── Hybrid /think: gather → analyze → synthesize ──
                        think_task = result.split(":", 1)[1]
                        # Echo the user's /think command so it stays visible
                        # during the multi-minute Stage 1/2 run.  Normal prompts
                        # reach print_user_turn via the main handler, but slash
                        # commands hit `continue` before that path — so we must
                        # echo explicitly here.
                        if console is not None:
                            from cogtrix_core.ui.turns import print_user_turn

                            print_user_turn(console, user_input)
                        try:
                            from datetime import date as _date

                            _today = _date.today().strftime("%B %d, %Y")

                            # Classify the task to pick specialised prompts.
                            # Start the spinner immediately so the user sees
                            # activity — the LLM call inside classify_think_task
                            # can take several seconds with no other feedback.
                            _spinner.start()
                            _spinner.set_context("Classifying")
                            try:
                                think_cat = classify_think_task(think_task, llm=llm)
                            finally:
                                _spinner.stop()

                            # Stage 1: Gather data via the agent
                            if console is not None:
                                console.print(
                                    f"[dim]Stage 1/2:[/dim] Gathering data "
                                    f"[dim](strategy: {think_cat.name})[/dim]…"
                                )
                            else:
                                print(f"Stage 1/2: Gathering data (strategy: {think_cat.name})…")

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
                                _gather_run_config = AgentRunConfig.from_app_config(config)
                                _gather_run_config.llm = llm
                                _gather_run_config.system_prompt = system_prompt
                                _gather_run_config.available_tools = (
                                    dict(available_tools)
                                    if (available_tools or TOOL_PRESETS.get(config.memory_mode))
                                    else None
                                )
                                _gather_run_config.active_tools_list = tools
                                _gather_run_config.max_context_tokens = max_context_tokens
                                _gather_run_config.preset_tools = TOOL_PRESETS.get(
                                    config.memory_mode, set()
                                )
                                _gather_run_config.compression_llm = _think_compression_llm
                                _gather_run_config.session_state = _session
                                _gather_run_config.memory_manager = memory_manager
                                _gather_run_config.confirmation_ui = _rich_ui
                                _gather_run_config.on_tool_expansion = _tool_expansion_ui
                                _gather_run_config.tools_ready = _MCP_TOOLS_READY_EVENT
                                _gather_run_config.checkpoint_store = _session.checkpoint_store
                                gather_output = _run_agent_cli(
                                    gather_prompt,
                                    gather_context.messages,
                                    registry,
                                    approvals,
                                    context_prefix=gather_context.context_prefix,
                                    callbacks=callbacks if callbacks else None,
                                    result_messages=gather_msgs,
                                    config=_gather_run_config,
                                )
                            finally:
                                _spinner.stop()

                            # Stage 2: Deep analysis with gathered data
                            if console is not None:
                                console.print("[dim]Stage 2/2:[/dim] Deep analysis…")
                            else:
                                print("Stage 2/2: Deep analysis…")

                            from cogtrix_core.tools.deep_think import deep_think

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
                            try:
                                _exc_prov = config.get_active_model().provider
                            except Exception:
                                _exc_prov = provider_config.name if provider_config else "unknown"
                            friendly = _friendly_error(exc, provider=_exc_prov)
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
                            try:
                                _exc_prov = config.get_active_model().provider
                            except Exception:
                                _exc_prov = provider_config.name if provider_config else "unknown"
                            friendly = _friendly_error(exc, provider=_exc_prov)
                            if console is not None:
                                console.print(f"[red]Delegation failed:[/red] {friendly}")
                            else:
                                print(f"Delegation failed: {friendly}")
                            if config.debug:
                                import traceback

                                traceback.print_exc()

                    elif result == "run_setup":
                        from cogtrix_core.setup_wizard import run_setup_wizard

                        try:
                            run_setup_wizard()
                        except SystemExit:
                            continue

                        # Reload config and rebuild LLM connection
                        try:
                            config = load_config(args)
                            provider_config, model_config = config.resolve_llm_config()
                            new_llm = create_chat_model_from_configs(provider_config, model_config)
                            _close_llm(llm)
                            invalidate_llm_caches()
                            if llm in _cleanup_resources:
                                _cleanup_resources.remove(llm)
                            llm = new_llm
                            max_context_tokens = (
                                model_config.context_window or DEFAULT_CONTEXT_WINDOW
                            )
                            _cleanup_resources.append(llm)
                            _cron_llm_ref[:] = [llm]

                            memory_manager.set_llm(llm)
                            _reconfigure_all_tools(config, max_context_tokens, tools)

                            mode_adds = memory_manager.get_system_prompt_additions()
                            system_prompt = build_system_prompt(
                                mode_additions=mode_adds,
                                models=config.models,
                                delegation_models=config.delegate_allowed_models,
                                tool_instructions=provider_config.tool_instructions,
                                active_tool_names={getattr(t, "name", "") for t in tools}
                                | set(available_tools),
                                decision_accountability_prompt=(
                                    ACCOUNTABILITY_PROMPT
                                    if config.decision_accountability_enabled
                                    else None
                                ),
                                pre_action_confirmation_prompt=(
                                    PRE_ACTION_CONFIRMATION_PROMPT
                                    if config.pre_action_confirmation_enabled
                                    else None
                                ),
                            )
                            if _project_context:
                                system_prompt = f"## Project Context (from COGTRIX.md)\n\n{_project_context}\n\n---\n\n{system_prompt}"
                            slash_cmds.system_prompt = system_prompt

                            if config.context_compression:
                                try:
                                    compression_llm = create_compression_llm(
                                        config.context_compression_model, config
                                    )
                                except Exception:
                                    compression_llm = None
                            else:
                                compression_llm = None
                            memory_manager._compression_llm = compression_llm  # type: ignore[attr-defined]
                            memory_manager._max_context_tokens = max_context_tokens  # type: ignore[attr-defined]

                            slash_cmds.config = config
                            actual_model = model_config.model
                            _reload_prov = provider_config.name
                            if console is not None:
                                console.print(
                                    f"\n[green]Config reloaded — using "
                                    f"[bold]{_reload_prov}[/bold] "
                                    f"[dim](model: {actual_model})[/dim][/green]"
                                )
                            else:
                                print(
                                    f"\nConfig reloaded — using {_reload_prov} "
                                    f"(model: {actual_model})"
                                )
                        except Exception as exc:
                            try:
                                _exc_prov = config.get_active_model().provider
                            except Exception:
                                _exc_prov = provider_config.name if provider_config else "unknown"
                            friendly = _friendly_error(exc, provider=_exc_prov)
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
                        _pending_plan = optimize_prompt(
                            user_input, llm, force=True, plan_milestones=True
                        )
                        user_input = _pending_plan.text
                        already_optimized = True
                    elif isinstance(result, str) and result.startswith("retry:"):
                        # Re-run the last prompt, then fall through to agent
                        user_input = result.split(":", 1)[1]
                    else:
                        continue

                    # All slash-command handlers return to the prompt except
                    # "optimize:" and "retry:" which fall through to agent execution.
                    if not (
                        isinstance(result, str)
                        and (result.startswith("optimize:") or result.startswith("retry:"))
                    ):
                        continue

            # Backward compat: bare exit/quit/q still works
            if user_input.lower() in ["exit", "quit", "q"]:
                log.info("Session ended by user")
                print("\nGoodbye!")
                break

            # Expand @file / @folder references before sending to the agent
            _expanded_input, _at_injected = _expand_at_references(user_input)
            if _at_injected:
                if console is not None:
                    for _p in _at_injected:
                        console.print(f"  [dim]↳ Injected: {_p}[/dim]")
                else:
                    for _p in _at_injected:
                        print(f"  ↳ Injected: {_p}")
                user_input = _expanded_input

            # Start new request tracking
            new_request_id()

            # Reset per-prompt state and unload agent-loaded (non-pinned)
            # tools so the LLM starts each turn with a clean tool set.
            _prev_loaded = set(_session.loaded_tools)
            _session.reset_for_new_prompt()
            _unloaded = _prev_loaded - _session.loaded_tools
            if _unloaded:
                for _uname in _unloaded:
                    # Return tool to on-demand pool
                    _orig = _session.all_tool_originals.get(_uname)
                    if _orig is not None:
                        available_tools[_uname] = _orig
                    registry.tools.pop(_uname, None)
                # Remove from active tools list
                _unloaded_names = _unloaded
                tools[:] = [t for t in tools if getattr(t, "name", None) not in _unloaded_names]
                log.debug("Auto-unloaded %d agent-loaded tools: %s", len(_unloaded), _unloaded)

            # Adaptive memory mode auto-selection
            if config.adaptive_memory and not _user_manually_set_mode:
                _recent_prompts.append(user_input)
                if len(_recent_prompts) > 3:
                    _recent_prompts.pop(0)
                _prompt_count += 1
                _auto_mode: str | None = None
                if _prompt_count == 1 and config.memory_mode == "conversation":
                    # Session-start: classify the first prompt
                    _classified = classify_memory_mode(user_input)
                    if _classified != "conversation":
                        _auto_mode = _classified
                elif _prompt_count % 5 == 0:
                    # Mid-session: check last 3 prompts every 5 turns
                    _auto_mode = should_switch_mode(config.memory_mode, _recent_prompts)
                if _auto_mode is not None:
                    if _do_mode_switch(_auto_mode):
                        if console is not None:
                            console.print(
                                f"[dim]Memory mode auto-switched to "
                                f"[bold]{_auto_mode}[/bold].[/dim]"
                            )
                        else:
                            print(f"Memory mode auto-switched to {_auto_mode}.")
                        log.info("Adaptive mode switch: %s", _auto_mode)

            # Log user message
            log_user_message(user_input)

            # Preserve the user's original phrasing before the optimizer
            # rewrites it — memory, classification, delegation, and the
            # output file must all record what the user actually typed.
            # Set before try so KeyboardInterrupt handlers can use it.
            original_input = user_input
            slash_cmds.last_input = original_input

            if console is not None:
                from cogtrix_core.ui.turns import print_user_turn

                print_user_turn(console, original_input)

            _progress_tool_interactive: object = None
            _run_sys_prompt_interactive = system_prompt
            _agent_t0 = _time_mod.monotonic()
            _spinner.start()
            if _quick_mode:
                if console is not None:
                    console.print("  [dim]⚡ quick: optimizer · memory · compression skipped[/dim]")
                else:
                    print("  [quick] optimizer · memory · compression skipped")
            try:
                _t0_prep = _time_mod.monotonic()
                # Run prepare_context and optimize_prompt concurrently when optimizer is enabled.
                # The two operations are independent: the optimizer rewrites the prompt text
                # while prepare_context reads conversation history — no data dependency.
                if _quick_mode:
                    context = MemoryContext(mode="quick")
                elif config.prompt_optimizer and not already_optimized:
                    # Use explicit ThreadPoolExecutor (not `with`) so shutdown(wait=False)
                    # can be used on timeout — `__exit__` calls shutdown(wait=True) which
                    # blocks on hung threads.
                    _pool = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep")
                    try:
                        _ctx_future = _pool.submit(memory_manager.prepare_context, user_input)
                        _opt_future = _pool.submit(
                            optimize_prompt, user_input, llm, plan_milestones=True
                        )
                        try:
                            context = _ctx_future.result(timeout=60)
                            plan = _opt_future.result(timeout=60)
                        except _cf.TimeoutError:
                            _ctx_future.cancel()
                            _opt_future.cancel()
                            _pool.shutdown(wait=False)
                            log.warning(
                                "Prompt prep timed out after 60s — falling back to sequential path"
                            )
                            context = memory_manager.prepare_context(user_input)
                        else:
                            user_input = plan.text
                            already_optimized = True
                            _progress_tool_interactive, _run_sys_prompt_interactive = (
                                _inject_milestones(plan, tools, _active_milestones, system_prompt)
                            )
                    finally:
                        _pool.shutdown(wait=False)
                elif _pending_plan is not None:
                    context = memory_manager.prepare_context(user_input)
                    _progress_tool_interactive, _run_sys_prompt_interactive = _inject_milestones(
                        _pending_plan, tools, _active_milestones, system_prompt
                    )
                    _pending_plan = None
                else:
                    context = memory_manager.prepare_context(user_input)

                log.debug(
                    "⏱ prepare_context: %.0fms",
                    (_time_mod.monotonic() - _t0_prep) * 1000,
                )

                # Debug: log context details
                log.debug(
                    f"Context: mode={context.mode}, "
                    f"{context.context_messages_count} messages"
                    + (f", ~{context.token_estimate} tokens" if context.token_estimate else "")
                )

                wants_deep = user_wants_deep_think(original_input)
                _task_complexity_i = classify_task_complexity(original_input)

                # Auto-promote COMPLEX_RESEARCH to delegation when user has not
                # already requested deep thinking or explicit delegation.
                if (
                    not wants_deep
                    and not user_wants_delegation(original_input)
                    and _task_complexity_i == TaskComplexity.COMPLEX_RESEARCH
                ):
                    log.info("Complex research task detected — auto-promoting to delegation")
                    # Delegate pipeline is triggered via the post-run check that
                    # calls force_delegation when user_wants_delegation returns True.
                    # Override original_input delegation signal by mutating the flag
                    # directly in the post-run condition via a closure variable.
                    _auto_delegation_i = True
                else:
                    _auto_delegation_i = False

                # Auto model routing: use fast LLM for simple queries
                _routed_llm = llm
                if _auto_route_enabled and _fast_llm is not None and not wants_deep:
                    _complexity = _classify_query_complexity(original_input)
                    if _complexity == "simple":
                        _routed_llm = _fast_llm
                        if console is not None:
                            console.print(f"  [dim]⚡ auto: {_fast_model_name}[/dim]")
                        else:
                            print(f"  [auto] fast: {_fast_model_name}")
                        log.debug("Auto-route: simple query → %s", _fast_model_name)
                    else:
                        log.debug("Auto-route: complex query → primary model")

                agent_msgs: list = []

                # ── Research delegate pre-flight ─────────────────────────────
                # When research_delegate_auto is enabled and session context is
                # above the threshold, route research queries to a subagent to
                # prevent raw web content from filling the main context.
                _preflight_output_i: str | None = None
                if getattr(config, "research_delegate_auto", False):
                    if max_context_tokens and max_context_tokens > 0:
                        _i_rd_threshold = getattr(config, "research_delegate_auto_threshold", 0.50)
                        _i_ctx_est = getattr(context, "token_estimate", 0) or 0
                        _i_session_ratio = _i_ctx_est / max_context_tokens
                        if _i_session_ratio >= _i_rd_threshold and _looks_like_research_query(
                            original_input
                        ):
                            from cogtrix_core.tools.delegate import (
                                delegate_task,
                                get_delegate_tools,
                            )

                            if get_delegate_tools():
                                log.info(
                                    "research_delegate_auto: pre-flighting research query "
                                    "(session context %.0f%%, threshold %.0f%%)",
                                    _i_session_ratio * 100,
                                    _i_rd_threshold * 100,
                                )
                                _spinner.start()
                                try:
                                    _preflight_output_i = delegate_task(
                                        task=original_input,
                                        use_tools=True,
                                        timeout=getattr(config, "research_delegate_timeout", 300),
                                    )
                                except Exception as _e:
                                    log.warning(
                                        "research_delegate_auto pre-flight failed: "
                                        "%s — falling back",
                                        _e,
                                    )
                                    _preflight_output_i = None
                                finally:
                                    _spinner.stop()

                _acc = _TokenAccumulator()
                _agent_cbs = (callbacks or []) + [_acc]
                _t0_agent = _time_mod.monotonic()
                if _preflight_output_i:
                    output = _preflight_output_i
                    _spinner.stop()
                else:
                    _repl_run_config = AgentRunConfig.from_app_config(config)
                    _repl_run_config.llm = _routed_llm
                    _repl_run_config.system_prompt = _run_sys_prompt_interactive
                    _repl_run_config.available_tools = (
                        dict(available_tools)
                        if (available_tools or TOOL_PRESETS.get(config.memory_mode))
                        else None
                    )
                    _repl_run_config.active_tools_list = tools
                    _repl_run_config.max_context_tokens = max_context_tokens
                    _repl_run_config.preset_tools = TOOL_PRESETS.get(config.memory_mode, set())
                    _repl_run_config.context_compression = (
                        False if _quick_mode else config.context_compression
                    )
                    _repl_run_config.compression_llm = compression_llm
                    _repl_run_config.session_state = _session
                    _repl_run_config.memory_manager = memory_manager
                    _repl_run_config.confirmation_ui = _rich_ui
                    _repl_run_config.on_tool_expansion = _tool_expansion_ui
                    _repl_run_config.tools_ready = _MCP_TOOLS_READY_EVENT
                    _repl_run_config.checkpoint_store = _session.checkpoint_store
                    output = _run_agent_cli(
                        user_input,
                        context.messages,
                        registry,
                        approvals,
                        context_prefix=context.context_prefix,
                        callbacks=_agent_cbs,
                        result_messages=agent_msgs,
                        config=_repl_run_config,
                        task_complexity=_task_complexity_i,
                    )
                log.debug(
                    "⏱ run_agent total: %.0fms",
                    (_time_mod.monotonic() - _t0_agent) * 1000,
                )
                _spinner.stop()
                _maybe_write_session_metrics(getattr(config, "log_file", None))

                # ── Enforce deep_think when the user requested it ──
                # Force-call if agent skipped it OR called with bad context.
                # Skip for tool-intensive tasks where the agent's tool
                # work is the primary deliverable.
                _research_output: str = ""
                wants_deep = _maybe_skip_force_deep_think_for_tool_intensive_task(
                    wants_deep, original_input, llm, log
                )
                if wants_deep and output:
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
                # Skip if the model already produced a substantial response.
                _resp_substantial_i = len(output or "") > 500
                if (
                    not wants_deep
                    and output
                    and not _resp_substantial_i
                    and (user_wants_delegation(original_input) or _auto_delegation_i)
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
                        _exec_run_config = AgentRunConfig.from_app_config(config)
                        _exec_run_config.llm = llm
                        _exec_run_config.system_prompt = system_prompt
                        _exec_run_config.available_tools = (
                            dict(available_tools) if available_tools else available_tools
                        )
                        _exec_run_config.active_tools_list = tools
                        _exec_run_config.max_context_tokens = max_context_tokens
                        _exec_run_config.preset_tools = TOOL_PRESETS.get(config.memory_mode, set())
                        _exec_run_config.session_state = _session
                        _exec_run_config.on_tool_expansion = _tool_expansion_ui
                        exec_output, exec_msgs = run_execution_phase(
                            output,
                            user_input,
                            context.messages,
                            registry,
                            approvals,
                            context_prefix=context.context_prefix,
                            callbacks=_agent_cbs,
                            config=_exec_run_config,
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
                    if _pt_session is not None:
                        from prompt_toolkit.patch_stdout import patch_stdout as _patch_stdout

                        _patch_ctx = _patch_stdout(raw=True)
                        _patch_ctx.__enter__()
                    else:
                        _patch_ctx = None
                    try:
                        from cogtrix_core.ui.turns import print_assistant_turn_header

                        print_assistant_turn_header(console)
                        console.print(
                            Padding(
                                Markdown(preserve_tables_for_markdown(output)),
                                (0, 0, 0, 2),
                            )
                        )
                        _session_tokens.input_tokens += _acc.input_tokens
                        _session_tokens.output_tokens += _acc.output_tokens
                        slash_cmds.last_input_tokens = _acc.last_input_tokens or _acc.input_tokens
                        # print_stats_footer stores stats for the toolbar display
                        from cogtrix_core.ui.stats import print_stats_footer as _print_stats_footer

                        _print_stats_footer(
                            console=console,
                            session_tokens=_acc.last_input_tokens or _acc.input_tokens,
                            max_context_tokens=max_context_tokens,
                            input_tokens=_acc.input_tokens,
                            output_tokens=_acc.output_tokens,
                        )
                    finally:
                        if _patch_ctx is not None:
                            _patch_ctx.__exit__(None, None, None)
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
                        log.error("Error appending to output file: %s", e)

                # Only save valid responses to history (skip empty/error).
                # Pass the full agent chain so the agent can continue
                # iterating on complex tasks across restarts (Ralph Loop).
                if _is_valid_response(output) and not _quick_mode:
                    memory_manager.update(original_input, output, agent_messages=turn_msgs or None)
                    memory_manager.save()
                elif not _is_valid_response(output):
                    log.warning("Skipping history save: empty or error response")

            except UserCancelledRun:
                _spinner.stop()
                _cleanup_milestones(_progress_tool_interactive, tools, _active_milestones)
                if console:
                    console.print("[yellow]Workflow cancelled.[/yellow]")
                else:
                    print("Workflow cancelled.")
                prefill_next_input(original_input)
                continue

            except KeyboardInterrupt:
                _spinner.stop()
                _cleanup_milestones(_progress_tool_interactive, tools, _active_milestones)
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
                _spinner.stop()
                _cleanup_milestones(_progress_tool_interactive, tools, _active_milestones)
                log_error(e, context="Agent execution error", include_trace=True)
                try:
                    _exc_prov = config.get_active_model().provider
                except Exception:
                    _exc_prov = provider_config.name if provider_config else "unknown"
                friendly = _friendly_error(e, provider=_exc_prov)
                if console:
                    console.print(f"[red]Error:[/red] {friendly}")
                else:
                    print(f"Error: {friendly}")
                if config.debug:
                    import traceback

                    traceback.print_exc()
                continue

            else:
                _cleanup_milestones(_progress_tool_interactive, tools, _active_milestones)

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

    # Close MCP connections BEFORE shutting down the thread-pool executor.
    # The atexit-registered close_all() fires too late (after Python has
    # already torn down the executor), which causes the MCP SSE post_writer
    # to raise "RuntimeError: cannot schedule new futures after shutdown"
    # and print a full traceback to the terminal.  Calling it here, while
    # the executor is still live, gives MCP a clean shutdown window.
    if _mcp_manager is not None:
        try:
            _mcp_manager.close_all()
        except Exception:  # noqa: BLE001
            pass  # best-effort; process is exiting

    # Cancel any in-flight background summarization and persist memory,
    # then force-exit.  Python's internal _python_exit() joins ALL
    # ThreadPoolExecutor threads, including those running background
    # summarization LLM calls — this blocks exit for 5-10+ seconds.
    # By running cleanup explicitly and calling os._exit(), we avoid
    # that wait entirely.
    try:
        memory_manager.shutdown()
    except Exception:  # noqa: BLE001
        pass  # best-effort; process is exiting
    _cleanup()
    save_input_history()
    if _escape_monitor is not None and _escape_monitor.available:
        try:
            _escape_monitor._restore_terminal()
        except Exception:  # noqa: BLE001
            pass
    import os as _os_exit

    _os_exit._exit(0)


if __name__ == "__main__":
    main()
