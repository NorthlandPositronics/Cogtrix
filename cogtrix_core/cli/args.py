import argparse
import os
import sys

from cogtrix_core._version import get_version_string


def color_enabled() -> bool:
    """Check if ANSI color output is supported."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def bold(text: str) -> str:
    """Bold text (ANSI) when color is enabled."""
    return f"\033[1m{text}\033[0m" if color_enabled() else text


def dim(text: str) -> str:
    """Dim text (ANSI) when color is enabled."""
    return f"\033[2m{text}\033[0m" if color_enabled() else text


class ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Argparse formatter with bold section headers and wider help columns."""

    def __init__(self, prog, **kwargs):
        kwargs.setdefault("max_help_position", 34)
        super().__init__(prog, **kwargs)

    def start_section(self, heading):
        if heading and color_enabled():
            heading = f"\033[1m{heading}\033[0m"
        super().start_section(heading)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (without calling parse_args)."""
    B = bold
    D = dim

    desc = (
        "Cogtrix \u2014 AI agent with extensible tools, memory, "
        "and multi-provider LLM support.\n\n"
        f"  {B('New to Cogtrix?')} Run:  cogtrix.py --setup"
    )

    epilog = (
        f"{B('examples:')}\n"
        f"\n"
        f"  {B('Getting started:')}\n"
        f"    cogtrix.py --setup                         {D('Generate config with setup wizard')}\n"
        f"    cogtrix.py --check-config                  {D('Validate config and exit')}\n"
        f"\n"
        f"  {B('Interactive (default):')}\n"
        f"    cogtrix.py                                 {D('Start interactive session')}\n"
        f"    cogtrix.py -m qwen3:8b                     {D('Use a specific model')}\n"
        f"    cogtrix.py -m reasoning -M reasoning       {D('Model alias + memory mode')}\n"
        f"    cogtrix.py -s project-alpha                {D('Named session (preserves history)')}\n"
        f"\n"
        f"  {B('Config file:')}\n"
        f"    cogtrix.py -c /etc/cogtrix/prod.yaml       {D('Use explicit config file')}\n"
        f"    cogtrix.py -c config.yml -m my-model        {D('Config file + model override')}\n"
        f"\n"
        f"  {B('Non-interactive (scripting):')}\n"
        f'    cogtrix.py --prompt "What is 2+2?"         {D("Single prompt, print result, exit")}\n'
        f'    cogtrix.py --prompt "..." -o out.md        {D("Save response to file")}\n'
        f"    cogtrix.py --prompt-file task.txt -o res.md {D('Read prompt from file, save result')}\n"
        f'    cogtrix.py --prompt "..." --no-stream      {D("Suppress streaming (clean stdout)")}\n'
        f"\n"
        f"  {B('Assistant mode:')}\n"
        f"    cogtrix.py --assistant                     {D('Start headless messaging daemon')}\n"
        f"    cogtrix.py --assistant --log --debug       {D('Assistant with debug logging')}\n"
        f"\n"
        f"  {B('Tool activation:')}\n"
        f"    cogtrix.py --activate-tools web_search     {D('Pin a tool as active at startup')}\n"
        f"    cogtrix.py --activate-tools shell,write_file {D('Pin multiple tools')}\n"
        f"\n"
        f"  {B('Safety and output:')}\n"
        f"    cogtrix.py -y                              {D('Skip all tool confirmations')}\n"
        f"    cogtrix.py -o session.md                   {D('Append every response to file')}\n"
        f"    cogtrix.py -y -o log.md                    {D('No confirmations + transcript')}\n"
        f"\n"
        f"  {B('Logging:')}\n"
        f"    cogtrix.py --log                           {D('Log to cogtrix.log')}\n"
        f"    cogtrix.py --log myrun.log -v              {D('Verbose log to custom file')}\n"
        f"    cogtrix.py --debug                         {D('Full debug — alias for --verbosity 1')}\n"
        f"    cogtrix.py --verbosity 2                   {D('Set verbosity: 0=normal 1=debug 2=verbose 3=trace')}\n"
        f"\n"
        f"  {B('RAG:')}\n"
        f"    cogtrix.py --ingest --docs-dir ./docs      {D('Build RAG vector database')}\n"
        f"\n"
        f"{B('quick reference:')}\n"
        f"\n"
        f"  Memory modes      conversation {D('(default)')}, code, reasoning\n"
        f"  Slash commands    /help /tools /info /mode /model /provider /session /quit\n"
        f'  Paste mode        /paste or """\n'
        f"\n"
        f"{B('config file search order')} {D('(first found wins):')}\n"
        f"\n"
        f"  --config-file <path>                         {D('Explicit (skips search)')}\n"
        f"  ./.cogtrix.json                              {D('Current dir \u2014 JSON')}\n"
        f"  ./.cogtrix.yml  |  ./.cogtrix.yaml           {D('Current dir \u2014 YAML')}\n"
        f"  ~/.cogtrix.json                              {D('Home dir \u2014 JSON')}\n"
        f"  ~/.cogtrix.yml  |  ~/.cogtrix.yaml           {D('Home dir \u2014 YAML')}\n"
        f"  ~/.config/cogtrix/cogtrix.json               {D('XDG config \u2014 JSON')}\n"
        f"  ~/.config/cogtrix/cogtrix.yml  |  .yaml      {D('XDG config \u2014 YAML')}\n"
        f"\n"
        f"{B('configuration priority')} {D('(highest to lowest):')}\n"
        f"\n"
        f"  1. Command-line arguments\n"
        f"  2. Environment variables {D('(COGTRIX_MODEL, COGTRIX_SESSION, etc.)')}\n"
        f"  3. Config file {D('(JSON or YAML \u2014 see search order above)')}\n"
        f"  4. Built-in defaults\n"
        f"\n"
        f"{B('environment variables:')}\n"
        f"\n"
        f"  COGTRIX_MODEL          Model alias from config\n"
        f"  COGTRIX_SESSION        Session ID\n"
        f"  COGTRIX_MEMORY_MODE    Memory mode {D('(conversation/code/reasoning)')}\n"
        f"  OPENAI_API_KEY         OpenAI API key\n"
        f"  OLLAMA_BASE_URL        Ollama base URL\n"
        f"  OPENWEATHER_API_KEY    OpenWeather API key\n"
        f"  TAVILY_API_KEY         Tavily search API key\n"
        f"  EXA_API_KEY            Exa search API key\n"
        f"  BRAVE_API_KEY          Brave Search API key\n"
        f"  SERPAPI_API_KEY        SerpAPI search key\n"
        f"  GOOGLE_API_KEY         Google Custom Search API key\n"
        f"  GOOGLE_CSE_ID          Google Programmable Search Engine ID\n"
    )

    parser = argparse.ArgumentParser(
        usage="cogtrix.py [OPTIONS]",
        description=desc,
        formatter_class=ColorHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"cogtrix {get_version_string()}",
    )

    # ── Getting started ──────────────────────────────────────────────
    start_group = parser.add_argument_group("Getting started")
    start_group.add_argument(
        "--setup",
        action="store_true",
        default=False,
        help="Setup wizard \u2014 generate a config file",
    )
    start_group.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit",
    )

    # ── Core ─────────────────────────────────────────────────────────
    core_group = parser.add_argument_group("Core")
    core_group.add_argument(
        "-m",
        "--model",
        metavar="NAME",
        help="Model name or alias",
    )
    core_group.add_argument(
        "-s",
        "--session",
        metavar="ID",
        help="Session ID for history (default: 'default')",
    )
    core_group.add_argument(
        "-M",
        "--memory-mode",
        choices=["conversation", "code", "reasoning"],
        help="Memory mode (default: 'conversation')",
    )
    core_group.add_argument(
        "-c",
        "--config-file",
        type=str,
        metavar="FILE",
        help="Path to config file (JSON or YAML)",
    )
    core_group.add_argument(
        "--data-dir",
        type=str,
        metavar="PATH",
        help="Root directory for all data storage (default: data)",
    )

    # ── Run modes ────────────────────────────────────────────────────
    mode_group = parser.add_argument_group("Run modes")
    mode_group.add_argument(
        "--prompt",
        type=str,
        metavar="TEXT",
        help="Single prompt, then exit",
    )
    mode_group.add_argument(
        "--prompt-file",
        type=str,
        metavar="FILE",
        help="Read prompt from file, then exit",
    )
    mode_group.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (for piping)",
    )
    mode_group.add_argument(
        "-S",
        "--silent",
        action="store_true",
        default=False,
        help=(
            "Silent scripting mode: no spinner/ANSI, plain stdout, "
            "tool confirmations auto-denied (use -y to auto-approve). "
            "Prompt via --prompt, --prompt-file, positional arg, or stdin."
        ),
    )
    mode_group.add_argument(
        "--assistant",
        action="store_true",
        default=False,
        help="Headless WhatsApp/Telegram daemon",
    )

    # ── Assistant mode ───────────────────────────────────────────────
    asst_group = parser.add_argument_group("Assistant mode")
    asst_group.add_argument(
        "--system-prompt",
        type=str,
        metavar="TEXT",
        help="Custom system prompt (overrides config)",
    )
    asst_group.add_argument(
        "--system-prompt-file",
        type=str,
        metavar="FILE",
        help="Read system prompt from file",
    )

    # ── Output and safety ────────────────────────────────────────────
    out_group = parser.add_argument_group("Output and safety")
    out_group.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="FILE",
        help="Save response to file",
    )
    out_group.add_argument(
        "-y",
        "--no-confirm",
        action="store_true",
        help="Auto-approve all tool confirmations",
    )
    out_group.add_argument(
        "-R",
        "--auto-route",
        action="store_true",
        help="Route simple queries to a fast model (requires auto_route_fast_model in config)",
    )
    out_group.add_argument(
        "-Q",
        "--quick",
        action="store_true",
        help="Skip optimizer, memory, and compression (fast one-off queries)",
    )
    out_group.add_argument(
        "-G",
        "--git-native",
        action="store_true",
        help="Auto stage and commit after each file write (requires git repo)",
    )
    out_group.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner",
    )
    out_group.add_argument(
        "-I",
        "--pipe",
        action="store_true",
        help=(
            "Read prompt from stdin, run once, exit. "
            "Suppresses the startup banner when stdout is not a tty."
        ),
    )
    out_group.add_argument(
        "-P",
        "--profile",
        metavar="NAME",
        help="Apply a named config profile (defined in config file)",
    )

    # ── Logging ──────────────────────────────────────────────────────
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Log to file (default: cogtrix.log)",
    )
    log_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode — verbosity 1 (LLM interactions)",
    )
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Full debug mode — verbosity 2 (LLM interactions + debug logs)",
    )
    log_group.add_argument(
        "--verbosity",
        type=int,
        metavar="N",
        choices=[0, 1, 2, 3],
        default=None,
        help="Verbosity level: 0=normal, 1=debug, 2=verbose, 3=trace",
    )

    # ── Tools ────────────────────────────────────────────────────────
    tool_group = parser.add_argument_group("Tools")
    tool_group.add_argument(
        "--tools",
        type=str,
        metavar="LIST",
        help="all, none, minimal, or comma-separated",
    )
    tool_group.add_argument(
        "--activate-tools",
        type=str,
        metavar="LIST",
        help="Comma-separated tools to pin as active on startup",
    )
    tool_group.add_argument(
        "--allow-write-path",
        action="append",
        metavar="DIR",
        help="Allow file writes to DIR (repeatable)",
    )
    tool_group.add_argument(
        "--allow-read-path",
        action="append",
        metavar="DIR",
        help="Allow file reads from DIR (repeatable)",
    )

    # ── RAG ──────────────────────────────────────────────────────────
    rag_group = parser.add_argument_group("RAG")
    rag_group.add_argument(
        "--ingest",
        action="store_true",
        help="Build vector database and exit",
    )
    rag_group.add_argument(
        "--docs-dir",
        type=str,
        metavar="PATH",
        help="Documents directory (default: docs)",
    )
    rag_group.add_argument(
        "--vectordb-dir",
        type=str,
        metavar="PATH",
        help="Vector DB directory (default: data/vectordb)",
    )
    rag_group.add_argument(
        "--embedding-provider",
        choices=["openai", "ollama"],
        help="Embedding provider (default: from config)",
    )
    rag_group.add_argument(
        "--embedding-model",
        type=str,
        metavar="NAME",
        help="Embedding model name",
    )

    # ── Shell completion ──────────────────────────────────────────────
    comp_group = parser.add_argument_group("Shell completion")
    comp_group.add_argument(
        "--install-completion",
        nargs="?",
        const="auto",
        metavar="SHELL",
        help="Print shell completion script (bash/zsh). Source it to enable tab-completion.",
    )

    # ── Setup wizard options ─────────────────────────────────────────
    setup_group = parser.add_argument_group("Setup wizard options")
    setup_group.add_argument(
        "--setup-docs",
        type=str,
        metavar="URL",
        help="Fetch docs from URL instead of embedded",
    )
    setup_group.add_argument(
        "--setup-output",
        type=str,
        metavar="FILE",
        help="Output path (default: ~/.cogtrix.yaml)",
    )

    # Positional prompt — accepted as a convenience shorthand for --silent mode.
    # Hidden from --help to avoid cluttering the usage line; examples section above
    # documents the usage pattern.
    parser.add_argument(
        "inline_prompt",
        nargs="?",
        default=None,
        metavar="PROMPT",
        help=argparse.SUPPRESS,
    )

    return parser


def parse_arguments():
    """Parse command line arguments."""
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "prompt", None) and getattr(args, "prompt_file", None):
        parser.error("--prompt and --prompt-file are mutually exclusive")
    if getattr(args, "system_prompt", None) and getattr(args, "system_prompt_file", None):
        parser.error("--system-prompt and --system-prompt-file are mutually exclusive")
    if (
        getattr(args, "inline_prompt", None)
        and not getattr(args, "silent", False)
        and not getattr(args, "prompt", None)
    ):
        # Positional prompt without --silent implies --silent for convenience
        args.silent = True

    return args
