import argparse
import os
import sys


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


def parse_arguments():
    """Parse command line arguments."""
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
        f"    cogtrix.py -p ollama -m qwen3:8b          {D('Use Ollama with a specific model')}\n"
        f"    cogtrix.py -m reasoning -M reasoning       {D('Model alias + memory mode')}\n"
        f"    cogtrix.py -s project-alpha                {D('Named session (preserves history)')}\n"
        f"\n"
        f"  {B('Config file:')}\n"
        f"    cogtrix.py -c /etc/cogtrix/prod.yaml       {D('Use explicit config file')}\n"
        f"    cogtrix.py -c config.yml -p my-server      {D('Config file + CLI override')}\n"
        f"\n"
        f"  {B('Non-interactive (scripting):')}\n"
        f"    cogtrix.py --prompt \"What is 2+2?\"         {D('Single prompt, print result, exit')}\n"
        f"    cogtrix.py --prompt \"...\" -o out.md        {D('Save response to file')}\n"
        f"    cogtrix.py --prompt-file task.txt -o res.md {D('Read prompt from file, save result')}\n"
        f"    cogtrix.py --prompt \"...\" --no-stream      {D('Suppress streaming (clean stdout)')}\n"
        f"\n"
        f"  {B('Assistant mode:')}\n"
        f"    cogtrix.py --assistant                     {D('Start headless messaging daemon')}\n"
        f"    cogtrix.py --assistant --log --debug       {D('Assistant with debug logging')}\n"
        f"\n"
        f"  {B('Safety and output:')}\n"
        f"    cogtrix.py -y                              {D('Skip all tool confirmations')}\n"
        f"    cogtrix.py -o session.md                   {D('Append every response to file')}\n"
        f"    cogtrix.py -y -o log.md                    {D('No confirmations + transcript')}\n"
        f"\n"
        f"  {B('Logging:')}\n"
        f"    cogtrix.py --log                           {D('Log to cogtrix.log')}\n"
        f"    cogtrix.py --log myrun.log -v              {D('Verbose log to custom file')}\n"
        f"    cogtrix.py --debug                         {D('Full debug (implies --log -v)')}\n"
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
        f"  2. Environment variables {D('(COGTRIX_PROVIDER, COGTRIX_MODEL, etc.)')}\n"
        f"  3. Config file {D('(JSON or YAML \u2014 see search order above)')}\n"
        f"  4. Built-in defaults\n"
        f"\n"
        f"{B('environment variables:')}\n"
        f"\n"
        f"  COGTRIX_PROVIDER       LLM provider name\n"
        f"  COGTRIX_MODEL          Model name or alias\n"
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
        "-p",
        "--provider",
        metavar="NAME",
        help="LLM provider name (default: from config)",
    )
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
        help="Log full LLM interactions",
    )
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Full debug mode (implies --log -v)",
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
        "--allow-write-path",
        action="append",
        metavar="DIR",
        help="Allow file writes to DIR (repeatable)",
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

    return parser.parse_args()
