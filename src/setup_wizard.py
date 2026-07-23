"""
Interactive setup wizard for Cogtrix.

Bootstraps an LLM via scripted prompts, then uses it to conversationally
generate a configuration file based on the project documentation.

Usage:
    python cogtrix.py --setup
    python cogtrix.py --setup --setup-docs https://example.com/docs
    python cogtrix.py --setup --setup-output ./my-config.yaml
"""

from __future__ import annotations

import contextlib
import getpass
import ipaddress
import json
import logging
import os
import re
import socket
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from string import Template
from typing import Any

import yaml

from src.cli.args import color_enabled

# Optional Rich imports for markdown rendering in wizard output
try:
    from rich.console import Console as _RichConsole
    from rich.markdown import Markdown as _RichMarkdown
    from rich.panel import Panel as _RichPanel
    from rich.syntax import Syntax as _RichSyntax

    _rich_console: Any = _RichConsole()
except ImportError:
    _RichConsole = None  # type: ignore[assignment, misc]
    _RichMarkdown = None  # type: ignore[assignment, misc]
    _RichPanel = None  # type: ignore[assignment, misc]
    _RichSyntax = None  # type: ignore[assignment, misc]
    _rich_console = None

log = logging.getLogger("cogtrix")

# ── ANSI helpers ─────────────────────────────────────────────────────


def _B(t: str) -> str:
    """Bold."""
    return f"\033[1m{t}\033[0m" if color_enabled() else t


def _D(t: str) -> str:
    """Dim."""
    return f"\033[2m{t}\033[0m" if color_enabled() else t


def _G(t: str) -> str:
    """Green."""
    return f"\033[32m{t}\033[0m" if color_enabled() else t


def _R(t: str) -> str:
    """Red."""
    return f"\033[31m{t}\033[0m" if color_enabled() else t


def _DM(t: str) -> str:
    """Dim magenta."""
    return f"\033[2;35m{t}\033[0m" if color_enabled() else t


def _C(t: str) -> str:
    """Cyan."""
    return f"\033[36m{t}\033[0m" if color_enabled() else t


def _BC(t: str) -> str:
    """Bold cyan."""
    return f"\033[1;36m{t}\033[0m" if color_enabled() else t


def _BG(t: str) -> str:
    """Bold green."""
    return f"\033[1;32m{t}\033[0m" if color_enabled() else t


def _mask_api_key(key: str) -> str:
    """Return a display-safe masked version of an API key.

    Shows the first 3 and last 4 characters when the key is at least 10
    characters long; otherwise returns ``***`` to avoid revealing short keys.
    """
    if len(key) < 10:
        return "***"
    return key[:3] + "***" + key[-4:]


# ── Spinner ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def _spinner(label: str = "Thinking"):
    """Show a spinner while a blocking operation runs.

    Uses Rich ``Console.status`` when available, otherwise falls back to a
    simple threaded dot-printer on stdout.
    """
    if _rich_console is not None:
        with _rich_console.status(f"  {label}\u2026", spinner="dots"):
            yield
        return

    # Plain fallback: print dots until the block finishes
    stop = threading.Event()

    def _dots() -> None:
        sys.stdout.write(f"  {label}")
        sys.stdout.flush()
        while not stop.wait(0.5):
            sys.stdout.write(".")
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    t = threading.Thread(target=_dots, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


# ── Box drawing ──────────────────────────────────────────────────────


def _print_banner() -> None:
    """Print a centered banner with rounded box-drawing characters."""
    title = "Cogtrix Setup Wizard"
    sub = "Generate your configuration file"
    w = max(len(title), len(sub)) + 6  # inner width with padding

    t_pad_l = (w - len(title)) // 2
    t_pad_r = w - t_pad_l - len(title)
    s_pad_l = (w - len(sub)) // 2
    s_pad_r = w - s_pad_l - len(sub)

    bar = "\u2500" * w
    print()
    print(f"  \u256d{bar}\u256e")
    print(f"  \u2502{' ' * w}\u2502")
    print(f"  \u2502{' ' * t_pad_l}{_BC(title)}{' ' * t_pad_r}\u2502")
    print(f"  \u2502{' ' * s_pad_l}{_D(sub)}{' ' * s_pad_r}\u2502")
    print(f"  \u2502{' ' * w}\u2502")
    print(f"  \u2570{bar}\u256f")
    print()


def _print_config_box(yaml_text: str) -> None:
    """Print YAML content with syntax highlighting at full terminal width."""
    if _rich_console is not None and _RichPanel is not None and _RichSyntax is not None:
        _rich_console.print()
        _rich_console.print(
            _RichPanel(
                _RichSyntax(
                    yaml_text.rstrip(), "yaml", theme="monokai", background_color="default"
                ),
                title="Generated Configuration",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        return

    # Plain-text fallback: use terminal width
    import shutil

    term_w = shutil.get_terminal_size((80, 24)).columns
    inner_w = max(term_w - 4, 30)  # 2 chars indent + 2 border chars

    header = " Generated Configuration "
    bar_len = inner_w - len(header)
    bar_l = 2
    bar_r = max(bar_len - bar_l, 1)

    print(
        f"  {_C('\u256d')}{_C('\u2500' * bar_l)}{_BC(header)}{_C('\u2500' * bar_r)}{_C('\u256e')}"
    )
    masked = _mask_secrets(yaml_text)
    for line in masked.rstrip().split("\n"):
        padded = line.ljust(inner_w)
        print(  # codeql[py/clear-text-logging-sensitive-data] yaml_text is passed through _mask_secrets() before this display; only redacted text reaches here
            f"  {_C('\u2502')}{padded}{_C('\u2502')}"
        )
    print(f"  {_C('\u2570')}{_C('\u2500' * inner_w)}{_C('\u256f')}")


def _step(n: int, label: str) -> None:
    """Print a step header like: Step 1 of 3 \u00b7 Connect to LLM"""
    print(f"\n  {_BC(f'Step {n} of 3')} {_D('\u00b7')} {_B(label)}\n")


# ── Constants ────────────────────────────────────────────────────────

_DOCS_PATH = Path(__file__).resolve().parent.parent / "docs" / "CONFIGURATION.md"
_MAX_DOC_SIZE = 10 * 1024 * 1024  # 10 MB
_DEFAULT_OUTPUT_PATH = Path.home() / ".cogtrix.yaml"

# ── Doc section helpers ───────────────────────────────────────────────

#: Matches H1/H2/H3 headings for doc section splitting.
_SECTION_RE = re.compile(r"^#{1,3} (.+)$", re.MULTILINE)


def _index_docs(docs_text: str) -> dict[str, str]:
    """Split docs into sections keyed by heading text (lowercase).

    Returns an ordered dict so the first entry is always the intro/overview.
    Returns an empty dict when *docs_text* is empty or has no recognisable headings.
    """
    if not docs_text.strip():
        return {}
    matches = list(_SECTION_RE.finditer(docs_text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(docs_text)
        content = docs_text[start:end].strip()
        key = heading.lower()
        # Deduplicate keys that share the same heading text
        if key in sections:
            key = f"{key}_{i}"
        sections[key] = content
    return sections


def _retrieve_relevant_sections(
    query: str,
    docs_index: dict[str, str],
    max_chars: int = 6000,
) -> str:
    """Return doc sections most relevant to *query*, up to *max_chars* total.

    Matching: a section is included when its heading (key) contains any
    non-trivial word (length > 2) from *query*, case-insensitively.

    Always prepends the first section (intro/overview) when there are matches
    and space allows.  Falls back to the first *max_chars* of the full docs
    when no sections match.  Returns ``""`` for an empty index.
    """
    if not docs_index:
        return ""

    keys = list(docs_index.keys())
    words = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]

    matched_keys = [k for k in keys if any(word in k for word in words)]

    if not matched_keys:
        # Fallback: first max_chars of full docs (sections in order)
        full = "\n\n".join(docs_index.values())
        return full[:max_chars]

    # Build ordered result: intro first, then matched sections (no duplicates)
    result_keys: list[str] = []
    seen: set[str] = set()
    if keys:
        result_keys.append(keys[0])
        seen.add(keys[0])
    for k in matched_keys:
        if k not in seen:
            result_keys.append(k)
            seen.add(k)

    parts: list[str] = []
    total = 0
    for k in result_keys:
        content = docs_index[k]
        gap = 2 if parts else 0  # "\n\n" separator
        if total + gap + len(content) > max_chars:
            remaining = max_chars - total - gap
            if remaining > 200:
                parts.append(content[:remaining])
            break
        parts.append(content)
        total += gap + len(content)

    return "\n\n".join(parts)


_WIZARD_SYSTEM_PROMPT = Template("""\
You are the Cogtrix setup wizard. Your job is to help the user create a \
configuration file for Cogtrix by asking targeted questions.

Reference the documentation sections provided in each user message when relevant. \
Ask one question at a time. Output only valid YAML in fenced blocks when ready.

## Existing Configuration

$existing_config

## Bootstrap Provider

The user completed Step 1 and already has a verified, working LLM connection:
- Provider name: $bootstrap_provider
- Provider type: $bootstrap_type
- Base URL: $bootstrap_base_url
- Model: $bootstrap_model
- API key configured: $bootstrap_has_key

**Use these values directly in the YAML config. Do NOT ask the user about the \
bootstrap provider — type, base URL, model name, and API key are already known.**

## Active (Production) Model

$production_context

All API keys are managed outside the config and will be injected automatically. \
Use the literal placeholder ``"your-api-key-here"`` for every api_key field. \
Do NOT ask for any key value, do NOT include any real key or secret — not in \
the YAML values, not in comments, not in "Next steps" or anywhere else.

Include all configured providers as entries (connection info only) and create \
model entries referencing them. Use ``models.default`` to set the active model.

## Instructions

- Be direct and professional. Do not use filler words like "Sure!", "Great!", \
"Awesome!", "Absolutely!" or similar. Just ask the question.
- Ask ONE question at a time. Wait for the user's response before asking the next.
- Start by asking what the user wants to use Cogtrix for (interactive assistant, \
WhatsApp bot, Telegram bot, research tool, etc.)
- Based on their answer, ask only the relevant follow-up questions. \
Do not ask about features the user does not need.
- For each config section, offer sensible defaults and explain trade-offs briefly.
- When you have enough information, produce the COMPLETE configuration as a \
YAML block enclosed in ```yaml``` and ``` markers.
- Include comments in the YAML explaining each section.
- Never include actual API keys or secrets anywhere in your output (not in values, \
not in comments). Use ``"your-api-key-here"`` as the only placeholder — the real \
key is injected automatically and the user never needs to copy-paste it.
- If editing an existing config, preserve settings the user does not want to change.
- Providers should contain only connection info (type, base_url, api_key). \
Model settings (model name, temperature, context_window, max_tokens) go in the models section.
- Use ``models.default: <alias>`` to set the active model.
- Do not use top-level ``provider`` or ``model`` keys — those are deprecated.\
""")


# ── YAML secret sanitization ─────────────────────────────────────────


#: Case-insensitive substrings that identify secret-bearing keys.
_SECRET_KEY_PATTERNS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "private_key",
    "client_secret",
)


def _sanitize_yaml_for_prompt(yaml_str: str) -> str:
    """Redact secret values from a YAML string before injecting into an LLM prompt.

    Walks the parsed YAML tree recursively and replaces the value of any key
    whose name contains a secret-bearing substring (case-insensitive) with
    ``"***"``.  Non-secret fields (model, provider, base_url, etc.) are
    preserved unchanged so the wizard can still read the existing config.

    Returns ``"[existing config redacted — could not parse]"`` if the YAML
    cannot be parsed, and an empty string unchanged if the input is empty.
    """
    if not yaml_str or not yaml_str.strip():
        return yaml_str

    def _is_secret_key(key: str) -> bool:
        low = str(key).lower()
        return any(pat in low for pat in _SECRET_KEY_PATTERNS)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: ("***" if _is_secret_key(k) else _walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    try:
        parsed = yaml.safe_load(yaml_str)
        if parsed is None:
            return yaml_str  # empty YAML (e.g. whitespace-only) — return as-is
        sanitized = _walk(parsed)
        return yaml.dump(sanitized, default_flow_style=False, allow_unicode=True)
    except yaml.YAMLError:
        return "[existing config redacted — could not parse]"


# ── Main wizard ──────────────────────────────────────────────────────


def run_setup_wizard(
    setup_docs_url: str | None = None,
    output_path: Path | None = None,
) -> None:
    """Main entry point for the --setup wizard."""
    output = output_path or _DEFAULT_OUTPUT_PATH

    _print_banner()

    # ── Preamble: detect environment and existing config ─────────
    env = _detect_environment()
    _print_detections(env)

    existing_yaml, existing_path = _load_existing_config()
    existing_info: dict[str, Any] = {}
    if existing_path:
        print(f"  {_G(chr(0x2713))} Found existing config: {_B(str(existing_path))}")
        existing_info = _extract_config_info(existing_yaml)
        if existing_info:
            print(
                f"    model: {existing_info.get('model', '?')} "
                f"({existing_info.get('provider', '?')})"
            )
        choice = _ask_choice(
            "Mode",
            choices=["edit existing", "create new"],
            default="edit existing",
        )
        if choice == "create new":
            existing_yaml = ""
            existing_info = {}

    # ── Step 1: bootstrap LLM ────────────────────────────────────
    _step(1, "Connect to LLM")
    llm, bootstrap_info = _bootstrap_llm(env, existing_info)

    # Load docs
    docs = _load_docs(setup_docs_url)

    # Build system prompt — Template uses $placeholder syntax so curly braces in
    # docs/config content are passed through without escaping.
    # Sanitize the existing config so API keys are not sent to the LLM.
    existing_config_raw = (
        _sanitize_yaml_for_prompt(existing_yaml) if existing_yaml else "No existing configuration."
    )
    production_info = _maybe_configure_production_model(bootstrap_info, env)
    system_prompt = _WIZARD_SYSTEM_PROMPT.substitute(
        existing_config=existing_config_raw,
        bootstrap_provider=bootstrap_info["provider"],
        bootstrap_type=bootstrap_info.get("type", "openai"),
        bootstrap_base_url=bootstrap_info.get("base_url") or "(default)",
        bootstrap_model=bootstrap_info["model"],
        bootstrap_has_key="yes" if bootstrap_info.get("api_key") else "no",
        production_context=_format_production_context(bootstrap_info, production_info),
    )

    docs_index = _index_docs(docs)

    # ── Step 2: LLM conversation ─────────────────────────────────
    _step(2, "Configure")
    print(f"  {_D('Type quit to cancel at any time.')}\n")
    final_response = _run_conversation(llm, system_prompt, docs_index=docs_index)

    # ── Step 3: validate and write ───────────────────────────────
    _step(3, "Save")
    yaml_content = _extract_yaml(final_response)
    _validate_and_write(yaml_content, bootstrap_info, output, production_info=production_info)

    print(f"\n  {_BG(chr(0x2713))} Config written to {_B(str(output))}")
    print(
        f"  {_D('Run')} python cogtrix.py {_D('to start Cogtrix.')}\n"
    )  # codeql[py/clear-text-logging-sensitive-data] no credentials in this message; api_key from bootstrap_info is masked before any display via _mask_secrets()


# ── Phase 1: bootstrap ──────────────────────────────────────────────


def _detect_environment() -> dict[str, Any]:
    """Auto-detect available LLM providers from env vars and running services."""
    env: dict[str, Any] = {}

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        env["openai_key"] = openai_key

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        env["anthropic_key"] = anthropic_key

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        env["gemini_key"] = gemini_key

    xai_key = os.environ.get("XAI_API_KEY")
    if xai_key:
        env["xai_key"] = xai_key

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        env["deepseek_key"] = deepseek_key

    # COGTRIX_OLLAMA accepts "host", "host:port", IPv6, or full URL
    from src.config import _parse_ollama_address

    cogtrix_ollama = os.environ.get("COGTRIX_OLLAMA")
    if cogtrix_ollama:
        ollama_url = _parse_ollama_address(cogtrix_ollama.strip())
    else:
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if _is_safe_ollama_url(ollama_url):
        try:
            req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    env["ollama_running"] = True
                    env["ollama_url"] = ollama_url
        except Exception:
            pass
    else:
        log.warning(
            "Skipping Ollama auto-detection — OLLAMA_BASE_URL %r resolves to a "
            "restricted address and will not be probed automatically.",
            _redact_url_creds(ollama_url),
        )

    return env


def _print_detections(env: dict[str, Any]) -> None:
    """Print auto-detected environment info."""
    if env.get("openai_key"):
        print(f"  {_G(chr(0x2713))} Detected OPENAI_API_KEY")
    if env.get("anthropic_key"):
        print(f"  {_G(chr(0x2713))} Detected ANTHROPIC_API_KEY")
    if env.get("gemini_key"):
        print(f"  {_G(chr(0x2713))} Detected GEMINI_API_KEY")
    if env.get("xai_key"):
        print(f"  {_G(chr(0x2713))} Detected XAI_API_KEY")
    if env.get("deepseek_key"):
        print(f"  {_G(chr(0x2713))} Detected DEEPSEEK_API_KEY")
    if env.get("ollama_running"):
        url = env.get("ollama_url", "http://127.0.0.1:11434")
        print(f"  {_G(chr(0x2713))} Detected Ollama at {url}")
    if env:
        print()


def _extract_config_info(yaml_content: str) -> dict[str, Any]:
    """Extract provider/model info from existing config YAML for defaults."""
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            return {}
        info: dict[str, Any] = {}

        # New format: models.default → model alias → model entry → provider
        models = data.get("models", {})
        if isinstance(models, dict):
            default_alias = models.get("default")
            if isinstance(default_alias, str) and default_alias in models:
                model_entry = models[default_alias]
                if isinstance(model_entry, dict):
                    info["model"] = model_entry.get("model")
                    provider_name = model_entry.get("provider")
                    if provider_name:
                        info["provider"] = provider_name
                        providers = data.get("providers", data.get("inference", {}))
                        if isinstance(providers, dict) and provider_name in providers:
                            pcfg = providers[provider_name]
                            if isinstance(pcfg, dict):
                                info["type"] = pcfg.get("type", "openai")
                                if pcfg.get("base_url"):
                                    info["base_url"] = pcfg["base_url"]
                                if pcfg.get("api_key"):
                                    info["api_key"] = pcfg["api_key"]

        # Legacy fallback: top-level provider/model
        if not info.get("provider"):
            provider_name = data.get("provider")
            if provider_name:
                info["provider"] = provider_name
            model = data.get("model")
            if model:
                info["model"] = model
            providers = data.get("providers", data.get("inference", {}))
            if isinstance(providers, dict) and provider_name and provider_name in providers:
                pcfg = providers[provider_name]
                if isinstance(pcfg, dict):
                    info["type"] = pcfg.get("type", "openai")
                    if not info.get("model"):
                        info["model"] = pcfg.get("model")
                    if pcfg.get("base_url"):
                        info["base_url"] = pcfg["base_url"]
                    if pcfg.get("api_key"):
                        info["api_key"] = pcfg["api_key"]

        return info
    except Exception:
        return {}


def _maybe_configure_production_model(
    bootstrap_info: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    """Offer to configure a separate production model after Step 1.

    The bootstrap model (used to run the wizard) may be small or cheap.
    This scripted prompt lets the user configure a more capable model for
    Cogtrix's day-to-day use without leaving the wizard.

    Returns either *bootstrap_info* unchanged (same model for both roles)
    or a fresh ``production_info`` dict from a second ``_bootstrap_llm`` run.
    """
    model = bootstrap_info["model"]
    provider = bootstrap_info["provider"]
    print(
        f"\n  {_D('Wizard ran with:')} {_B(model)} "
        f"{_D('on')} {_B(provider)}"
        f"  {_D('(this will be your active Cogtrix model)')}"
    )
    choice = _ask_choice(
        "Configure a different model for Cogtrix",
        choices=["yes", "no"],
        default="no",
    )
    if choice != "yes":
        return bootstrap_info

    print(
        f"\n  {_D('Enter the connection details for your production model.')}\n"
        f"  {_D('The wizard will test the connection before proceeding.')}\n"
    )
    _, production_info = _bootstrap_llm(env, {})
    return production_info


def _format_production_context(
    bootstrap_info: dict[str, Any],
    production_info: dict[str, Any],
) -> str:
    """Render the production-model block for the wizard system prompt.

    When bootstrap == production the LLM is told to use the bootstrap as
    the active model.  When they differ it receives the full production
    connection context so it can write a two-provider config without asking.
    """
    if production_info is bootstrap_info or (
        production_info.get("provider") == bootstrap_info.get("provider")
        and production_info.get("model") == bootstrap_info.get("model")
    ):
        return (
            f"Same as bootstrap: use {bootstrap_info['provider']} / "
            f"{bootstrap_info['model']} as models.default."
        )

    return (
        "The user configured a separate production model in Step 1:\n"
        f"- Provider name: {production_info['provider']}\n"
        f"- Provider type: {production_info.get('type', 'openai')}\n"
        f"- Base URL: {production_info.get('base_url') or '(default)'}\n"
        f"- Model: {production_info['model']}\n"
        f"- API key configured: {'yes' if production_info.get('api_key') else 'no'}\n"  # codeql[py/clear-text-logging-sensitive-data] only presence is reported, not the key value itself
        "\n"
        "Add this as a second provider entry in the config and set it as "
        "models.default. Keep the bootstrap provider in the providers section "
        "too, but do NOT set it as the active model."
    )


def _list_ollama_models(base_url: str) -> list[str]:
    """Fetch and display installed Ollama models. Returns model names."""
    from urllib.parse import urlparse as _urlparse

    _parsed = _urlparse(base_url)
    _hostname = _parsed.hostname or ""
    _port = _parsed.port or 80
    try:
        socket.getaddrinfo(_hostname, _port)
    except OSError:
        log.warning(
            "Cannot reach Ollama at %r — hostname did not resolve. "
            "If running inside Docker, try --network host.",
            _redact_url_creds(base_url),
        )
        return []
    if not _is_safe_ollama_url(base_url, allow_private=True):
        log.warning(
            "Skipping model list — %r resolved to a restricted address "
            "(link-local or reserved range). Use a loopback or LAN address.",
            _redact_url_creds(base_url),
        )
        return []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            return []
        print(f"\n  {_D('Installed models:')}")
        names: list[str] = []
        for m in models:
            name = m.get("name", "?")
            size = m.get("size", 0)
            if size > 1e9:
                size_s = f"{size / 1e9:.1f}GB"
            else:
                size_s = f"{size / 1e6:.0f}MB"
            print(f"    {name} {_D(chr(0x00B7))} {_D(size_s)}")
            names.append(name)
        print()
        return names
    except Exception:
        return []


def _extract_connection_error(exc: Exception) -> str:
    """Extract a human-readable message from an LLM API exception.

    The openai SDK builds ``exc.message`` as
    ``"Error code: {status} - {body_repr}"`` which is noisy.  The clean
    human-readable message is nested inside ``exc.body``.

    For ``APIConnectionError`` (host unreachable) there is no HTTP status code,
    so ``exc.message`` is already a clean short string like ``"Connection error."``
    with no ``"Error code:"`` prefix — that is returned as-is.
    """
    # openai SDK: APIStatusError.body is the parsed JSON response body.
    # The actual message is at body['error']['message'] or body['message'].
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if body.get("message"):
            return str(body["message"])
    # APIConnectionError and similar non-HTTP errors: exc.message is already
    # a clean short string ("Connection error.", "Timeout.", etc.) with no
    # "Error code:" prefix.
    msg = getattr(exc, "message", None)
    if isinstance(msg, str) and msg and not msg.startswith("Error code:"):
        return msg
    return str(exc)


def _test_connection(
    provider_type: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> Any | None:
    """Test LLM connectivity. Returns the LLM instance on success, None on failure."""

    from src.providers import create_chat_model

    try:
        # Use max_retries=0: we want the connection test to fail fast rather than
        # retrying internally on network errors (which would make the wizard appear
        # to hang when the host is truly unreachable).
        llm = create_chat_model(
            provider_type,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
    except Exception as exc:
        print(f"  {_R(chr(0x2717))} Provider setup failed: {_extract_connection_error(exc)}")
        return None

    try:
        from langchain_core.messages import HumanMessage

        with _spinner("Testing connection"):
            response = llm.invoke([HumanMessage(content="Say 'ok' in one word.")])
        text = response.content if hasattr(response, "content") else str(response)
        if not text.strip():
            raise RuntimeError("Empty response from LLM")
        print(f"  {_BG(chr(0x2713))} Connected to {_B(provider_type)}/{_B(model)}\n")
        return llm
    except Exception as exc:
        print(f"  {_R(chr(0x2717))} Connection failed: {_extract_connection_error(exc)}\n")
        return None


def _bootstrap_llm(
    env: dict[str, Any],
    existing_info: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Phase 1: scripted prompts to get a working LLM instance.

    Wraps the provider/model/key questions in a retry loop so that
    connection failures don't kill the wizard.  All entered values are
    carried forward as defaults on each retry attempt.
    """
    # Determine initial default provider type from existing config or environment
    if existing_info.get("type"):
        default_type = existing_info["type"]
    elif env.get("ollama_running"):
        default_type = "ollama"
    elif env.get("openai_key"):
        default_type = "openai"
    elif env.get("anthropic_key"):
        default_type = "anthropic"
    elif env.get("gemini_key"):
        default_type = "google"
    elif env.get("xai_key"):
        default_type = "xai"
    elif env.get("deepseek_key"):
        default_type = "deepseek"
    else:
        default_type = "openai"

    last: dict[str, Any] = {}  # carries entered values across retry iterations

    while True:
        selected_type = _ask_choice(
            "Provider type",
            choices=["ollama", "openai", "anthropic", "google", "xai", "deepseek"],
            default=last.get("selected_type") or default_type,
        )

        api_key: str | None = None
        base_url: str | None = None
        provider_name: str = selected_type
        provider_type: str = selected_type  # may be remapped (xai → openai)

        if selected_type == "ollama":
            default_url = (
                last.get("base_url")
                or (
                    existing_info.get("base_url") if existing_info.get("type") == "ollama" else None
                )
                or env.get("ollama_url", "http://127.0.0.1:11434")
            )
            base_url = _ask_input("Ollama URL", default=default_url)
            available = _list_ollama_models(base_url)
            default_model = last.get("model") or existing_info.get("model") or "qwen3:8b"
            if available and default_model not in available:
                default_model = available[0]
            model = _ask_input("Model", default=default_model)

        elif selected_type == "openai":
            prior_key = last.get("api_key") or (
                existing_info.get("api_key") if existing_info.get("type") == "openai" else None
            )
            if env.get("openai_key"):
                print(f"  {_G(chr(0x2713))} Using OPENAI_API_KEY from environment")
                api_key = env["openai_key"]
            elif prior_key:
                masked = _mask_api_key(prior_key)
                entered = _ask_input("API key", default=masked, secret=True)
                api_key = prior_key if entered == masked else (entered or None)
            else:
                entered = _ask_input("API key", secret=True)
                api_key = entered or None

            default_base = (
                last.get("base_url")
                or (
                    existing_info.get("base_url") if existing_info.get("type") == "openai" else None
                )
                or "https://api.openai.com/v1"
            )
            base_url = _ask_input("Base URL", default=default_base)
            if base_url != "https://api.openai.com/v1":
                provider_name = _ask_input(
                    "Provider name",
                    default=last.get("provider_name") or existing_info.get("provider") or "openai",
                )
            else:
                provider_name = "openai"
            default_model = last.get("model") or existing_info.get("model") or "gpt-4.1-mini"
            model = _ask_input("Model", default=default_model)

        elif selected_type == "anthropic":
            prior_key = last.get("api_key") or (
                existing_info.get("api_key") if existing_info.get("type") == "anthropic" else None
            )
            if env.get("anthropic_key"):
                print(f"  {_G(chr(0x2713))} Using ANTHROPIC_API_KEY from environment")
                api_key = env["anthropic_key"]
            elif prior_key:
                masked = _mask_api_key(prior_key)
                entered = _ask_input("API key", default=masked, secret=True)
                api_key = prior_key if entered == masked else (entered or None)
            else:
                entered = _ask_input("API key", secret=True)
                api_key = entered or None
            provider_name = "anthropic"
            default_model = last.get("model") or existing_info.get("model") or "claude-sonnet-4-5"
            model = _ask_input("Model", default=default_model)

        elif selected_type == "google":
            prior_key = last.get("api_key") or (
                existing_info.get("api_key") if existing_info.get("type") == "google" else None
            )
            if env.get("gemini_key"):
                print(f"  {_G(chr(0x2713))} Using GEMINI_API_KEY from environment")
                api_key = env["gemini_key"]
            elif prior_key:
                masked = _mask_api_key(prior_key)
                entered = _ask_input("API key", default=masked, secret=True)
                api_key = prior_key if entered == masked else (entered or None)
            else:
                entered = _ask_input("API key", secret=True)
                api_key = entered or None
            provider_name = "google"
            default_model = last.get("model") or existing_info.get("model") or "gemini-2.5-flash"
            model = _ask_input("Model", default=default_model)

        elif selected_type == "xai":
            from src.providers.defaults import OPENAI_PRESETS as _P

            _preset = _P["xai"]
            prior_key = last.get("api_key") or (
                existing_info.get("api_key")
                if existing_info.get("type") == "openai" and existing_info.get("provider") == "xai"
                else None
            )
            if env.get("xai_key"):
                print(f"  {_G(chr(0x2713))} Using XAI_API_KEY from environment")
                api_key = env["xai_key"]
            elif prior_key:
                masked = _mask_api_key(prior_key)
                entered = _ask_input("API key", default=masked, secret=True)
                api_key = prior_key if entered == masked else (entered or None)
            else:
                entered = _ask_input("API key", secret=True)
                api_key = entered or None
            provider_name = "xai"
            base_url = _preset["base_url"]
            provider_type = "openai"
            default_model = last.get("model") or existing_info.get("model") or _preset["model"]
            model = _ask_input("Model", default=default_model)

        elif selected_type == "deepseek":
            from src.providers.defaults import OPENAI_PRESETS as _P

            _preset = _P["deepseek"]
            prior_key = last.get("api_key") or (
                existing_info.get("api_key")
                if existing_info.get("type") == "openai"
                and existing_info.get("provider") == "deepseek"
                else None
            )
            if env.get("deepseek_key"):
                print(f"  {_G(chr(0x2713))} Using DEEPSEEK_API_KEY from environment")
                api_key = env["deepseek_key"]
            elif prior_key:
                masked = _mask_api_key(prior_key)
                entered = _ask_input("API key", default=masked, secret=True)
                api_key = prior_key if entered == masked else (entered or None)
            else:
                entered = _ask_input("API key", secret=True)
                api_key = entered or None
            provider_name = "deepseek"
            base_url = _preset["base_url"]
            provider_type = "openai"
            default_model = last.get("model") or existing_info.get("model") or _preset["model"]
            model = _ask_input("Model", default=default_model)

        else:
            # Should not happen while choices list and elif chain are in sync.
            # Log loudly and retry rather than crashing the wizard.
            import logging as _wiz_log

            _wiz_log.getLogger("cogtrix.setup_wizard").error(
                "Unhandled provider type %r — choices list and elif chain "
                "are out of sync. Please report this bug.",
                selected_type,
            )
            print(
                f"\n  [!] Provider type {selected_type!r} is not yet supported "
                "by the wizard. Please choose a different provider.",
                flush=True,
            )
            last["selected_type"] = None
            continue

        # Persist all entered values so the next retry uses them as defaults
        last = {
            "selected_type": selected_type,
            "type": provider_type,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "provider_name": provider_name,
        }

        # Test connection
        llm = _test_connection(provider_type, model, api_key, base_url)
        if llm is not None:
            bootstrap_info: dict[str, Any] = {
                "provider": provider_name,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "type": provider_type,
            }
            return llm, bootstrap_info

        # Connection failed — offer retry
        retry = _ask_choice("Retry", choices=["yes", "no"], default="yes")
        if retry == "no":
            raise SystemExit(1)
        print()


# ── Documentation & config loading ───────────────────────────────────


def _redact_url_creds(url: str) -> str:
    """Return *url* with any embedded username/password stripped for safe logging."""
    try:
        p = urllib.parse.urlparse(url)
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        return urllib.parse.urlunparse(p._replace(netloc=netloc))
    except Exception:
        return "<unparseable URL>"


def _is_safe_ollama_url(url: str, *, allow_private: bool = False) -> bool:
    """Return True if *url* is safe to probe as an Ollama endpoint.

    Like ``_is_safe_url`` but permits loopback addresses (127.x.x.x /
    ::1) so that a locally-running Ollama instance is always reachable.

    *allow_private* controls RFC-1918 handling:

    - ``False`` (default) — used for untrusted sources such as the
      ``OLLAMA_BASE_URL`` env-var probe (BUG-229): RFC-1918 private
      addresses are blocked alongside link-local, reserved, and unspecified.
    - ``True`` — used for user-typed wizard input (BUG-230): RFC-1918
      addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x) are allowed so
      that Ollama servers on a LAN are reachable.  Link-local (169.254.x.x /
      AWS metadata), reserved, and unspecified are always blocked.

    Returns False on any parse or DNS resolution failure.
    """
    from urllib.parse import urlparse as _urlparse

    try:
        parsed = _urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not hostname:
            return False
        resolved = socket.getaddrinfo(hostname, port)
        for _, _, _, _, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_loopback:
                continue  # always allow localhost (canonical Ollama address)
            if ip.is_link_local or ip.is_reserved or ip.is_unspecified:
                return False  # always block — never legitimate Ollama targets
            if not allow_private and ip.is_private:
                return False  # block RFC-1918 in strict (env-var) mode only
    except Exception:
        return False
    return True


def _is_safe_url(url: str, *, allow_local: bool = False) -> bool:
    """Return True if all resolved IP addresses for *url* are safe to fetch.

    *allow_local* controls what is permitted:

    - ``False`` (default) — strict mode for untrusted URL sources: blocks
      loopback, RFC-1918 private, link-local, reserved, and unspecified.
    - ``True`` — relaxed mode for explicitly user-provided URLs (e.g.
      ``--setup-docs``, API admin wizard ``docs_url``): allows loopback and
      RFC-1918 private so local/LAN doc servers work. Link-local (169.254.x.x
      / AWS metadata), reserved, and unspecified are still blocked.

    Returns False on any DNS resolution failure.
    """
    from urllib.parse import urlparse as _urlparse

    try:
        parsed = _urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not hostname:
            return False
        resolved = socket.getaddrinfo(hostname, port)
        for _, _, _, _, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_link_local or ip.is_reserved or ip.is_unspecified:
                return False  # always block — never a legitimate doc server
            if not allow_local and (ip.is_loopback or ip.is_private):
                return False  # block private ranges in strict mode only
    except Exception:
        return False
    return True


def _load_docs(url: str | None = None) -> str:
    """Load configuration documentation.

    If *url* is provided, fetch from that URL first. Falls back to the
    embedded ``docs/CONFIGURATION.md`` file on any failure.
    """
    if url:
        try:
            from urllib.parse import urlparse as _urlparse

            parsed = _urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(
                    f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)"
                )
            if not _is_safe_url(url, allow_local=True):
                log.warning(
                    "Docs URL %s could not be fetched (address not permitted) — using built-in documentation.",
                    url,
                )
            else:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as resp:  # nosec B310
                    raw = resp.read(_MAX_DOC_SIZE + 1)
                    if len(raw) > _MAX_DOC_SIZE:
                        log.warning(
                            "Docs URL response exceeded %d bytes, using embedded docs",
                            _MAX_DOC_SIZE,
                        )
                    else:
                        content = raw.decode("utf-8", errors="replace")
                        if content.strip():
                            log.debug("Loaded docs from URL: %s", url)
                            return content
        except Exception as exc:
            log.debug("Failed to fetch docs from %s: %s (using embedded)", url, exc)

    if _DOCS_PATH.exists():
        return _DOCS_PATH.read_text(encoding="utf-8")

    return "(Configuration documentation not found.)"


def _load_existing_config() -> tuple[str, Path | None]:
    """Load existing config file if one is found.

    Returns:
        (yaml_content, path) or ("", None) if no config exists.
    """
    from src.config import find_config_file

    path = find_config_file()
    if path is None:
        return "", None

    try:
        content = path.read_text(encoding="utf-8")
        return content, path
    except Exception:
        return "", None


# ── Phase 2: conversation ───────────────────────────────────────────


def _run_conversation(
    llm: Any,
    system_prompt: str,
    docs_index: dict[str, str] | None = None,
) -> str:
    """Phase 2: interactive LLM conversation loop.

    The LLM asks questions one at a time; the user responds. When the LLM
    produces a ```yaml``` block, the user is asked to confirm.

    Args:
        docs_index: Optional section index built by :func:`_index_docs`.
            When provided, relevant sections are injected as a prefix to each
            human message so the LLM has just-in-time documentation context
            without embedding the entire file upfront.

    Returns:
        The final LLM response containing the YAML configuration.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    def _augment(user_input: str) -> str:
        """Prepend relevant doc sections to *user_input* when available."""
        if not docs_index:
            return user_input
        relevant = _retrieve_relevant_sections(user_input, docs_index)
        if not relevant:
            return user_input
        return f"[Relevant documentation]\n{relevant}\n\n[User question]\n{user_input}"

    # Strict OpenAI-compatible backends (vLLM, LiteLLM) reject a messages list
    # that contains only a SystemMessage — they require at least one HumanMessage.
    # Seed the conversation with a minimal trigger so the wizard LLM opens first.
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Start."),
    ]

    with _spinner("Thinking"):
        response = llm.invoke(messages)
    ai_text: str = response.content if hasattr(response, "content") else str(response)
    messages.append(AIMessage(content=ai_text))
    _print_wizard(ai_text)

    while True:
        if _has_yaml_block(ai_text):
            confirm = _ask_choice(
                "Accept this configuration",
                choices=["yes", "no, continue editing"],
                default="yes",
            )
            if confirm == "yes":
                return ai_text
            # Let the user describe what to change or ask a question
            try:
                user_input = input(f"\n  {_DM('\u25cb')} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                raise SystemExit(0) from None
            if not user_input:
                user_input = "I'd like to make changes to the config."
            if user_input.lower() in ("quit", "exit", "cancel"):
                print(f"  {_D('Setup cancelled.')}")
                raise SystemExit(0)
            messages.append(HumanMessage(content=_augment(user_input)))
            with _spinner("Thinking"):
                response = llm.invoke(messages)
            ai_text = response.content if hasattr(response, "content") else str(response)
            messages.append(AIMessage(content=ai_text))
            _print_wizard(ai_text)
            continue

        try:
            user_input = input(f"\n  {_DM('\u25cb')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0) from None

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "cancel"):
            print(f"  {_D('Setup cancelled.')}")
            raise SystemExit(0)

        messages.append(HumanMessage(content=_augment(user_input)))
        with _spinner("Thinking"):
            response = llm.invoke(messages)
        ai_text = response.content if hasattr(response, "content") else str(response)
        messages.append(AIMessage(content=ai_text))
        _print_wizard(ai_text)

    return ai_text  # unreachable but satisfies type checker


def _print_wizard(text: str) -> None:
    """Print wizard output with Rich markdown rendering."""
    if _rich_console is not None and _RichMarkdown is not None:
        _rich_console.print()
        _rich_console.print(
            _RichPanel(
                _RichMarkdown(text),
                title="Wizard",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    else:
        print(f"\n  {_BC('Wizard:')}\n")
        for line in text.split("\n"):
            print(f"  {line}")


# ── YAML extraction ─────────────────────────────────────────────────


def _has_yaml_block(text: str) -> bool:
    """Check if text contains a complete ```yaml``` code fence."""
    if "```yaml" not in text:
        return False
    after = text.split("```yaml", 1)[1]
    return "```" in after


def _strip_nulls(obj: Any) -> Any:
    """Recursively remove None values and empty dicts from a config structure.

    Prevents ``services: null`` and similar LLM-generated placeholders from
    appearing in the written config file.
    """
    if not isinstance(obj, dict):
        return obj
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if v is None:
            continue
        stripped = _strip_nulls(v)
        if isinstance(stripped, dict) and not stripped:
            continue
        result[k] = stripped
    return result


def _extract_yaml(text: str) -> str:
    """Extract YAML content from the last ```yaml``` code fence in *text*.

    Uses plain string search so code blocks that appear after the YAML (e.g.
    shell commands in a "Next steps" section) never bleed into the result.

    Raises:
        ValueError: If no ```yaml``` block is found.
    """
    start_marker = "```yaml"
    last_start = text.rfind(start_marker)
    if last_start == -1:
        raise ValueError("No ```yaml``` block found in LLM response")

    # Skip past the ```yaml header line to the content
    newline_pos = text.find("\n", last_start)
    if newline_pos == -1:
        raise ValueError("No ```yaml``` block found in LLM response")
    content_start = newline_pos + 1

    # Stop at the very next ``` (the closing fence), not the last one in the file
    end_pos = text.find("```", content_start)
    if end_pos == -1:
        # Unclosed block — take everything to end of string
        return text[content_start:].strip()

    return text[content_start:end_pos].strip()


# ── Phase 3: validate & write ────────────────────────────────────────


def _validate_and_write(
    yaml_content: str,
    bootstrap_info: dict[str, Any],
    output_path: Path,
    production_info: dict[str, Any] | None = None,
) -> None:
    """Phase 3: parse, inject secrets, validate, and write config."""
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        print(f"  {_R('\u2717')} Invalid YAML: {exc}")
        print(f"  {_D('Please run --setup again.')}")
        raise SystemExit(1) from exc

    if not isinstance(data, dict):
        print(f"  {_R('\u2717')} Generated config is not a valid mapping.")
        raise SystemExit(1)

    _inject_bootstrap(data, bootstrap_info, production_info=production_info)
    data = _strip_nulls(data)

    # Validate by round-tripping through load_config
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = Path(tmp.name)
            yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)

        from src.config import Config, _apply_config_file

        test_config = Config()
        _apply_config_file(test_config, tmp_path)
        log.debug("Config validation passed")
    except Exception as exc:
        print(f"  {_R('\u26a0')} Validation warning: {exc}")
        print(f"  {_D('The config may still work but could have issues.')}")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Serialize the bootstrap-injected data (not the original LLM text)
    final_yaml = yaml.dump(data, default_flow_style=False, sort_keys=False)
    masked = _mask_secrets(
        final_yaml
    )  # codeql[py/clear-text-logging-sensitive-data] api_key values are replaced by _mask_secrets() before any display
    print()
    _print_config_box(masked)
    print()

    confirm = _ask_choice("Write this config", choices=["yes", "no"], default="yes")
    if confirm != "yes":
        print(f"  {_D('Setup cancelled. Config was not written.')}")
        raise SystemExit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"# Generated by Cogtrix setup wizard\n{final_yaml}".encode())
    finally:
        os.close(fd)


def _inject_bootstrap(
    data: dict[str, Any],
    bootstrap_info: dict[str, Any],
    production_info: dict[str, Any] | None = None,
) -> None:
    """Inject real connection credentials from bootstrap (and optionally production).

    *bootstrap_info* is always written into the providers section.
    When *production_info* is provided and differs from *bootstrap_info*,
    it is also written and used as models.default; otherwise the bootstrap
    provider is the active model.
    """

    def _write_provider(info: dict[str, Any]) -> None:
        providers = data.setdefault("providers", {})
        cfg = providers.setdefault(info["provider"], {})
        cfg["type"] = info["type"]
        if info.get("api_key"):
            cfg["api_key"] = info["api_key"]
        if info.get("base_url"):
            cfg["base_url"] = info["base_url"]
        for legacy in ("model", "temperature", "num_ctx", "context_window", "max_tokens"):
            cfg.pop(legacy, None)

    _write_provider(bootstrap_info)

    # Determine which provider/model is the active (production) one
    is_separate = (
        production_info is not None
        and production_info is not bootstrap_info
        and (
            production_info.get("provider") != bootstrap_info.get("provider")
            or production_info.get("model") != bootstrap_info.get("model")
        )
    )
    active: dict[str, Any] = bootstrap_info
    if is_separate:
        assert production_info is not None  # narrowed by is_separate condition above
        _write_provider(production_info)
        active = production_info

    # Ensure a default model entry exists in the models registry
    models = data.setdefault("models", {})
    alias = "default_model"
    if alias not in models:
        models[alias] = {
            "provider": active["provider"],
            "model": active["model"],
        }
    models["default"] = alias

    # Remove legacy top-level fields
    data.pop("provider", None)
    data.pop("model", None)


def _mask_secrets(yaml_text: str) -> str:
    """Mask API keys and tokens in YAML text for display."""
    _SECRET_KEYS = r"api_key|api_secret|token|password|secret"

    def _replace_inline(m: re.Match[str]) -> str:
        key = m.group(1)
        quote = m.group(2)
        val = m.group(3)
        masked = _mask_api_key(val)
        return f"{key}: {quote}{masked}{quote}"

    # Inline values (plain, single-quoted, double-quoted)
    result = re.sub(
        rf"({_SECRET_KEYS}):\s*([\"']?)([^\s\"'#]+)\2",
        _replace_inline,
        yaml_text,
        flags=re.IGNORECASE,
    )
    # Block scalar values (| or > with optional chomping indicator)
    result = re.sub(
        rf"({_SECRET_KEYS}):\s*[|>]-?\n(?:[ \t]+\S[^\n]*\n?)+",
        lambda m: f"{m.group(1)}: ***",
        result,
        flags=re.IGNORECASE,
    )
    return result


# ── Input helpers ────────────────────────────────────────────────────


def _read_masked_input(prompt: str) -> str:
    """Read a line of input displaying '*' for each character typed.

    Uses character-by-character raw terminal reads so the user sees ``*``
    feedback without the actual characters being echoed.  Falls back to
    ``getpass`` when stdin is not a TTY or ``termios`` is unavailable
    (e.g. Windows or piped input).

    Correctly handles multi-byte UTF-8 characters and ANSI escape sequences
    of arbitrary length (BUG-228, BUG-235).
    """
    if not sys.stdin.isatty():
        return getpass.getpass(prompt)
    try:
        import select
        import termios
        import tty
    except ImportError:
        return getpass.getpass(prompt)

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars: list[str] = []
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def _read_byte() -> bytes:
        return os.read(fd, 1)

    def _read_char() -> str:
        """Read one complete UTF-8 character from fd."""
        b0 = _read_byte()
        if not b0:
            return ""
        v = b0[0]
        if v < 0x80:
            return b0.decode("utf-8", errors="replace")
        extra = 1 if v < 0xE0 else (2 if v < 0xF0 else 3)
        data = b0 + b"".join(_read_byte() for _ in range(extra))
        return data.decode("utf-8", errors="replace")

    def _drain_escape() -> None:
        """Consume the remainder of an ANSI/VT escape sequence.

        Uses select with a short timeout so a lone ESC keypress doesn't block.
        CSI sequences (ESC [) end at the first byte in 0x40–0x7E.
        SS3 sequences (ESC O, used by F1–F4) have a single final byte.
        """
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            return  # standalone ESC key — nothing more to read
        nxt = _read_byte()
        if nxt == b"[":  # CSI — read until final byte (0x40–0x7E)
            while True:
                b = _read_byte()
                if 0x40 <= b[0] <= 0x7E:
                    break
        elif nxt == b"O":  # SS3 — one final byte
            _read_byte()
        # other two-byte sequences: nxt already consumed

    try:
        tty.setcbreak(fd)
        while True:
            ch = _read_char()
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            if ch in ("\x7f", "\x08"):  # backspace / delete
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x1b":  # escape sequence (arrow keys etc.) — consume and ignore
                _drain_escape()
            elif ch.isprintable():
                chars.append(ch)
                sys.stdout.write("*")
                sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "".join(chars)


def _ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Ask the user to pick from inline choices.

    Displays choices as ``(a / b / c)`` with the default bolded.
    Accepts prefix text match or Enter for the default.
    """
    parts: list[str] = []
    for c in choices:
        parts.append(_B(c) if c == default else c)
    inline = " / ".join(parts)

    while True:
        suffix = f" {_D(f'[{default}]')}" if default else ""
        try:
            raw = input(f"  {_B(prompt)} {_D('(')} {inline} {_D(')')}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0) from None

        if not raw and default:
            return default

        raw_lower = raw.lower()
        matches = [c for c in choices if c.lower().startswith(raw_lower)]
        if len(matches) == 1:
            return matches[0]

        # Also accept exact match
        exact = [c for c in choices if c.lower() == raw_lower]
        if exact:
            return exact[0]

        print(f"  {_D('Type one of:')} {' / '.join(choices)}")


def _ask_input(prompt: str, default: str | None = None, secret: bool = False) -> str:
    """Ask the user for free-text input.

    When *secret* is ``True`` each typed character is echoed as ``*``.
    An empty response accepts *default* when provided, or returns an empty
    string — callers treat an empty API key as "no authentication required".
    """
    suffix = f" {_D(f'[{default}]')}" if default else ""
    full_prompt = f"  {_B(prompt)}{suffix}: "

    try:
        if secret:
            raw = _read_masked_input(full_prompt)
        else:
            raw = input(full_prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0) from None

    raw = raw.strip()
    if not raw and default:
        return default
    return raw
