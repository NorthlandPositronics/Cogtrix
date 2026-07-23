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

    _rich_console: Any = _RichConsole()
except ImportError:
    _RichConsole = None  # type: ignore[assignment, misc]
    _RichMarkdown = None  # type: ignore[assignment, misc]
    _RichPanel = None  # type: ignore[assignment, misc]
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
    """Print YAML content inside a box-drawn frame."""
    lines = yaml_text.rstrip().split("\n")
    inner_w = max((len(line) for line in lines), default=0) + 2
    inner_w = max(inner_w, 30)  # minimum width

    header = " Generated Configuration "
    bar_len = inner_w - len(header)
    bar_l = 2
    bar_r = max(bar_len - bar_l, 1)

    print(
        f"  {_C('\u256d')}{_C('\u2500' * bar_l)}{_BC(header)}{_C('\u2500' * bar_r)}{_C('\u256e')}"
    )
    for line in lines:
        padded = line.ljust(inner_w)
        print(f"  {_C('\u2502')}{padded}{_C('\u2502')}")
    print(f"  {_C('\u2570')}{_C('\u2500' * inner_w)}{_C('\u256f')}")


def _step(n: int, label: str) -> None:
    """Print a step header like: Step 1 of 3 \u00b7 Connect to LLM"""
    print(f"\n  {_BC(f'Step {n} of 3')} {_D('\u00b7')} {_B(label)}\n")


# ── Constants ────────────────────────────────────────────────────────

_DOCS_PATH = Path(__file__).resolve().parent.parent / "docs" / "CONFIGURATION.md"
_MAX_DOC_SIZE = 10 * 1024 * 1024  # 10 MB
_DEFAULT_OUTPUT_PATH = Path.home() / ".cogtrix.yaml"

_WIZARD_SYSTEM_PROMPT = Template("""\
You are the Cogtrix setup wizard. Your job is to help the user create a \
configuration file for Cogtrix by asking targeted questions.

## Documentation

$docs

## Existing Configuration

$existing_config

## Bootstrap Provider

The user already has a working LLM connection with these settings:
- Provider name: $bootstrap_provider
- Model: $bootstrap_model

Include this as a provider entry (connection info only) and create a model entry \
referencing it in the generated config. Use ``models.default`` to set it as active.

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
- Never include actual API keys in the output \u2014 use placeholder values like \
"your-api-key-here" and tell the user to replace them.
- If editing an existing config, preserve settings the user does not want to change.
- Providers should contain only connection info (type, base_url, api_key). \
Model settings (model name, temperature, context_window, max_tokens) go in the models section.
- Use ``models.default: <alias>`` to set the active model.
- Do not use top-level ``provider`` or ``model`` keys — those are deprecated.\
""")


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
    existing_config_raw = existing_yaml or "No existing configuration."
    system_prompt = _WIZARD_SYSTEM_PROMPT.substitute(
        docs=docs,
        existing_config=existing_config_raw,
        bootstrap_provider=bootstrap_info["provider"],
        bootstrap_model=bootstrap_info["model"],
    )

    # ── Step 2: LLM conversation ─────────────────────────────────
    _step(2, "Configure")
    print(f"  {_D('Type quit to cancel at any time.')}\n")
    final_response = _run_conversation(llm, system_prompt)

    # ── Step 3: validate and write ───────────────────────────────
    _step(3, "Save")
    yaml_content = _extract_yaml(final_response)
    _validate_and_write(yaml_content, bootstrap_info, output)

    print(f"\n  {_BG(chr(0x2713))} Config written to {_B(str(output))}")
    print(f"  {_D('Run')} python cogtrix.py {_D('to start Cogtrix.')}\n")


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
            "_detect_environment: skipping Ollama probe — URL resolves to a "
            "non-public address (BUG-229): %s",
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


def _list_ollama_models(base_url: str) -> list[str]:
    """Fetch and display installed Ollama models. Returns model names."""
    if not _is_safe_ollama_url(base_url):
        log.warning(
            "_list_ollama_models: skipping model fetch — URL resolves to a "
            "non-public address (BUG-230): %s",
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
            print(f"    {name} {_D(chr(0x00b7))} {_D(size_s)}")
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
    else:
        default_type = "openai"

    last: dict[str, Any] = {}  # carries entered values across retry iterations

    while True:
        selected_type = _ask_choice(
            "Provider type",
            choices=["ollama", "openai", "anthropic", "google", "xai"],
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

        else:  # xai
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
            base_url = "https://api.x.ai/v1"
            provider_type = "openai"
            default_model = last.get("model") or existing_info.get("model") or "grok-4.1-fast"
            model = _ask_input("Model", default=default_model)

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


def _is_safe_ollama_url(url: str) -> bool:
    """Return True if *url* is safe to probe as an Ollama endpoint.

    Like ``_is_safe_url`` but permits loopback addresses (127.x.x.x /
    ::1) so that a locally-running Ollama instance is always reachable.
    All other non-public ranges — RFC-1918 private (except loopback),
    link-local, reserved, and unspecified — are blocked to prevent SSRF
    via ``OLLAMA_BASE_URL`` or user-supplied wizard input (BUG-229/230).
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
            if ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
                return False
    except Exception:
        return False
    return True


def _is_safe_url(url: str) -> bool:
    """Return True if all resolved IP addresses for *url* are publicly routable.

    Blocks loopback, RFC-1918 private, link-local, reserved, and unspecified
    addresses to prevent SSRF attacks when the wizard fetches docs from a URL.
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
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False
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
            if not _is_safe_url(url):
                log.warning("Blocked potentially unsafe docs URL: %s (using embedded docs)", url)
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


def _run_conversation(llm: Any, system_prompt: str) -> str:
    """Phase 2: interactive LLM conversation loop.

    The LLM asks questions one at a time; the user responds. When the LLM
    produces a ```yaml``` block, the user is asked to confirm.

    Returns:
        The final LLM response containing the YAML configuration.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages: list[Any] = [SystemMessage(content=system_prompt)]

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
            messages.append(HumanMessage(content=user_input))
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

        messages.append(HumanMessage(content=user_input))
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


def _extract_yaml(text: str) -> str:
    """Extract YAML content from the last ```yaml``` code fence in *text*.

    Raises:
        ValueError: If no ```yaml``` block is found.
    """
    # Non-greedy first: handles clean cases without nested backticks
    pattern = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)
    iter_matches = list(pattern.finditer(text))
    if not iter_matches:
        # Greedy fallback: from first ```yaml to last ```
        greedy = re.compile(r"```yaml\s*\n(.*)```", re.DOTALL)
        greedy_matches = greedy.findall(text)
        if not greedy_matches:
            raise ValueError("No ```yaml``` block found in LLM response")
        return greedy_matches[-1].strip()
    last = iter_matches[-1]
    if "```" in text[last.end() :]:
        # Non-greedy terminated early at a nested ``` inside the block; use greedy
        # to span from the first ```yaml opener to the last ``` in the text.
        greedy = re.compile(r"```yaml\s*\n(.*)```", re.DOTALL)
        greedy_matches = greedy.findall(text)
        if greedy_matches:
            return greedy_matches[-1].strip()
    return last.group(1).strip()


# ── Phase 3: validate & write ────────────────────────────────────────


def _validate_and_write(
    yaml_content: str,
    bootstrap_info: dict[str, Any],
    output_path: Path,
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

    _inject_bootstrap(data, bootstrap_info)

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
    masked = _mask_secrets(final_yaml)
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


def _inject_bootstrap(data: dict[str, Any], bootstrap_info: dict[str, Any]) -> None:
    """Replace placeholder API keys with real values from bootstrap."""
    provider = bootstrap_info["provider"]
    api_key = bootstrap_info.get("api_key")
    model = bootstrap_info["model"]
    base_url = bootstrap_info.get("base_url")

    # Ensure provider entry has connection info only (no model fields)
    providers = data.setdefault("providers", {})
    provider_cfg = providers.setdefault(provider, {})
    provider_cfg["type"] = bootstrap_info["type"]
    if api_key:
        provider_cfg["api_key"] = api_key
    if base_url:
        provider_cfg["base_url"] = base_url
    # Remove any legacy model field from provider entry
    provider_cfg.pop("model", None)
    provider_cfg.pop("temperature", None)
    provider_cfg.pop("num_ctx", None)
    provider_cfg.pop("context_window", None)
    provider_cfg.pop("max_tokens", None)

    # Ensure a default model entry exists in the models registry
    models = data.setdefault("models", {})
    alias = "default_model"
    if alias not in models:
        models[alias] = {
            "provider": provider,
            "model": model,
        }
    models["default"] = alias

    # Remove legacy top-level fields
    data.pop("provider", None)
    data.pop("model", None)


def _mask_secrets(yaml_text: str) -> str:
    """Mask API keys and tokens in YAML text for display."""
    _SECRET_KEYS = r"api_key|api_secret|token|password|secret"
    # Inline values (plain, single-quoted, double-quoted) — trailing quote consumed
    result = re.sub(
        rf"({_SECRET_KEYS}):\s*[\"']?[^\s\"'#]+[\"']?",
        lambda m: f"{m.group(1)}: ***",
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
