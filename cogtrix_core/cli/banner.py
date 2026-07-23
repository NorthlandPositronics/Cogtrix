import logging
from typing import Any

from cogtrix_core._version import get_version_string

log = logging.getLogger("cogtrix")

try:
    from rich.console import Console

    _console: Any = Console()
except ImportError:
    _console = None


def startup_compact(config: Any, **extra: Any) -> None:
    """Render a single-line compact startup indicator."""
    try:
        mc = config.get_active_model()
        model_str = mc.model
        provider_str = mc.provider
    except Exception:
        model_str = "?"
        provider_str = "?"

    session_id = extra.get("session_id", getattr(config, "session", "default"))
    msg_count = extra.get("msg_count", 0)
    has_embeddings = extra.get("has_embeddings", False)

    if msg_count == 0:
        status = "new"
    elif has_embeddings:
        status = f"{msg_count} msgs + recall"
    else:
        status = f"{msg_count} msgs"

    ver = f"[bold steel_blue1]v{get_version_string()}[/bold steel_blue1]"
    status_styled = f"[bold steel_blue1]{status}[/bold steel_blue1]"
    base = f"cogtrix v{get_version_string()} · {provider_str}/{model_str} · {session_id} ({status})"
    rich_base = (
        f"[dim]cogtrix {ver} · {provider_str}/{model_str} · {session_id} ({status_styled})[/dim]"
    )

    if _console is not None:
        try:
            if msg_count == 0:
                _console.print(f"{rich_base}  [dim]/help for commands[/dim]", highlight=False)
            else:
                _console.print(rich_base, highlight=False)
        except Exception:
            print(base)
    else:
        print(base)


def print_startup(config: Any, **extra: Any) -> None:
    """Print the startup banner with configuration summary.

    Uses Rich for a compact, well-formatted display when available,
    with a plain-text fallback.

    Optional keyword arguments (forwarded to renderers):
        tools_text, session_id, msg_count, no_confirm, confirm_count
    """
    banner_mode = getattr(config, "banner", "compact")
    if banner_mode == "off":
        return
    if banner_mode == "compact":
        startup_compact(config, **extra)
        return
    if _console is not None:
        startup_rich(config, **extra)
    else:
        startup_plain(config, **extra)


def startup_rich(config: Any, **extra: Any) -> None:
    """Render startup info using Rich.

    Optional keyword arguments (passed after tool/session init):
        tools_text:    e.g. "12 active (+23 on request)"
        session_id:    e.g. "default"
        msg_count:     e.g. 0
        no_confirm:    True if safety confirmations are disabled
        confirm_count: number of confirm-gated tools
    """
    if _console is None:
        startup_plain(config, **extra)
        return

    alias = getattr(config, "active_model_alias", None)
    try:
        mc = config.get_active_model()
        model_str = mc.model
        provider_str = mc.provider
    except Exception as exc:
        log.debug("Failed to resolve active model for banner: %s", exc)
        model_str = "?"
        provider_str = "?"

    memory_mode = getattr(config, "memory_mode", "conversation")
    session_id = extra.get("session_id", getattr(config, "session", "default"))
    msg_count = extra.get("msg_count", 0)
    has_embeddings = extra.get("has_embeddings", False)
    no_confirm = extra.get("no_confirm", False)
    tools_text = extra.get("tools_text")

    try:
        _console.rule(
            f"[bold steel_blue1]cogtrix v{get_version_string()}[/bold steel_blue1]",
            style="steel_blue1",
        )
        _console.print()

        # Primary info line
        if alias and alias != model_str:
            model_part = f"[bold]{alias}[/bold] [dim]({model_str})[/dim]"
        else:
            model_part = f"[bold]{model_str}[/bold]"
        provider_part = f"via [cyan]{provider_str}[/cyan]"
        mode_part = f"[dim]{memory_mode} memory[/dim]"

        info_parts = [f"  {model_part} {provider_part}", mode_part]
        if tools_text:
            info_parts.append(f"[dim]{tools_text}[/dim]")
        _console.print("  ·  ".join(info_parts))

        # Session line — only when non-trivial
        if session_id and (session_id != "default" or msg_count > 0):
            if msg_count == 0:
                sess_text = f"  [dim]session: {session_id} · new[/dim]"
            elif has_embeddings:
                sess_text = (
                    f"  [dim]session: {session_id} · resumed · {msg_count} msgs + recall[/dim]"
                )
            else:
                sess_text = f"  [dim]session: {session_id} · resumed · {msg_count} msgs[/dim]"
            _console.print(sess_text)

        if getattr(config, "config_file_path", None):
            _console.print(f"  [dim]config: {config.config_file_path}[/dim]")

        if no_confirm:
            _console.print("  [yellow]⚠ Tool confirmations disabled (auto-approved)[/yellow]")

        _console.print()

        if msg_count == 0:
            _console.print(
                "  [dim]Type [bold white]/help[/bold white] for commands  "
                "· [bold white]!cmd[/bold white] to run shell  "
                "· [bold white]/setup[/bold white] to reconfigure[/dim]"
            )
            _console.print()

        _console.rule(style="dim steel_blue1")
        _console.print()
    except Exception:  # pragma: no cover
        startup_plain(config, **extra)


def startup_plain(config: Any, **extra: Any) -> None:
    """Plain-text startup banner (no Rich/ANSI)."""
    try:
        mc = config.get_active_model()
        model_str = f"{mc.provider}/{mc.model}"
    except Exception:
        model_str = "unknown"
    session_id = extra.get("session_id", getattr(config, "session", "default"))
    msg_count = extra.get("msg_count", 0)
    version_line = f"cogtrix v{get_version_string()} · {model_str} · session: {session_id}"
    print("-" * len(version_line))
    print(version_line)
    print("-" * len(version_line))
    if msg_count == 0:
        print("Type /help for commands.")
    print()
