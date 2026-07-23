from typing import Any

from src._version import __copyright__, __version__

try:
    from rich import box as rich_box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    _console: Any = Console()
except ImportError:
    rich_box = None  # type: ignore[assignment]
    Align = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    _console = None


LOGO_LINES = [
    "░█▀▀░█▀█░█▀▀░▀█▀░█▀▄░▀█▀░█░█",
    "░█░░░█░█░█░█░░█░░█▀▄░░█░░▄▀▄",
    "░▀▀▀░▀▀▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀░▀",
]


def print_startup(config: Any, **extra: Any) -> None:
    """Print the startup banner with configuration summary.

    Uses Rich for a compact, well-formatted display when available,
    with a plain-text fallback.

    Optional keyword arguments (forwarded to renderers):
        tools_text, session_id, msg_count, no_confirm, confirm_count
    """
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
    if _console is None or Align is None or Group is None or Text is None:  # pragma: no cover
        return

    # Provider / model line
    prov_cfg = None
    try:
        prov_cfg = config.get_provider_config()
    except ValueError:
        pass

    model = config.model or (prov_cfg.get_model() if prov_cfg else "?")
    prov_type = prov_cfg.type if prov_cfg else "?"

    # ── Build renderables ─────────────────────────────────────
    parts: list = []

    # Centered ASCII art logo
    logo_text = Text("\n".join(LOGO_LINES), style="bright_blue")
    parts.append(Align.center(logo_text))
    parts.append(Text())  # blank line

    # Left-aligned config section (with leading indent)
    lbl = 12  # label column width
    info = Text()
    info.append(f"    {'Provider':<{lbl}}: ", style="bold")
    info.append(f"{config.provider} ")
    info.append(f"({prov_type})", style="dim")
    info.append(f"\n    {'Model':<{lbl}}: {model}", style="bold")
    # re-apply: the bold covered the whole line; build per-line instead
    info = Text()
    info.append(f"    {'Provider':<{lbl}}", style="bold")
    info.append(f": {config.provider} ")
    info.append(f"({prov_type})\n", style="dim")
    info.append(f"    {'Model':<{lbl}}", style="bold")
    info.append(f": {model}\n")
    info.append(f"    {'Mode':<{lbl}}", style="bold")
    info.append(f": {config.memory_mode}\n")
    if config.config_file_path:
        info.append(f"    {'Config':<{lbl}}", style="bold")
        info.append(f": {config.config_file_path}\n", style="dim")

    # Tools & session (bar = configured / total registered)
    tools_text = extra.get("tools_text")
    configured_count = extra.get("configured_count", 0)
    total_registered = extra.get("total_registered", 0)
    session_id = extra.get("session_id")
    if tools_text:
        bar_len = 12
        if tools_text == "disabled":
            filled = 0
        elif total_registered > 0:
            filled = max(1, round(configured_count / total_registered * bar_len))
        else:
            filled = bar_len if configured_count > 0 else 0
        bar_str = "█" * filled + "░" * (bar_len - filled)
        info.append(f"    {'Tools':<{lbl}}", style="bold")
        info.append(f": [{bar_str}] {tools_text}\n")
    if session_id is not None:
        msg_count = extra.get("msg_count", 0)
        info.append(f"    {'Session':<{lbl}}", style="bold")
        info.append(f": {session_id} ")
        info.append(f"({msg_count} messages)\n", style="dim")
    if extra.get("no_confirm") and extra.get("confirm_count"):
        info.append(
            f"    ⚡ Safety confirmations disabled for {extra['confirm_count']} tool(s)\n",
            style="yellow",
        )

    # Strip trailing newline
    info.rstrip()
    parts.append(info)

    # Centered copyright at the bottom
    parts.append(Text())  # blank line
    copyright_text = Text(f"{__copyright__} · v{__version__}", style="dim")
    parts.append(Align.center(copyright_text))

    body = Group(*parts)
    _console.print()
    _console.print(
        Panel(
            body,
            box=rich_box.DOUBLE if rich_box else None,  # type: ignore[arg-type]
            border_style="bright_blue",
            expand=False,
            padding=(1, 2),
        )
    )


def startup_plain(config: Any, **extra: Any) -> None:
    """Render startup info as plain text."""
    print()
    for logo_line in LOGO_LINES:
        print(f"  {logo_line}")
    print()

    prov_cfg = None
    try:
        prov_cfg = config.get_provider_config()
    except ValueError:
        pass

    model = config.model or (prov_cfg.get_model() if prov_cfg else "?")
    prov_type = prov_cfg.type if prov_cfg else "?"

    lbl = 12
    print(f"  {'Provider':<{lbl}}: {config.provider} ({prov_type})")
    print(f"  {'Model':<{lbl}}: {model}")
    print(f"  {'Mode':<{lbl}}: {config.memory_mode}")
    if config.config_file_path:
        print(f"  {'Config':<{lbl}}: {config.config_file_path}")

    tools_text = extra.get("tools_text")
    configured_count = extra.get("configured_count", 0)
    total_registered = extra.get("total_registered", 0)
    session_id = extra.get("session_id")
    if tools_text:
        bar_len = 12
        if tools_text == "disabled":
            filled = 0
        elif total_registered > 0:
            filled = max(1, round(configured_count / total_registered * bar_len))
        else:
            filled = bar_len if configured_count > 0 else 0
        bar_str = "█" * filled + "░" * (bar_len - filled)
        print(f"  {'Tools':<{lbl}}: [{bar_str}] {tools_text}")
    if session_id is not None:
        msg_count = extra.get("msg_count", 0)
        print(f"  {'Session':<{lbl}}: {session_id} ({msg_count} messages)")
    if extra.get("no_confirm") and extra.get("confirm_count"):
        print(f"  ⚡ Safety confirmations disabled for {extra['confirm_count']} tool(s)")
    print()
    print(f"  {__copyright__} · v{__version__}")
    print()
