"""Agent-to-agent communication tools — send and read messages between agents.

Tools:
    send_to_agent    — append a message to another agent's inbox
    read_agent_inbox — read and acknowledge messages from an inbox

Configuration:
    TOOL_SETUP(config) is called automatically by ToolRegistry after this
    module is loaded.  It sets _data_dir from config.data_dir.
    Do not add configure_messaging_tools() to configure.py or cogtrix.py.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

from src.utils.atomic_write import atomic_write_json

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("cogtrix.tools.agent_messaging")

# ── Module-level state (set by TOOL_SETUP) ────────────────────────────────────

_data_dir: Path = Path("data")

_TTL_SECONDS: float = 86400.0  # 24 hours
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ── Configuration ─────────────────────────────────────────────────────────────


def configure_messaging_tools(config: Config, data_dir: Path | None = None) -> None:
    """Wire data_dir from *config*."""
    global _data_dir
    if data_dir is not None:
        _data_dir = data_dir
    elif hasattr(config, "data_dir"):
        _data_dir = Path(config.data_dir)


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    configure_messaging_tools(
        config,
        data_dir=Path(config.data_dir) if hasattr(config, "data_dir") else None,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _validate_agent_name(agent_name: str) -> str | None:
    """Return an error string if *agent_name* is invalid, else None."""
    if not _NAME_RE.match(agent_name):
        return (
            f"Invalid agent_name {agent_name!r}. "
            "Must contain only letters, digits, underscores, or hyphens (1–64 characters)."
        )
    return None


def _inbox_path(agent_name: str) -> Path | str:
    """Return the resolved inbox Path, or an error string if it escapes data_dir."""
    path = (_data_dir / "tasks" / "inbox" / f"{agent_name}.json").resolve()
    if not path.is_relative_to(_data_dir.resolve()):
        return f"Path traversal detected for agent_name {agent_name!r}."
    return path


def _load_messages(path: Path) -> list[dict]:
    """Load messages from *path*; return [] if the file is absent or malformed."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning("Failed to load inbox %s: %s", path, exc)
    return []


def _prune(messages: list[dict]) -> list[dict]:
    """Return only messages younger than _TTL_SECONDS."""
    cutoff = time.time() - _TTL_SECONDS
    return [
        m for m in messages if isinstance(m.get("sent_at"), (int, float)) and m["sent_at"] >= cutoff
    ]


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ── Input schemas ─────────────────────────────────────────────────────────────


class SendToAgentInput(BaseModel):
    agent_name: str = Field(
        ...,
        description="Name of the destination agent (letters, digits, underscores, hyphens; 1–64 chars).",
    )
    message: str = Field(..., description="Message text to deliver to the agent.")
    from_agent: str = Field(
        default="",
        description="Name of the sending agent. Leave empty to send anonymously.",
    )


class ReadAgentInboxInput(BaseModel):
    agent_name: str = Field(
        default="",
        description="Name of the agent whose inbox to read (1–64 chars). Required.",
    )


# ── Tool functions ────────────────────────────────────────────────────────────


def send_to_agent(agent_name: str, message: str, from_agent: str = "") -> str:
    """Append a message to the named agent's inbox."""
    err = _validate_agent_name(agent_name)
    if err:
        return err

    path_or_err = _inbox_path(agent_name)
    if isinstance(path_or_err, str):
        return path_or_err
    path: Path = path_or_err

    messages = _prune(_load_messages(path))
    messages.append(
        {
            "from_agent": from_agent,
            "message": message,
            "sent_at": time.time(),
            "read": False,
        }
    )

    with atomic_write_json(path) as f:
        json.dump(messages, f, indent=2)

    return f"Message sent to agent '{agent_name}'"


def read_agent_inbox(agent_name: str = "") -> str:
    """Read and acknowledge all messages in the named agent's inbox."""
    if not agent_name:
        return "agent_name is required"

    err = _validate_agent_name(agent_name)
    if err:
        return err

    path_or_err = _inbox_path(agent_name)
    if isinstance(path_or_err, str):
        return path_or_err
    path: Path = path_or_err

    if not path.exists():
        return "Inbox empty."

    messages = _prune(_load_messages(path))

    if not messages:
        # All messages expired; write back empty list to clean up the file
        with atomic_write_json(path) as f:
            json.dump([], f)
        return "Inbox empty."

    # Mark all as read and persist
    for msg in messages:
        msg["read"] = True

    with atomic_write_json(path) as f:
        json.dump(messages, f, indent=2)

    # Format output
    lines: list[str] = []
    for i, msg in enumerate(messages, start=1):
        read_label = "yes" if msg.get("read") else "no"
        ts = _fmt_ts(msg.get("sent_at", 0.0))
        sender = msg.get("from_agent") or "unknown"
        text = msg.get("message", "")
        lines.append(f"[{i}] From: {sender} | {ts} | READ: {read_label}\n    {text}")

    return "\n".join(lines)


# ── Tool registry entries ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "send_to_agent",
        "description": (
            "Send a message to another agent's inbox. "
            "The message is stored persistently and can be read by the recipient agent "
            "using read_agent_inbox. Messages expire after 24 hours."
        ),
        "input_schema": SendToAgentInput,
        "requires_confirmation": False,
        "function": send_to_agent,
    },
    {
        "name": "read_agent_inbox",
        "description": (
            "Read all pending messages from an agent's inbox. "
            "Messages are marked as read after retrieval. "
            "Expired messages (older than 24 hours) are automatically removed."
        ),
        "input_schema": ReadAgentInboxInput,
        "requires_confirmation": False,
        "function": read_agent_inbox,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "TOOL_SETUP",
    "configure_messaging_tools",
    "send_to_agent",
    "read_agent_inbox",
    "SendToAgentInput",
    "ReadAgentInboxInput",
    "TOOL_CONFIGS",
    "TOOL_CONFIG",
]
