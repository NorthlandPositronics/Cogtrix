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
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.tools.delegate import register_tool_categories

# Best-effort file locking for concurrent inbox access.
# fcntl is POSIX-only; on Windows we fall back to a threading lock (which at
# least guards the in-process case).
try:
    import fcntl as _fcntl

    def _lock_file(fd: int) -> None:
        _fcntl.flock(fd, _fcntl.LOCK_EX)

    def _unlock_file(fd: int) -> None:
        _fcntl.flock(fd, _fcntl.LOCK_UN)

except ImportError:  # Windows / non-POSIX
    _fcntl = None  # type: ignore[assignment]

    def _lock_file(fd: int) -> None:  # type: ignore[misc]
        pass

    def _unlock_file(fd: int) -> None:  # type: ignore[misc]
        pass


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

# In-process per-inbox locks so concurrent threads don't interleave writes.
# Combined with OS-level flock (see _lock_file) this guards both in-process
# and multi-process concurrent access to the same inbox file.
#
# LRU eviction: _inbox_locks is bounded to _INBOX_LOCKS_MAX_SIZE entries.
_INBOX_LOCKS_MAX_SIZE = 1024
_inbox_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_inbox_locks_meta = threading.Lock()


def _get_inbox_lock(agent_name: str) -> threading.Lock:
    """Return the in-process threading.Lock for *agent_name* (LRU-bounded)."""
    with _inbox_locks_meta:
        if agent_name in _inbox_locks:
            _inbox_locks.move_to_end(agent_name)
            return _inbox_locks[agent_name]
        while len(_inbox_locks) >= _INBOX_LOCKS_MAX_SIZE:
            _inbox_locks.popitem(last=False)
        lock = threading.Lock()
        _inbox_locks[agent_name] = lock
        return lock


def _locked_read_modify_write(path: Path, modify_fn):
    """Acquire in-process and OS-level locks, then read-modify-write *path*.

    *modify_fn* receives the current message list and should return the new
    message list.  The write uses tempfile + os.replace for atomicity.

    Returns the modified message list.
    """
    with _get_inbox_lock(path.stem):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        lock_fd = -1
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
            if lock_fd >= 0:
                _lock_file(lock_fd)
            messages = _prune(_load_messages(path))
            messages = modify_fn(messages)
            with atomic_write_json(path) as f:
                json.dump(messages, f, indent=2)
            return messages
        finally:
            if lock_fd >= 0:
                _unlock_file(lock_fd)
                try:
                    os.close(lock_fd)
                except OSError:
                    pass


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

    def _append(messages: list[dict]) -> list[dict]:
        messages.append(
            {
                "from_agent": from_agent,
                "message": message,
                "sent_at": time.time(),
                "read": False,
            }
        )
        return messages

    _locked_read_modify_write(path, _append)
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

    def _mark_read(messages: list[dict]) -> list[dict]:
        messages = _prune(messages)
        for msg in messages:
            msg["read"] = True
        return messages

    messages = _locked_read_modify_write(path, _mark_read)

    if not messages:
        # All messages expired; write back empty list to clean up the file
        with atomic_write_json(path) as f:
            json.dump([], f)
        return "Inbox empty."

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
        "category": "messaging",
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
        "category": "privacy",
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]


register_tool_categories({"send_to_agent": "messaging", "read_agent_inbox": "privacy"})

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
