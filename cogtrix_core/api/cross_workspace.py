"""Cross-workspace agent communication (Enterprise Phase 1 — task 1.3.3).

Allows agents and sessions in one workspace to send structured messages to
agents in another workspace within the **same organization**.

Rules
-----
- Only workspaces in the same org may communicate (``CROSS_ORG_BLOCKED``).
- Both source and target workspaces must be active.
- Communication may be restricted to an allowlist (``CrossWorkspacePolicy``).
- The message is written to the target workspace's inbox store
  (``/data/cross_workspace/{to_workspace_id}/<id>.json``).

The inbox is intentionally file-based for simplicity; a later task can
migrate it to a DB table or a message queue when throughput requires it.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("cogtrix.api.cross_workspace")

# ---------------------------------------------------------------------------
# UUID v4 validation — explicit sanitiser for CodeQL CWE-22
# ---------------------------------------------------------------------------

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _validate_uuid4(value: str, name: str) -> None:
    """Raise ValueError if *value* is not a UUID v4.

    This explicit check is the primary CodeQL CWE-22 sanitiser — it runs before
    any path construction and is recognised by static analysers as a taint-kill.
    """
    if not _UUID4_RE.match(value):
        raise ValueError(f"Invalid {name} — expected UUID v4: {value!r}")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class CrossWorkspacePolicy:
    """Governs which workspaces may communicate.

    Attributes:
        enabled:         Master switch.
        allowed_pairs:   Explicit (from_id, to_id) allowlist.  Empty = all
                         pairs within the org are permitted (when enabled).
    """

    enabled: bool = True
    allowed_pairs: list[tuple[str, str]] = field(default_factory=list)

    def is_allowed(self, from_workspace_id: str, to_workspace_id: str) -> bool:
        if not self.enabled:
            return False
        if not self.allowed_pairs:
            return True  # open within org
        return (from_workspace_id, to_workspace_id) in self.allowed_pairs


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class CrossWorkspaceMessage:
    """A message sent from one workspace to another.

    Attributes:
        id:                  UUID v4.
        from_workspace_id:   Source workspace.
        to_workspace_id:     Destination workspace.
        sender_user_id:      User who triggered the message.
        subject:             Short subject line (≤ 128 chars).
        body:                Arbitrary JSON-serialisable payload.
        sent_at:             UTC creation timestamp.
    """

    from_workspace_id: str
    to_workspace_id: str
    sender_user_id: str
    subject: str
    body: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sent_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_workspace_id": self.from_workspace_id,
            "to_workspace_id": self.to_workspace_id,
            "sender_user_id": self.sender_user_id,
            "subject": self.subject,
            "body": self.body,
            "sent_at": self.sent_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Inbox helpers
# ---------------------------------------------------------------------------


def _inbox_dir(data_root: Path, workspace_id: str) -> Path:
    """Return the inbox directory for *workspace_id*, validated against traversal.

    Raises ValueError when the resolved path escapes the cross_workspace root,
    which would indicate a path-traversal attempt via a crafted workspace_id.
    """
    _validate_uuid4(workspace_id, "workspace_id")
    base = data_root / "cross_workspace"
    candidate = (base / workspace_id).resolve()
    # Ensure the resolved path stays inside base (handles ../ sequences)
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        raise ValueError(
            f"Invalid workspace_id — path traversal detected: {workspace_id!r}"
        ) from None
    return candidate


def write_to_inbox(
    message: CrossWorkspaceMessage,
    *,
    data_root: Path | None = None,
) -> Path:
    """Write *message* as a JSON file to the target workspace's inbox.

    Args:
        message:   The message to deliver.
        data_root: Override for the data directory (defaults to ``/data``).

    Returns:
        Path to the written message file.
    """
    import os

    root = data_root or Path(os.environ.get("COGTRIX_DATA_DIR", "/data"))
    inbox = _inbox_dir(root, message.to_workspace_id)
    inbox.mkdir(parents=True, exist_ok=True)
    dest = inbox / f"{message.id}.json"
    dest.write_text(json.dumps(message.to_dict(), indent=2))
    log.info(
        "cross-workspace: %s → %s msg=%s subject=%r",
        message.from_workspace_id,
        message.to_workspace_id,
        message.id,
        message.subject,
    )
    return dest


def read_inbox(
    workspace_id: str,
    *,
    data_root: Path | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return up to *limit* messages from the workspace inbox, newest first."""
    import os

    root = data_root or Path(os.environ.get("COGTRIX_DATA_DIR", "/data"))
    inbox = _inbox_dir(root, workspace_id)
    if not inbox.exists():
        return []
    files = sorted(inbox.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    messages = []
    for f in files[:limit]:
        try:
            messages.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            pass
    return messages


def delete_message(
    workspace_id: str,
    message_id: str,
    *,
    data_root: Path | None = None,
) -> bool:
    """Delete a message from the inbox.  Returns True if the file existed."""
    import os

    _validate_uuid4(message_id, "message_id")
    root = data_root or Path(os.environ.get("COGTRIX_DATA_DIR", "/data"))
    inbox = _inbox_dir(root, workspace_id)
    msg_file = inbox / f"{message_id}.json"
    try:
        msg_file.resolve().relative_to(inbox.resolve())
    except ValueError:
        raise ValueError(f"Invalid message_id — path traversal detected: {message_id!r}") from None
    if msg_file.exists():
        msg_file.unlink()
        return True
    return False
