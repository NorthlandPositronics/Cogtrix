"""
Scheduled reply delivery for Cogtrix assistant mode.

Provides MessageScheduler, which queues agent-generated replies for deferred
delivery via a background daemon thread, plus the ScheduleReplyState/tool
factory used by MessageHandler to capture per-call scheduling intent.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool as _StructuredToolType  # noqa: F401

    from src.assistant.channel import Channel

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc, assignment]

log = logging.getLogger("cogtrix")

_BACKOFF_SECONDS: tuple[int, ...] = (30, 120, 600)
_MIN_DISPATCH_INTERVAL: float = 1.0
_STALE_THRESHOLD: float = 2 * 3600.0  # 2 hours
_CLEANUP_AGE: float = 24 * 3600.0  # 24 hours
_MAX_QUEUE_ITEMS_PER_TURN: int = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScheduledMessage:
    id: str
    channel: str
    chat_id: str
    text: str
    send_at: float  # wall-clock time.time()
    created_at: float  # wall-clock time.time()
    recipient: str | None = None  # human-readable: phone, username, or name
    status: str = "pending"  # pending | sending | sent | cancelled | failed | expired
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledMessage:
        coerced = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for float_field in ("send_at", "created_at"):
            if float_field in coerced:
                coerced[float_field] = float(coerced[float_field])
        for int_field in ("attempts", "max_attempts"):
            if int_field in coerced:
                coerced[int_field] = int(coerced[int_field])
        return cls(**coerced)


@dataclass
class ScheduleReplyState:
    """Mutable per-call state set by the schedule_reply tool closure."""

    was_called: bool = False
    scheduled_text: str = ""
    delay_minutes: int = 0


@dataclass
class EditReplyState:
    """Mutable per-call state set by the edit_last_reply tool closure."""

    was_called: bool = False
    new_text: str = ""


@dataclass
class QueueReplyState:
    """Per-call state for queue_reply. Supports multiple calls per turn."""

    @dataclass
    class Item:
        text: str
        gap_minutes: int

    items: list[Item] = field(default_factory=list)


@dataclass
class QuietHoursPolicy:
    start_hour: int  # e.g. 23
    end_hour: int  # e.g. 8
    timezone: str  # e.g. "Asia/Dubai"


# ---------------------------------------------------------------------------
# Pydantic schema (module-level, created once)
# ---------------------------------------------------------------------------


class ScheduleReplyInput(BaseModel):
    text: str = Field(description="The reply message to send later")
    delay_minutes: int = Field(
        description="Minutes to wait before sending (e.g., 180 for 3 hours)",
        ge=1,
        le=1440,
    )


class QueueReplyInput(BaseModel):
    text: str = Field(description="The message text to deliver")
    gap_minutes: int = Field(
        default=0,
        description=(
            "Extra delay in minutes after the last pending message for this contact. "
            "0 means deliver immediately after the previous queued message. "
            "Use a positive value for spacing (e.g., 60 for an extra hour between messages)."
        ),
        ge=0,
        le=1440,
    )


class EditReplyInput(BaseModel):
    new_text: str = Field(description="The corrected/updated text to replace your last reply")


class ListScheduledInput(BaseModel):
    recipient: str = Field(
        default="",
        description=(
            "Filter by recipient phone number, username, or display name (substring match). "
            "Leave empty to skip this filter."
        ),
    )
    chat_id: str = Field(
        default="",
        description=(
            "Filter by exact conversation ID "
            "(e.g. '971503308667@c.us' for WhatsApp, '123456789' for Telegram). "
            "Leave empty to skip this filter."
        ),
    )
    contact_name: str = Field(
        default="",
        description=(
            "Filter by contact name from the phonebook configuration "
            "(e.g. 'shraddha', 'alice') or a sender ID / username. "
            "Resolves to phone/chat ID via phonebook, falls back to substring match. "
            "Leave empty to skip this filter."
        ),
    )


class EditScheduledInput(BaseModel):
    message_id: str = Field(
        min_length=1,
        description="ID of the scheduled message (from list_scheduled_messages)",
    )
    new_text: str | None = Field(
        default=None,
        description="Updated message text. Leave empty to keep the current text.",
    )
    reschedule_minutes: int | None = Field(
        default=None,
        ge=1,
        le=1440,
        description=(
            "Reschedule delivery to this many minutes from now. "
            "Leave empty to keep current time."
        ),
    )


class CancelScheduledInput(BaseModel):
    message_id: str = Field(
        min_length=1,
        description="ID of the scheduled message to cancel (from list_scheduled_messages)",
    )


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def create_schedule_reply_tool(state: ScheduleReplyState) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    When the agent calls this tool the closure mutates *state* in-place
    so the handler can route the reply after agent execution completes.
    This function does NOT enqueue directly.
    """

    _lock = threading.Lock()

    def _schedule_reply(text: str, delay_minutes: int) -> str:
        with _lock:
            if state.was_called:
                return (
                    f"Reply already scheduled for delivery in {state.delay_minutes} minute(s). "
                    "Only one scheduled reply is allowed per turn."
                )
            state.was_called = True
            state.scheduled_text = text
            state.delay_minutes = delay_minutes
        return (
            f"Reply scheduled for delivery in {delay_minutes} minute(s). "
            "Do not repeat the message — it will be sent automatically."
        )

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_schedule_reply,
        name="schedule_reply",
        description=(
            "Schedule your reply for delayed delivery instead of sending immediately. "
            "Use when your instructions require delayed or timed responses. "
            "Provide the full reply text and the delay in minutes (e.g., 180 for 3 hours). "
            "Your reply text will NOT be sent to the user directly — it will be "
            "delivered automatically after the specified delay."
        ),
        args_schema=ScheduleReplyInput,
    )


def create_edit_reply_tool(state: EditReplyState) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    When the agent calls this tool the closure mutates *state* in-place
    so the handler can route the edit after agent execution completes.
    """

    _lock = threading.Lock()

    def _edit_last_reply(new_text: str) -> str:
        with _lock:
            if state.was_called:
                return "Last reply already queued for update — only one edit per turn is allowed."
            state.was_called = True
            state.new_text = new_text
        return "Last reply will be updated with the new text."

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_edit_last_reply,
        name="edit_last_reply",
        description=(
            "Edit/replace your most recently sent reply in this chat. "
            "Use when you need to correct a mistake, update information, "
            "or improve your previous response. Provide the complete new text."
        ),
        args_schema=EditReplyInput,
    )


def create_queue_reply_tool(state: QueueReplyState) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    Unlike schedule_reply, this tool supports multiple calls per turn.
    Each call appends an item to *state.items*.
    """

    _lock = threading.Lock()

    def _queue_reply(text: str, gap_minutes: int = 0) -> str:
        with _lock:
            if len(state.items) >= _MAX_QUEUE_ITEMS_PER_TURN:
                return (
                    f"Queue limit reached: at most {_MAX_QUEUE_ITEMS_PER_TURN} messages "
                    "may be queued per turn."
                )
            position = len(state.items) + 1
            state.items.append(QueueReplyState.Item(text=text, gap_minutes=gap_minutes))
        gap_note = f" with a {gap_minutes}-minute gap" if gap_minutes else ""
        return (
            f"Message #{position} queued for delivery after the current queue tail{gap_note}. "
            "It will be sent automatically — do not repeat it."
        )

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_queue_reply,
        name="queue_reply",
        description=(
            "Add a reply to the end of the delivery queue for this contact. "
            "The message will be sent after all currently pending messages for this chat. "
            "Use gap_minutes to add extra delay after the previous queued message. "
            "Use this instead of schedule_reply when you want messages sent in order "
            "without computing absolute delay times."
        ),
        args_schema=QueueReplyInput,
    )


def _resolve_message_id(scheduler: MessageScheduler, short_id: str) -> str | None:
    """Resolve a short ID prefix to the full message UUID, or None if not found."""
    if not short_id:
        return None
    with scheduler._lock:
        for msg_id, msg in scheduler._queue.items():
            if msg_id.startswith(short_id) and msg.status == "pending":
                return msg_id
    return None


def _merge_phonebooks(services_config: dict[str, Any]) -> dict[str, list[str]]:
    """Build a flat contact_name -> [normalized_identifiers] map from all channel phonebooks."""
    merged: dict[str, list[str]] = {}
    for _channel_key, channel_cfg in services_config.items():
        if not isinstance(channel_cfg, dict):
            continue
        phonebook = channel_cfg.get("phonebook", {})
        if not isinstance(phonebook, dict):
            continue
        for name, identifier in phonebook.items():
            key = str(name).strip().lower()
            normalized = (
                str(identifier)
                .strip()
                .replace("+", "")
                .replace("@c.us", "")
                .replace("@s.whatsapp.net", "")
                .lower()
            )
            merged.setdefault(key, []).append(normalized)
    return merged


def create_list_scheduled_tool(
    scheduler: MessageScheduler,
    services_config: dict[str, Any] | None = None,
    caller_chat_id: str = "",
) -> StructuredTool:  # type: ignore[valid-type]
    """Return a tool that lists pending scheduled messages.

    ``caller_chat_id`` scopes the listing to the calling session's chat — callers
    see only their own messages unless they explicitly provide a ``chat_id`` filter.
    """
    import datetime

    _phonebook = _merge_phonebooks(services_config or {})

    def _list_scheduled(recipient: str = "", chat_id: str = "", contact_name: str = "") -> str:
        effective_recipient = recipient or None
        # Restrict to caller's own chat when no explicit chat_id filter is given (BUG-040)
        effective_chat_id = chat_id or caller_chat_id or None

        if contact_name:
            key = contact_name.strip().lower()
            identifiers = _phonebook.get(key)
            if identifiers:
                seen_ids: set[str] = set()
                msgs: list[ScheduledMessage] = []
                for ident in identifiers:
                    for m in scheduler.get_pending(recipient=ident, chat_id=effective_chat_id):
                        if m.id not in seen_ids:
                            seen_ids.add(m.id)
                            msgs.append(m)
                if effective_recipient:
                    needle = effective_recipient.lower().replace("+", "").replace("@c.us", "")
                    msgs = [
                        m
                        for m in msgs
                        if needle in (m.recipient or "").lower().replace("+", "")
                        or needle in m.chat_id.lower().replace("@c.us", "").replace("@lid", "")
                    ]
                msgs.sort(key=lambda m: m.send_at)
            else:
                # No phonebook hit: treat contact_name as a recipient substring
                msgs = scheduler.get_pending(recipient=contact_name, chat_id=effective_chat_id)
                # Apply additional recipient filter for AND semantics
                if effective_recipient and msgs:
                    needle = effective_recipient.lower().replace("+", "").replace("@c.us", "")
                    msgs = [
                        m
                        for m in msgs
                        if needle in (m.recipient or "").lower().replace("+", "")
                        or needle in m.chat_id.lower().replace("@c.us", "").replace("@lid", "")
                    ]
        else:
            msgs = scheduler.get_pending(recipient=effective_recipient, chat_id=effective_chat_id)

        if not msgs:
            parts = []
            if recipient:
                parts.append(f"recipient '{recipient}'")
            if chat_id:
                parts.append(f"chat '{chat_id}'")
            if contact_name:
                parts.append(f"contact '{contact_name}'")
            who = " for " + " and ".join(parts) if parts else ""
            return f"No pending scheduled messages{who}."

        lines = [f"{len(msgs)} pending scheduled message(s):\n"]
        now = time.time()
        for i, msg in enumerate(msgs, 1):
            mins_left = max(0, int((msg.send_at - now) / 60))
            dt = datetime.datetime.fromtimestamp(msg.send_at, tz=datetime.UTC)
            time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            who = msg.recipient or msg.chat_id
            preview = msg.text[:80] + ("..." if len(msg.text) > 80 else "")
            lines.append(
                f"{i}. [ID: {msg.id[:8]}] To: {who}\n"
                f'   Text: "{preview}"\n'
                f"   Delivery: in {mins_left} min ({time_str})"
            )
        return "\n".join(lines)

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_list_scheduled,
        name="list_scheduled_messages",
        description=(
            "List all messages currently queued for scheduled delivery. "
            "Filter by recipient (phone/name), chat_id (exact conversation ID), "
            "or contact_name (phonebook key like 'shraddha'). "
            "All filters are optional and combined with AND logic. "
            "Returns message IDs needed for edit_scheduled_message and cancel_scheduled_message."
        ),
        args_schema=ListScheduledInput,
    )


def create_edit_scheduled_tool(
    scheduler: MessageScheduler,
    caller_chat_id: str = "",
) -> StructuredTool:  # type: ignore[valid-type]
    """Return a tool that edits a pending scheduled message.

    ``caller_chat_id`` enforces per-session authorization — callers may only edit
    their own messages (BUG-041).
    """

    def _edit_scheduled(
        message_id: str, new_text: str | None = None, reschedule_minutes: int | None = None
    ) -> str:
        full_id = _resolve_message_id(scheduler, message_id)
        if full_id is None:
            return f"No pending message found with ID starting with '{message_id}'."
        # Authorization: callers may only edit messages belonging to their own chat (BUG-041)
        if caller_chat_id:
            with scheduler._lock:
                msg_obj = scheduler._queue.get(full_id)
            if msg_obj and msg_obj.chat_id != caller_chat_id:
                return f"No pending message found with ID starting with '{message_id}'."
        new_send_at = time.time() + reschedule_minutes * 60 if reschedule_minutes else None
        ok = scheduler.edit_message(full_id, new_text=new_text, new_send_at=new_send_at)
        if not ok:
            return (
                f"Could not edit message {message_id} — "
                "it may have already been sent or cancelled."
            )
        parts = []
        if new_text is not None:
            parts.append("text updated")
        if reschedule_minutes is not None:
            parts.append(f"rescheduled to {reschedule_minutes} minutes from now")
        return f"Message {message_id} {' and '.join(parts)}."

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_edit_scheduled,
        name="edit_scheduled_message",
        description=(
            "Edit a queued message's text and/or delivery time. "
            "Use the message ID from list_scheduled_messages. "
            "You can update the text, reschedule the delivery, or both."
        ),
        args_schema=EditScheduledInput,
    )


def create_cancel_scheduled_tool(
    scheduler: MessageScheduler,
    caller_chat_id: str = "",
) -> StructuredTool:  # type: ignore[valid-type]
    """Return a tool that cancels a specific pending scheduled message.

    ``caller_chat_id`` enforces per-session authorization — callers may only cancel
    their own messages (BUG-041).
    """

    def _cancel_scheduled(message_id: str) -> str:
        full_id = _resolve_message_id(scheduler, message_id)
        if full_id is None:
            return f"No pending message found with ID starting with '{message_id}'."
        # Authorization: callers may only cancel messages belonging to their own chat (BUG-041)
        if caller_chat_id:
            with scheduler._lock:
                msg_obj = scheduler._queue.get(full_id)
            if msg_obj and msg_obj.chat_id != caller_chat_id:
                return f"No pending message found with ID starting with '{message_id}'."
        ok = scheduler.cancel_message(full_id)
        if not ok:
            return (
                f"Could not cancel message {message_id} — "
                "it may have already been sent or cancelled."
            )
        return f"Message {message_id} cancelled. It will not be delivered."

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_cancel_scheduled,
        name="cancel_scheduled_message",
        description=(
            "Cancel a specific scheduled message so it will not be delivered. "
            "Use the message ID from list_scheduled_messages."
        ),
        args_schema=CancelScheduledInput,
    )


# ---------------------------------------------------------------------------
# MessageScheduler
# ---------------------------------------------------------------------------


def _parse_quiet_hours(cfg: dict[str, Any]) -> QuietHoursPolicy | None:
    """Parse a single contact/default timing config dict into a policy."""
    qh = cfg.get("quiet_hours")
    tz = cfg.get("timezone", "UTC")
    if not qh or not isinstance(qh, (list, tuple)) or len(qh) != 2:
        return None
    try:
        s, e = int(qh[0]), int(qh[1])
        if not (0 <= s <= 23 and 0 <= e <= 23) or s == e:
            log.warning(
                "Invalid quiet_hours [%s, %s] — must be 0-23 and start != end", qh[0], qh[1]
            )
            return None
        return QuietHoursPolicy(start_hour=s, end_hour=e, timezone=str(tz))
    except (TypeError, ValueError):
        return None


def _is_in_quiet_window(policy: QuietHoursPolicy, wall_time: float) -> bool:
    """Return True if *wall_time* falls within the quiet window."""
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(policy.timezone)
    except Exception:
        import datetime

        tz = datetime.UTC  # type: ignore[assignment]

    import datetime

    dt = datetime.datetime.fromtimestamp(wall_time, tz=tz)
    hour = dt.hour
    start = policy.start_hour
    end = policy.end_hour

    if start == end:
        return False

    if start < end:
        # e.g. 9..17
        return start <= hour < end
    else:
        # wraps midnight, e.g. 23..8
        return hour >= start or hour < end


def _next_quiet_end(policy: QuietHoursPolicy, wall_time: float) -> float:
    """Return the wall-clock timestamp when the quiet window ends."""
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(policy.timezone)
    except Exception:
        import datetime

        tz = datetime.UTC  # type: ignore[assignment]

    import datetime

    dt = datetime.datetime.fromtimestamp(wall_time, tz=tz)
    end_today = dt.replace(hour=policy.end_hour, minute=0, second=0, microsecond=0)
    if end_today <= dt:
        end_today += datetime.timedelta(days=1)
    return end_today.timestamp()


class MessageScheduler:
    """Background scheduler that delivers queued replies via messaging channels.

    Args:
        channels: Mapping of channel name to Channel instance.
        persist_path: Path for the JSON persistence file.
        quiet_hours_cfg: Dict from ``services.<channel>.response_timing``
            (per-contact overrides keyed by contact name; ``_default`` key
            applies to all contacts without a specific entry).
    """

    def __init__(
        self,
        channels: dict[str, Channel],
        persist_path: Path,
        quiet_hours_cfg: dict[str, Any] | None = None,
        dispatch_interval: float = 30.0,
    ) -> None:
        self._channels = channels
        self._persist_path = persist_path
        self._quiet_cfg: dict[str, Any] = quiet_hours_cfg or {}
        self._dispatch_interval = dispatch_interval
        self._queue: dict[str, ScheduledMessage] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._load()
        self._expire_stale()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        channel: str,
        chat_id: str,
        text: str,
        send_at: float,
        recipient: str | None = None,
    ) -> str:
        """Add a message to the queue and persist. Returns the message ID."""
        msg = ScheduledMessage(
            id=str(uuid.uuid4()),
            channel=channel,
            chat_id=chat_id,
            text=text,
            send_at=send_at,
            created_at=time.time(),
            recipient=recipient,
        )
        with self._lock:
            self._queue[msg.id] = msg
        self.save()
        log.debug("Scheduled message %s for %s@%s at %.0f", msg.id, chat_id, channel, send_at)
        return msg.id

    def queue_after_tail(
        self,
        channel: str,
        chat_id: str,
        text: str,
        gap_seconds: float = 0.0,
        recipient: str | None = None,
        *,
        persist: bool = True,
    ) -> str:
        """Atomically find the queue tail for *chat_id* and insert after it.

        If no pending messages exist for this chat, the base time is
        ``time.time()``.  Pass ``persist=False`` to skip the disk write
        when batching multiple inserts (caller must call ``save()``
        afterward).  Returns the message ID.
        """
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            tail_time = now
            for m in self._queue.values():
                if (
                    m.channel == channel
                    and m.chat_id == chat_id
                    and m.status == "pending"
                    and m.send_at >= tail_time
                ):
                    tail_time = m.send_at
            send_at = tail_time + gap_seconds
            msg = ScheduledMessage(
                id=msg_id,
                channel=channel,
                chat_id=chat_id,
                text=text,
                send_at=send_at,
                created_at=now,
                recipient=recipient,
            )
            self._queue[msg_id] = msg
        if persist:
            self.save()
        log.debug(
            "Queued message %s after tail (send_at=%.0f) for %s@%s",
            msg_id,
            send_at,
            chat_id,
            channel,
        )
        return msg_id

    def cancel_pending(self, channel: str, chat_id: str) -> int:
        """Cancel all pending or in-flight messages for the given chat. Returns the count cancelled."""
        cancelled = 0
        with self._lock:
            for msg in self._queue.values():
                if (
                    msg.channel == channel
                    and msg.chat_id == chat_id
                    and msg.status in ("pending", "sending")
                ):
                    msg.status = "cancelled"
                    cancelled += 1
        if cancelled:
            self.save()
        return cancelled

    def get_pending(
        self,
        recipient: str | None = None,
        chat_id: str | None = None,
        include_all: bool = False,
    ) -> list[ScheduledMessage]:
        """Return queued messages, optionally filtered by recipient or chat_id.

        By default returns only 'pending' status. Set include_all=True to
        include sent/cancelled/failed/expired as well.
        """
        with self._lock:
            result = []
            for msg in self._queue.values():
                if not include_all and msg.status != "pending":
                    continue
                if chat_id and msg.chat_id != chat_id:
                    continue
                if recipient:
                    needle = recipient.lower().replace("+", "").replace("@c.us", "")
                    haystack_r = (msg.recipient or "").lower().replace("+", "")
                    haystack_c = msg.chat_id.lower().replace("@c.us", "").replace("@lid", "")
                    if needle not in haystack_r and needle not in haystack_c:
                        continue
                result.append(msg)
        result.sort(key=lambda m: m.send_at)
        return result

    def edit_message(
        self, msg_id: str, new_text: str | None = None, new_send_at: float | None = None
    ) -> bool:
        """Edit a pending message's text and/or scheduled time. Returns True on success."""
        with self._lock:
            msg = self._queue.get(msg_id)
            if msg is None or msg.status != "pending":
                return False
            if new_text is not None:
                msg.text = new_text
            if new_send_at is not None:
                msg.send_at = new_send_at
        self.save()
        return True

    def cancel_message(self, msg_id: str) -> bool:
        """Cancel a specific pending message by ID. Returns True on success."""
        with self._lock:
            msg = self._queue.get(msg_id)
            if msg is None or msg.status != "pending":
                return False
            msg.status = "cancelled"
        self.save()
        return True

    def start(self) -> None:
        """Launch the background dispatch thread (daemon)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="scheduler-dispatch",
        )
        self._thread.start()
        log.info("MessageScheduler started")

    def stop(self) -> None:
        """Signal the dispatch thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        log.info("MessageScheduler stopped")

    def save(self) -> None:
        """Persist the queue to disk atomically."""
        if self._persist_path is None:
            return
        with self._lock:
            snapshot = {mid: m.to_dict() for mid, m in self._queue.items()}
        self._atomic_write(snapshot)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for mid, data in raw.items():
                try:
                    self._queue[mid] = ScheduledMessage.from_dict(data)
                except Exception as exc:
                    log.warning("Skipping malformed scheduled message %s: %s", mid, exc)
            # Recover in-flight messages interrupted by a crash (at-least-once delivery).
            for msg in self._queue.values():
                if msg.status == "sending":
                    msg.status = "pending"
                    log.debug("Recovered in-flight message %s back to pending", msg.id)
            log.info("MessageScheduler: loaded %d messages from disk", len(self._queue))
        except Exception as exc:
            log.warning("MessageScheduler: failed to load queue: %s", exc)

    def _expire_stale(self) -> None:
        """Mark messages that are pending but overdue by > 2 h as expired."""
        now = time.time()
        expired_any = False
        with self._lock:
            for msg in self._queue.values():
                if msg.status == "pending" and (now - msg.send_at) > _STALE_THRESHOLD:
                    msg.status = "expired"
                    expired_any = True
                    log.debug("Expired stale message %s (overdue %.0fs)", msg.id, now - msg.send_at)
        if expired_any:
            self.save()

    def _get_quiet_policy(self, _channel: str, chat_id: str) -> QuietHoursPolicy | None:
        """Resolve quiet-hours policy for a chat.

        Priority: per-contact entry > _default entry > None.
        The quiet_cfg dict is keyed by contact name; since we only have chat_id
        here, we check chat_id directly then fall back to _default.
        """
        if not self._quiet_cfg:
            return None
        contact_cfg = self._quiet_cfg.get(chat_id) or self._quiet_cfg.get("_default")
        if not contact_cfg:
            return None
        return _parse_quiet_hours(contact_cfg)

    def _next_wake_interval(self) -> float:
        """Compute sleep duration until the next pending message is due."""
        now = time.time()
        with self._lock:
            earliest = min(
                (m.send_at for m in self._queue.values() if m.status == "pending"),
                default=None,
            )
        if earliest is None:
            return self._dispatch_interval
        remaining = earliest - now
        return max(_MIN_DISPATCH_INTERVAL, min(remaining, self._dispatch_interval))

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._dispatch_due()
                self._cleanup_old()
            except Exception as exc:
                log.error("MessageScheduler dispatch error: %s", exc)
            self._stop_event.wait(timeout=self._next_wake_interval())

    def _dispatch_due(self) -> None:
        now = time.time()
        due: list[ScheduledMessage] = []

        with self._lock:
            for msg in self._queue.values():
                if msg.status == "pending" and msg.send_at <= now:
                    due.append(msg)

        for msg in due:
            policy = self._get_quiet_policy(msg.channel, msg.chat_id)
            if policy and _is_in_quiet_window(policy, now):
                new_send_at = _next_quiet_end(policy, now)
                with self._lock:
                    # re-check status under lock before modifying
                    if self._queue.get(msg.id) and self._queue[msg.id].status == "pending":
                        self._queue[msg.id].send_at = new_send_at
                log.debug("Deferred message %s to %.0f due to quiet hours", msg.id, new_send_at)
                continue

            self._send_message(msg)

        if due:
            self.save()

    def _send_message(self, msg: ScheduledMessage) -> None:
        """Attempt to send a single message with retry bookkeeping."""
        with self._lock:
            current = self._queue.get(msg.id)
            if current is None or current.status != "pending":
                return
            current.status = "sending"

        channel = self._channels.get(msg.channel)
        if channel is None:
            log.warning("Scheduler: channel '%s' not found for message %s", msg.channel, msg.id)
            with self._lock:
                if self._queue.get(msg.id) and self._queue[msg.id].status == "sending":
                    self._queue[msg.id].status = "failed"
            return

        try:
            result = channel.send(msg.chat_id, msg.text)
            success = result.ok
        except Exception as exc:
            log.warning("Scheduler: send error for message %s: %s", msg.id, exc)
            success = False

        with self._lock:
            current = self._queue.get(msg.id)
            if current is None:
                return
            if current.status == "cancelled":
                log.debug("Scheduler: message %s was cancelled during send", msg.id)
                return
            current.attempts += 1
            if success:
                current.status = "sent"
                log.info("Scheduler: sent message %s to %s@%s", msg.id, msg.chat_id, msg.channel)
            elif current.attempts >= current.max_attempts:
                current.status = "failed"
                log.warning(
                    "Scheduler: message %s failed after %d attempts", msg.id, current.attempts
                )
            else:
                backoff = _BACKOFF_SECONDS[min(current.attempts - 1, len(_BACKOFF_SECONDS) - 1)]
                current.send_at = time.time() + backoff
                current.status = "pending"
                log.debug(
                    "Scheduler: message %s will retry in %ds (attempt %d/%d)",
                    msg.id,
                    backoff,
                    current.attempts,
                    current.max_attempts,
                )

    def _cleanup_old(self) -> None:
        """Remove terminal-state messages older than 24 hours."""
        cutoff = time.time() - _CLEANUP_AGE
        terminal = {"sent", "cancelled", "failed", "expired"}
        with self._lock:
            stale = [
                mid
                for mid, m in self._queue.items()
                if m.status in terminal and m.created_at < cutoff
            ]
            for mid in stale:
                del self._queue[mid]
        if stale:
            self.save()

    def _atomic_write(self, data: dict[str, Any]) -> None:
        try:
            from src.utils.atomic_write import atomic_write_json

            with atomic_write_json(self._persist_path) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.warning("MessageScheduler: failed to persist queue: %s", exc)
