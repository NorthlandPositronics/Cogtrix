"""
Scheduled reply delivery for Cogtrix assistant mode.

Provides MessageScheduler, which queues agent-generated replies for deferred
delivery via a background daemon thread, plus the ScheduleReplyState/tool
factory used by MessageHandler to capture per-call scheduling intent.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
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
    status: str = "pending"  # pending | sending | sent | cancelled | failed | expired
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledMessage:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ScheduleReplyState:
    """Mutable per-call state set by the schedule_reply tool closure."""

    was_called: bool = False
    scheduled_text: str = ""
    delay_minutes: int = 0


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


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def create_schedule_reply_tool(state: ScheduleReplyState) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    When the agent calls this tool the closure mutates *state* in-place
    so the handler can route the reply after agent execution completes.
    This function does NOT enqueue directly.
    """

    def _schedule_reply(text: str, delay_minutes: int) -> str:
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

    def schedule(self, channel: str, chat_id: str, text: str, send_at: float) -> str:
        """Add a message to the queue and persist. Returns the message ID."""
        msg = ScheduledMessage(
            id=str(uuid.uuid4()),
            channel=channel,
            chat_id=chat_id,
            text=text,
            send_at=send_at,
            created_at=time.time(),
        )
        with self._lock:
            self._queue[msg.id] = msg
        self.save()
        log.debug("Scheduled message %s for %s@%s at %.0f", msg.id, chat_id, channel, send_at)
        return msg.id

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
                    log.debug("Skipping malformed scheduled message %s: %s", mid, exc)
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
            success = channel.send(msg.chat_id, msg.text)
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
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._persist_path.parent), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._persist_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            log.debug("MessageScheduler: failed to persist queue: %s", exc)
