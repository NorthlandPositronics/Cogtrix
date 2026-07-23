"""
Deferred message processing for Cogtrix assistant mode.

Provides DeferralManager, which tracks deferred re-processing passes for
(channel, chat_id) pairs. When the agent calls defer_processing, the handler
registers a DeferredRecord here; the background thread fires a reprocess_callback
when the timer expires. New messages arriving for a deferred chat are coalesced
into the pending record instead of being processed immediately.

Also provides the create_defer_processing_tool and create_suppress_reply_tool
factories used by MessageHandler to inject per-call tools.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool as _StructuredToolType  # noqa: F401

    from src.assistant.scheduler import ScheduleReplyState

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc, assignment]

log = logging.getLogger("cogtrix")

_BACKOFF_SECONDS: float = 30.0
_DEFAULT_CHECK_INTERVAL: float = 10.0
_DEFAULT_MAX_DEPTH: int = 3
_DEFAULT_STALE_THRESHOLD: float = 7200.0  # 2 hours


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DeferredRecord:
    """Represents a pending deferred re-processing pass for a single chat session."""

    id: str
    channel: str
    chat_id: str
    fire_at: float  # wall-clock time.time()
    created_at: float
    pending_messages: list[dict[str, Any]] = field(default_factory=list)
    deferral_depth: int = 0
    status: str = "pending"  # pending | firing | cancelled

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredRecord:
        coerced = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for float_field in ("fire_at", "created_at"):
            if float_field in coerced:
                coerced[float_field] = float(coerced[float_field])
        for int_field in ("deferral_depth",):
            if int_field in coerced:
                coerced[int_field] = int(coerced[int_field])
        return cls(**coerced)


@dataclass
class DeferReplyState:
    """Mutable per-call state set by the defer_processing tool closure."""

    was_called: bool = False
    delay_seconds: float = 0.0


@dataclass
class SuppressReplyState:
    """Mutable per-call state set by the suppress_reply tool closure."""

    was_called: bool = False


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DeferProcessingInput(BaseModel):
    delay_minutes: int = Field(
        description=(
            "Minutes to wait before re-processing this conversation. "
            "Use 3–8 for partial messages where the user seems to be still typing. "
            "Use 15–30 when waiting for information the user said they would provide. "
            "Use 30–90 for situational delays (waiting for a specific event or time). "
            "Use 1440 only for explicit next-day scenarios. "
            "Prefer shorter delays — the conversation re-processes sooner and you can "
            "defer again if needed."
        ),
        ge=1,
        le=1440,
    )
    reason: str = Field(
        default="",
        description=(
            "Brief internal note explaining why processing is deferred. "
            "Not sent to the user. Used for logging only."
        ),
    )


class SuppressReplyInput(BaseModel):
    reason: str = Field(
        default="",
        description=(
            "Brief internal note explaining why no reply is needed. "
            "Not sent to the user. Used for logging only."
        ),
    )


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def create_defer_processing_tool(
    state: DeferReplyState,
    schedule_state: ScheduleReplyState | None = None,
) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    The tool does NOT contact the channel. It sets state.was_called so
    handle() can skip delivery and register the deferral with DeferralManager.
    Accepts optional schedule_state to detect and warn on co-invocation.
    """

    _lock = threading.Lock()

    def _defer_processing(delay_minutes: int, reason: str = "") -> str:
        with _lock:
            if state.was_called:
                return (
                    f"Processing already deferred by {int(state.delay_seconds // 60)} minute(s). "
                    "Only one deferral is allowed per turn."
                )
            if schedule_state is not None and schedule_state.was_called:
                log.warning(
                    "defer_processing called in same turn as schedule_reply — "
                    "deferral takes precedence; scheduled reply will be discarded."
                )
            state.was_called = True
            state.delay_seconds = float(delay_minutes * 60)
        if reason:
            log.debug("defer_processing: %s (delay=%dm)", reason, delay_minutes)
        return (
            f"Processing deferred for {delay_minutes} minute(s). "
            "No reply will be sent now. The conversation will be re-processed at the specified time."
        )

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_defer_processing,
        name="defer_processing",
        description=(
            "Defer re-processing of this conversation to a later time without sending any reply. "
            "Use when more context is expected (the user may continue writing), when the right "
            "moment to respond has not yet arrived, or when you need time to gather information. "
            "The conversation will be re-processed at the specified time with full history, "
            "including any messages that arrive in the meantime. "
            "Do not use this as a substitute for schedule_reply: schedule_reply composes a reply "
            "now and delivers it later; this tool postpones the reasoning pass itself. "
            "Do not call both defer_processing and schedule_reply in the same turn."
        ),
        args_schema=DeferProcessingInput,
    )


def create_suppress_reply_tool(state: SuppressReplyState) -> StructuredTool:  # type: ignore[valid-type]
    """Return a StructuredTool whose closure captures *state*.

    When called, sets state.was_called so handle() can skip delivery and
    memory update entirely. Injected only during re-processing passes.
    """

    _lock = threading.Lock()

    def _suppress_reply(reason: str = "") -> str:
        with _lock:
            if state.was_called:
                return "Reply already suppressed for this turn."
            state.was_called = True
        if reason:
            log.debug("suppress_reply: %s", reason)
        return "Reply suppressed. No message will be sent to the user."

    return StructuredTool.from_function(  # type: ignore[union-attr]
        func=_suppress_reply,
        name="suppress_reply",
        description=(
            "Suppress the reply for this turn — send nothing to the user and skip "
            "memory update. Use only when you have determined that no reply is "
            "warranted (e.g., the question resolved itself, the user answered their "
            "own question in a follow-up). Do not use to avoid difficult questions "
            "or skip tasks — use it only when sending any reply would be worse than "
            "silence. Do not call in combination with schedule_reply or queue_reply."
        ),
        args_schema=SuppressReplyInput,
    )


# ---------------------------------------------------------------------------
# Elapsed time formatting
# ---------------------------------------------------------------------------


def format_elapsed(seconds: float) -> str:
    """Return human-readable elapsed time.

    Examples:
        ``format_elapsed(30)`` -> ``"<1 min"``
        ``format_elapsed(90)`` -> ``"1 min"``
        ``format_elapsed(3661)`` -> ``"1 h 1 min"``
        ``format_elapsed(7200)`` -> ``"2 h"``
    """
    if seconds < 60:
        return "<1 min"
    total_minutes = int(seconds // 60)
    if total_minutes < 60:
        return f"{total_minutes} min"
    hours = total_minutes // 60
    remaining_minutes = total_minutes % 60
    if remaining_minutes == 0:
        return f"{hours} h"
    return f"{hours} h {remaining_minutes} min"


# ---------------------------------------------------------------------------
# DeferralManager
# ---------------------------------------------------------------------------


class DeferralManager:
    """Tracks deferred re-processing passes for (channel, chat_id) pairs.

    At most one DeferredRecord per session_key (channel::chat_id) may be
    pending at any time. A second defer() call for the same chat merges into
    the existing record rather than creating a second timer.

    Args:
        persist_path: Path for the JSON persistence file.
        reprocess_callback: Callable[[list[IncomingMessage], Channel, int], None].
            Called on the background thread when a deferred timer fires.
            Receives (messages, channel, deferral_depth).
        channels: Mapping of channel name to Channel instance.
        max_depth: Maximum number of consecutive deferrals allowed (default 3).
        check_interval: How often (seconds) the background thread wakes (default 10.0).
        stale_threshold: Seconds past fire_at after which a pending record expires
            (default 7200.0).
    """

    def __init__(
        self,
        persist_path: Path,
        reprocess_callback: Callable[..., None] | None,
        channels: dict[str, Any],
        max_depth: int = _DEFAULT_MAX_DEPTH,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
        stale_threshold: float = _DEFAULT_STALE_THRESHOLD,
    ) -> None:
        assert isinstance(
            persist_path, Path
        ), f"persist_path must be a Path, got {type(persist_path).__name__}"
        self._persist_path = persist_path
        self._reprocess_callback = reprocess_callback
        self._channels = channels
        self.max_depth = max_depth
        self._check_interval = check_interval
        self._stale_threshold = stale_threshold

        self._records: dict[str, DeferredRecord] = {}  # session_key -> DeferredRecord
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reprocess_callback(self, callback: Callable[..., None]) -> None:
        """Set (or replace) the reprocess callback after construction."""
        self._reprocess_callback = callback

    def _on_reprocess_success(self, session_key: str) -> None:
        """Called by the done callback when reprocessing completes successfully."""
        with self._lock:
            current = self._records.get(session_key)
            if current is not None and current.status == "firing":
                self._records.pop(session_key, None)
            # Call save inside the lock to avoid TOCTOU between pop and persistence
            self._save_locked()
        log.info("DeferralManager: reprocessing completed for %s", session_key)

    def _on_reprocess_failure(self, session_key: str) -> None:
        """Called by the done callback when reprocessing fails."""
        retry_scheduled = False
        with self._lock:
            current = self._records.get(session_key)
            if current is not None and current.status == "firing":
                current.status = "pending"
                current.fire_at = time.time() + _BACKOFF_SECONDS
                retry_scheduled = True
            # Call save inside the lock to avoid TOCTOU between status change and persistence
            self._save_locked()
        if retry_scheduled:
            log.warning(
                "DeferralManager: reprocessing failed for %s, retrying in %.0fs",
                session_key,
                _BACKOFF_SECONDS,
            )
        else:
            log.debug(
                "DeferralManager: reprocess failed for %s but record was already "
                "cancelled/completed; no retry scheduled",
                session_key,
            )

    def defer(
        self,
        msg: Any,  # IncomingMessage at runtime
        delay_seconds: float,
        depth: int = 0,
    ) -> str:
        """Register or merge a deferred re-processing pass.

        If a pending record exists for msg.session_key, replaces its fire_at
        with the later of existing.fire_at and now + delay_seconds, appends msg
        to pending_messages, and updates depth.

        Returns the record ID.
        """
        now = time.time()
        fire_at = now + delay_seconds
        session_key = msg.session_key

        with self._lock:
            existing = self._records.get(session_key)
            if existing is not None and existing.status in ("pending", "firing"):
                # Merge: extend fire_at to the later of the two timers
                existing.fire_at = max(existing.fire_at, fire_at)
                existing.pending_messages.append(self._msg_to_dict(msg))
                existing.deferral_depth = depth
                record_id = existing.id
                log.debug(
                    "Merged deferral for %s (depth=%d, fire_at=%.0f, status=%s)",
                    session_key,
                    depth,
                    existing.fire_at,
                    existing.status,
                )
            else:
                record = DeferredRecord(
                    id=str(uuid.uuid4()),
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    fire_at=fire_at,
                    created_at=now,
                    pending_messages=[self._msg_to_dict(msg)],
                    deferral_depth=depth,
                    status="pending",
                )
                self._records[session_key] = record
                record_id = record.id
                log.debug(
                    "Registered deferral for %s (depth=%d, fire_at=%.0f)",
                    session_key,
                    depth,
                    fire_at,
                )

        self.save()
        return record_id

    def add_message(self, msg: Any) -> bool:  # IncomingMessage at runtime
        """Append *msg* to any pending or firing deferred record for its session_key.

        Returns True if the message was appended (a deferred record is pending
        or firing for this chat), False otherwise.
        """
        session_key = msg.session_key
        with self._lock:
            record = self._records.get(session_key)
            if record is None or record.status not in ("pending", "firing"):
                return False
            record.pending_messages.append(self._msg_to_dict(msg))

        self.save()
        log.debug("add_message: appended to deferred record for %s", session_key)
        return True

    def has_pending(self, session_key: str) -> bool:
        """Return True if a pending deferred record exists for session_key."""
        with self._lock:
            record = self._records.get(session_key)
            return record is not None and record.status == "pending"

    def current_depth(self, session_key: str) -> int:
        """Return the deferral_depth of a pending record, or 0 if none."""
        with self._lock:
            record = self._records.get(session_key)
            if record is None or record.status != "pending":
                return 0
            return record.deferral_depth

    def cancel(self, session_key: str) -> bool:
        """Cancel any pending or in-flight deferred record for session_key.

        Covers both "pending" and "firing" states so that partial-absorption
        detection in handle_batch can cancel a record that transitioned to
        "firing" between the two add_message calls (BUG-094 partial fix).
        Returns True if a record was cancelled.
        """
        with self._lock:
            record = self._records.get(session_key)
            if record is None or record.status not in ("pending", "firing"):
                return False
            prev_status = record.status
            record.status = "cancelled"

        self.save()
        log.debug("Cancelled deferred record for %s (was %s)", session_key, prev_status)
        return True

    def start(self) -> None:
        """Launch the background dispatch thread (daemon)."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._reprocess_callback is None:
            raise RuntimeError("DeferralManager.start() called before set_reprocess_callback()")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="deferral-dispatch",
        )
        self._thread.start()
        log.info("DeferralManager started (check_interval=%.0fs)", self._check_interval)

    def stop(self) -> None:
        """Signal the dispatch thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        log.info("DeferralManager stopped")

    def save(self) -> None:
        """Persist all records to disk atomically."""
        with self._lock:
            snapshot = {key: rec.to_dict() for key, rec in self._records.items()}
        self._atomic_write(snapshot)

    def _save_locked(self) -> None:
        """Persist all records to disk while already holding the lock.

        This avoids a TOCTOU window where status changes are made outside the lock
        and then save() is called, which would acquire the lock again and potentially
        overwrite those changes made by another thread in between.
        """
        snapshot = {key: rec.to_dict() for key, rec in self._records.items()}
        self._atomic_write(snapshot)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_to_dict(msg: Any) -> dict[str, Any]:
        """Convert an IncomingMessage to a JSON-serialisable dict."""
        # Normalize metadata to dict or empty dict, handling non-dict values gracefully
        meta_raw = msg.metadata
        try:
            meta_dict: dict[str, Any] = dict(meta_raw) if meta_raw else {}
        except (TypeError, ValueError):
            meta_dict = {}
        return {
            "channel": msg.channel,
            "chat_id": msg.chat_id,
            "message_id": msg.message_id,
            "sender_id": msg.sender_id,
            "sender_name": msg.sender_name,
            "text": msg.text,
            "timestamp": msg.timestamp,
            "metadata": meta_dict,
            "resolved_phone": msg.resolved_phone,
        }

    @staticmethod
    def _dict_to_msg(data: dict[str, Any]) -> Any:
        """Reconstruct an IncomingMessage from a persisted dict."""
        from src.assistant.channel import IncomingMessage

        return IncomingMessage(
            channel=data.get("channel", ""),
            chat_id=data.get("chat_id", ""),
            message_id=data.get("message_id", ""),
            sender_id=data.get("sender_id", ""),
            sender_name=data.get("sender_name"),
            text=(data.get("text") or ""),
            timestamp=float(data.get("timestamp", 0.0)),
            metadata=data.get("metadata", {}),
            resolved_phone=data.get("resolved_phone"),
        )

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for session_key, data in raw.items():
                try:
                    record = DeferredRecord.from_dict(data)
                    # Recover in-flight records interrupted by a crash (at-least-once semantics).
                    if record.status == "firing":
                        record.status = "pending"
                        log.debug("Recovered firing deferral for %s back to pending", session_key)
                    self._records[session_key] = record
                except Exception as exc:
                    log.warning("Skipping malformed deferred record %s: %s", session_key, exc)
            log.info("DeferralManager: loaded %d records from disk", len(self._records))
        except Exception as exc:
            log.warning("DeferralManager: failed to load records: %s", exc)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._dispatch_due()
            except Exception as exc:
                log.error("DeferralManager dispatch error: %s", exc)
            self._stop_event.wait(timeout=self._check_interval)

    def _dispatch_due(self) -> None:
        # BUG-092: the original elif branch ended with `pass` (dead code) and used
        # an incorrect formula. Fixed: cancel future-dated stale records inside the
        # lock-held block with the correct (now - created_at) > stale_threshold formula.
        # BUG-099: re-read fire_at under lock before the overdue check to close the
        # race with a concurrent defer() call that may have extended fire_at.
        now = time.time()
        due: list[tuple[str, DeferredRecord]] = []
        to_cancel_stale: list[tuple[str, DeferredRecord]] = []

        with self._lock:
            for session_key, record in self._records.items():
                if record.status != "pending":
                    continue
                if record.fire_at <= now:
                    due.append((session_key, record))
                elif (now - record.created_at) > self._stale_threshold:
                    # Record is too old to keep even though fire_at is in the future.
                    to_cancel_stale.append((session_key, record))

            for session_key, record in to_cancel_stale:
                if self._records.get(session_key) is record and record.status == "pending":
                    record.status = "cancelled"
                    log.warning(
                        "DeferralManager: future-dated stale record for %s (age %.0fs) — cancelled",
                        session_key,
                        now - record.created_at,
                    )

        if to_cancel_stale:
            self.save()

        for session_key, record in due:
            # Re-validate fire_at under lock to guard against a concurrent defer()
            # call that extended fire_at between the snapshot above and this read.
            with self._lock:
                current = self._records.get(session_key)
                if current is None or current.status != "pending":
                    continue
                fire_at_snapshot = current.fire_at

            # If fire_at was extended by a concurrent defer() call, skip this cycle.
            if fire_at_snapshot > now:
                continue

            overdue = now - fire_at_snapshot
            if overdue > self._stale_threshold:
                with self._lock:
                    if (
                        self._records.get(session_key) is not None
                        and self._records[session_key].status == "pending"
                    ):
                        self._records[session_key].status = "cancelled"
                log.warning(
                    "DeferralManager: stale record for %s (overdue %.0fs) — cancelled",
                    session_key,
                    overdue,
                )
                self.save()
                continue

            self._fire_record(session_key, record, now)

    def _fire_record(self, session_key: str, record: DeferredRecord, now: float) -> None:
        """Attempt to fire a due deferral record."""
        with self._lock:
            current = self._records.get(session_key)
            if current is None or current.status != "pending":
                return
            current.status = "firing"

        self.save()

        channel = self._channels.get(record.channel)
        if channel is None:
            log.warning(
                "DeferralManager: channel '%s' not found for record %s",
                record.channel,
                record.id,
            )
            with self._lock:
                if self._records.get(session_key) and self._records[session_key].status == "firing":
                    self._records[session_key].status = "pending"
                    self._records[session_key].fire_at = now + _BACKOFF_SECONDS
            self.save()
            return

        # Build messages from the pending list, prepend re-processing prefix to the first.
        messages = [self._dict_to_msg(d) for d in record.pending_messages]
        if not messages:
            # Nothing to re-process; just remove the record.
            with self._lock:
                self._records.pop(session_key, None)
            self.save()
            return

        elapsed = format_elapsed(now - record.created_at)
        n = len(messages)
        depth = record.deferral_depth
        prefix = (
            f"[Re-processing — deferred {elapsed} ago | {n} message(s) in batch"
            f" | depth {depth}/{self.max_depth}]"
        )
        # Prepend prefix to first message text.
        first = messages[0]
        from src.assistant.channel import IncomingMessage

        messages[0] = IncomingMessage(
            channel=first.channel,
            chat_id=first.chat_id,
            message_id=first.message_id,
            sender_id=first.sender_id,
            sender_name=first.sender_name,
            text=f"{prefix}\n{first.text}",
            timestamp=first.timestamp,
            metadata=first.metadata,
            resolved_phone=first.resolved_phone,
        )

        try:
            if self._reprocess_callback is None:
                raise RuntimeError(
                    "DeferralManager._fire_record called with no reprocess callback set; "
                    "ensure set_reprocess_callback() is called before start()."
                )
            self._reprocess_callback(messages, channel, depth, session_key)
            # Success: the record will be removed by _on_reprocess_success via the done callback.
            # Do NOT pop or save here - that would create a TOCTOU with the callback's state update.
        except Exception as exc:
            log.error(
                "DeferralManager: reprocess callback raised on submit for %s: %s — retrying in %.0fs",
                session_key,
                exc,
                _BACKOFF_SECONDS,
                exc_info=True,
            )
            with self._lock:
                current = self._records.get(session_key)
                if current is not None and current.status == "firing":
                    current.status = "pending"
                    current.fire_at = now + _BACKOFF_SECONDS
            self.save()

    def _atomic_write(self, data: dict[str, Any]) -> None:
        try:
            from src.utils.atomic_write import atomic_write_json

            with atomic_write_json(self._persist_path) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.warning("DeferralManager: failed to persist records: %s", exc)
