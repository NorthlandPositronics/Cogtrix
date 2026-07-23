"""
JSON file-based memory store.
Saves conversation history to data/history/{session_id}.json.
"""

import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from cogtrix_core.utils.path_safety import _sanitize_session_id

# Best-effort file locking for concurrent same-session writes.
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


from cogtrix_core.memory.base import BaseMemoryStore

log = logging.getLogger("cogtrix")

# Optional LangChain message classes
try:  # pragma: no cover
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
except ImportError:  # pragma: no cover
    BaseMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore
    ToolMessage = None  # type: ignore


def _message_to_dict(msg: Any) -> dict:
    """Serialize a message to a dict representation.

    Supports HumanMessage, AIMessage (including tool_calls), and
    ToolMessage so that the full agent chain — not just the final
    text — survives a save/load round-trip.
    """
    ts: str | None = None

    # ── ToolMessage ───────────────────────────────────────────────
    if ToolMessage is not None and isinstance(msg, ToolMessage):
        d: dict[str, Any] = {
            "type": "tool",
            "content": (
                msg.content
                if isinstance(msg.content, list)
                else (msg.content if msg.content else "")
            ),
            "name": getattr(msg, "name", ""),
            "tool_call_id": getattr(msg, "tool_call_id", ""),
        }
        return d

    # ── BaseMessage (Human / AI) ──────────────────────────────────
    if BaseMessage is not None and isinstance(msg, BaseMessage):
        is_human = HumanMessage is not None and isinstance(msg, HumanMessage)
        role = "human" if is_human else "ai"
        ts = (msg.additional_kwargs or {}).get("_ts")
        # Preserve list content as JSON array; string stays as string
        content: str | list = (
            msg.content if isinstance(msg.content, list) else (msg.content if msg.content else "")
        )
        d = {"type": role, "content": content}

        # Preserve tool_calls on AIMessages so the agent can see its
        # previous tool-calling chain on restart (Ralph Loop support).
        if not is_human:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                        "name": (
                            tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                        ),
                        "args": (
                            tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        ),
                    }
                    for tc in tool_calls
                ]
            # Preserve reasoning_content for DeepSeek round-trip: the API
            # requires this field to be echoed back in subsequent calls.
            # Without serialization it is lost on save/load, causing 400 errors.
            # Cap at 8192 chars to prevent multi-KB CoT from compounding into
            # very large JSON files that are fully loaded on every session restore.
            rc = (msg.additional_kwargs or {}).get("reasoning_content")
            if rc:
                if len(rc) > 8192:
                    rc = rc[:8192] + " … [truncated]"
                d["reasoning_content"] = rc

        if ts:
            d["timestamp"] = ts
        return d

    # ── Plain dict (fallback) ─────────────────────────────────────
    if isinstance(msg, dict) and "content" in msg:
        d = {"type": msg.get("type", "human"), "content": msg["content"]}
        ts = msg.get("timestamp")
        if ts:
            d["timestamp"] = ts
        # Forward tool-related fields
        for key in ("tool_calls", "name", "tool_call_id"):
            if key in msg:
                d[key] = msg[key]
        return d

    return {"type": "human", "content": str(msg)}


def _dict_to_message(data: dict) -> Any:
    """Deserialize dict back to a message object.

    Handles ``type`` values: ``"human"``, ``"ai"`` (optionally with
    ``tool_calls``), and ``"tool"``.
    """
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    ts = data.get("timestamp")
    additional: dict[str, Any] = {"_ts": ts} if ts else {}

    # ── ToolMessage ───────────────────────────────────────────────
    if msg_type == "tool" and ToolMessage is not None:
        # content can be str or list; pass through as-is
        return ToolMessage(
            content=content,
            name=data.get("name", ""),
            tool_call_id=data.get("tool_call_id", ""),
        )

    # ── AIMessage (optionally with tool_calls) ────────────────────
    if msg_type == "ai" and AIMessage is not None:
        # Restore reasoning_content so DeepSeek re-injection works after reload
        rc = data.get("reasoning_content")
        if rc:
            additional["reasoning_content"] = rc
        tool_calls_data = data.get("tool_calls")
        # content can be str or list; pass through as-is
        if tool_calls_data:
            return AIMessage(
                content=content,
                additional_kwargs=additional,
                tool_calls=tool_calls_data,
            )
        return AIMessage(content=content, additional_kwargs=additional)

    # ── HumanMessage ──────────────────────────────────────────────
    if HumanMessage is not None:
        # content can be str or list; pass through as-is
        return HumanMessage(content=content, additional_kwargs=additional)

    # ── Fallback (no LangChain) ───────────────────────────────────
    d: dict[str, Any] = {"type": msg_type, "content": content}
    if ts:
        d["timestamp"] = ts
    return d


class JsonFileMemoryStore(BaseMemoryStore):
    """Persist conversation history as JSON files."""

    # In-process per-session locks so concurrent threads sharing the same
    # JsonFileMemoryStore instance don't interleave writes.  Combined with
    # OS-level flock (see _lock_file) this guards both in-process and
    # multi-process concurrent access to the same session file.
    #
    # LRU eviction: _session_locks is bounded to _SESSION_LOCKS_MAX_SIZE entries.
    # When capacity is reached, the least-recently-used entry is evicted before
    # a new one is added. This prevents unbounded memory growth in long-running
    # server processes that handle many ephemeral session IDs.
    _SESSION_LOCKS_MAX_SIZE = 1024
    _session_locks: OrderedDict[str, threading.Lock] = OrderedDict()
    _session_locks_meta = threading.Lock()

    @classmethod
    def _get_session_lock(cls, session_id: str) -> threading.Lock:
        with cls._session_locks_meta:
            if session_id in cls._session_locks:
                # Move to end to mark as most recently used
                cls._session_locks.move_to_end(session_id)
                return cls._session_locks[session_id]
            # Evict oldest entries if at capacity
            while len(cls._session_locks) >= cls._SESSION_LOCKS_MAX_SIZE:
                cls._session_locks.popitem(last=False)
            lock = threading.Lock()
            cls._session_locks[session_id] = lock
            return lock

    def __init__(self, base_dir: str = "data/history"):
        self.base_path = Path(base_dir)
        self._save_disabled = False
        self._consecutive_save_failures = 0
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._save_disabled = True
            log.warning(
                "Cannot create history directory %s: %s. History will not be saved.", base_dir, exc
            )

    def _session_path(self, session_id: str) -> Path:
        # Use the shared sanitizer from path_safety.py for consistent encoding
        safe_id = _sanitize_session_id(session_id)

        # Resolve and verify the path stays inside base_path
        full_path = (self.base_path / f"{safe_id}.json").resolve()
        try:
            full_path.relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(f"Invalid session ID: {session_id}") from None

        return full_path

    def load_history(self, session_id: str):
        path = self._session_path(session_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [_dict_to_message(item) for item in data]
        except json.JSONDecodeError as e:
            log.error("Corrupt session file %s: %s", path, e)
            return []
        except Exception as e:
            log.error("Error loading session %s: %s", session_id, e)
            return []

    def save_history(self, session_id: str, messages: list[Any]):
        if self._save_disabled:
            return
        path = self._session_path(session_id)
        serializable = [_message_to_dict(m) for m in messages]
        # Acquire in-process lock first, then OS-level flock on the lock file
        # so concurrent writes from different processes don't corrupt history.
        with self._get_session_lock(session_id):
            self._save_history_locked(path, serializable, session_id)

    def _save_history_locked(self, path: Path, serializable: list[Any], session_id: str) -> None:
        """Write history while holding the in-process lock for this session."""
        lock_path = path.with_suffix(".lock")
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        except OSError:
            lock_fd = -1
        try:
            if lock_fd >= 0:
                _lock_file(lock_fd)
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(serializable, f, ensure_ascii=False)
                    os.replace(tmp_path, path)
                    self._consecutive_save_failures = 0
                except Exception:
                    # Do NOT call os.close(tmp_fd) here — os.fdopen took ownership
                    # of the fd and its context-manager __exit__ already closed it
                    # when the 'with' block exited on exception.  An explicit close
                    # would be a double-close (fd leak on repeated close is harmless
                    # but os.close mock counts expose the bug in tests).
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except Exception as exc:
                self._consecutive_save_failures += 1
                if self._consecutive_save_failures >= 3:
                    self._save_disabled = True
                    log.warning(
                        "Cannot save history to %s: %s. Disabling saves after 3 consecutive failures.",
                        path,
                        exc,
                    )
                else:
                    log.warning(
                        "History save failed for %s (attempt %d/3): %s",
                        path,
                        self._consecutive_save_failures,
                        exc,
                    )
        finally:
            if lock_fd >= 0:
                _unlock_file(lock_fd)
                try:
                    os.close(lock_fd)
                except OSError:
                    pass

    def delete_lock(self, session_id: str) -> None:
        """Remove the per-session ``.lock`` file created during save.

        The flock guard file is ``O_CREAT``'d on every ``save_history`` and is
        otherwise never removed, so a server cycling through many session ids
        accumulates one ``{id}.lock`` per session forever (#2131 C5).
        ``BaseMemoryManager.clear()`` calls this to clean it up. Best-effort:
        a missing file or path error is ignored.
        """
        try:
            lock_path = self._session_path(session_id).with_suffix(".lock")
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
