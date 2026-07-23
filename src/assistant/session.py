"""
Chat session lifecycle management for Cogtrix assistant mode.

Each unique (channel, chat_id) pair gets its own isolated ChatSession with
independent memory.  ChatSessionManager is the thread-safe registry that
creates, retrieves, evicts, and saves sessions.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("cogtrix")

#: Maximum overflow multiplier over _max_sessions before the hard cap fires.
#: Allows transient burst growth when all sessions are busy, but caps sustained
#: overflow at max_sessions * this value to prevent unbounded memory growth.
_MAX_SESSION_OVERFLOW_MULTIPLIER = 1.5


@dataclass
class ChatSession:
    """Per-chat isolated context, memory manager, and concurrency guard."""

    session_key: str
    channel: str
    chat_id: str
    memory_manager: Any
    last_activity: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    guardrail_violations: int = 0
    last_sent_message_id: str | None = None
    workflow_id: str | None = None


class ChatSessionManager:
    """Thread-safe registry for chat sessions.

    Args:
        config: Full application Config object.
        llm: LLM instance for memory summarization.
        system_prompt: Base system prompt (unused here, stored for reference).
        registry: Tool registry (unused here, stored for reference).
        max_sessions: Maximum number of concurrent sessions before eviction.
        idle_timeout: Seconds of inactivity before a session is evicted.
    """

    def __init__(
        self,
        config: Any,
        llm: Any,
        system_prompt: str,
        registry: Any,
        max_sessions: int = 50,
        idle_timeout: float = 3600.0,
    ) -> None:
        self._config = config
        self._llm = llm
        self._system_prompt = system_prompt
        self._registry = registry
        self._max_sessions = max_sessions
        self._idle_timeout = idle_timeout
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.RLock()

    def get_or_create(self, msg: Any) -> ChatSession:
        """Return the existing session for *msg*, creating one if needed.

        Thread-safe at the registry level. Individual session operations
        are serialized by the per-session lock.
        """
        key = msg.session_key
        idle_evicted: list[tuple[str, ChatSession]] = []
        evicted_session: ChatSession | None = None
        evicted_key: str | None = None
        try:
            with self._lock:
                if key in self._sessions:
                    session = self._sessions[key]
                    session.last_activity = time.monotonic()
                    return session

                if len(self._sessions) >= self._max_sessions:
                    # Inline idle eviction: collect candidates and remove from registry
                    # under the lock, but save outside to avoid blocking other threads.
                    now = time.monotonic()
                    to_evict = [
                        k
                        for k, s in self._sessions.items()
                        if (now - s.last_activity) > self._idle_timeout
                    ]
                    for k in to_evict:
                        s = self._sessions[k]
                        if not s.lock.acquire(blocking=False):
                            log.debug("Skipping eviction of busy session %s", k)
                            continue
                        del self._sessions[k]
                        idle_evicted.append((k, s))

                    if not idle_evicted and self._sessions:
                        oldest_key = min(
                            self._sessions, key=lambda k: self._sessions[k].last_activity
                        )
                        oldest = self._sessions[oldest_key]
                        if oldest.lock.acquire(blocking=False):
                            del self._sessions[oldest_key]
                            evicted_session = oldest
                            evicted_key = oldest_key
                        else:
                            log.debug(
                                "Skipping eviction of busy session %s; allowing over cap",
                                oldest_key,
                            )

                    # Hard cap: reject new sessions once we've exceeded the overflow
                    # threshold. This prevents unbounded growth under sustained concurrent
                    # load when every session is busy and non-blocking eviction fails.
                    if len(self._sessions) >= int(
                        self._max_sessions * _MAX_SESSION_OVERFLOW_MULTIPLIER
                    ):
                        raise RuntimeError(
                            f"Session cap exceeded: {len(self._sessions)} active "
                            f"sessions (hard cap at {int(self._max_sessions * _MAX_SESSION_OVERFLOW_MULTIPLIER)}). "
                            f"Reduce concurrent load or increase max_sessions."
                        )

                session = self._create_session(msg)
                self._sessions[key] = session
                log.info("Created session %s", key)

        finally:
            # BUG-106: cleanup always runs — even if _create_session raises —
            # so evicted session locks are always released and saves always attempted.
            # All save I/O happens outside the registry lock.
            for k, s in idle_evicted:
                try:
                    s.memory_manager.save()
                    log.info("Evicted idle session %s", k)
                except Exception as exc:
                    log.warning("Failed to save session %s on eviction: %s", k, exc)
                finally:
                    s.lock.release()

            if evicted_session is not None:
                try:
                    evicted_session.memory_manager.save()
                except Exception as exc:
                    log.warning(
                        "Failed to save session %s before eviction: %s",
                        evicted_key,
                        exc,
                    )
                finally:
                    evicted_session.lock.release()
                log.warning("Session cap reached; evicted oldest session %s", evicted_key)

        return session

    def _create_session(self, msg: Any) -> ChatSession:
        import src.memory.modes as _modes  # noqa: F401 — registers modes with MemoryFactory

        _ = _modes  # ensure pyright sees it as used
        from src.memory.factory import MemoryFactory
        from src.memory.json_store import JsonFileMemoryStore

        asst_cfg = (
            self._config.services.get("assistant", {}) if hasattr(self._config, "services") else {}
        )
        memory_cfg: dict[str, Any] = asst_cfg.get("memory", {})

        history_dir = str(Path(getattr(self._config, "data_dir", "data")) / "history")
        store = JsonFileMemoryStore(history_dir)
        mm = MemoryFactory.create(
            mode="conversation",
            store=store,
            session_id=msg.session_key,
            config=memory_cfg or None,
        )
        mm.load()
        mm.set_llm(self._llm)

        return ChatSession(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            memory_manager=mm,
        )

    def evict_idle(self) -> int:
        """Save and remove sessions idle longer than *idle_timeout*. Returns count evicted.

        Acquires each session's lock (non-blocking) before saving to avoid
        racing with an active message handler.  Busy sessions are skipped and
        will be retried on the next eviction pass.

        Disk I/O (save) runs outside the registry lock to avoid blocking
        other threads that need get_or_create.
        """
        evicted_sessions: list[tuple[str, ChatSession]] = []
        with self._lock:
            now = time.monotonic()
            to_evict = [
                key
                for key, session in self._sessions.items()
                if (now - session.last_activity) > self._idle_timeout
            ]
            for key in to_evict:
                session = self._sessions[key]
                if not session.lock.acquire(blocking=False):
                    log.debug("Skipping eviction of busy session %s", key)
                    continue
                del self._sessions[key]
                evicted_sessions.append((key, session))

        for key, session in evicted_sessions:
            try:
                session.memory_manager.save()
            except Exception as exc:
                log.warning("Failed to save session %s on eviction: %s", key, exc)
            finally:
                session.lock.release()
            log.info("Evicted idle session %s", key)

        return len(evicted_sessions)

    def save_all(self) -> None:
        """Persist all active sessions (called on graceful shutdown).

        Snapshots the session registry under the registry lock, then acquires
        each session's lock individually outside the registry lock to avoid
        deadlocking with get_or_create or evict_idle.
        """
        with self._lock:
            items = list(self._sessions.items())

        for key, session in items:
            acquired = session.lock.acquire(timeout=10.0)
            if not acquired:
                log.warning("Failed to acquire lock for session %s on shutdown; skipping", key)
                continue
            try:
                session.memory_manager.save()
                log.debug("Saved session %s", key)
            except Exception as exc:
                log.warning("Failed to save session %s: %s", key, exc)
            finally:
                session.lock.release()
