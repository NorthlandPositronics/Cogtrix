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
        with self._lock:
            if key in self._sessions:
                return self._sessions[key]

            if len(self._sessions) >= self._max_sessions:
                evicted = self.evict_idle()
                if evicted == 0 and self._sessions:
                    oldest_key = min(self._sessions, key=lambda k: self._sessions[k].last_activity)
                    oldest = self._sessions[oldest_key]
                    if oldest.lock.acquire(blocking=False):
                        try:
                            oldest.memory_manager.save()
                        except Exception as exc:
                            log.warning(
                                "Failed to save session %s before eviction: %s",
                                oldest_key,
                                exc,
                            )
                        finally:
                            oldest.lock.release()
                        del self._sessions[oldest_key]
                        log.warning("Session cap reached; evicted oldest session %s", oldest_key)
                    else:
                        log.debug(
                            "Skipping eviction of busy session %s; allowing over cap", oldest_key
                        )

            session = self._create_session(msg)
            self._sessions[key] = session
            log.info("Created session %s", key)
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
        """
        with self._lock:
            now = time.monotonic()
            to_evict = [
                key
                for key, session in self._sessions.items()
                if (now - session.last_activity) > self._idle_timeout
            ]
            evicted = 0
            for key in to_evict:
                session = self._sessions[key]
                if not session.lock.acquire(blocking=False):
                    log.debug("Skipping eviction of busy session %s", key)
                    continue
                try:
                    session.memory_manager.save()
                except Exception as exc:
                    log.warning("Failed to save session %s on eviction: %s", key, exc)
                finally:
                    session.lock.release()
                del self._sessions[key]
                evicted += 1
                log.info("Evicted idle session %s", key)
            return evicted

    def save_all(self) -> None:
        """Persist all active sessions (called on graceful shutdown).

        Acquires each session lock with a timeout to avoid hanging if a
        handler is stuck.
        """
        with self._lock:
            for key, session in self._sessions.items():
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
