"""Session bridge: in-memory ApiSession registry backed by the database.

``ApiSession`` holds live LLM / memory / AgentRunConfig objects for a web
session.  ``ApiSessionRegistry`` manages lifecycle — lazy warm from DB,
idle eviction, and graceful shutdown save.

Design invariants (from the implementation plan):
- ``SessionState.no_confirm = True`` for all API sessions — confirmation is
  handled in future phases via ``ApiConfirmationUI``.
- ``MemoryManager.save()`` is called before any eviction or archive.
- The tool catalog is shared globally on ``app.state``; per-session state
  lives only in ``SessionState``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.db.models import ApiSessionRecord
    from src.orchestration.run_config import AgentRunConfig

log = logging.getLogger("cogtrix.api.session_bridge")

# How long (seconds) an idle ApiSession stays in memory before being evicted.
_DEFAULT_MAX_IDLE_AGE = 1800  # 30 minutes
# How often the background eviction task runs.
_EVICTION_INTERVAL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# ApiSession dataclass
# ---------------------------------------------------------------------------


@dataclass
class ApiSession:
    """In-memory representation of a live web session.

    Holds all objects needed to run an agent turn — LLM, memory, tools.
    The corresponding ``ApiSessionRecord`` in the DB is the durable backup;
    this object is the authoritative live view.
    """

    id: str
    user_id: str
    name: str
    config: dict = field(default_factory=dict)
    session_state: Any = None  # SessionState
    memory_manager: Any = None  # BaseMemoryManager
    llm: Any = None  # LangChain chat model
    run_config: Any = None  # AgentRunConfig
    registry: Any = None  # ToolRegistry — needed by run_agent for requires_confirmation checks
    agent_state: str = "idle"
    token_counts: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "context_window": 0}
    )
    turn_task: Any = None  # asyncio.Task | None
    drain_task: Any = None  # asyncio.Task | None — active WS drain task
    cancel_event: Any = None  # asyncio.Event
    ws_queue: Any = None  # asyncio.Queue
    _lock: Any = None  # asyncio.Lock
    last_activity: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()
        if self.ws_queue is None:
            self.ws_queue = asyncio.Queue()


# ---------------------------------------------------------------------------
# warm_session helper
# ---------------------------------------------------------------------------


async def warm_session(record: ApiSessionRecord, app_state: Any) -> ApiSession:
    """Construct a live ``ApiSession`` from a DB record and app state.

    Steps:
    1. Deserialize the session config from ``record.config_json``.
    2. Create ``SessionState(no_confirm=True)``.
    3. Build a memory manager from the configured mode.
    4. Create the LLM from the provider/model config (falls back to app defaults).
    5. Build ``AgentRunConfig`` with tools from ``app.state.tool_registry``.
    6. Return a populated ``ApiSession``.
    """
    from src.orchestration.session_state import SessionState

    # 1. Parse session config
    try:
        config = json.loads(record.config_json) if record.config_json else {}
    except (json.JSONDecodeError, TypeError):
        config = {}

    # 2. Session state — API sessions never use CLI confirmation
    session_state = SessionState(no_confirm=True)

    # 3. Build memory manager — may perform file I/O (manager.load()); run in thread.
    memory_manager = await asyncio.to_thread(_build_memory_manager, record.id, config, app_state)

    # 4. Create LLM — may perform network I/O; run in thread.
    llm = await asyncio.to_thread(_build_llm, config, app_state)

    # 5. Build AgentRunConfig
    run_config = _build_run_config(llm, session_state, config, app_state)

    # 6. Parse token counts
    try:
        token_counts = json.loads(record.token_counts_json) if record.token_counts_json else {}
    except (json.JSONDecodeError, TypeError):
        token_counts = {}
    token_counts.setdefault("input_tokens", 0)
    token_counts.setdefault("output_tokens", 0)
    token_counts.setdefault("context_window", 0)

    session = ApiSession(
        id=record.id,
        user_id=record.user_id,
        name=record.name,
        config=config,
        session_state=session_state,
        memory_manager=memory_manager,
        llm=llm,
        run_config=run_config,
        registry=getattr(app_state, "tool_registry", None),
        agent_state=record.state or "idle",
        token_counts=token_counts,
    )
    log.debug("Warmed session %s for user %s", record.id, record.user_id)
    return session


def _build_memory_manager(session_id: str, config: dict, app_state: Any) -> Any:
    """Create a memory manager from the session config."""
    try:
        # Ensure all modes are registered
        import src.memory.modes  # noqa: F401
        from src.memory import JsonFileMemoryStore, MemoryFactory

        mode = config.get("memory_mode") or "conversation"
        if not MemoryFactory.is_registered(mode):
            mode = "conversation"

        store = JsonFileMemoryStore()
        manager = MemoryFactory.create(mode, store, session_id)
        manager.load()
        return manager
    except Exception as exc:
        log.warning("Could not build memory manager for session %s: %s", session_id, exc)
        return None


def _build_llm(config: dict, app_state: Any) -> Any:
    """Create an LLM from the session config or fall back to app defaults."""
    try:
        from src.config import ProviderConfig
        from src.providers import create_chat_model_from_config

        app_cfg = getattr(app_state, "config", None)

        provider_type = (
            config.get("provider")
            or (getattr(app_cfg, "provider", None) if app_cfg else None)
            or "ollama"
        )
        model_name = config.get("model") or (getattr(app_cfg, "model", None) if app_cfg else None)

        # Resolve api_key from app config if available
        api_key: str | None = None
        if app_cfg is not None:
            try:
                provider_cfg = app_cfg.get_provider_config()
                if provider_cfg is not None:
                    api_key = provider_cfg.api_key
            except Exception:
                pass

        pc = ProviderConfig(
            name=provider_type,
            type=provider_type,
            model=model_name,
            api_key=api_key,
        )
        return create_chat_model_from_config(pc)
    except Exception as exc:
        log.warning("Could not build LLM for session: %s", exc)
        return None


def _build_run_config(
    llm: Any,
    session_state: Any,
    config: dict,
    app_state: Any,
) -> AgentRunConfig:
    """Build an AgentRunConfig from session config and app state."""
    from src.orchestration.run_config import AgentRunConfig

    # Gather tools from the registry
    available_tools: dict[str, Any] = {}
    active_tools_list: list[Any] = []
    tool_registry = getattr(app_state, "tool_registry", None)
    if tool_registry is not None:
        try:
            all_tools = getattr(tool_registry, "tools", {})
            available_tools = dict(all_tools)
        except Exception as exc:
            log.warning("Could not read tool registry: %s", exc)

    # Resolve settings from session config with fallbacks
    app_cfg = getattr(app_state, "config", None)

    def _cfg_get(key: str, default: Any) -> Any:
        val = config.get(key)
        if val is not None:
            return val
        if app_cfg is not None:
            return getattr(app_cfg, key, default)
        return default

    parallel = _cfg_get("parallel_tool_execution", True)
    context_compression = _cfg_get("context_compression", True)

    # Determine context window size from app config
    max_context_tokens: int | None = None
    if app_cfg is not None:
        max_context_tokens = getattr(app_cfg, "max_context_tokens", None)
    if max_context_tokens is None:
        max_context_tokens = 131072  # safe default

    return AgentRunConfig(
        llm=llm,
        available_tools=available_tools,
        active_tools_list=active_tools_list,
        max_context_tokens=max_context_tokens,
        context_compression=bool(context_compression),
        parallel_tool_execution=bool(parallel),
        session_state=session_state,
    )


# ---------------------------------------------------------------------------
# ApiSessionRegistry
# ---------------------------------------------------------------------------


class ApiSessionRegistry:
    """Process-level registry mapping session UUID → live ApiSession.

    Thread-safe via an ``asyncio.Lock`` on all mutating operations.
    A background ``asyncio.Task`` evicts sessions idle longer than
    ``max_idle_age`` every ``_EVICTION_INTERVAL`` seconds.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state
        self._sessions: dict[str, ApiSession] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Event] = {}
        self._eviction_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_eviction_loop(self) -> None:
        """Start the background eviction task.  Call once from app lifespan."""
        if self._eviction_task is None or self._eviction_task.done():
            self._eviction_task = asyncio.create_task(
                self._eviction_loop(), name="session-eviction"
            )

    async def stop_eviction_loop(self) -> None:
        """Cancel the background eviction task and save all sessions."""
        if self._eviction_task is not None and not self._eviction_task.done():
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
        await self.save_all()

    async def _eviction_loop(self) -> None:
        """Periodically evict sessions that have been idle too long."""
        while True:
            try:
                await asyncio.sleep(_EVICTION_INTERVAL)
                await self.evict_idle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Session eviction error: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_warm(self, session_id: str, db_session: AsyncSession) -> ApiSession | None:
        """Return the cached ApiSession or warm it from the DB.

        Returns ``None`` if the session does not exist in the DB.

        Uses a per-session ``asyncio.Event`` to ensure only one coroutine
        performs the expensive ``warm_session()`` call — concurrent callers
        for the same session await the event instead of duplicating work.
        """
        async with self._lock:
            if session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.last_activity = time.time()
                return sess

            # Another coroutine is already warming this session — wait for it.
            pending_event = self._pending.get(session_id)
            if pending_event is not None:
                event = pending_event
            else:
                # We are the first — register an event so others can wait.
                event = asyncio.Event()
                self._pending[session_id] = event

        if pending_event is not None:
            # Wait for the warming coroutine to finish, then return from cache.
            await event.wait()
            async with self._lock:
                sess = self._sessions.get(session_id)
                if sess is not None:
                    sess.last_activity = time.time()
                return sess

        # We own the warming responsibility.
        try:
            from src.api.db.repositories.sessions import SessionRepository

            repo = SessionRepository(db_session)
            record = await repo.get_by_id(session_id)
            if record is None:
                return None

            session = await warm_session(record, self._app_state)

            discarded: ApiSession | None = None
            async with self._lock:
                if session_id not in self._sessions:
                    self._sessions[session_id] = session
                else:
                    # A concurrent warmer already inserted a session for this id.
                    # Save and discard the one we just built to prevent a memory
                    # manager resource leak.
                    discarded = session
                existing = self._sessions[session_id]
                existing.last_activity = time.time()

            if discarded is not None:
                await _save_memory(discarded)

            return existing
        finally:
            # Set the event BEFORE removing from _pending so that any coroutine
            # that arrives between pop() and set() does not start a duplicate
            # warm.  Waiters that already hold a reference to the event will
            # be unblocked correctly; new arrivals after the pop() will either
            # find the session in _sessions (success path) or register a fresh
            # event and re-warm (failure path), both of which are correct.
            event.set()
            async with self._lock:
                self._pending.pop(session_id, None)

    async def evict_idle(self, max_age_seconds: float = _DEFAULT_MAX_IDLE_AGE) -> int:
        """Save and evict sessions idle longer than ``max_age_seconds``.

        Returns the number of sessions evicted.
        """
        now = time.time()
        to_evict: list[str] = []

        async with self._lock:
            for sid, sess in list(self._sessions.items()):
                if now - sess.last_activity > max_age_seconds:
                    to_evict.append(sid)

        evicted = 0
        for sid in to_evict:
            async with self._lock:
                sess = self._sessions.pop(sid, None)
            if sess is not None:
                await _save_memory(sess)
                evicted += 1
                log.debug("Evicted idle session %s", sid)

        return evicted

    async def save_all(self) -> None:
        """Persist all in-memory sessions before shutdown."""
        async with self._lock:
            sessions = list(self._sessions.values())

        for sess in sessions:
            try:
                await _save_memory(sess)
            except Exception as exc:
                log.warning("Error saving session %s on shutdown: %s", sess.id, exc)

    async def remove(self, session_id: str) -> None:
        """Save and remove a specific session from the registry."""
        async with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is not None:
            await _save_memory(sess)

    async def put(self, session: ApiSession) -> None:
        """Store a session in the registry (used after warm_session)."""
        async with self._lock:
            self._sessions[session.id] = session

    def get_cached(self, session_id: str) -> ApiSession | None:
        """Return the cached session synchronously (no DB fallback)."""
        return self._sessions.get(session_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _save_memory(sess: ApiSession) -> None:
    """Persist the memory manager for a session, if available.

    Runs ``mm.save()`` in a thread to avoid blocking the event loop during
    eviction or shutdown.
    """
    mm = getattr(sess, "memory_manager", None)
    if mm is not None:
        try:
            await asyncio.to_thread(mm.save)
        except Exception as exc:
            log.warning("Memory save failed for session %s: %s", sess.id, exc)
