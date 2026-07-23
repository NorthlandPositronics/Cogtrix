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
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
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
    turn_lock: Any = None  # asyncio.Lock — serializes concurrent agent turns per session
    last_activity: float = field(default_factory=time.time)
    active_confirmation_ui: Any = None  # ApiConfirmationUI | None — set for the duration of a turn

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self.turn_lock is None:
            self.turn_lock = asyncio.Lock()
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()
        if self.ws_queue is None:
            # Bounded queue: prevents unbounded growth when no WebSocket drain
            # task is active (e.g. REST-only clients that never open a WS
            # connection) — BUG-FORGE-004.
            self.ws_queue = asyncio.Queue(maxsize=10_000)


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
    from src.logging_config import clear_session_id, set_session_id
    from src.orchestration.session_state import SessionState

    set_session_id(record.id)
    try:
        # 1. Parse session config
        try:
            config = json.loads(record.config_json) if record.config_json else {}
        except (json.JSONDecodeError, TypeError):
            config = {}

        # 2. Session state — API sessions never use CLI confirmation
        session_state = SessionState(no_confirm=True)
        # Deny shell/exec tools unless explicitly enabled in config
        app_config = getattr(app_state, "config", None)
        if not getattr(app_config, "api_dangerous_tools", False):
            for _t in ("shell", "bash", "python_exec"):
                session_state.deny_tool(_t)

        # 3 & 4. Build memory manager and LLM concurrently — both are I/O-bound and independent.
        memory_manager, llm = await asyncio.gather(
            asyncio.to_thread(_build_memory_manager, record.id, config, app_state),
            asyncio.to_thread(_build_llm, config, app_state),
        )

        if llm is None:
            raise RuntimeError(
                f"LLM could not be created for session {record.id} — check provider config and API keys"
            )

        if memory_manager is None:
            raise RuntimeError(f"Memory manager could not be created for session {record.id}")

        # 5. Build AgentRunConfig
        run_config = _build_run_config(llm, session_state, memory_manager, config, app_state)

        # Wire TCC helper attributes onto the memory manager so that background
        # roll-forward jobs use the correct context budget and compression LLM.
        if memory_manager is not None:
            memory_manager.configure_compression(
                max_context_tokens=run_config.max_context_tokens,
                compression_llm=run_config.compression_llm,
            )
            if llm is not None:
                memory_manager.set_llm(llm)

        # The web_search stage-5 synthesiser LLM is scoped inside
        # ``run_agent`` itself — see src/orchestration/runner.py. We
        # don't wire it here because the session can outlive a single
        # agent turn while the ContextVar should be re-set per turn.

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
    finally:
        clear_session_id()


def _build_memory_manager(session_id: str, config: dict, app_state: Any) -> Any:
    """Create a memory manager from the session config."""
    try:
        # Ensure all modes are registered
        import src.memory.modes  # noqa: F401
        from src.memory import JsonFileMemoryStore, MemoryFactory

        mode = config.get("memory_mode") or "conversation"
        if not MemoryFactory.is_registered(mode):
            mode = "conversation"

        app_cfg = getattr(app_state, "config", None)
        history_dir = (
            str(Path(app_cfg.data_dir) / "history") if app_cfg is not None else "data/history"
        )
        store = JsonFileMemoryStore(history_dir)
        manager = MemoryFactory.create(mode, store, session_id)
        manager.load()
        return manager
    except Exception as exc:
        raise RuntimeError(
            f"Could not build memory manager for session {session_id}: {exc}"
        ) from exc


def _build_llm(config: dict, app_state: Any) -> Any:
    """Create an LLM from the session config or fall back to app defaults.

    Uses ``resolve_llm_config_for`` when a per-session model alias is set,
    otherwise falls back to ``resolve_llm_config`` for the app default.
    """
    try:
        from src.providers import create_chat_model_from_configs

        app_cfg = getattr(app_state, "config", None)
        if app_cfg is None:
            raise RuntimeError("Could not build LLM for session: app config not loaded")

        session_model = config.get("model")

        try:
            if session_model:
                pc, mc = app_cfg.resolve_llm_config_for(session_model)
            else:
                pc, mc = app_cfg.resolve_llm_config()
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(f"Could not resolve LLM config: {exc}") from exc

        return create_chat_model_from_configs(pc, mc, streaming=True)
    except Exception as exc:
        raise RuntimeError(f"Could not build LLM for session: {exc}") from exc


def _build_run_config(
    llm: Any,
    session_state: Any,
    memory_manager: Any,
    config: dict,
    app_state: Any,
) -> AgentRunConfig:
    """Build an AgentRunConfig from session config and app state."""
    from src.agent.registry import filter_tools_for_agent
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

    # Populate all_tool_originals so unload/disable can restore the canonical
    # (unwrapped) tool object — mirrors CLI behaviour (BUG-199).
    if available_tools:
        session_state.all_tool_originals = dict(available_tools)

    # Extract agent_name from session config to enable tool filtering
    agent_name = config.get("agent_name")

    # Filter tools based on agent's tools_include/tools_exclude configuration
    if agent_name and available_tools:
        filtered_tools, filtered_list = filter_tools_for_agent(agent_name, available_tools)
        if filtered_tools != available_tools:
            log.info(
                "Agent %r tool filtering: %d tools after filtering (included: %s, excluded: %s)",
                agent_name,
                len(filtered_tools),
                config.get("tools_include", []),
                config.get("tools_exclude", []),
            )
        available_tools = filtered_tools
        active_tools_list = filtered_list

    # Seed active_tools_list with request_tools so the agent can load on-demand tools.
    # Auto-activate query_knowledge_base when a knowledge base exists.
    if available_tools:
        try:
            from src.tools.configure import (
                _update_rag_tool_description,
                build_tool_catalog,
                create_request_tools_tool,
                rag_should_auto_activate,
            )

            if rag_should_auto_activate() and "query_knowledge_base" in available_tools:
                rag_tool = available_tools.pop("query_knowledge_base")
                _update_rag_tool_description(rag_tool)
                active_tools_list.append(rag_tool)

            catalog = build_tool_catalog(available_tools)
            rt_tool = create_request_tools_tool(available_tools, catalog)
            if rt_tool is not None:
                active_tools_list.insert(0, rt_tool)
        except Exception as exc:
            log.warning("Could not create request_tools for session: %s", exc)

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
    context_max_messages = _cfg_get("context_max_messages", 200)
    context_max_tokens = _cfg_get("context_max_tokens", 40_000)
    tool_trust = _cfg_get("tool_trust", None)

    # Build system prompt: use session override if set, otherwise construct
    # from DEFAULT_SYSTEM_PROMPT with model and tool context — matching CLI behavior.
    custom_prompt = config.get("system_prompt")
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        try:
            from src.agent.core import build_system_prompt

            tool_names: set[str] = set()
            for t in active_tools_list:
                name = getattr(t, "name", "")
                if name:
                    tool_names.add(name)
            tool_names.update(available_tools.keys())

            models_dict = getattr(app_cfg, "models", None) if app_cfg else None
            delegation_models = (
                getattr(app_cfg, "delegate_allowed_models", None) if app_cfg else None
            )
            from src.orchestration.reflection_delegate import (
                ACCOUNTABILITY_PROMPT,
                PRE_ACTION_CONFIRMATION_PROMPT,
            )

            _da_enabled = app_cfg.decision_accountability_enabled if app_cfg is not None else False
            _pac_enabled = app_cfg.pre_action_confirmation_enabled if app_cfg is not None else False
            system_prompt = build_system_prompt(
                models=models_dict,
                delegation_models=delegation_models,
                active_tool_names=tool_names,
                decision_accountability_prompt=(ACCOUNTABILITY_PROMPT if _da_enabled else None),
                pre_action_confirmation_prompt=(
                    PRE_ACTION_CONFIRMATION_PROMPT if _pac_enabled else None
                ),
            )
        except Exception as exc:
            log.warning("Could not build system prompt: %s", exc)
            from src.agent.core import DEFAULT_SYSTEM_PROMPT

            system_prompt = DEFAULT_SYSTEM_PROMPT

    # Determine context window from the active model's context_window — matching CLI behavior.
    max_context_tokens: int | None = None
    if app_cfg is not None:
        session_model = config.get("model")
        try:
            if session_model:
                _, mc = app_cfg.resolve_llm_config_for(session_model)
            else:
                _, mc = app_cfg.resolve_llm_config()
            max_context_tokens = mc.context_window
        except Exception as exc:
            log.warning(
                "Could not resolve context window for session model %r: %s — using fallback %d",
                session_model,
                exc,
                32_768,
            )
    if max_context_tokens is None:
        max_context_tokens = 32_768  # matches _DEFAULT_CONTEXT_WINDOW in core.py

    tier_cache_enabled = _cfg_get("tier_cache_enabled", True)

    return AgentRunConfig(
        llm=llm,
        system_prompt=system_prompt or None,
        available_tools=available_tools,
        active_tools_list=active_tools_list,
        max_context_tokens=max_context_tokens,
        context_max_messages=context_max_messages,
        context_max_tokens=context_max_tokens,
        context_compression=bool(context_compression),
        parallel_tool_execution=bool(parallel),
        tier_cache_enabled=bool(tier_cache_enabled),
        session_state=session_state,
        memory_manager=memory_manager,
        tools_ready=(
            getattr(getattr(app_state, "mcp_manager", None), "tools_ready", None)
            if app_state
            else None
        ),
        bound_cache=OrderedDict(),
        compression_cache=OrderedDict(),
        decision_accountability_enabled=(
            app_cfg.decision_accountability_enabled if app_cfg is not None else False
        ),
        decision_accountability_report_uncertainty=(
            app_cfg.decision_accountability_report_uncertainty if app_cfg is not None else True
        ),
        decision_accountability_min_confidence=(
            app_cfg.decision_accountability_min_confidence if app_cfg is not None else 7.0
        ),
        task_ownership_classifier_enabled=(
            app_cfg.task_ownership_classifier_enabled if app_cfg is not None else True
        ),
        task_ownership_classifier_llm_fallback=(
            app_cfg.task_ownership_classifier_llm_fallback if app_cfg is not None else False
        ),
        task_ownership_ambiguous_action=(
            app_cfg.task_ownership_ambiguous_action if app_cfg is not None else "ask"
        ),
        pre_action_confirmation_enabled=(
            app_cfg.pre_action_confirmation_enabled if app_cfg is not None else False
        ),
        tool_trust=tool_trust,
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
                log.debug("Session %s: warm hit", session_id)
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
        log.debug("Session %s: cold start", session_id)
        try:
            from src.api.db.repositories.sessions import SessionRepository

            repo = SessionRepository(db_session)
            record = await repo.get_by_id(session_id)
            if record is None:
                return None

            try:
                session = await warm_session(record, self._app_state)
            except Exception as exc:
                log.error("Failed to warm session %s: %s", session_id, exc)
                return None

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

        Sessions with an active agent turn are skipped — evicting them would save
        a stale memory snapshot that the running turn would overwrite moments later,
        producing a divergence between on-disk state and what the turn commits.

        Returns the number of sessions evicted.
        """
        now = time.time()
        to_evict: list[str] = []

        async with self._lock:
            for sid, sess in list(self._sessions.items()):
                if now - sess.last_activity > max_age_seconds:
                    # Skip sessions with an agent turn actively running.
                    turn_task = getattr(sess, "turn_task", None)
                    if turn_task is not None and not turn_task.done():
                        log.debug("Skipping eviction of session %s: agent turn in progress", sid)
                        continue
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

    async def get_cached(self, session_id: str) -> ApiSession | None:
        """Return the cached session without a DB fallback."""
        async with self._lock:
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
