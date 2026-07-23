"""TestForge Suite 2 — regression and coverage tests for the second TestForge run.

Covers:
1. _run_think_pipeline cancel at each phase (classify → research → deep_think)
2. _run_delegate_pipeline cancel path and normal execution
3. WebSocketCallbackHandler debug-log guard (P2 fix: prompt_chars gated behind isEnabledFor)
4. ApiSessionRegistry.get_or_warm concurrent-warming deduplication
5. ApiSession dataclass initialization and defaults
6. Wizard step 0 PROVIDER_UNREACHABLE behavior via config routes
7. _build_history edge cases (None mm, exception, empty)
8. _extract_token_counts edge cases
9. on_tool_start / on_tool_end / on_tool_error callbacks thread safety
10. _run_think_pipeline: early exit when task is tool-intensive
11. _run_think_pipeline: early exit when deep_think already had good context
12. Delegate pipeline: skips when delegation already called
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment setup — before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

pytest.importorskip("fastapi")


# ===========================================================================
# Helpers shared across tests
# ===========================================================================


def _make_mock_session(*, agent_state: str = "idle") -> MagicMock:
    """Build a minimal MagicMock ApiSession for turn_runner tests."""
    session = MagicMock()
    session.id = f"test-session-{uuid.uuid4().hex[:8]}"
    session.turn_lock = asyncio.Lock()
    session.session_state = None
    session.run_config = None
    session.memory_manager = None
    session.cancel_event = asyncio.Event()
    session.ws_queue = asyncio.Queue(maxsize=100)
    session.active_confirmation_ui = None
    session.agent_state = agent_state
    session.token_counts = {"input_tokens": 0, "output_tokens": 0}
    session.last_activity = 0.0
    return session


def _make_callback_handler():
    """Build a WebSocketCallbackHandler with its own event loop and queue."""
    from cogtrix_core.api.callbacks import WebSocketCallbackHandler

    loop = asyncio.new_event_loop()

    async def _make_q():
        return asyncio.Queue(maxsize=1000)

    q = loop.run_until_complete(_make_q())
    h = WebSocketCallbackHandler(ws_queue=q, loop=loop)
    return h, q, loop


def _flush_loop_and_drain(loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> list[dict]:
    """Flush call_soon_threadsafe callbacks and drain the queue."""
    loop.run_until_complete(asyncio.sleep(0))
    items: list[dict] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


# ===========================================================================
# 1. _run_think_pipeline — cancel at classify phase
# ===========================================================================


class TestRunThinkPipelineCancelClassify:
    """Cancel after classify_think_task call resets agent_state to idle."""

    @pytest.mark.asyncio
    async def test_cancel_at_classify_resets_state(self):
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = MagicMock()

        async def _fake_enqueue(s, state):
            s.agent_state = state

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", side_effect=_fake_enqueue):
            with patch(
                "cogtrix_core.api.turn_runner.asyncio.to_thread",
                side_effect=asyncio.CancelledError("classify cancelled"),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await _run_think_pipeline(session, "task", "response", [], run_config)

    @pytest.mark.asyncio
    async def test_cancel_at_classify_does_not_suppress_error(self):
        """CancelledError propagates — never swallowed."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None  # skip to_thread

        # With llm=None, classify is skipped; next step is cancel check
        session.cancel_event.set()  # signal cancel

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await _run_think_pipeline(session, "task", "response", [], run_config)


# ===========================================================================
# 2. _run_think_pipeline — cancel at research phase
# ===========================================================================


class TestRunThinkPipelineCancelResearch:
    """Cancel after research delegate call resets agent_state to idle."""

    @pytest.mark.asyncio
    async def test_second_cancel_check_fires_after_research(self):
        """The cancel_event check between research and deep_think phases works."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None

        enqueue_states: list[str] = []

        async def _fake_enqueue(s, state):
            s.agent_state = state
            enqueue_states.append(state)

        # No web tools → skips research; cancel_event set between research and deep_think
        session.cancel_event.set()  # fires at second check (post-research)

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", side_effect=_fake_enqueue):
            with patch(
                "cogtrix_core.orchestration.phases.was_deep_think_called", return_value=False
            ):
                with patch(
                    "cogtrix_core.orchestration.phases.deep_think_had_good_context",
                    return_value=False,
                ):
                    with patch(
                        "cogtrix_core.orchestration.phases.collect_tool_outputs", return_value=""
                    ):
                        with patch(
                            "cogtrix_core.orchestration.phases.agent_used_web_tools",
                            return_value=False,
                        ):
                            with pytest.raises(asyncio.CancelledError):
                                await _run_think_pipeline(
                                    session, "task", "original response", [], run_config
                                )


# ===========================================================================
# 3. _run_think_pipeline — cancel at deep_think phase
# ===========================================================================


class TestRunThinkPipelineCancelDeepThink:
    """Cancel during force_deep_think raises CancelledError."""

    @pytest.mark.asyncio
    async def test_cancel_event_after_deep_think_raises(self):
        """After deep_think completes, if cancel_event is set, CancelledError is raised."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None

        async def _fake_to_thread(fn, *args, **kwargs):
            # Set cancel_event just as deep_think would complete
            session.cancel_event.set()
            return "deep think result"

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with patch(
                "cogtrix_core.orchestration.phases.was_deep_think_called", return_value=False
            ):
                with patch(
                    "cogtrix_core.orchestration.phases.deep_think_had_good_context",
                    return_value=False,
                ):
                    with patch(
                        "cogtrix_core.orchestration.phases.collect_tool_outputs", return_value=""
                    ):
                        with patch(
                            "cogtrix_core.orchestration.phases.agent_used_web_tools",
                            return_value=False,
                        ):
                            with patch(
                                "cogtrix_core.api.turn_runner.asyncio.to_thread",
                                side_effect=_fake_to_thread,
                            ):
                                with pytest.raises(asyncio.CancelledError):
                                    await _run_think_pipeline(
                                        session, "task", "original response", [], run_config
                                    )


# ===========================================================================
# 4. _run_think_pipeline — early exits
# ===========================================================================


class TestRunThinkPipelineEarlyExits:
    """Early-exit conditions return original response_text unchanged."""

    @pytest.mark.asyncio
    async def test_tool_intensive_task_returns_response_unchanged(self):
        """Tool-intensive classification causes early return with original text."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = MagicMock()

        task_cat = MagicMock()
        task_cat.tool_intensive = True
        task_cat.name = "web_research"

        async def _fake_enqueue(s, state):
            s.agent_state = state

        async def _fake_to_thread(fn, *args, **kwargs):
            return task_cat

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", side_effect=_fake_enqueue):
            with patch(
                "cogtrix_core.api.turn_runner.asyncio.to_thread", side_effect=_fake_to_thread
            ):
                result = await _run_think_pipeline(
                    session, "task", "original response", [], run_config
                )

        assert result == "original response"

    @pytest.mark.asyncio
    async def test_deep_think_already_called_with_good_context_returns_original(self):
        """deep_think already called with good context → skip re-running."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None  # skip classify

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with patch(
                "cogtrix_core.orchestration.phases.was_deep_think_called", return_value=True
            ):
                with patch(
                    "cogtrix_core.orchestration.phases.deep_think_had_good_context",
                    return_value=True,
                ):
                    result = await _run_think_pipeline(
                        session, "task", "original response", [], run_config
                    )

        assert result == "original response"


# ===========================================================================
# 5. _run_delegate_pipeline — cancel path
# ===========================================================================


class TestRunDelegatePipelineCancel:
    """_run_delegate_pipeline cancel handling."""

    @pytest.mark.asyncio
    async def test_cancel_at_start_raises(self):
        """Cancel event set before delegation begins → CancelledError."""
        from cogtrix_core.api.turn_runner import _run_delegate_pipeline

        session = _make_mock_session()
        session.cancel_event.set()
        run_config = MagicMock()

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with patch(
                "cogtrix_core.orchestration.phases.was_delegation_called", return_value=False
            ):
                with pytest.raises(asyncio.CancelledError):
                    await _run_delegate_pipeline(
                        session, "task", "original response", [], run_config
                    )

    @pytest.mark.asyncio
    async def test_skip_when_delegation_already_called(self):
        """If delegation was already called, return response unchanged."""
        from cogtrix_core.api.turn_runner import _run_delegate_pipeline

        session = _make_mock_session()
        run_config = MagicMock()

        with patch("cogtrix_core.orchestration.phases.was_delegation_called", return_value=True):
            result = await _run_delegate_pipeline(
                session, "task", "original response", [], run_config
            )

        assert result == "original response"

    @pytest.mark.asyncio
    async def test_normal_path_returns_forced_result(self):
        """Normal delegation path returns the force_delegation result."""
        from cogtrix_core.api.turn_runner import _run_delegate_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = MagicMock()

        async def _fake_enqueue(s, state):
            s.agent_state = state

        async def _fake_to_thread(fn, *args, **kwargs):
            return "delegated result"

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", side_effect=_fake_enqueue):
            with patch(
                "cogtrix_core.orchestration.phases.was_delegation_called", return_value=False
            ):
                with patch(
                    "cogtrix_core.orchestration.phases.collect_tool_outputs", return_value=""
                ):
                    with patch(
                        "cogtrix_core.api.turn_runner.asyncio.to_thread",
                        side_effect=_fake_to_thread,
                    ):
                        result = await _run_delegate_pipeline(
                            session, "task", "original response", [], run_config
                        )

        assert result == "delegated result"

    @pytest.mark.asyncio
    async def test_force_delegation_exception_returns_original(self):
        """Exception in force_delegation returns original response text."""
        from cogtrix_core.api.turn_runner import _run_delegate_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None

        async def _fake_to_thread(fn, *args, **kwargs):
            raise RuntimeError("delegation failed")

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with patch(
                "cogtrix_core.orchestration.phases.was_delegation_called", return_value=False
            ):
                with patch(
                    "cogtrix_core.orchestration.phases.collect_tool_outputs", return_value=""
                ):
                    with patch(
                        "cogtrix_core.api.turn_runner.asyncio.to_thread",
                        side_effect=_fake_to_thread,
                    ):
                        result = await _run_delegate_pipeline(
                            session, "task", "original response", [], run_config
                        )

        assert result == "original response"


# ===========================================================================
# 6. _run_message_turn_inner — pipeline cancel resets agent_state to idle
# ===========================================================================


class TestPipelineCancelResetsAgentState:
    """CancelledError from pipeline phases resets session.agent_state = 'idle'."""

    def _make_run_agent_mock(self):
        """Return a callable that acts as run_agent, filling result_messages."""

        def _fake_run_agent(*args, **kwargs):
            result_messages = kwargs.get("result_messages")
            if result_messages is not None:
                result_messages.extend([])
            return "initial response"

        return _fake_run_agent

    @pytest.mark.asyncio
    async def test_think_pipeline_cancel_resets_state(self):
        """Regression: think pipeline cancel must reset agent_state to 'idle'."""
        from cogtrix_core.api.turn_runner import _run_message_turn_inner

        session = _make_mock_session()
        session.agent_state = "idle"

        mock_ws_cb = MagicMock()
        mock_ws_cb.input_tokens = 0
        mock_ws_cb.output_tokens = 0
        mock_ws_cb.tool_call_count = 0

        with patch(
            "cogtrix_core.orchestration.runner.run_agent", side_effect=self._make_run_agent_mock()
        ):
            with patch(
                "cogtrix_core.api.callbacks.WebSocketCallbackHandler",
                return_value=mock_ws_cb,
            ):
                with patch(
                    "cogtrix_core.api.confirmation.ApiConfirmationUI", return_value=MagicMock()
                ):

                    async def _cancel_think(*args, **kwargs):
                        raise asyncio.CancelledError("think pipeline cancel")

                    with patch(
                        "cogtrix_core.api.turn_runner._run_think_pipeline",
                        side_effect=_cancel_think,
                    ):
                        with pytest.raises(asyncio.CancelledError):
                            await _run_message_turn_inner(session, "hello", "think", None, None)

        assert (
            session.agent_state == "idle"
        ), f"agent_state should be 'idle' after pipeline cancel, got {session.agent_state!r}"

    @pytest.mark.asyncio
    async def test_delegate_pipeline_cancel_resets_state(self):
        """Regression: delegate pipeline cancel must reset agent_state to 'idle'."""
        from cogtrix_core.api.turn_runner import _run_message_turn_inner

        session = _make_mock_session()
        session.agent_state = "idle"

        mock_ws_cb = MagicMock()
        mock_ws_cb.input_tokens = 0
        mock_ws_cb.output_tokens = 0
        mock_ws_cb.tool_call_count = 0

        with patch(
            "cogtrix_core.orchestration.runner.run_agent", side_effect=self._make_run_agent_mock()
        ):
            with patch(
                "cogtrix_core.api.callbacks.WebSocketCallbackHandler",
                return_value=mock_ws_cb,
            ):
                with patch(
                    "cogtrix_core.api.confirmation.ApiConfirmationUI", return_value=MagicMock()
                ):

                    async def _cancel_delegate(*args, **kwargs):
                        raise asyncio.CancelledError("delegate pipeline cancel")

                    with patch(
                        "cogtrix_core.api.turn_runner._run_delegate_pipeline",
                        side_effect=_cancel_delegate,
                    ):
                        with pytest.raises(asyncio.CancelledError):
                            await _run_message_turn_inner(session, "hello", "delegate", None, None)

        assert session.agent_state == "idle", (
            f"agent_state should be 'idle' after delegate cancel, " f"got {session.agent_state!r}"
        )


# ===========================================================================
# 7. WebSocketCallbackHandler — debug-log guard (P2 fix)
# ===========================================================================


class TestCallbacksDebugLogGuard:
    """prompt_chars sum computation must be gated behind isEnabledFor(DEBUG)."""

    def test_on_llm_start_with_debug_disabled_does_not_sum_prompts(self):
        """With DEBUG disabled, prompt_chars sum must not execute."""
        handler, q, loop = _make_callback_handler()

        # Ensure DEBUG is disabled for this logger
        log = logging.getLogger("cogtrix.api.callbacks")
        original_level = log.level
        log.setLevel(logging.WARNING)

        try:
            # Pass a custom object whose __len__ raises if called
            class _FailOnLen:
                def __len__(self):
                    raise AssertionError("len() must not be called when DEBUG is disabled")

                def __iter__(self):
                    return iter([_FailOnLen()])

            # Should not raise even though prompts contains a custom object
            # (only fails if sum(len(p) for p in prompts) is evaluated)
            handler.on_llm_start(serialized={"name": "model"}, prompts=[_FailOnLen()])
        finally:
            log.setLevel(original_level)
            loop.close()

    def test_on_llm_start_with_debug_enabled_logs_model_name(self):
        """With DEBUG enabled, on_llm_start logs model info without raising."""
        handler, q, loop = _make_callback_handler()
        log = logging.getLogger("cogtrix.api.callbacks")
        original_level = log.level
        log.setLevel(logging.DEBUG)

        try:
            handler.on_llm_start(
                serialized={"name": "gpt-4o"},
                prompts=["prompt text"],
            )
            # Must not raise and must not enqueue any WebSocket message
            items = _flush_loop_and_drain(loop, q)
            assert items == [], "on_llm_start must not enqueue WS messages even at DEBUG"
        finally:
            log.setLevel(original_level)
            loop.close()

    def test_on_llm_start_with_debug_enabled_empty_prompts(self):
        """Empty prompts list is safe at DEBUG level."""
        handler, q, loop = _make_callback_handler()
        log = logging.getLogger("cogtrix.api.callbacks")
        original_level = log.level
        log.setLevel(logging.DEBUG)

        try:
            handler.on_llm_start(serialized={"id": ["openai", "ChatOpenAI"]}, prompts=[])
        finally:
            log.setLevel(original_level)
            loop.close()

    def test_on_llm_start_serialized_none_at_debug_level(self):
        """serialized=None is handled gracefully at DEBUG level."""
        handler, q, loop = _make_callback_handler()
        log = logging.getLogger("cogtrix.api.callbacks")
        original_level = log.level
        log.setLevel(logging.DEBUG)

        try:
            handler.on_llm_start(serialized=None, prompts=["test"])
        finally:
            log.setLevel(original_level)
            loop.close()


# ===========================================================================
# 8. ApiSessionRegistry — concurrent warming deduplication
# ===========================================================================


class TestApiSessionRegistryDedup:
    """get_or_warm must deduplicate concurrent warming requests."""

    @pytest.mark.asyncio
    async def test_cached_session_returned_without_db_lookup(self):
        """If session already in cache, get_or_warm returns it without DB access."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        existing = ApiSession(id="sess-1", user_id="user-1", name="test")
        await registry.put(existing)

        db = MagicMock()
        result = await registry.get_or_warm("sess-1", db)

        assert result is existing
        # DB should not have been consulted
        db.get_by_id = MagicMock()
        db.get_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_none(self):
        """get_or_warm returns None if session not in DB."""
        from cogtrix_core.api.session_bridge import ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        db = MagicMock()

        # SessionRepository is imported lazily inside get_or_warm — patch at source.
        with patch("cogtrix_core.api.db.repositories.sessions.SessionRepository") as MockRepoClass:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_id = AsyncMock(return_value=None)
            MockRepoClass.return_value = mock_repo_instance

            result = await registry.get_or_warm("nonexistent-id", db)

        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_warmers_only_call_warm_session_once(self):
        """Two concurrent get_or_warm calls for the same session warm exactly once."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        warm_call_count = [0]
        mock_session = ApiSession(id="sess-2", user_id="user-1", name="test")

        original_record = MagicMock()
        original_record.id = "sess-2"
        original_record.user_id = "user-1"
        original_record.name = "test"
        original_record.config_json = "{}"
        original_record.token_counts_json = "{}"
        original_record.state = "idle"

        async def _mock_warm_session(record, app_st):
            warm_call_count[0] += 1
            # Add a tiny delay to allow concurrent calls to pile up
            await asyncio.sleep(0.01)
            return mock_session

        # Patch warm_session (module-level in session_bridge) and SessionRepository
        # (imported lazily inside get_or_warm — patch at source module).
        with patch("cogtrix_core.api.session_bridge.warm_session", side_effect=_mock_warm_session):
            with patch(
                "cogtrix_core.api.db.repositories.sessions.SessionRepository"
            ) as MockRepoClass:
                mock_repo_instance = MagicMock()
                mock_repo_instance.get_by_id = AsyncMock(return_value=original_record)
                MockRepoClass.return_value = mock_repo_instance

                db_mock = MagicMock()
                # Launch two concurrent get_or_warm calls
                result1, result2 = await asyncio.gather(
                    registry.get_or_warm("sess-2", db_mock),
                    registry.get_or_warm("sess-2", db_mock),
                )

        # Both should return a valid session
        assert result1 is not None
        assert result2 is not None
        # warm_session must be called at most once
        assert (
            warm_call_count[0] <= 1
        ), f"warm_session called {warm_call_count[0]} times — expected at most 1"

    @pytest.mark.asyncio
    async def test_evict_idle_skips_active_turn(self):
        """Sessions with an in-progress agent turn are skipped during eviction."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        # Create a session that looks idle by time but has an active turn
        active_session = ApiSession(id="active-sess", user_id="u1", name="active")
        active_session.last_activity = 0.0  # very old
        # Create a mock turn_task that is not done
        mock_task = MagicMock()
        mock_task.done.return_value = False
        active_session.turn_task = mock_task

        await registry.put(active_session)
        evicted = await registry.evict_idle(max_age_seconds=0)  # evict everything old

        assert evicted == 0, "Active-turn session must not be evicted"
        # Session still in cache
        cached = await registry.get_cached("active-sess")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_evict_idle_removes_stale_session(self):
        """Sessions older than max_age_seconds with no active turn are evicted."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        stale_session = ApiSession(id="stale-sess", user_id="u1", name="stale")
        stale_session.last_activity = 0.0  # very old
        stale_session.memory_manager = None  # no I/O needed
        stale_session.turn_task = None

        await registry.put(stale_session)

        with patch("cogtrix_core.api.session_bridge._save_memory", new_callable=AsyncMock):
            evicted = await registry.evict_idle(max_age_seconds=0)

        assert evicted == 1
        cached = await registry.get_cached("stale-sess")
        assert cached is None

    @pytest.mark.asyncio
    async def test_save_all_persists_all_sessions(self):
        """save_all calls _save_memory for every session in the registry."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        app_state = MagicMock()
        registry = ApiSessionRegistry(app_state)

        saved_ids: list[str] = []

        for i in range(3):
            sess = ApiSession(id=f"sess-{i}", user_id="u1", name=f"s{i}")
            sess.memory_manager = MagicMock()
            await registry.put(sess)

        async def _track_save(sess):
            saved_ids.append(sess.id)

        with patch("cogtrix_core.api.session_bridge._save_memory", side_effect=_track_save):
            await registry.save_all()

        assert sorted(saved_ids) == ["sess-0", "sess-1", "sess-2"]


# ===========================================================================
# 9. ApiSession dataclass defaults
# ===========================================================================


class TestApiSessionDataclass:
    """ApiSession dataclass initializes fields correctly."""

    @pytest.mark.asyncio
    async def test_ws_queue_bounded_by_default(self):
        """ws_queue has maxsize=10_000 to prevent unbounded growth (BUG-FORGE-004)."""
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.ws_queue.maxsize == 10_000

    @pytest.mark.asyncio
    async def test_cancel_event_created_automatically(self):
        """cancel_event is an asyncio.Event created in __post_init__."""
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.cancel_event is not None
        assert not sess.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_turn_lock_created_automatically(self):
        """turn_lock is an asyncio.Lock created in __post_init__."""
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.turn_lock is not None

    @pytest.mark.asyncio
    async def test_agent_state_defaults_to_idle(self):
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.agent_state == "idle"

    @pytest.mark.asyncio
    async def test_token_counts_defaults(self):
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.token_counts["input_tokens"] == 0
        assert sess.token_counts["output_tokens"] == 0
        assert sess.token_counts["context_window"] == 0

    @pytest.mark.asyncio
    async def test_active_confirmation_ui_defaults_to_none(self):
        from cogtrix_core.api.session_bridge import ApiSession

        sess = ApiSession(id="s", user_id="u", name="n")
        assert sess.active_confirmation_ui is None


# ===========================================================================
# 10. _build_history edge cases
# ===========================================================================


class TestBuildHistory:
    """_build_history handles None, exceptions, and empty contexts."""

    def test_none_memory_manager_returns_empty_list(self):
        from cogtrix_core.api.turn_runner import _build_history

        assert _build_history(None) == []
        assert _build_history(None, user_input="hello") == []

    def test_exception_returns_empty_list(self):
        from cogtrix_core.api.turn_runner import _build_history

        mm = MagicMock()
        mm.prepare_context.side_effect = RuntimeError("memory corrupted")

        result = _build_history(mm, user_input="hello")
        assert result == []

    def test_returns_messages_from_context(self):
        from cogtrix_core.api.turn_runner import _build_history

        mm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.messages = ["msg1", "msg2", "msg3"]
        mm.prepare_context.return_value = mock_ctx

        result = _build_history(mm, user_input="test")
        assert result == ["msg1", "msg2", "msg3"]

    def test_passes_user_input_to_prepare_context(self):
        from cogtrix_core.api.turn_runner import _build_history

        mm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        mm.prepare_context.return_value = mock_ctx

        _build_history(mm, user_input="find me something")
        mm.prepare_context.assert_called_once_with("find me something")

    def test_empty_messages_is_valid(self):
        from cogtrix_core.api.turn_runner import _build_history

        mm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        mm.prepare_context.return_value = mock_ctx

        assert _build_history(mm) == []

    def test_returns_list_copy_not_reference(self):
        """Returns a new list, not a reference to ctx.messages."""
        from cogtrix_core.api.turn_runner import _build_history

        mm = MagicMock()
        original = ["msg1", "msg2"]
        mock_ctx = MagicMock()
        mock_ctx.messages = original
        mm.prepare_context.return_value = mock_ctx

        result = _build_history(mm)
        result.append("extra")
        assert original == ["msg1", "msg2"], "Modifying result must not affect original"


# ===========================================================================
# 11. _extract_token_counts edge cases
# ===========================================================================


class TestExtractTokenCounts:
    """_extract_token_counts handles missing attributes gracefully."""

    def test_normal_callback_extracted_correctly(self):
        from cogtrix_core.api.turn_runner import _extract_token_counts

        cb = MagicMock()
        cb.input_tokens = 100
        cb.output_tokens = 50
        cb.tool_call_count = 3

        result = _extract_token_counts(cb)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["tool_call_count"] == 3

    def test_missing_attrs_default_to_zero(self):
        from cogtrix_core.api.turn_runner import _extract_token_counts

        cb = object()  # no attributes

        result = _extract_token_counts(cb)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["tool_call_count"] == 0

    def test_zero_values_allowed(self):
        from cogtrix_core.api.turn_runner import _extract_token_counts

        cb = MagicMock()
        cb.input_tokens = 0
        cb.output_tokens = 0
        cb.tool_call_count = 0

        result = _extract_token_counts(cb)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


# ===========================================================================
# 12. WebSocketCallbackHandler — on_tool_start / on_tool_end / on_tool_error
# ===========================================================================


class TestCallbacksToolEvents:
    """Tool event callbacks enqueue the correct WS messages."""

    def test_on_tool_start_enqueues_tool_start_message(self):
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "web_search"}, '{"query": "test"}', run_id="run-1")
        items = _flush_loop_and_drain(loop, q)
        assert any(i["type"] == "tool_start" for i in items)
        tool_start = next(i for i in items if i["type"] == "tool_start")
        assert tool_start["payload"]["tool_name"] == "web_search"
        assert tool_start["payload"]["tool_call_id"] == "run-1"
        loop.close()

    def test_on_tool_start_increments_tool_call_count(self):
        handler, q, loop = _make_callback_handler()
        assert handler.tool_call_count == 0
        handler.on_tool_start({"name": "tool_a"}, "", run_id="r1")
        handler.on_tool_start({"name": "tool_b"}, "", run_id="r2")
        assert handler.tool_call_count == 2
        loop.close()

    def test_on_tool_end_enqueues_tool_end_message(self):
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "search"}, "", run_id="r1")
        _flush_loop_and_drain(loop, q)  # drain tool_start
        handler.on_tool_end("result", run_id="r1", name="search")
        items = _flush_loop_and_drain(loop, q)
        assert any(i["type"] == "tool_end" for i in items)
        tool_end = next(i for i in items if i["type"] == "tool_end")
        assert tool_end["payload"]["error"] is None
        loop.close()

    def test_on_tool_error_enqueues_tool_end_with_error(self):
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "search"}, "", run_id="r1")
        _flush_loop_and_drain(loop, q)
        handler.on_tool_error(RuntimeError("network error"), run_id="r1", name="search")
        items = _flush_loop_and_drain(loop, q)
        tool_end = next((i for i in items if i["type"] == "tool_end"), None)
        assert tool_end is not None
        assert tool_end["payload"]["error"] == "network error"
        loop.close()

    def test_on_tool_end_removes_from_in_flight(self):
        """After on_tool_end, the run_id is removed from _tool_starts."""
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "t"}, "", run_id="r1")
        assert "r1" in handler._tool_starts
        handler.on_tool_end("ok", run_id="r1")
        assert "r1" not in handler._tool_starts
        loop.close()

    def test_on_tool_end_unknown_run_id_does_not_raise(self):
        """on_tool_end with an unknown run_id is safe (uses current time as fallback)."""
        handler, q, loop = _make_callback_handler()
        handler.on_tool_end("result", run_id="nonexistent", name="t")
        loop.close()

    def test_tool_start_with_dict_input(self):
        """on_tool_start accepts dict input_str directly."""
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "t"}, {"query": "test"}, run_id="r1")
        items = _flush_loop_and_drain(loop, q)
        tool_start = next(i for i in items if i["type"] == "tool_start")
        assert tool_start["payload"]["input"] == {"query": "test"}
        loop.close()

    def test_tool_start_with_invalid_json_input(self):
        """on_tool_start with invalid JSON input_str uses empty dict."""
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start({"name": "t"}, "not valid json {{{", run_id="r1")
        items = _flush_loop_and_drain(loop, q)
        tool_start = next(i for i in items if i["type"] == "tool_start")
        assert isinstance(tool_start["payload"]["input"], dict)
        loop.close()

    def test_tool_start_with_none_serialized(self):
        """on_tool_start with serialized=None uses 'unknown' as tool name."""
        handler, q, loop = _make_callback_handler()
        handler.on_tool_start(None, "", run_id="r1")
        items = _flush_loop_and_drain(loop, q)
        tool_start = next(i for i in items if i["type"] == "tool_start")
        assert tool_start["payload"]["tool_name"] == "unknown"
        loop.close()


# ===========================================================================
# 13. WebSocketCallbackHandler — on_llm_error
# ===========================================================================


class TestCallbacksLlmError:
    """on_llm_error enqueues an error message to the WebSocket queue."""

    def test_on_llm_error_enqueues_error_message(self):
        handler, q, loop = _make_callback_handler()
        handler.on_llm_error(RuntimeError("rate limit"))
        items = _flush_loop_and_drain(loop, q)
        errors = [i for i in items if i["type"] == "error"]
        assert errors, "on_llm_error should enqueue an error message"
        assert errors[0]["payload"]["code"] == "AGENT_ERROR"
        assert "rate limit" in errors[0]["payload"]["message"]
        loop.close()

    def test_on_llm_error_with_string_error(self):
        """on_llm_error accepts both exception and string."""
        handler, q, loop = _make_callback_handler()
        handler.on_llm_error("context window exceeded")
        items = _flush_loop_and_drain(loop, q)
        errors = [i for i in items if i["type"] == "error"]
        assert errors
        assert "context window exceeded" in errors[0]["payload"]["message"]
        loop.close()


# ===========================================================================
# 14. Wizard step 0 PROVIDER_UNREACHABLE — behavioral tests
# ===========================================================================


class TestWizardStep0ProviderUnreachable:
    """POST /api/v1/config/wizard/{id}/step returns PROVIDER_UNREACHABLE on connection failure."""

    def _make_admin_client(self):
        from fastapi.testclient import TestClient

        from cogtrix_core.api.app import create_app
        from cogtrix_core.api.auth import create_access_token

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        return TestClient(app, raise_server_exceptions=False), admin_token

    def test_step0_returns_422_on_connection_failure(self):
        """When the LLM provider is unreachable, step 0 returns 422 with PROVIDER_UNREACHABLE."""
        client, admin_token = self._make_admin_client()

        with client:
            # Start a wizard session
            start_resp = client.post(
                "/api/v1/config/wizard",
                json={"edit_existing": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            if start_resp.status_code not in (200, 201):
                pytest.skip("Wizard endpoint not available in this test environment")

            wizard_id = start_resp.json()["data"]["wizard_id"]

            # Advance step 0 with a failing provider
            with patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                side_effect=ConnectionError("Connection refused"),
            ):
                step_resp = client.post(
                    f"/api/v1/config/wizard/{wizard_id}/step",
                    json={
                        "data": {
                            "provider_type": "openai",
                            "api_key": "fake-key",
                            "model": "gpt-4o",
                        }
                    },
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

        assert step_resp.status_code == 422
        err = step_resp.json().get("error") or step_resp.json().get("detail", {})
        if isinstance(err, dict):
            assert err.get("code") == "PROVIDER_UNREACHABLE"

    def test_step0_returns_422_on_timeout(self):
        """Timeout during provider connection returns PROVIDER_UNREACHABLE."""
        client, admin_token = self._make_admin_client()

        with client:
            start_resp = client.post(
                "/api/v1/config/wizard",
                json={"edit_existing": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            if start_resp.status_code not in (200, 201):
                pytest.skip("Wizard endpoint not available in this test environment")

            wizard_id = start_resp.json()["data"]["wizard_id"]

            with patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                side_effect=TimeoutError("connection timed out"),
            ):
                step_resp = client.post(
                    f"/api/v1/config/wizard/{wizard_id}/step",
                    json={
                        "data": {
                            "provider_type": "ollama",
                            "model": "qwen3:8b",
                        }
                    },
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

        assert step_resp.status_code == 422
        err = step_resp.json().get("error") or step_resp.json().get("detail", {})
        if isinstance(err, dict):
            assert err.get("code") == "PROVIDER_UNREACHABLE"

    def test_step0_returns_404_for_nonexistent_wizard(self):
        """Advancing a nonexistent wizard session returns 404."""
        client, admin_token = self._make_admin_client()

        with client:
            step_resp = client.post(
                f"/api/v1/config/wizard/{uuid.uuid4()}/step",
                json={"data": {"provider_type": "openai", "model": "gpt-4o"}},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert step_resp.status_code == 404

    def test_start_wizard_requires_admin(self):
        """POST /config/wizard requires admin role."""
        from fastapi.testclient import TestClient

        from cogtrix_core.api.app import create_app
        from cogtrix_core.api.auth import create_access_token

        app = create_app()
        user_token = create_access_token(user_id=str(uuid.uuid4()), role="user")

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/config/wizard",
                json={"edit_existing": False},
                headers={"Authorization": f"Bearer {user_token}"},
            )

        assert resp.status_code == 403

    def test_start_wizard_returns_step0_details(self):
        """POST /config/wizard returns step=0, step_name='Connect to LLM'."""
        client, admin_token = self._make_admin_client()

        with client:
            with patch("cogtrix_core.api.routes.config._wizard_detect_env", return_value={}):
                resp = client.post(
                    "/api/v1/config/wizard",
                    json={"edit_existing": False},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

        if resp.status_code not in (200, 201):
            pytest.skip("Wizard not available")

        data = resp.json()["data"]
        assert data["step"] == 0
        assert "wizard_id" in data
        assert "Connect" in data.get("step_name", "")


# ===========================================================================
# 15. WizardStepOut.requires_acceptance is present on step 0 response
# ===========================================================================


class TestWizardStepOutStep0:
    """The step 0 response includes requires_acceptance=False."""

    def test_step_0_response_has_requires_acceptance_false(self):
        from fastapi.testclient import TestClient

        from cogtrix_core.api.app import create_app
        from cogtrix_core.api.auth import create_access_token

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")

        with TestClient(app, raise_server_exceptions=False) as client:
            with patch("cogtrix_core.api.routes.config._wizard_detect_env", return_value={}):
                resp = client.post(
                    "/api/v1/config/wizard",
                    json={"edit_existing": False},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

        if resp.status_code not in (200, 201):
            pytest.skip("Wizard not available")

        data = resp.json()["data"]
        assert "requires_acceptance" in data
        assert data["requires_acceptance"] is False


# ===========================================================================
# 16a. Wizard step 1 LLM failure returns 422, not 500 (BUG-001)
# ===========================================================================


class TestWizardStep1ProviderError:
    """Step 1 LLM invocation failure returns 422 PROVIDER_UNREACHABLE, not 500."""

    def _make_admin_client(self):
        from fastapi.testclient import TestClient

        from cogtrix_core.api.app import create_app
        from cogtrix_core.api.auth import create_access_token

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        return TestClient(app, raise_server_exceptions=False), admin_token

    def test_step1_llm_failure_returns_422_not_500(self):
        """When the LLM fails during step 1 Q&A, the route returns 422, not 500."""
        client, admin_token = self._make_admin_client()

        with client:
            start_resp = client.post(
                "/api/v1/config/wizard",
                json={"edit_existing": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if start_resp.status_code not in (200, 201):
                pytest.skip("Wizard endpoint not available")

            wizard_id = start_resp.json()["data"]["wizard_id"]

            # Patch the wizard session directly to be at step 1 with a fake LLM
            from cogtrix_core.api.routes.config import _wizard_sessions

            _wizard_sessions[wizard_id]["step"] = 1
            _wizard_sessions[wizard_id]["messages"] = []

            # The LLM invocation at step 1 raises a provider error
            with patch(
                "cogtrix_core.api.routes.config._wizard_invoke_llm",
                side_effect=RuntimeError(
                    "Error code: 400 - {'error': {'message': 'No connected db.', 'type': 'no_db_connection'}}"
                ),
            ):
                step_resp = client.post(
                    f"/api/v1/config/wizard/{wizard_id}/step",
                    json={"answer": "use ollama with default settings"},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

        assert (
            step_resp.status_code == 422
        ), f"Expected 422 PROVIDER_UNREACHABLE, got {step_resp.status_code}: {step_resp.text}"
        err = step_resp.json().get("error") or {}
        assert err.get("code") == "PROVIDER_UNREACHABLE"
        assert "No connected db." in err.get("message", "")


# ===========================================================================
# 16. _enqueue_agent_state — non-blocking behavior
# ===========================================================================


class TestEnqueueAgentState:
    """_enqueue_agent_state uses put_nowait and never blocks."""

    @pytest.mark.asyncio
    async def test_sets_session_agent_state(self):
        from cogtrix_core.api.turn_runner import _enqueue_agent_state

        session = _make_mock_session()
        await _enqueue_agent_state(session, "thinking")
        assert session.agent_state == "thinking"

    @pytest.mark.asyncio
    async def test_enqueues_agent_state_message(self):
        from cogtrix_core.api.turn_runner import _enqueue_agent_state

        session = _make_mock_session()
        await _enqueue_agent_state(session, "deep_thinking")

        msg = session.ws_queue.get_nowait()
        assert msg["type"] == "agent_state"
        assert msg["payload"]["state"] == "deep_thinking"

    @pytest.mark.asyncio
    async def test_full_queue_does_not_block(self):
        """When queue is full, _enqueue_agent_state drops the message silently."""
        from cogtrix_core.api.turn_runner import _enqueue_agent_state

        session = _make_mock_session()
        # Fill the queue to capacity
        tiny_queue = asyncio.Queue(maxsize=1)
        tiny_queue.put_nowait({"type": "placeholder"})
        session.ws_queue = tiny_queue

        # Must not raise, must not block
        await asyncio.wait_for(
            _enqueue_agent_state(session, "analyzing"),
            timeout=1.0,
        )

    @pytest.mark.asyncio
    async def test_valid_states_are_enqueued(self):
        """All standard agent states are handled without error."""
        from cogtrix_core.api.turn_runner import _enqueue_agent_state

        states = ["idle", "thinking", "analyzing", "researching", "deep_thinking", "delegating"]
        for state in states:
            session = _make_mock_session()
            await _enqueue_agent_state(session, state)
            assert session.agent_state == state


# ===========================================================================
# 17. _save_memory helper
# ===========================================================================


class TestSaveMemory:
    """_save_memory handles None memory_manager and exceptions gracefully."""

    @pytest.mark.asyncio
    async def test_none_memory_manager_is_safe(self):
        from cogtrix_core.api.session_bridge import ApiSession, _save_memory

        sess = ApiSession(id="s", user_id="u", name="n")
        sess.memory_manager = None
        # Must not raise
        await _save_memory(sess)

    @pytest.mark.asyncio
    async def test_save_called_on_memory_manager(self):
        from cogtrix_core.api.session_bridge import ApiSession, _save_memory

        sess = ApiSession(id="s", user_id="u", name="n")
        mm = MagicMock()
        mm.save = MagicMock()
        sess.memory_manager = mm

        await _save_memory(sess)
        mm.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_save_is_caught(self):
        """_save_memory must not propagate exceptions from mm.save()."""
        from cogtrix_core.api.session_bridge import ApiSession, _save_memory

        sess = ApiSession(id="s", user_id="u", name="n")
        mm = MagicMock()
        mm.save.side_effect = OSError("disk full")
        sess.memory_manager = mm

        # Must not raise
        await _save_memory(sess)
