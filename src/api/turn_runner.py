"""Agent turn execution for the API layer.

``run_message_turn`` is the single entry point for executing an agent turn
inside a WebSocket session.  It:

1. Enqueues an ``agent_state`` → "thinking" message.
2. Builds per-turn callbacks (``WebSocketCallbackHandler``, ``ApiConfirmationUI``).
3. Reads conversation history from the memory manager.
4. Runs ``run_agent`` on a worker thread via ``asyncio.to_thread``.
5. Persists the AI message to the database.
6. Updates session token counts.
7. Enqueues the ``done`` sentinel message.

``run_agent`` is always called inside ``asyncio.to_thread`` — never from an
async context — to stay compatible with LangGraph's internal threading.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

log = logging.getLogger("cogtrix.api.turn_runner")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.session_bridge import ApiSession


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_history(memory_manager: Any, user_input: str = "") -> list:
    """Extract the message history from the memory manager.

    ``prepare_context`` is the public API for all BaseMemoryManager subclasses.
    It returns a ``MemoryContext`` dataclass (not a dict), so the ``messages``
    attribute is accessed directly.  ``user_input`` is forwarded so that
    relevance-filtering modes (e.g. vector recall) can select appropriate context.
    """
    if memory_manager is None:
        return []
    try:
        ctx = memory_manager.prepare_context(user_input)
        return list(ctx.messages)
    except Exception as exc:
        log.warning(
            "Could not prepare context from memory manager — starting with empty history: %s",
            exc,
            exc_info=True,
        )
        return []


def _extract_token_counts(ws_callback: Any) -> dict[str, int]:
    """Extract accumulated token counts and tool call count from the WebSocket callback."""
    return {
        "input_tokens": getattr(ws_callback, "input_tokens", 0),
        "output_tokens": getattr(ws_callback, "output_tokens", 0),
        "tool_call_count": getattr(ws_callback, "tool_call_count", 0),
    }


async def _enqueue_agent_state(session: ApiSession, state: str) -> None:
    """Set session.agent_state and enqueue an agent_state message on the session queue."""
    session.agent_state = state
    try:
        session.ws_queue.put_nowait({"type": "agent_state", "payload": {"state": state}})
    except asyncio.QueueFull:
        log.debug("Queue full, dropping agent_state for session %s", session.id)
    except Exception as exc:  # pragma: no cover
        log.debug("Could not enqueue agent_state for session %s: %s", session.id, exc)


def _extract_final_solution(report: str) -> str:
    """Extract the Final Solution section from a Tree-of-Thought report.

    ``force_deep_think`` returns a full formatted report that includes branch
    scores, iteration summaries, and a ``## Final Solution`` section.  Only the
    final solution text should be stored as the AI response; the full report is
    only useful for debug inspection.

    Falls back to the full report if the section is absent or empty so content
    is never silently lost.
    """
    match = re.search(
        # Match any confidence value (int, float, or locale-specific decimal).
        # Lookahead stops at a horizontal rule (---) on its own line or end-of-string.
        r"^##\s+Final Solution\b[^\n]*\n+(.*?)(?=\n---|\Z)",
        report,
        re.DOTALL | re.MULTILINE,
    )
    if match:
        sol = match.group(1).strip()
        # BUG-252: when best_solution is empty, \n+ consumes all blank lines and
        # the regex captures the footer separator ("---\n*N iterations...*") as the
        # "solution".  Reject any match that starts with the horizontal-rule marker.
        if sol and not sol.lstrip().startswith("---"):
            return sol
    return report


async def _run_think_pipeline(
    session: ApiSession,
    user_input: str,
    response_text: str,
    agent_msgs: list,
    run_config: Any,
) -> str:
    """Post-process an agent turn with the deep-think pipeline.

    Mirrors the CLI logic in ``cogtrix.py`` lines 2649-2714:
    classify task → check if deep_think was already called → collect tool
    outputs → optionally run research delegate → force deep_think.
    """
    from src.orchestration.intent import classify_think_task
    from src.orchestration.phases import (
        agent_used_web_tools,
        collect_tool_outputs,
        deep_think_had_good_context,
        extract_fetched_urls,
        force_deep_think,
        run_research_delegate,
        was_deep_think_called,
    )

    llm = getattr(run_config, "llm", None) if run_config else None

    # Classify to log category and update agent_state.  Unlike the automatic
    # deep_think trigger path in the CLI, explicit think mode (mode="think") must
    # NEVER skip force_deep_think based on tool_intensive classification — the user
    # explicitly requested deep reasoning regardless of task category (BUG-248).
    try:
        await _enqueue_agent_state(session, "analyzing")
        task_cat = await asyncio.to_thread(classify_think_task, user_input, llm) if llm else None
    except Exception as exc:
        log.warning("classify_think_task failed: %s", exc)
        task_cat = None

    if task_cat:
        log.info("Think task classified as '%s'", task_cat.name)

    if session.cancel_event.is_set():
        raise asyncio.CancelledError("Cancel requested between pipeline phases")

    called = was_deep_think_called(agent_msgs)
    if called and deep_think_had_good_context(agent_msgs):
        return response_text

    if called:
        log.info("deep_think was called but with inadequate context — forcing re-call")

    tool_data = collect_tool_outputs(agent_msgs)

    # Optionally run research delegate when web tools were used.
    # BUG-249/250: delegate tools are stored in threading.local and are never set
    # in the API layer.  Capture the available tool set from run_config and inject
    # it into the worker thread via set_delegate_tools before the delegate runs.
    research_output = ""
    rd_enabled = getattr(run_config, "research_delegate_enabled", True) if run_config else True
    if rd_enabled and agent_used_web_tools(agent_msgs):
        fetched_urls = extract_fetched_urls(agent_msgs)
        if fetched_urls:
            max_ctx = getattr(run_config, "max_context_tokens", None) or 128_000
            rd_timeout = (
                getattr(run_config, "research_delegate_timeout", 300) if run_config else 300
            )
            rd_cap = (
                getattr(run_config, "research_delegate_cap_ratio", 0.85) if run_config else 0.85
            )
            _active = list(getattr(run_config, "active_tools_list", None) or [])
            _avail = dict(getattr(run_config, "available_tools", None) or {})

            def _run_research_with_tools(
                urls: list,
                task: str,
                active: list = _active,
                avail: dict = _avail,
                max_context_tokens: int | None = max_ctx,
                timeout: int = rd_timeout,
                cap_ratio: float = rd_cap,
            ) -> str:
                """Inject delegate tools into this worker thread, then run the delegate."""
                from src.tools.delegate import set_delegate_tools

                set_delegate_tools(active, avail)
                return run_research_delegate(
                    urls,
                    task,
                    max_context_tokens=max_context_tokens,
                    timeout=timeout,
                    cap_ratio=cap_ratio,
                )

            try:
                await _enqueue_agent_state(session, "researching")
                research_output = await asyncio.to_thread(
                    _run_research_with_tools,
                    fetched_urls,
                    user_input,
                )
            except Exception as exc:
                log.warning("Research delegate failed: %s", exc)

    if session.cancel_event.is_set():
        raise asyncio.CancelledError("Cancel requested between pipeline phases")

    try:
        await _enqueue_agent_state(session, "deep_thinking")
        result = await asyncio.to_thread(
            force_deep_think,
            user_input,
            response_text,
            tool_data,
            log,
            research_context=research_output or None,
            llm=llm,
        )
        if session.cancel_event.is_set():
            raise asyncio.CancelledError("Cancel requested between pipeline phases")
        return _extract_final_solution(result)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("force_deep_think failed: %s", exc)
        return response_text


async def _run_delegate_pipeline(
    session: ApiSession,
    user_input: str,
    response_text: str,
    agent_msgs: list,
    run_config: Any,
) -> str:
    """Post-process an agent turn with forced delegation.

    Mirrors the CLI logic in ``cogtrix.py`` lines 2717-2736:
    check if delegation was already called → collect tool outputs →
    force delegation.
    """
    from src.orchestration.phases import (
        collect_tool_outputs,
        force_delegation,
        was_delegation_called,
    )

    if was_delegation_called(agent_msgs):
        return response_text

    if session.cancel_event.is_set():
        raise asyncio.CancelledError("Cancel requested between pipeline phases")

    log.info("Forcing parallel delegation for API turn")
    tool_data = collect_tool_outputs(agent_msgs)

    try:
        await _enqueue_agent_state(session, "delegating")
        _delegation_llm = getattr(run_config, "llm", None) if run_config else None
        forced = await asyncio.to_thread(
            force_delegation,
            user_input,
            response_text,
            tool_data,
            run_config,
            log,
            _delegation_llm,
        )
        if forced and forced != response_text:
            return forced
    except Exception as exc:
        log.warning("force_delegation failed: %s", exc)

    return response_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_message_turn(
    session: ApiSession,
    text: str,
    mode: str = "normal",
    db: AsyncSession | None = None,
    app_state: Any = None,
) -> None:
    """Execute one agent turn for the given session.

    The function is designed to run as an ``asyncio.Task`` (via
    ``asyncio.create_task``).  It blocks on the agent thread via
    ``asyncio.to_thread`` and streams results back through ``session.ws_queue``.

    Args:
        session: Live ApiSession holding LLM, memory, and run config.
        text: The user's message text.
        mode: Execution mode — "normal", "think", or "delegate".
        db: Async DB session for persisting the AI message.
        app_state: FastAPI app.state (unused currently; reserved for future hooks).
    """
    if mode not in ("normal", "think", "delegate"):
        log.warning("run_message_turn: unknown mode %r — treating as 'normal'", mode)
        mode = "normal"

    # Auto-promote COMPLEX_RESEARCH tasks to delegation when the caller did not
    # explicitly request a mode.  This mirrors the interactive-loop promotion in
    # cogtrix.py so API sessions benefit from the same adaptive strategy.
    if mode == "normal":
        from src.orchestration.intent import TaskComplexity, classify_task_complexity

        if classify_task_complexity(text) == TaskComplexity.COMPLEX_RESEARCH:
            log.info("Complex research task detected — auto-promoting API turn to delegate mode")
            mode = "delegate"

    async with session.turn_lock:
        await _run_message_turn_inner(session, text, mode, db, app_state)


async def _run_message_turn_inner(
    session: ApiSession,
    text: str,
    mode: str,
    db: Any,
    app_state: Any,
) -> None:
    """Execute one agent turn while holding ``session.turn_lock``.

    Separated from ``run_message_turn`` so the lock scope is obvious and
    tests can call this directly when they already hold the lock.
    """
    from src.api.callbacks import WebSocketCallbackHandler
    from src.api.confirmation import ApiConfirmationUI
    from src.orchestration.runner import run_agent

    turn_start = time.monotonic()
    log.debug("Turn start: session=%s mode=%s", session.id, mode)
    await _enqueue_agent_state(session, "thinking")

    # Clear ephemeral per-prompt state: remove agent-loaded (non-pinned) tools
    # from loaded_tools and reset deny_all — matching the CLI prompt boundary
    # behaviour (BUG-198).
    if session.session_state is not None:
        session.session_state.reset_for_new_prompt()

    loop = asyncio.get_running_loop()
    ws_callback = WebSocketCallbackHandler(session.ws_queue, loop)
    confirmation_ui = ApiConfirmationUI(session.ws_queue, loop)

    # Publish the confirmation UI on the session so the WebSocket handler can route
    # tool_confirm messages to it.  The field is cleared in the finally block so stale
    # UI references never linger between turns (BUG-FORGE-001).
    session.active_confirmation_ui = confirmation_ui

    # Build a per-turn copy of the run config with the confirmation UI wired in.
    # Never mutate the shared session.run_config — concurrent REST + WebSocket turns
    # would race on the same object (BUG-117).
    # Deep-copy mutable fields so tool expansion in process_tools doesn't mutate the
    # session-level lists/dicts (BUG-134).
    run_config = session.run_config
    if run_config is not None:
        run_config = dataclasses.replace(
            run_config,
            confirmation_ui=confirmation_ui,
            active_tools_list=(
                list(run_config.active_tools_list) if run_config.active_tools_list else []
            ),
            available_tools=dict(run_config.available_tools) if run_config.available_tools else {},
        )

    # Read history from memory manager.
    history_messages = _build_history(session.memory_manager, text)
    log.debug("Context prepared: session=%s history_msgs=%d", session.id, len(history_messages))

    # The tool registry is needed by run_agent for requires_confirmation() checks
    # during tool auto-expansion. Pass the registry stored on the session (populated
    # from app.state.tool_registry at warm_session time), NOT available_tools dict.
    tool_registry = getattr(session, "registry", None)

    approvals: set = set()
    if session.session_state is not None:
        approvals = set(getattr(session.session_state, "approvals", set()))

    agent_msgs: list = []
    try:
        try:
            response_text: str = await asyncio.to_thread(
                run_agent,
                text,
                history_messages,
                tool_registry,
                approvals,
                callbacks=[ws_callback],
                config=run_config,
                result_messages=agent_msgs,
            )
        except asyncio.CancelledError:
            session.agent_state = "idle"
            await _enqueue_agent_state(session, "idle")
            try:
                session.ws_queue.put_nowait(
                    {
                        "type": "error",
                        "payload": {"code": "CANCELLED", "message": "Agent turn cancelled."},
                    }
                )
            except asyncio.QueueFull:
                log.warning("Queue full, dropping CANCELLED error for session %s", session.id)
            raise
        except Exception as exc:
            log.exception("Agent turn failed for session %s: %s", session.id, exc)
            session.agent_state = "error"
            await _enqueue_agent_state(session, "error")
            try:
                session.ws_queue.put_nowait(
                    {"type": "error", "payload": {"code": "AGENT_ERROR", "message": str(exc)}}
                )
            except asyncio.QueueFull:
                log.warning("Queue full, dropping AGENT_ERROR for session %s", session.id)
            # Emit done so the sync drain loop in send_message terminates correctly
            # and can surface the error rather than returning HTTP 200 with empty text.
            try:
                session.ws_queue.put_nowait(
                    {"type": "done", "payload": {"text": "", "error": str(exc)}}
                )
            except asyncio.QueueFull:
                log.warning("Queue full, dropping done-on-error for session %s", session.id)
            session.agent_state = "idle"
            await _enqueue_agent_state(session, "idle")
            return

        # ── Think / delegate post-processing ──────────────────────────
        try:
            if mode == "think" and response_text:
                response_text = await _run_think_pipeline(
                    session, text, response_text, agent_msgs, run_config
                )
            elif mode == "delegate" and response_text:
                response_text = await _run_delegate_pipeline(
                    session, text, response_text, agent_msgs, run_config
                )
        except asyncio.CancelledError:
            # Pipeline phase was cancelled — reset session state to idle before
            # re-raising so the WebSocket client sees the session as available
            # and not stuck in "analyzing", "researching", or "deep_thinking".
            session.agent_state = "idle"
            await _enqueue_agent_state(session, "idle")
            try:
                session.ws_queue.put_nowait(
                    {
                        "type": "error",
                        "payload": {
                            "code": "CANCELLED",
                            "message": "Agent turn cancelled during post-processing.",
                        },
                    }
                )
            except asyncio.QueueFull:
                log.warning("Queue full, dropping CANCELLED error for session %s", session.id)
            raise

        # BUG-253: deep_think / delegation run inside asyncio.to_thread without the
        # ws_callback, so no incremental tokens are streamed during these phases.
        # Emit the final result as a single token message now so the frontend has
        # content to display regardless of whether it clears the earlier run_agent
        # tokens on the "analyzing" state transition.
        if mode in ("think", "delegate") and response_text:
            try:
                session.ws_queue.put_nowait(
                    {"type": "token", "payload": {"text": response_text, "final": True}}
                )
            except asyncio.QueueFull:
                log.debug("Queue full, dropping think-result token for session %s", session.id)

        # Update memory with the new exchange.
        if session.memory_manager is not None:
            try:
                # update() is the public API for all BaseMemoryManager subclasses.
                # save() performs file I/O — run both off the event loop thread.
                mm = session.memory_manager

                def _update_and_save() -> None:
                    mm.update(text, response_text)
                    mm.save()

                await asyncio.to_thread(_update_and_save)
            except Exception as exc:
                log.warning("Memory update failed for session %s: %s", session.id, exc)

        # Persist AI message to DB.
        ai_message_id = str(uuid.uuid4())
        if db is not None:
            try:
                from src.api.db.repositories.messages import MessageRepository

                msg_repo = MessageRepository(db)
                ai_msg = await msg_repo.create(
                    session_id=session.id,
                    role="assistant",
                    content_json=json.dumps({"text": response_text}),
                )
                ai_message_id = ai_msg.id
                await db.commit()
            except Exception as exc:
                log.warning("Could not persist AI message for session %s: %s", session.id, exc)
                try:
                    await db.rollback()
                except Exception:
                    pass

        # Update session token counts from the callback accumulator.
        # Use .get() with a default for the LHS so a missing key (e.g. from an
        # incomplete token_counts_json in the DB or test construction) never
        # raises KeyError.
        token_counts = _extract_token_counts(ws_callback)
        session.token_counts["input_tokens"] = session.token_counts.get(
            "input_tokens", 0
        ) + token_counts.get("input_tokens", 0)
        session.token_counts["output_tokens"] = session.token_counts.get(
            "output_tokens", 0
        ) + token_counts.get("output_tokens", 0)

        duration_ms = int((time.monotonic() - turn_start) * 1000)
        log.debug(
            "Turn complete: session=%s mode=%s duration_ms=%d in=%d out=%d tools=%d",
            session.id,
            mode,
            duration_ms,
            token_counts.get("input_tokens", 0),
            token_counts.get("output_tokens", 0),
            token_counts.get("tool_call_count", 0),
        )
        session.last_activity = time.time()
        session.agent_state = "idle"
        await _enqueue_agent_state(session, "idle")

        done_msg = {
            "type": "done",
            "payload": {
                "message_id": ai_message_id,
                "total_tokens": token_counts.get("input_tokens", 0)
                + token_counts.get("output_tokens", 0),
                "input_tokens": token_counts.get("input_tokens", 0),
                "output_tokens": token_counts.get("output_tokens", 0),
                "duration_ms": duration_ms,
                "tool_calls": token_counts.get("tool_call_count", 0),
                "text": response_text,
            },
        }
        try:
            await asyncio.wait_for(session.ws_queue.put(done_msg), timeout=5.0)
        except TimeoutError:
            log.warning("Queue put timeout, dropping done message for session %s", session.id)
    finally:
        # Clear the active confirmation UI so stale references never linger
        # after the turn completes or is cancelled (BUG-FORGE-001).
        session.active_confirmation_ui = None
