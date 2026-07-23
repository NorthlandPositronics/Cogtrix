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
import json
import logging
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
        log.warning("Could not prepare context from memory manager: %s", exc)
        return []


def _extract_token_counts(ws_callback: Any) -> dict[str, int]:
    """Extract accumulated token counts and tool call count from the WebSocket callback."""
    return {
        "input_tokens": getattr(ws_callback, "input_tokens", 0),
        "output_tokens": getattr(ws_callback, "output_tokens", 0),
        "tool_call_count": getattr(ws_callback, "tool_call_count", 0),
    }


async def _enqueue_agent_state(session: ApiSession, state: str) -> None:
    """Enqueue an agent_state message on the session queue."""
    try:
        await session.ws_queue.put({"type": "agent_state", "payload": {"state": state}})
    except Exception as exc:  # pragma: no cover
        log.debug("Could not enqueue agent_state for session %s: %s", session.id, exc)


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

    # Classify — skip force-think for tool-intensive tasks.
    try:
        await _enqueue_agent_state(session, "analyzing")
        task_cat = await asyncio.to_thread(classify_think_task, user_input, llm) if llm else None
    except Exception as exc:
        log.warning("classify_think_task failed: %s", exc)
        task_cat = None

    if task_cat and task_cat.tool_intensive:
        log.info(
            "Skipping force deep_think: task classified as '%s' (tool-intensive)",
            task_cat.name,
        )
        return response_text

    called = was_deep_think_called(agent_msgs)
    if called and deep_think_had_good_context(agent_msgs):
        return response_text

    if called:
        log.info("deep_think was called but with inadequate context — forcing re-call")

    tool_data = collect_tool_outputs(agent_msgs)

    # Optionally run research delegate when web tools were used.
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
            try:
                await _enqueue_agent_state(session, "researching")
                research_output = await asyncio.to_thread(
                    run_research_delegate,
                    fetched_urls,
                    user_input,
                    max_context_tokens=max_ctx,
                    timeout=rd_timeout,
                    cap_ratio=rd_cap,
                )
            except Exception as exc:
                log.warning("Research delegate failed: %s", exc)

    try:
        await _enqueue_agent_state(session, "deep_thinking")
        result = await asyncio.to_thread(
            force_deep_think,
            user_input,
            response_text,
            tool_data,
            log,
            research_context=research_output or None,
        )
        return result
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

    log.info("Forcing parallel delegation for API turn")
    tool_data = collect_tool_outputs(agent_msgs)

    try:
        await _enqueue_agent_state(session, "delegating")
        forced = await asyncio.to_thread(
            force_delegation, user_input, response_text, tool_data, run_config, log
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
    from src.api.callbacks import WebSocketCallbackHandler
    from src.api.confirmation import ApiConfirmationUI
    from src.orchestration.runner import run_agent

    if mode not in ("normal", "think", "delegate"):
        log.warning("run_message_turn: unknown mode %r — treating as 'normal'", mode)
        mode = "normal"

    turn_start = time.monotonic()
    session.agent_state = "thinking"
    await _enqueue_agent_state(session, "thinking")

    loop = asyncio.get_running_loop()
    ws_callback = WebSocketCallbackHandler(session.ws_queue, loop)
    confirmation_ui = ApiConfirmationUI(session.ws_queue, loop)

    # Wire confirmation UI into the run config so SafeTool uses it.
    run_config = session.run_config
    if run_config is not None:
        run_config.confirmation_ui = confirmation_ui

    # Read history from memory manager.
    history_messages = _build_history(session.memory_manager, text)

    # The tool registry is needed by run_agent for requires_confirmation() checks
    # during tool auto-expansion. Pass the registry stored on the session (populated
    # from app.state.tool_registry at warm_session time), NOT available_tools dict.
    tool_registry = getattr(session, "registry", None)

    approvals: set = set()
    if session.session_state is not None:
        approvals = set(getattr(session.session_state, "approvals", set()))

    agent_msgs: list = []
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
        await session.ws_queue.put(
            {"type": "error", "payload": {"code": "CANCELLED", "message": "Agent turn cancelled."}}
        )
        raise
    except Exception as exc:
        log.exception("Agent turn failed for session %s: %s", session.id, exc)
        session.agent_state = "error"
        await _enqueue_agent_state(session, "error")
        await session.ws_queue.put(
            {"type": "error", "payload": {"code": "AGENT_ERROR", "message": str(exc)}}
        )
        return

    # ── Think / delegate post-processing ──────────────────────────
    if mode == "think" and response_text:
        response_text = await _run_think_pipeline(
            session, text, response_text, agent_msgs, run_config
        )
    elif mode == "delegate" and response_text:
        response_text = await _run_delegate_pipeline(
            session, text, response_text, agent_msgs, run_config
        )

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
    token_counts = _extract_token_counts(ws_callback)
    session.token_counts["input_tokens"] += token_counts.get("input_tokens", 0)
    session.token_counts["output_tokens"] += token_counts.get("output_tokens", 0)

    duration_ms = int((time.monotonic() - turn_start) * 1000)
    session.last_activity = time.time()
    session.agent_state = "idle"
    await _enqueue_agent_state(session, "idle")

    await session.ws_queue.put(
        {
            "type": "done",
            "payload": {
                "message_id": ai_message_id,
                "total_tokens": token_counts.get("input_tokens", 0)
                + token_counts.get("output_tokens", 0),
                "input_tokens": token_counts.get("input_tokens", 0),
                "output_tokens": token_counts.get("output_tokens", 0),
                "duration_ms": duration_ms,
                "tool_calls": token_counts.get("tool_call_count", 0),
            },
        }
    )
