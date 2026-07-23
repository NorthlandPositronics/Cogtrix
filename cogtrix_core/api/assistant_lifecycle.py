"""Shared assistant service lifecycle helpers.

Extracted from ``routes/assistant.py`` so that both the lifespan auto-start
in ``app.py`` and the ``POST /api/v1/assistant/start`` endpoint use the same
code path.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger("cogtrix.api")


async def create_and_start_assistant(
    config: Any,
    tool_registry: Any,
) -> Any:
    """Build an ``AssistantService``, start its background threads, and return it.

    Raises ``RuntimeError`` on failure so callers can translate to the
    appropriate HTTP or log-level response.
    """
    from cogtrix_core.assistant.service import AssistantService
    from cogtrix_core.providers import create_chat_model_from_configs

    # Resolve the LLM config from the app Config object and build the LLM.
    # create_chat_model_from_configs may perform network I/O (e.g. Ollama API
    # introspection); run it off the event loop thread to prevent stalls.
    pc, mc = config.resolve_llm_config()
    llm = await asyncio.to_thread(create_chat_model_from_configs, pc, mc)
    tools_dict: dict[str, Any] = dict(tool_registry.tools)
    svc = AssistantService(
        config=config,
        llm=llm,
        registry=tool_registry,
        system_prompt="",
        available_tools=tools_dict,
        active_tools=[],
    )
    # Start background threads without blocking (do not call run() which blocks)
    svc._poller.start()
    svc._scheduler.start()
    if svc._deferral_mgr is not None:
        svc._deferral_mgr.start()
    svc._started_at = datetime.now(UTC)  # type: ignore[attr-defined]
    return svc


def shutdown_assistant_sync(svc: Any) -> None:
    """Gracefully stop an ``AssistantService`` (blocking, run via ``to_thread``).

    Mirrors the shutdown sequence in ``routes/assistant.py:stop_assistant``.
    """
    poller = getattr(svc, "_poller", None)
    scheduler = getattr(svc, "_scheduler", None)
    deferral_mgr = getattr(svc, "_deferral_mgr", None)
    executor = getattr(svc, "_executor", None)
    session_mgr = getattr(svc, "_session_mgr", None)
    knowledge_store = getattr(svc, "_knowledge_store", None)

    try:
        if poller is not None:
            poller.stop()
    except Exception as exc:
        log.warning("Error stopping poller: %s", exc)
    try:
        if scheduler is not None:
            scheduler.stop()
            scheduler.save()
    except Exception as exc:
        log.warning("Error stopping scheduler: %s", exc)
    try:
        if deferral_mgr is not None:
            deferral_mgr.stop()
            deferral_mgr.save()
    except Exception as exc:
        log.warning("Error stopping deferral manager: %s", exc)
    try:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    except Exception as exc:
        log.warning("Error shutting down executor: %s", exc)
    try:
        if session_mgr is not None:
            session_mgr.save_all()
    except Exception as exc:
        log.warning("Error saving sessions: %s", exc)
    try:
        if knowledge_store is not None:
            knowledge_store.save()
    except Exception as exc:
        log.warning("Error saving knowledge store: %s", exc)
