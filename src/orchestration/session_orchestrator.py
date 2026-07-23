"""Session-switching orchestrator for mode/model/provider/session switches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cogtrix")


@dataclass
class SessionSnapshot:
    """Captured state before a switch attempt, for rollback on failure."""

    # Config fields
    model: str | None = None
    provider: str | None = None
    active_model: Any = None
    memory_mode: str | None = None
    memory_config: Any = None
    session: str | None = None

    # Runtime objects
    system_prompt: str | None = None
    memory_manager: Any = None

    # Tool state
    registry_tools: dict[str, Any] = field(default_factory=dict)
    available_tools: dict[str, Any] = field(default_factory=dict)
    tools: list[Any] = field(default_factory=list)


class SessionOrchestrator:
    """Consolidates snapshot-try-rollback logic for session switches.

    Wraps the live session objects so handlers can call ``snapshot()``
    before attempting a switch and ``rollback(snap)`` if the switch fails.

    The orchestrator does not perform the switches itself — it only captures
    and restores state.  All switch logic stays in ``cogtrix.py``.
    """

    def __init__(
        self,
        config: Any,
        slash_cmds_ref: Any,
    ) -> None:
        self._config = config
        self._slash_cmds_ref = slash_cmds_ref

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        *,
        memory_manager: Any = None,
        system_prompt: str | None = None,
        registry_tools: dict[str, Any] | None = None,
        available_tools: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> SessionSnapshot:
        """Capture the current session state for potential rollback.

        Callers pass the local variables that only exist inside ``main()``
        (``memory_manager``, ``system_prompt``, tool lists) as keyword
        arguments; config-level state is read directly from ``self._config``.
        """
        cfg = self._config
        return SessionSnapshot(
            model=cfg.model,
            provider=cfg.provider,
            active_model=getattr(cfg, "_active_model", None),
            memory_mode=cfg.memory_mode,
            memory_config=cfg.memory_config,
            session=cfg.session,
            system_prompt=system_prompt,
            memory_manager=memory_manager,
            registry_tools=dict(registry_tools) if registry_tools is not None else {},
            available_tools=dict(available_tools) if available_tools is not None else {},
            tools=list(tools) if tools is not None else [],
        )

    def rollback(
        self,
        snap: SessionSnapshot,
        *,
        tools_list: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Restore config-level state from a snapshot.

        Returns a dict of the local-variable values that ``main()`` must
        reassign itself (``memory_manager``, ``system_prompt``,
        ``available_tools``, and optionally the tool list).

        ``tools_list`` is the live ``tools`` list from ``main()``.  When
        provided, it is cleared and refilled in-place from the snapshot so
        the same list object is mutated (matching the existing behaviour).
        """
        cfg = self._config

        # Restore config-level fields
        cfg.model = snap.model
        cfg.provider = snap.provider
        if hasattr(cfg, "_active_model"):
            cfg._active_model = snap.active_model
        cfg.memory_mode = snap.memory_mode
        cfg.memory_config = snap.memory_config
        cfg.session = snap.session

        # Restore tool list in-place when the caller supplies it
        if tools_list is not None and snap.tools is not None:
            tools_list.clear()
            tools_list.extend(snap.tools)

        # Update slash command references
        sc = self._slash_cmds_ref
        if snap.system_prompt is not None:
            sc.system_prompt = snap.system_prompt
        if snap.memory_manager is not None:
            sc.memory_manager = snap.memory_manager
        sc.available_tools = snap.available_tools

        return {
            "memory_manager": snap.memory_manager,
            "system_prompt": snap.system_prompt,
            "available_tools": snap.available_tools,
        }
