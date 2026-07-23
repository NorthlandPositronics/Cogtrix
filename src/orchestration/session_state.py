"""Session-scoped mutable state for a single Cogtrix session.

Consolidates the 7 module-level globals that were previously scattered
across cogtrix.py into a single dataclass, enabling proper session
isolation and eliminating the need for ``global`` declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """All mutable state that belongs to one interactive session.

    Lifetime mapping:
    - ``denials``, ``loaded_tools``, ``approvals`` — session-scoped; cleared on session switch.
    - ``deny_all`` — per-prompt; reset at the start of each new prompt cycle.
    - ``pinned_tools`` — session-scoped; only cleared manually or on session switch.
    - ``no_confirm`` — process-scoped; set once at startup from CLI flags.
    - ``all_tool_descriptions``, ``all_tool_originals`` — process-scoped; populated once at startup.

    Tool loading tiers:
    - **Agent-loaded** — tools loaded by the LLM via ``request_tools`` during a turn.
      Tracked in ``loaded_tools``.  Cleared at the start of each new prompt cycle
      so the agent doesn't carry stale tools between turns.
    - **Pinned** — tools loaded manually by the user (``/tools load`` in CLI or
      ``PATCH /sessions/{id}/tools`` in API).  Tracked in ``pinned_tools``.
      Persist across prompt cycles until explicitly unloaded.
    """

    denials: set[str] = field(default_factory=set)
    deny_all: bool = False
    no_confirm: bool = False
    approvals: set[str] = field(default_factory=set)
    loaded_tools: set[str] = field(default_factory=set)
    pinned_tools: set[str] = field(default_factory=set)
    all_tool_descriptions: dict[str, str] = field(default_factory=dict)
    all_tool_originals: dict[str, Any] = field(default_factory=dict)
    checkpoint_store: Any | None = None  # CheckpointStore for checkpoint tool

    def reset_for_new_session(self) -> None:
        """Clear session-scoped state. Preserves no_confirm and tool catalogs."""
        self.denials.clear()
        self.deny_all = False
        self.loaded_tools.clear()
        self.pinned_tools.clear()
        self.approvals.clear()

    def reset_for_new_prompt(self) -> None:
        """Reset per-prompt state.

        Clears ``deny_all`` and removes agent-loaded (non-pinned) tools from
        ``loaded_tools`` so the LLM starts each turn with a clean tool set.
        Pinned tools remain in ``loaded_tools``.
        """
        self.deny_all = False
        self.loaded_tools &= self.pinned_tools
