"""Session-scoped mutable state for a single Cogtrix session.

Consolidates the 7 module-level globals that were previously scattered
across cogtrix.py into a single dataclass, enabling proper session
isolation and eliminating the need for ``global`` declarations.
"""

from __future__ import annotations

import threading
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

    # Internal lock — not exposed in repr/equality; guards concurrent denial reads/writes.
    # Tool execution runs in a ThreadPoolExecutor (8 threads); API handlers mutate
    # denials/deny_all from the asyncio event loop thread via asyncio.to_thread.
    # Without this lock, budget-enforcement writes (graph.py) and API disable calls
    # (routes/tools.py) race against safety-wrapper reads (safety.py).
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def is_denied(self, tool_name: str) -> bool:
        """Atomically check deny_all and per-tool denial."""
        with self._lock:
            return self.deny_all or tool_name in self.denials

    def deny_tool(self, tool_name: str) -> None:
        """Atomically add tool_name to per-tool denials."""
        with self._lock:
            self.denials.add(tool_name)

    def allow_tool(self, tool_name: str) -> None:
        """Atomically remove tool_name from per-tool denials."""
        with self._lock:
            self.denials.discard(tool_name)

    def set_deny_all(self) -> None:
        """Atomically set deny_all = True."""
        with self._lock:
            self.deny_all = True

    def get_denials_snapshot(self) -> frozenset[str]:
        """Return an immutable snapshot of current denials for safe off-lock inspection."""
        with self._lock:
            return frozenset(self.denials)

    def reset_for_new_session(self) -> None:
        """Clear session-scoped state. Preserves no_confirm and tool catalogs.

        .. warning::
            Do NOT use ``dataclasses.replace()`` on this dataclass.  ``replace``
            copies set references by value, producing two SessionState objects
            that share the same mutable sets guarded by two unrelated locks —
            updates in one would be invisible to the other.
        """
        with self._lock:
            self.denials.clear()
            self.deny_all = False
        # approvals/loaded_tools/pinned_tools are not guarded by _lock yet;
        # clear them outside the lock to avoid holding it for unbounded time.
        self.loaded_tools.clear()
        self.pinned_tools.clear()
        self.approvals.clear()

    def reset_for_new_prompt(self) -> None:
        """Reset per-prompt state.

        Clears ``deny_all`` and removes agent-loaded (non-pinned) tools from
        ``loaded_tools`` so the LLM starts each turn with a clean tool set.
        Pinned tools remain in ``loaded_tools``.
        """
        with self._lock:
            self.deny_all = False
        self.loaded_tools &= self.pinned_tools
