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
    - ``no_confirm`` — process-scoped; set once at startup from CLI flags.
    - ``all_tool_descriptions``, ``all_tool_originals`` — process-scoped; populated once at startup.
    """

    denials: set[str] = field(default_factory=set)
    deny_all: bool = False
    no_confirm: bool = False
    approvals: set[str] = field(default_factory=set)
    loaded_tools: set[str] = field(default_factory=set)
    all_tool_descriptions: dict[str, str] = field(default_factory=dict)
    all_tool_originals: dict[str, Any] = field(default_factory=dict)

    def reset_for_new_session(self) -> None:
        """Clear session-scoped state. Preserves no_confirm and tool catalogs."""
        self.denials.clear()
        self.deny_all = False
        self.loaded_tools.clear()
        self.approvals.clear()

    def reset_for_new_prompt(self) -> None:
        """Reset per-prompt blanket-forbid flag."""
        self.deny_all = False
