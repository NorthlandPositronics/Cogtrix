"""Checkpoint tool — records progress findings that survive context compression.

The agent calls ``checkpoint(finding)`` when it discovers something important
(a working tool path, a successful download, a key decision).  Checkpoints
are stored in a ``CheckpointStore`` that the graph injects before every LLM
call, ensuring they're always visible regardless of context window pressure.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class CheckpointStore:
    """Thread-safe store for progress checkpoints."""

    _findings: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    max_checkpoints: int = 15

    def add(self, finding: str) -> int:
        """Record a finding. Returns the 1-based checkpoint index."""
        with self._lock:
            if len(self._findings) >= self.max_checkpoints:
                # Keep most recent, drop oldest
                self._findings.pop(0)
            self._findings.append(finding.strip())
            return len(self._findings)

    def summary(self) -> str:
        """Build a summary string for context injection."""
        with self._lock:
            if not self._findings:
                return ""
            lines = [f"  {i + 1}. {f}" for i, f in enumerate(self._findings)]
            return (
                "[Progress checkpoints — these are confirmed findings, "
                "do NOT re-investigate them:]\n" + "\n".join(lines)
            )

    def clear(self) -> None:
        with self._lock:
            self._findings.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._findings)


class CheckpointInput(BaseModel):
    """Input schema for checkpoint tool."""

    finding: str = Field(
        description=(
            "A concise statement of what you discovered or accomplished. "
            "Examples: 'as (GNU Assembler 2.38) works at ~/bin/as with "
            "LD_LIBRARY_PATH=/tmp/libs', 'downloaded binutils source to /tmp/binutils.tar.gz', "
            "'port 8080 is available for the web server'."
        ),
    )


def create_checkpoint_tool(store: CheckpointStore) -> Any:
    """Factory: build a checkpoint StructuredTool bound to the given store."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return None

    def checkpoint(finding: str) -> str:
        idx = store.add(finding)
        return f"Checkpoint #{idx} recorded. Continue with the next step."

    return StructuredTool.from_function(
        func=checkpoint,
        name="checkpoint",
        description=(
            "Record a progress finding that should be remembered throughout the task. "
            "Use this for BOTH successes and important failures:\n"
            "- Success: 'Tool X confirmed working at /path/to/tool'\n"
            "- Failure: 'Approach Y failed because of dependency Z'\n"
            "- Decision: 'Switching to approach W since Y is incompatible'\n\n"
            "Checkpoints are always visible to you regardless of context length. "
            "Use them to avoid re-investigating things you already know and to "
            "track which approaches have been tried.\n\n"
            "When recording the ABSENCE of a tool or capability, use the format "
            "'CONFIRMED ABSENT: as, ld, gcc — do NOT run which/find for these again.'\n\n"
            "Call this IMMEDIATELY after each significant outcome — don't wait."
        ),
        args_schema=CheckpointInput,
    )


__all__ = ["CheckpointStore", "CheckpointInput", "create_checkpoint_tool"]
