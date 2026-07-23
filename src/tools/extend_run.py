"""Mid-run self-extension tool — lets the agent request more steps or delegate.

The agent calls ``extend_run`` when it realizes a task is too complex for the
remaining step budget.  Two modes:

* **continue** — signal the runner to re-invoke the graph with a higher step
  limit after the current run exhausts its budget.  The agent keeps working
  sequentially.
* **delegate** — provide a list of independent subtask descriptions.  After
  the current run ends, the runner spawns parallel sub-agents for each
  subtask, collects their results, and feeds the combined output back to the
  agent for synthesis.

The tool communicates with the runner via a shared ``ExtendRunState`` object
injected into the graph closure at build time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class ExtendRunState:
    """Shared state between the extend_run tool and the graph runner."""

    requested: bool = False
    mode: str = "continue"  # "continue" or "delegate"
    subtasks: list[str] = field(default_factory=list)
    reason: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request_extension(
        self, mode: str = "continue", subtasks: list[str] | None = None, reason: str = ""
    ) -> None:
        with self._lock:
            self.requested = True
            self.mode = mode
            self.subtasks = subtasks or []
            self.reason = reason

    def reset(self) -> None:
        with self._lock:
            self.requested = False
            self.mode = "continue"
            self.subtasks = []
            self.reason = ""


class ExtendRunInput(BaseModel):
    """Input schema for extend_run tool."""

    mode: str = Field(
        default="continue",
        description=(
            "Extension mode: 'continue' to request more steps for sequential work, "
            "or 'delegate' to split remaining work into parallel sub-agents."
        ),
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description=(
            "Required when mode='delegate': list of independent subtask descriptions "
            "that can be executed in parallel by sub-agents. Each subtask should be "
            "self-contained and produce a clear result."
        ),
    )
    reason: str = Field(
        default="",
        description="Brief explanation of why more steps or delegation is needed.",
    )


def create_extend_run_tool(state: ExtendRunState) -> Any:
    """Factory: build an extend_run StructuredTool bound to the given state."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return None

    def extend_run(
        mode: str = "continue", subtasks: list[str] | None = None, reason: str = ""
    ) -> str:
        if mode == "delegate" and not subtasks:
            return (
                "Error: mode='delegate' requires a non-empty 'subtasks' list. "
                "Provide 2-5 independent subtask descriptions."
            )
        state.request_extension(mode=mode, subtasks=subtasks or [], reason=reason)
        if mode == "delegate":
            return (
                f"Extension registered: {len(subtasks or [])} subtasks queued for parallel delegation. "
                "Continue with any remaining sequential work; delegation will execute after this run."
            )
        return (
            "Extension registered: the step budget will be increased when the current limit "
            "is reached. Continue working on the task."
        )

    return StructuredTool.from_function(
        func=extend_run,
        name="extend_run",
        description=(
            "Request more execution steps or delegate work to parallel sub-agents. "
            "Call this when you realize the task needs significantly more iterations "
            "than available, or when the task has independent subtasks that can run "
            "in parallel.\n\n"
            "Modes:\n"
            "- 'continue': Request more sequential steps (for builds, multi-step installs)\n"
            "- 'delegate': Split work into parallel sub-agents (for research from multiple "
            "angles, independent analysis tasks). Requires a 'subtasks' list.\n\n"
            "Call this EARLY — don't wait until you're almost out of steps."
        ),
        args_schema=ExtendRunInput,
    )


__all__ = ["ExtendRunState", "ExtendRunInput", "create_extend_run_tool"]
