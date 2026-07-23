"""Milestone progress reporting tool — not registry-discoverable.

Created by ``create_report_progress_tool()`` and injected directly into
``active_tools_list`` when a prompt plan with milestones is active.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

_progress_callback: Callable[[int, str], None] | None = None
_callback_lock = threading.Lock()


def set_progress_callback(cb: Callable[[int, str], None]) -> None:
    global _progress_callback
    with _callback_lock:
        _progress_callback = cb


class ReportProgressInput(BaseModel):
    milestone_index: int = Field(description="1-based index of milestone you are starting")
    status: str = Field(default="", description="Optional brief status note")


def report_progress(milestone_index: int, status: str = "") -> str:
    """Report progress — invokes the registered callback."""
    with _callback_lock:
        cb = _progress_callback
    if cb is not None:
        cb(milestone_index, status)
    return f"Progress reported: milestone {milestone_index}. Continue with the task."


def create_report_progress_tool(milestones: list) -> Any:
    """Factory: builds a StructuredTool with milestone list baked into description."""
    from langchain_core.tools import StructuredTool

    milestone_lines = "\n".join(f"{m.index}. {m.title}" for m in milestones)
    description = (
        "Report that you have started a milestone. Call this BEFORE beginning each milestone.\n\n"
        f"Milestones:\n{milestone_lines}\n\n"
        "Pass the 1-based milestone_index matching the milestone you are about to start."
    )

    return StructuredTool.from_function(
        func=report_progress,
        name="report_progress",
        description=description,
        args_schema=ReportProgressInput,
    )
