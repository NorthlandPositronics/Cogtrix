"""Session metrics computation from Cogtrix log files.

Reads a Cogtrix ``.log`` file and computes the 10 behavioral effectiveness
metrics defined in ``docs/testing/agent-effectiveness-metrics.md``.

Metrics that require manual classification return ``None`` with a descriptive
note so callers can distinguish "not yet implemented" from "computation error".
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"\[(\w+)\]\s+"
    r"\[([0-9a-z]{8})\]\s+"
    r"(.+)$"
)


@dataclass
class _LogEvent:
    timestamp: str
    level: str
    session_id: str
    message: str
    line_no: int = 0


@dataclass
class _SessionData:
    events: list[_LogEvent] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    call_model_rounds: int = 0
    checkpoint_nudges: int = 0
    thinking_breaks: int = 0
    stuck_detections: int = 0
    searches: list[int] = field(default_factory=list)  # line numbers
    writes: list[int] = field(default_factory=list)  # line numbers
    checkpoints: list[str] = field(default_factory=list)


# Tools considered "productive" for TTFPA (first tangible result)
_PRODUCTIVE_TOOLS = {
    "write_file",
    "filesystem_write_file",
    "create_or_update_file",
    "edit_file",
    "push_files",
    "create_directory",
    "run_shell",
    "shell",
    "download_file",
    "request_tools",  # loading search_web is productive research
    "search_web",
}

# Tools considered "environment discovery" (not productive for TTFPA)
_DISCOVERY_TOOLS = {
    "read_text_file",
    "read_file",
    "filesystem_read_file",
    "list_directory",
    "filesystem_list_directory",
    "get_file_info",
    "as",
    "gcc",
    "wget",
    "curl",
    "python",
    "list_issues",
    "list_pull_requests",
    "get_pull_request_status",
    "get_current_datetime",
    "slack_get_channel_history",
    "slack_list_channels",
    "slack_post_message",
    "slack_get_users",
}

# Tools that create/modify files
_WRITE_TOOLS = {
    "write_file",
    "filesystem_write_file",
    "create_or_update_file",
    "edit_file",
    "push_files",
    "create_directory",
    "move_file",
}


def _parse_log(log_path: str) -> dict[str, _SessionData]:
    """Parse a Cogtrix log file into per-session event buckets."""
    sessions: dict[str, _SessionData] = defaultdict(_SessionData)

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")
            m = _LOG_RE.match(line)
            if not m:
                continue

            ts, level, session_id, message = m.groups()
            event = _LogEvent(
                timestamp=ts,
                level=level,
                session_id=session_id,
                message=message,
                line_no=line_no,
            )
            sess = sessions[session_id]
            sess.events.append(event)

            # TOOL_START
            if message.startswith("TOOL_START: "):
                tool_name = message.split("TOOL_START: ", 1)[1].split()[0]
                sess.tool_calls.append(tool_name)
                if tool_name == "checkpoint":
                    sess.checkpoints.append(f"line_{line_no}")
                if tool_name in _WRITE_TOOLS:
                    sess.writes.append(line_no)
                if tool_name == "search_web":
                    sess.searches.append(line_no)

            # call_model rounds (count unique model.invoke lines)
            if "⏱ call_model model.invoke:" in message:
                sess.call_model_rounds += 1

            # Checkpoint nudge
            if "Checkpoint nudge fired" in message:
                sess.checkpoint_nudges += 1

            # Thinking break / stuck detection
            if "thinking_break" in message.lower() or "thinking break" in message.lower():
                sess.thinking_breaks += 1
            if "Stuck detected" in message or "forcing thinking break" in message:
                sess.stuck_detections += 1

    return dict(sessions)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def _compute_cd(data: _SessionData) -> float | None:
    """Checkpoint Density (%) — checkpoint calls / total tool calls."""
    total = len(data.tool_calls)
    if total == 0:
        return None
    ckpts = sum(1 for t in data.tool_calls if t == "checkpoint")
    return round(ckpts / total * 100, 2)


def _compute_rba(data: _SessionData) -> float | None:
    """Research-Before-Action Rate (%).

    Checks whether ``search_web`` was loaded/used before the first
    non-discovery tool call.
    """
    if not data.tool_calls:
        return None

    first_search_idx = None
    first_action_idx = None

    for idx, tool in enumerate(data.tool_calls):
        if tool == "search_web" and first_search_idx is None:
            first_search_idx = idx
        if tool not in _DISCOVERY_TOOLS and first_action_idx is None:
            first_action_idx = idx

    # If no search happened → 0%
    if first_search_idx is None:
        return 0.0

    # If no non-discovery action happened → can't score
    if first_action_idx is None:
        return None

    return 100.0 if first_search_idx <= first_action_idx else 0.0


def _compute_ttfpa(data: _SessionData) -> int | None:
    """Time-to-First-Productive-Action — rounds until first productive tool."""
    if not data.tool_calls:
        return None

    # Approximate rounds by counting productive tool calls up to first one
    productive_idx = None
    for idx, tool in enumerate(data.tool_calls):
        if tool in _PRODUCTIVE_TOOLS:
            productive_idx = idx
            break

    if productive_idx is None:
        return None

    # Round ≈ index + 1 (0-indexed, but rounds start at 1)
    # More accurately: count call_model rounds up to this point
    # We use the event list to correlate
    return productive_idx + 1


def _compute_wshr(data: _SessionData) -> float | None:
    """Web Search Hit Rate (%).

    For each ``search_web`` call, check if any result URL, package name,
    command, or approach from the search output appears in subsequent tool
    calls within the next 5 rounds.

    We approximate "next 5 rounds" as the next 5 tool calls after the search.
    """
    if not data.searches:
        return None

    searches_with_hits = 0
    start = 0
    for _search_line in data.searches:
        # Find the index of this search in tool_calls
        try:
            idx = data.tool_calls.index("search_web", start)
        except ValueError:
            continue

        start = idx + 1  # Next search starts after this one

        # Look at next 5 tool calls
        subsequent = data.tool_calls[idx + 1 : idx + 6]
        # A "hit" means a non-discovery, non-search tool was used next
        if any(t not in _DISCOVERY_TOOLS and t != "search_web" for t in subsequent):
            searches_with_hits += 1

    return round(searches_with_hits / len(data.searches) * 100, 2)


def _compute_crs(data: _SessionData) -> float | None:
    """Context Retention Score (%).

    Automated proxy: 100% minus penalty for re-checking tools used shortly
    after a checkpoint.  Full CRS requires manual analysis of contradictions.
    """
    if not data.tool_calls or not data.checkpoints:
        return None

    # Penalise if the same discovery tool is called >2× in a session
    # (simple proxy for re-verification)
    discovery_counts: dict[str, int] = defaultdict(int)
    for tool in data.tool_calls:
        if tool in _DISCOVERY_TOOLS:
            discovery_counts[tool] += 1

    violations = sum(1 for c in discovery_counts.values() if c > 2)
    decision_points = max(len(data.checkpoints), 1)
    score = max(0.0, 100.0 - (violations / decision_points * 100))
    return round(score, 2)


def _compute_tce_proxy(data: _SessionData) -> dict[str, Any]:
    """Tool Call Efficiency — returns proxy + note that manual review is needed."""
    total = len(data.tool_calls)
    if total == 0:
        return {"value": None, "note": "No tool calls found"}

    # Automated proxy: count non-discovery calls as "useful"
    useful = sum(1 for t in data.tool_calls if t not in _DISCOVERY_TOOLS)
    proxy = round(useful / total * 100, 2)
    return {
        "value": proxy,
        "note": (
            f"Proxy TCE={proxy}% (non-discovery / total). Manual review required for true TCE."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_session_metrics(log_path: str) -> dict[str, Any]:
    """Parse *log_path* and return metric name → score mapping.

    Metrics requiring manual classification include a ``note`` field instead
    of a raw ``value``.
    """
    sessions = _parse_log(log_path)

    # If multiple sessions in one log, aggregate or pick the largest
    if not sessions:
        return {"error": "No parseable session data found"}

    # Use the session with the most tool calls (primary session)
    primary_session_id = max(sessions, key=lambda s: len(sessions[s].tool_calls))
    data = sessions[primary_session_id]

    # Compute individual metrics
    tce = _compute_tce_proxy(data)
    cd = _compute_cd(data)
    rba = _compute_rba(data)
    ttfpa = _compute_ttfpa(data)
    wshr = _compute_wshr(data)
    crs = _compute_crs(data)

    result: dict[str, Any] = {
        "session_id": primary_session_id,
        "total_tool_calls": len(data.tool_calls),
        "total_call_model_rounds": data.call_model_rounds,
        "metrics": {
            "task_completion_rate": {
                "value": None,
                "note": "Requires manual classification of deliverables vs requirements",
            },
            "tool_call_efficiency": tce,
            "checkpoint_density": {"value": cd, "unit": "percent"},
            "research_before_action": {"value": rba, "unit": "percent"},
            "time_to_first_productive_action": {
                "value": ttfpa,
                "unit": "rounds",
            },
            "stuck_detection_accuracy": {
                "value": None,
                "note": (
                    f"Requires manual review of {data.thinking_breaks} thinking breaks / "
                    f"{data.stuck_detections} stuck detections"
                ),
            },
            "pivot_quality": {
                "value": None,
                "note": "Requires manual classification of post-break tool category changes",
            },
            "debug_loop_efficiency": {
                "value": None,
                "note": (f"Requires diff analysis of {len(data.writes)} file write events"),
            },
            "context_retention_score": {"value": crs, "unit": "percent"},
            "web_search_hit_rate": {"value": wshr, "unit": "percent"},
        },
    }

    # Composite score (only using available metrics)
    m = result["metrics"]
    available: dict[str, float] = {}
    if m["tool_call_efficiency"]["value"] is not None:
        available["tce"] = m["tool_call_efficiency"]["value"]
    if cd is not None:
        available["cd"] = cd
    if rba is not None:
        available["rba"] = rba
    if ttfpa is not None:
        available["ttfpa"] = max(0.0, (10 - ttfpa) / 10 * 100)
    if wshr is not None:
        available["wshr"] = wshr
    if crs is not None:
        available["crs"] = crs

    # Weights (same as docs/testing/agent-effectiveness-metrics.md)
    weights = {
        "tce": 0.15,
        "cd": 0.10,
        "rba": 0.15,
        "ttfpa": 0.05,
        "wshr": 0.05,
        "crs": 0.05,
    }
    total_weight = sum(weights.get(k, 0.0) for k in available)
    if total_weight > 0:
        composite = sum(available[k] * weights[k] for k in available if k in weights) / total_weight
        result["composite_score"] = round(composite, 2)
    else:
        result["composite_score"] = None
        result["composite_note"] = "Insufficient automated metrics for composite score"

    return result


def write_session_metrics(log_path: str, out_dir: str | None = None) -> Path:
    """Compute metrics for *log_path* and write JSON to disk.

    Output path: ``~/.cogtrix/data/metrics/{session_id}.json``
    """
    result = compute_session_metrics(log_path)
    session_id = result.get("session_id", "unknown")

    if out_dir is None:
        out_dir = os.path.expanduser("~/.cogtrix/data/metrics")

    out_path = Path(out_dir) / f"{session_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return out_path
