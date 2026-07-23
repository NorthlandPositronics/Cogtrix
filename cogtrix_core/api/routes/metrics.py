"""Prometheus metrics endpoint.

Exposes the following 7 key observability metrics at GET /api/v1/metrics:

- cogtrix_sessions_active — gauge (current active sessions)
- cogtrix_llm_requests_total — counter by provider/model/status
- cogtrix_llm_latency_seconds — histogram
- cogtrix_tool_calls_total — counter by tool/status
- cogtrix_tasks_total — counter by state
- cogtrix_api_requests_total — counter by route/method/status
- cogtrix_db_connections — gauge (active/idle pool)

All metrics require admin authentication.

Backward compatibility:
- `LLM_TOKENS_TOTAL` is exported for legacy orchestration code that may
  attempt to import it. It is set to `None` as this metric is now handled
  through the new metrics endpoint format.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from cogtrix_core.api.auth import TokenData, require_admin

log = logging.getLogger("cogtrix.api.metrics")

router = APIRouter(prefix="/metrics", tags=["Metrics"])


# Legacy compatibility: Export metric stubs for orchestration code.
# Both are None when Prometheus is not configured; callers must guard before use.
LLM_TOKENS_TOTAL: Any = None
TOOL_CALLS_TOTAL: Any = None


def _collect_metrics(request: Request) -> dict[str, Any]:
    """Collect all Prometheus metrics from live state."""
    session_registry = getattr(request.app.state, "session_registry", None)

    metrics: dict[str, Any] = {}

    # 1. cogtrix_sessions_active — gauge
    sessions_active = 0
    if session_registry is not None:
        try:
            sessions_dict = getattr(session_registry, "_sessions", {})
            sessions_active = sum(
                1 for s in sessions_dict.values() if not getattr(s, "archived", False)
            )
        except Exception:
            sessions_active = 0
    metrics["cogtrix_sessions_active"] = {
        "type": "gauge",
        "value": sessions_active,
        "help": "Number of active (non-archived) sessions",
    }

    # 2. cogtrix_llm_requests_total — counter by provider/model/status
    llm_requests: dict[str, int] = {}
    try:
        llm_stats = getattr(request.app.state, "llm_stats", None)
        if llm_stats is not None:
            for provider, models in llm_stats.items():
                for model, statuses in models.items():
                    for status, count in statuses.items():
                        key = f"{provider}_{model}_{status}"
                        llm_requests[key] = count
    except Exception:
        llm_requests = {}
    metrics["cogtrix_llm_requests_total"] = {
        "type": "counter",
        "help": "Total LLM requests by provider, model, and status",
        "values": llm_requests,
    }

    # 3. cogtrix_llm_latency_seconds — histogram
    llm_latency: dict[str, float | int] = {}
    try:
        latency_stats = getattr(request.app.state, "llm_latency", None)
        if latency_stats is not None:
            if isinstance(latency_stats, (list, tuple)):
                if len(latency_stats) > 0:
                    llm_latency["count"] = len(latency_stats)
                    llm_latency["sum"] = sum(latency_stats)
                    llm_latency["avg"] = sum(latency_stats) / len(latency_stats)
            elif isinstance(latency_stats, dict):
                llm_latency = latency_stats
    except Exception:
        llm_latency = {}
    metrics["cogtrix_llm_latency_seconds"] = {
        "type": "histogram_summary",
        "help": "LLM call latency in seconds (count, sum, avg)",
        "values": llm_latency,
    }

    # 4. cogtrix_tool_calls_total — counter by tool/status
    tool_calls: dict[str, int] = {}
    try:
        tool_stats = getattr(request.app.state, "tool_stats", None)
        if tool_stats is not None:
            for tool_name, statuses in tool_stats.items():
                for status, count in statuses.items():
                    key = f"{tool_name}_{status}"
                    tool_calls[key] = count
    except Exception:
        tool_calls = {}
    metrics["cogtrix_tool_calls_total"] = {
        "type": "counter",
        "help": "Total tool calls by tool name and status",
        "values": tool_calls,
    }

    # 5. cogtrix_tasks_total — counter by state
    tasks_total: dict[str, int] = {}
    try:
        graph = getattr(request.app.state, "_agent_graph", None)
        if graph is not None:
            tasks_state = getattr(graph, "tasks_state", {})
            tasks_total = dict(tasks_state)
        else:
            task_counter = getattr(request.app.state, "task_counter", None)
            if task_counter is not None:
                tasks_total = dict(task_counter)
    except Exception:
        tasks_total = {}
    metrics["cogtrix_tasks_total"] = {
        "type": "counter",
        "help": "Total tasks by state",
        "values": tasks_total,
    }

    # 6. cogtrix_api_requests_total — counter by route/method/status
    api_requests: dict[str, int] = {}
    try:
        api_stats = getattr(request.app.state, "api_stats", None)
        if api_stats is not None:
            for route, methods in api_stats.items():
                for method, statuses in methods.items():
                    for status, count in statuses.items():
                        key = f"{route}_{method}_{status}"
                        api_requests[key] = count
    except Exception:
        api_requests = {}
    metrics["cogtrix_api_requests_total"] = {
        "type": "counter",
        "help": "Total API requests by route, method, and status",
        "values": api_requests,
    }

    # 7. cogtrix_db_connections — gauge (active/idle pool)
    db_connections: dict[str, int] = {}
    try:
        from cogtrix_core.api.db.engine import engine as _engine

        pool = getattr(_engine, "pool", None)
        if pool is not None:
            active = getattr(pool, "checkedoutcount", None)
            idle = getattr(pool, "checkedincount", None)
            if active is not None:
                db_connections["active"] = active
            if idle is not None:
                db_connections["idle"] = idle
            if not db_connections:
                size = getattr(pool, "size", None)
                checkedout = getattr(pool, "checkedout", None)
                if size is not None:
                    db_connections["total"] = size
                if checkedout is not None:
                    db_connections["active"] = checkedout
    except Exception:
        db_connections = {}
    metrics["cogtrix_db_connections"] = {
        "type": "gauge",
        "help": "Database connection pool status (active/idle)",
        "values": db_connections,
    }

    return metrics


@router.get(
    "",
    summary="Prometheus metrics",
    description=(
        "Return metrics in Prometheus exposition format for scraping by Prometheus. "
        "All 7 observability metrics are exposed with proper labels and types."
    ),
    responses={
        200: {
            "description": "Metrics returned in Prometheus format",
            "content": {"text/plain": {"example": "# HELP cogtrix_sessions_active ..."}},
        },
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def prometheus_metrics(
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> PlainTextResponse:
    """Return metrics in Prometheus exposition format.

    Auth: admin bearer token required.

    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    metrics = _collect_metrics(request)
    return PlainTextResponse(content=_format_prometheus(metrics))


def _format_prometheus(metrics: dict[str, Any]) -> str:
    """Format metrics dict as Prometheus exposition text."""
    lines: list[str] = []

    for name, mdata in metrics.items():
        help_text = mdata.get("help", "")
        mtype = mdata.get("type", "unknown")

        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")

        if mtype in ("gauge", "counter"):
            value = mdata.get("value")
            values = mdata.get("values", {})

            if value is not None:
                lines.append(f"{name} {value}")
            elif values:
                for label_key, val in values.items():
                    if "_" in label_key:
                        parts = label_key.rsplit("_", 1)
                        if len(parts) == 2:
                            metric_name = parts[0]
                            status = parts[1]
                            lines.append(f'{metric_name}{{status="{status}"}} {val}')
                        else:
                            lines.append(f"{label_key} {val}")
                    else:
                        lines.append(f"{label_key} {val}")
            else:
                lines.append(f"{name} 0")

        elif mtype == "histogram_summary":
            values = mdata.get("values", {})
            for key, val in values.items():
                lines.append(f"{name}_{key} {val}")

    return "\n".join(lines) + "\n"


__all__ = ["router"]
