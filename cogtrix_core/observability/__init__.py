"""Observability module for Cogtrix.

Provides tracing, metrics, and logging infrastructure.
"""

from cogtrix_core.observability.tracing import (
    get_trace_id,
    get_tracer,
    scrub_span_attributes,
    start_db_span,
    start_http_span,
    start_llm_span,
    start_mcp_span,
    start_tool_span,
)

__all__ = [
    "get_tracer",
    "get_trace_id",
    "start_http_span",
    "start_llm_span",
    "start_tool_span",
    "start_db_span",
    "start_mcp_span",
    "scrub_span_attributes",
]
