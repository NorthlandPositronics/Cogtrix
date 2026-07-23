"""OpenTelemetry tracing implementation for Cogtrix.

Provides 5-layer distributed tracing:
- HTTP middleware (FastAPI) with trace_id propagation
- LLM call spans (provider, model, tokens, latency)
- Tool invocation spans (tool name, args, duration, success/failure)
- DB query spans (asyncpg instrumentation)
- MCP connection spans (connect, reconnect, tool discovery)

Configuration:
- OTEL_EXPORTER_OTLP_ENDPOINT — OTLP gRPC collector endpoint
- OTEL_SERVICE_NAME — service name (default: cogtrix)
- OTEL_TRACE_SAMPLING_RATE — sampling rate 0.0-1.0 (default: 0.1 for production)

No PII or secrets are included in span attributes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

from src.logging_config import _scrub_secrets

log = logging.getLogger("cogtrix.tracing")

# Module-level tracer instance
_tracer: trace.Tracer | None = None
_trace_id_context: dict[str, str] = {}


def _get_sampling_rate() -> float:
    """Get the trace sampling rate from environment or default to 10%."""
    raw = os.environ.get("OTEL_TRACE_SAMPLING_RATE", "").strip()
    if raw:
        try:
            rate = float(raw)
            return max(0.0, min(1.0, rate))
        except ValueError:
            pass
    # Default to 10% for production
    return 0.1


def _normalize_attribute_value(value: Any) -> Any:
    """Return a span-attribute-safe value, scrubbing secrets."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return _scrub_secrets(str(value))
    if isinstance(value, bytes):
        return _scrub_secrets(value.decode("utf-8", errors="replace"))
    if isinstance(value, (list, tuple)):
        normalized: list[Any] = []
        for item in value:
            normalized.append(_normalize_attribute_value(item))
        return normalized
    return _scrub_secrets(str(value))


def setup_tracing(service_name: str, otlp_endpoint: str | None) -> bool:
    """Configure OTLP tracing with sampling if an endpoint is provided.

    Returns True when tracing is enabled, False when no endpoint was supplied
    or initialization failed.

    Args:
        service_name: Service name for the tracer
        otlp_endpoint: OTLP gRPC endpoint (e.g., "localhost:4317")

    Returns:
        True if tracing is enabled, False otherwise
    """
    global _tracer

    if not otlp_endpoint:
        return False

    try:
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: service_name})

        # Configure sampling rate
        sampling_rate = _get_sampling_rate()
        log.info(f"OpenTelemetry sampling rate: {sampling_rate:.2%}")

        # Create tracer provider with sampling
        sampler = ParentBasedTraceIdRatio(rate=sampling_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Add OTLP exporter
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        return True
    except Exception as exc:  # pragma: no cover - defensive startup path
        log.warning(f"OpenTelemetry initialization failed: {exc}")
        return False


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance.

    Returns:
        OpenTelemetry Tracer instance. Returns a no-op tracer if tracing is
        not initialized (e.g., in test environments).
    """
    if _tracer is None:
        # Return a no-op tracer when tracing is not initialized
        # This allows tests and development to work without tracing
        return trace.get_tracer(__name__)
    return _tracer


def get_trace_id() -> str | None:
    """Get the current trace ID from the context.

    Returns:
        Current trace ID or None if not in a trace context
    """
    span = trace.get_current_span()
    if span is None:
        return None
    return format(span.get_span_context().trace_id, "032x")


def get_span_id() -> str | None:
    """Get the current span ID from the context.

    Returns:
        Current span ID or None if not in a trace context
    """
    span = trace.get_current_span()
    if span is None:
        return None
    return format(span.get_span_context().span_id, "016x")


def scrub_span_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Scrub secrets from span attributes.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        Dictionary with secrets scrubbed
    """
    scrubbed: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized = _normalize_attribute_value(value)
        if normalized is not None:
            scrubbed[key] = normalized
    return scrubbed


@contextmanager
def start_span(
    span_name: str,
    *,
    span_type: str = "default",
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start a span with normalized and scrubbed attributes.

    Args:
        span_name: Name of the span
        span_type: Type of span for instrumentation (not exposed to OTLP)
        attributes: Span attributes ( secrets are scrubbed)

    Yields:
        The active span

    Example:
        ```python
        with start_span("my_operation", span_type="http", attributes={"url": "..."}) as span:
            # Do work
            span.set_attribute("status", "success")
        ```
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            scrubbed = scrub_span_attributes(attributes)
            for key, value in scrubbed.items():
                span.set_attribute(key, value)
        yield span


# ============================================================================
# Layer-specific span helpers
# ============================================================================


@contextmanager
def start_http_span(
    method: str,
    path: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start an HTTP request span with FastAPI conventions.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        attributes: Additional span attributes

    Yields:
        The active HTTP span
    """
    span_name = f"{method} {path}"
    span_attributes: dict[str, Any] = {
        "http.method": method,
        "http.target": path,
    }
    if attributes:
        span_attributes.update(attributes)

    with start_span(
        span_name,
        span_type="http",
        attributes=span_attributes,
    ) as span:
        yield span


@contextmanager
def start_llm_span(
    provider: str,
    model: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start an LLM call span with provider/model/tokens/latency data.

    Args:
        provider: LLM provider name (openai, ollama, anthropic, etc.)
        model: Model name
        attributes: Additional span attributes (tokens_in, tokens_out, etc.)

    Yields:
        The active LLM span
    """
    span_name = f"LLM: {provider}/{model}"
    span_attributes: dict[str, Any] = {
        "llm.provider": provider,
        "llm.model": model,
    }
    if attributes:
        span_attributes.update(attributes)

    with start_span(
        span_name,
        span_type="llm",
        attributes=span_attributes,
    ) as span:
        yield span


@contextmanager
def start_tool_span(
    tool_name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start a tool invocation span with name/duration/result data.

    Args:
        tool_name: Name of the tool being invoked
        attributes: Additional span attributes (args_summary, duration, status)

    Yields:
        The active tool span
    """
    span_name = f"Tool: {tool_name}"
    span_attributes: dict[str, Any] = {
        "tool.name": tool_name,
    }
    if attributes:
        span_attributes.update(attributes)

    with start_span(
        span_name,
        span_type="tool",
        attributes=span_attributes,
    ) as span:
        yield span


@contextmanager
def start_db_span(
    query: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start a database query span with query text.

    Args:
        query: Database query text (truncated if too long)
        attributes: Additional span attributes (duration, rows_affected)

    Yields:
        The active DB span
    """
    # Truncate query to avoid PII exposure and span size limits
    max_query_len = 500
    query_preview = query[:max_query_len] + "..." if len(query) > max_query_len else query
    query_preview = _scrub_secrets(query_preview)

    span_name = "DB Query"
    span_attributes: dict[str, Any] = {
        "db.statement": query_preview,
        "db.type": "postgres",
    }
    if attributes:
        span_attributes.update(attributes)

    with start_span(
        span_name,
        span_type="db",
        attributes=span_attributes,
    ) as span:
        yield span


@contextmanager
def start_mcp_span(
    server_name: str,
    action: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span]:
    """Start an MCP connection span with server/action data.

    Args:
        server_name: MCP server name
        action: Action being performed (connect, reconnect, tool_discovery, etc.)
        attributes: Additional span attributes (tool_count, latency)

    Yields:
        The active MCP span
    """
    span_name = f"MCP: {server_name}/{action}"
    span_attributes: dict[str, Any] = {
        "mcp.server": server_name,
        "mcp.action": action,
    }
    if attributes:
        span_attributes.update(attributes)

    with start_span(
        span_name,
        span_type="mcp",
        attributes=span_attributes,
    ) as span:
        yield span
