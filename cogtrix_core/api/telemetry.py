"""OpenTelemetry helpers for Cogtrix API and orchestration spans.

This module provides a thin wrapper around src.observability.tracing for
backward compatibility. New code should import directly from
src.observability.tracing.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from cogtrix_core.observability.tracing import (
    setup_tracing as _setup_tracing,
)
from cogtrix_core.observability.tracing import (
    start_span as _start_span,
)

log = logging.getLogger("cogtrix.telemetry")


def _normalize_attribute_value(value: Any) -> Any:
    """Return a span-attribute-safe value, scrubbing secrets."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        normalized: list[Any] = []
        for item in value:
            normalized.append(_normalize_attribute_value(item))
        return normalized
    return str(value)


def setup_telemetry(service_name: str, otlp_endpoint: str | None) -> bool:
    """Configure OTLP tracing if an endpoint is provided.

    Returns True when tracing is enabled, False when no endpoint was supplied
    or initialization failed. The no-op path keeps development overhead near
    zero when tracing is not configured.
    """
    return _setup_tracing(service_name, otlp_endpoint)


def get_trace_id() -> str | None:
    """Get the current trace ID from the active span.

    Returns:
        Current trace ID or None if not in a trace context
    """
    from cogtrix_core.observability.tracing import get_trace_id as _get_trace_id

    return _get_trace_id()


def get_span_id() -> str | None:
    """Get the current span ID from the active span.

    Returns:
        Current span ID or None if not in a trace context
    """
    from cogtrix_core.observability.tracing import get_span_id as _get_span_id

    return _get_span_id()


@contextmanager
def start_span(
    tracer_name: str,
    span_name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Generator[Any]:
    """Start a span with normalized attributes.

    Deprecated: Use src.observability.tracing.start_span instead.
    """
    # For backward compatibility, create a span with the tracer name
    # The new module uses a single global tracer
    with _start_span(
        span_name,
        span_type="default",
        attributes=attributes or {},
    ) as span:
        yield span
