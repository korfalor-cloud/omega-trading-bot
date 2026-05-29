"""Distributed Tracing — span creation, context propagation, export.

Implements:
- Span creation and management
- Trace context propagation
- Operation timing
- Error recording
- Export format (JSON)
- Integration with logging
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpanEvent:
    """An event within a span (log point)."""
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to a related span in another trace."""
    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanContext:
    """Immutable propagation context for a span."""
    trace_id: str
    span_id: str
    is_remote: bool = False
    trace_flags: int = 1  # 1 = sampled


@dataclass
class Span:
    """Represents a unit of work in a trace."""
    name: str
    trace_id: str
    span_id: str
    parent_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status_code: str = "OK"  # OK, ERROR
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0

    @property
    def context(self) -> SpanContext:
        return SpanContext(trace_id=self.trace_id, span_id=self.span_id)

    @property
    def is_recording(self) -> bool:
        return self.end_time == 0.0

    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        return self

    def add_link(self, trace_id: str, span_id: str, attributes: dict[str, Any] | None = None) -> Span:
        self.links.append(SpanLink(trace_id=trace_id, span_id=span_id, attributes=attributes or {}))
        return self

    def set_status(self, code: str, message: str = "") -> None:
        self.status_code = code
        self.status_message = message

    def record_error(self, error: Exception) -> None:
        self.status_code = "ERROR"
        self.status_message = str(error)
        self.add_event("exception", {
            "type": type(error).__name__,
            "message": str(error),
        })

    def finish(self) -> None:
        if self.end_time == 0.0:
            self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "status_code": self.status_code,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "links": [
                {"trace_id": l.trace_id, "span_id": l.span_id, "attributes": l.attributes}
                for l in self.links
            ],
        }


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Creates spans, manages context, and exports traces."""

    def __init__(self, service_name: str = "omega-trading-bot"):
        self.service_name = service_name
        self._completed: list[Span] = []
        self._max_completed = 10_000
        self._lock = threading.Lock()
        self._local = threading.local()

    # ---- span creation ----

    def start_span(self, name: str, parent: Span | SpanContext | None = None,
                   attributes: dict[str, Any] | None = None) -> Span:
        trace_id = ""
        parent_id = ""

        if parent is None:
            # Check the active span on this thread
            active = self.get_active_span()
            if active:
                trace_id = active.trace_id
                parent_id = active.span_id
        elif isinstance(parent, Span):
            trace_id = parent.trace_id
            parent_id = parent.span_id
        elif isinstance(parent, SpanContext):
            trace_id = parent.trace_id
            parent_id = parent.span_id

        if not trace_id:
            trace_id = uuid.uuid4().hex

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            start_time=time.time(),
            attributes={},
        )
        if attributes:
            span.attributes.update(attributes)
        span.attributes["service.name"] = self.service_name

        # Set as active span on this thread
        self._local.active_span = span
        return span

    @contextmanager
    def trace(self, name: str, parent: Span | SpanContext | None = None,
              attributes: dict[str, Any] | None = None):
        """Context manager that creates a span, yields it, and finishes on exit."""
        span = self.start_span(name, parent=parent, attributes=attributes)
        try:
            yield span
        except Exception as exc:
            span.record_error(exc)
            raise
        finally:
            span.finish()
            self._record(span)

    def get_active_span(self) -> Span | None:
        return getattr(self._local, "active_span", None)

    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject trace context into a carrier dict (e.g., HTTP headers)."""
        span = self.get_active_span()
        if span:
            carrier["trace-id"] = span.trace_id
            carrier["span-id"] = span.span_id
            carrier["trace-flags"] = "1"

    def extract_context(self, carrier: dict[str, str]) -> SpanContext | None:
        """Extract trace context from a carrier dict."""
        trace_id = carrier.get("trace-id", "")
        span_id = carrier.get("span-id", "")
        if trace_id and span_id:
            return SpanContext(trace_id=trace_id, span_id=span_id, is_remote=True)
        return None

    # ---- recording ----

    def _record(self, span: Span) -> None:
        with self._lock:
            self._completed.append(span)
            if len(self._completed) > self._max_completed:
                self._completed = self._completed[-self._max_completed:]

        if span.status_code == "ERROR":
            logger.error(
                f"Trace span error: {span.name} | trace={span.trace_id} "
                f"| {span.status_message} | {span.duration_ms:.1f}ms"
            )
        else:
            logger.debug(
                f"Trace span: {span.name} | trace={span.trace_id} "
                f"| {span.duration_ms:.1f}ms"
            )

    # ---- retrieval ----

    def get_completed(self, limit: int = 100) -> list[Span]:
        with self._lock:
            return list(self._completed[-limit:])

    def get_trace(self, trace_id: str) -> list[Span]:
        with self._lock:
            return [s for s in self._completed if s.trace_id == trace_id]

    # ---- export ----

    def export_json(self, spans: list[Span] | None = None, indent: int | None = None) -> str:
        """Export spans as JSON."""
        if spans is None:
            spans = self.get_completed()
        data = {
            "service": self.service_name,
            "spans": [s.to_dict() for s in spans],
        }
        return json.dumps(data, indent=indent, default=str)

    def export_ndjson(self, spans: list[Span] | None = None) -> str:
        """Export spans as newline-delimited JSON (one span per line)."""
        if spans is None:
            spans = self.get_completed()
        lines = [json.dumps(s.to_dict(), default=str) for s in spans]
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._completed.clear()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class OperationTimer:
    """Measures wall-clock duration of an operation and records it."""
    name: str
    tracer: Tracer
    attributes: dict[str, Any] = field(default_factory=dict)
    _span: Span | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> OperationTimer:
        self._span = self.tracer.start_span(self.name, attributes=self.attributes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._span is None:
            return
        if exc_val is not None:
            self._span.record_error(exc_val)
        self._span.finish()
        self.tracer._record(self._span)


def timed(tracer: Tracer, name: str | None = None, **extra_attrs):
    """Decorator that wraps a function in a trace span."""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        def wrapper(*args, **kwargs):
            with tracer.trace(span_name, attributes=extra_attrs) as span:
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Module-level default tracer
# ---------------------------------------------------------------------------

DEFAULT_TRACER = Tracer()


def get_tracer() -> Tracer:
    """Return the module-level default tracer."""
    return DEFAULT_TRACER
