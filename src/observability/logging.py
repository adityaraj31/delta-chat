"""Structured logging via structlog."""

from __future__ import annotations

import logging
import uuid

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON rendering and request context merging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def new_correlation_id() -> str:
    """Generate a new correlation ID and bind it to structlog context."""
    cid = uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=cid, request_id=cid)
    return cid


def bind_correlation_id(cid: str) -> None:
    """Bind an existing correlation ID to structlog context."""
    structlog.contextvars.bind_contextvars(correlation_id=cid, request_id=cid)


def bind_trace_context(*, trace_id: str | None = None, trace_url: str | None = None) -> None:
    """Bind Langfuse trace identifiers to the structured log context."""
    values: dict[str, str] = {}
    if trace_id:
        values["trace_id"] = trace_id
    if trace_url:
        values["trace_url"] = trace_url
    if values:
        structlog.contextvars.bind_contextvars(**values)


def clear_trace_context() -> None:
    """Remove Langfuse trace identifiers from the structured log context."""
    structlog.contextvars.bind_contextvars(trace_id=None, trace_url=None)
