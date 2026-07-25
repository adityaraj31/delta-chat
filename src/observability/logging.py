"""Structured logging via structlog."""

from __future__ import annotations

import logging
import sys
import uuid

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON rendering and a console renderer for dev."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
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
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def bind_correlation_id(cid: str) -> None:
    """Bind an existing correlation ID to structlog context."""
    structlog.contextvars.bind_contextvars(correlation_id=cid)
