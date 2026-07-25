"""Langfuse tracing setup and helpers."""

from __future__ import annotations

import os
from pathlib import Path

from langfuse import Langfuse


def _langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

from langfuse import get_client

def init_tracer(trace_dir: Path | None = None) -> None:
    """Initialize Langfuse via decorators."""
    pass

def flush() -> None:
    """Flush Langfuse buffers."""
    client = get_client()
    if client:
        client.flush()