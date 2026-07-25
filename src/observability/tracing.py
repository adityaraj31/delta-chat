"""Langfuse tracing setup and helpers."""

from __future__ import annotations

import os
from pathlib import Path

from langfuse import Langfuse


def _langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

_lf = None

def init_tracer(trace_dir: Path | None = None) -> None:
    """Initialize Langfuse. (Local fallback traces via output directory can be set via env var)."""
    global _lf
    if _langfuse_configured():
        _lf = Langfuse()

def flush() -> None:
    """Flush Langfuse buffers."""
    if _lf:
        _lf.flush()