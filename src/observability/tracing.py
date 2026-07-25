"""Lightweight per-request tracing — writes spans as JSON lines."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class Tracer:
    """Collects trace spans for a single pipeline run."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self._spans: list[dict[str, Any]] = []
        self._trace_dir = trace_dir

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Generator[None]:
        """Context manager that records start/end time and optional attributes."""
        start = time.monotonic()
        log.info("trace.span.start", run_id=self.run_id, span=name, **attrs)
        try:
            yield
        except Exception as exc:
            log.error("trace.span.error", run_id=self.run_id, span=name, error=str(exc))
            raise
        finally:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            self._spans.append({
                "run_id": self.run_id,
                "span": name,
                "elapsed_ms": elapsed_ms,
                **attrs,
            })
            log.info("trace.span.end", run_id=self.run_id, span=name, elapsed_ms=elapsed_ms)

    def flush(self) -> None:
        """Write all collected spans to a JSON lines file."""
        if self._trace_dir is None:
            return
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        path = self._trace_dir / f"run-{self.run_id}.jsonl"
        with open(path, "w") as f:
            f.writelines(json.dumps(span) + "\n" for span in self._spans)
        log.info("trace.flushed", run_id=self.run_id, path=str(path), spans=len(self._spans))


# Module-level default tracer (initialised by init_tracer)
_default: Tracer | None = None


def init_tracer(trace_dir: Path | None = None) -> Tracer:
    global _default
    _default = Tracer(trace_dir)
    return _default


def get_tracer() -> Tracer:
    global _default
    if _default is None:
        _default = Tracer()
    return _default
