"""Langfuse-backed tracing helpers with local run artifacts."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.observability.logging import bind_trace_context

log = structlog.get_logger(__name__)


def _langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def estimate_cost_usd(model: str | None, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if not model or prompt_tokens is None and completion_tokens is None:
        return None

    rates: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.0025, 0.01),
        "text-embedding-3-small": (0.00002, 0.0),
        "text-embedding-3-large": (0.00013, 0.0),
    }
    input_rate, output_rate = rates.get(model, (0.0, 0.0))
    prompt = float(prompt_tokens or 0)
    completion = float(completion_tokens or 0)
    cost = (prompt / 1000.0) * input_rate + (completion / 1000.0) * output_rate
    return round(cost, 8)


class _NoopObservation:
    def update(self, **_: Any) -> None:
        return None

    def end(self) -> None:
        return None


@dataclass
class SpanRecord:
    name: str
    as_type: str
    elapsed_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Tracer:
    """Collects request traces for a single pipeline or API run."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self._trace_dir = trace_dir
        self._spans: list[SpanRecord] = []
        self._trace_id: str | None = None
        self._trace_url: str | None = None

        self._langfuse = None
        if _langfuse_configured():
            try:
                from langfuse import get_client

                self._langfuse = get_client()
            except Exception as exc:  # pragma: no cover - best-effort init
                log.warning("langfuse.disabled", reason=str(exc), run_id=self.run_id)

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    @property
    def trace_url(self) -> str | None:
        return self._trace_url

    @contextmanager
    def request(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        trace_context: dict[str, str] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Generator[Any]:
        """Create a request root span that becomes the Langfuse trace root."""
        start = time.monotonic()
        root_metadata = {"run_id": self.run_id}
        if metadata:
            root_metadata.update(metadata)

        if self._langfuse is None:
            span_ctx = nullcontext(_NoopObservation())
        else:
            from langfuse import propagate_attributes

            with propagate_attributes(
                session_id=session_id or self.run_id,
                metadata={k: str(v) for k, v in root_metadata.items()},
                tags=tags or [],
                trace_name=name,
            ):
                span_ctx = self._langfuse.start_as_current_observation(
                    as_type="span",
                    name=name,
                    input=input,
                    metadata=root_metadata,
                    trace_context=trace_context,
                )

        with span_ctx as span:
            self._capture_trace_identifiers()
            log.info("trace.request.start", run_id=self.run_id, trace_id=self._trace_id, span=name)
            try:
                yield span
            except Exception as exc:
                self._mark_error(span, str(exc))
                log.error("trace.request.error", run_id=self.run_id, trace_id=self._trace_id, span=name, error=str(exc))
                raise
            finally:
                elapsed_ms = round((time.monotonic() - start) * 1000, 1)
                self._spans.append(SpanRecord(name=name, as_type="span", elapsed_ms=elapsed_ms, attrs=root_metadata))
                self._capture_trace_identifiers()
                log.info(
                    "trace.request.end",
                    run_id=self.run_id,
                    trace_id=self._trace_id,
                    trace_url=self._trace_url,
                    span=name,
                    elapsed_ms=elapsed_ms,
                )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        trace_context: dict[str, str] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Generator[Any]:
        """Create a nested span or generation observation."""
        start = time.monotonic()
        span_metadata = metadata or {}

        if self._langfuse is None:
            span_ctx = nullcontext(_NoopObservation())
        else:
            span_ctx = self._langfuse.start_as_current_observation(
                as_type=as_type,
                name=name,
                input=input,
                metadata=span_metadata,
                model=model,
                model_parameters=model_parameters,
                usage_details=usage_details,
                cost_details=cost_details,
                trace_context=trace_context,
            )

        with span_ctx as obs:
            self._capture_trace_identifiers()
            log.info("trace.span.start", run_id=self.run_id, trace_id=self._trace_id, span=name, as_type=as_type)
            try:
                yield obs
            except Exception as exc:
                self._mark_error(obs, str(exc))
                log.error("trace.span.error", run_id=self.run_id, trace_id=self._trace_id, span=name, error=str(exc))
                raise
            finally:
                elapsed_ms = round((time.monotonic() - start) * 1000, 1)
                record_attrs = {"as_type": as_type, **span_metadata}
                if model:
                    record_attrs["model"] = model
                if model_parameters:
                    record_attrs["model_parameters"] = model_parameters
                if usage_details:
                    record_attrs["usage_details"] = usage_details
                if cost_details:
                    record_attrs["cost_details"] = cost_details
                self._spans.append(SpanRecord(name=name, as_type=as_type, elapsed_ms=elapsed_ms, attrs=record_attrs))
                self._capture_trace_identifiers()
                log.info(
                    "trace.span.end",
                    run_id=self.run_id,
                    trace_id=self._trace_id,
                    span=name,
                    as_type=as_type,
                    elapsed_ms=elapsed_ms,
                )

    def flush(self) -> None:
        """Flush Langfuse buffers and write a local JSONL trace artifact."""
        if self._langfuse is not None:
            self._langfuse.flush()

        if self._trace_dir is None:
            return

        self._trace_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._trace_dir / f"run-{self.run_id}.jsonl"
        summary_path = self._trace_dir / f"run-{self.run_id}.summary.json"

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for span in self._spans:
                handle.write(json.dumps({
                    "run_id": self.run_id,
                    "span": span.name,
                    "as_type": span.as_type,
                    "elapsed_ms": span.elapsed_ms,
                    **span.attrs,
                    **({"error": span.error} if span.error else {}),
                }) + "\n")

        summary = {
            "run_id": self.run_id,
            "trace_id": self._trace_id,
            "trace_url": self._trace_url,
            "span_count": len(self._spans),
            "spans": [span.__dict__ for span in self._spans],
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        log.info(
            "trace.flushed",
            run_id=self.run_id,
            trace_id=self._trace_id,
            trace_url=self._trace_url,
            jsonl_path=str(jsonl_path),
            summary_path=str(summary_path),
            spans=len(self._spans),
        )

    def _capture_trace_identifiers(self) -> None:
        if self._langfuse is None:
            return
        trace_id = self._langfuse.get_current_trace_id()
        if trace_id:
            self._trace_id = trace_id
            self._trace_url = self._langfuse.get_trace_url(trace_id=trace_id)
            bind_trace_context(trace_id=trace_id, trace_url=self._trace_url)

    @staticmethod
    def _mark_error(observation: Any, error: str) -> None:
        if hasattr(observation, "update"):
            try:
                observation.update(level="ERROR", status_message=error, metadata={"error": error})
            except Exception:
                return


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