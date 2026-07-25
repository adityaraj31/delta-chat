"""Retrieval index over canonical document elements and delta entries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from langfuse import observe

from src.canonical.model import CanonicalDocument, Element
from src.delta.engine import DeltaEntry

# Patterns for citation validation
_PID_CITE_RE = re.compile(r"\[PID:([^:]+):p(\d+):([^\]]+)\]")
_DELTA_CITE_RE = re.compile(r"\[delta:(\d+)\]")


@dataclass
class IndexEntry:
    """A single searchable chunk."""
    id: str
    text: str
    source_label: str  # e.g. "old_pid" / "new_pid" / "delta"
    page: int
    element_id: str | None = None
    delta_entry_index: int | None = None
    embedding: list[float] = field(default_factory=list)


@dataclass
class CitationValidation:
    """Result of validating a single citation string."""
    raw: str
    valid: bool
    entry_id: str | None = None
    reason: str = ""


def _element_to_entry(el: Element, source_label: str) -> IndexEntry:
    citation = f"[PID:{source_label}:p{el.page}:{el.id}]"
    return IndexEntry(
        id=el.id,
        text=f"{citation} {el.raw_text}",
        source_label=source_label,
        page=el.page,
        element_id=el.id,
    )


def _delta_to_entry(idx: int, entry: DeltaEntry) -> IndexEntry:
    citation = f"[delta:{idx}]"
    old_part = f"OLD: {entry.old.raw_text}" if entry.old else ""
    new_part = f"NEW: {entry.new.raw_text}" if entry.new else ""
    desc_part = f"DESC: {entry.description}" if entry.description else ""
    value_part = ""
    if entry.old and entry.new:
        old_val = entry.old.setpoint_value
        new_val = entry.new.setpoint_value
        if old_val is not None or new_val is not None:
            unit = entry.new.setpoint_unit or entry.old.setpoint_unit or ""
            unit_suffix = f" {unit}" if unit else ""
            value_part = f" SETPOINT: {old_val}{unit_suffix} -> {new_val}{unit_suffix}"
    return IndexEntry(
        id=f"delta-{idx}",
        text=f"{citation} {entry.kind.value}: {old_part} {new_part} {desc_part}{value_part}".strip(),
        source_label="delta",
        page=entry.page,
        delta_entry_index=idx,
    )


class RetrievalIndex:
    """In-memory vector store using simple cosine similarity (no external deps).

    For production, swap in a real vector DB. This is good enough for
    demo / take-home scope.
    """

    def __init__(self, embedding_dim: int = 1536) -> None:
        self._entries: list[IndexEntry] = []
        self._dim = embedding_dim
        self._element_ids: set[str] = set()
        self._delta_indices: set[int] = set()

    @property
    def entries(self) -> list[IndexEntry]:
        return list(self._entries)

    def add_document(self, doc: CanonicalDocument, label: str) -> None:
        for el in doc.elements:
            entry = _element_to_entry(el, label)
            self._entries.append(entry)
            self._element_ids.add(el.id)

    def add_delta_entries(self, entries: list[DeltaEntry]) -> None:
        for idx, entry in enumerate(entries):
            self._entries.append(_delta_to_entry(idx, entry))
            self._delta_indices.add(idx)

    def set_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Set pre-computed embeddings by entry id."""
        for entry in self._entries:
            if entry.id in embeddings:
                entry.embedding = embeddings[entry.id]

    @observe(as_type="span", name="retrieval.search")
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[IndexEntry, float]]:
        """Cosine similarity search. Returns (entry, score) pairs."""
        scored: list[tuple[IndexEntry, float]] = []
        for entry in self._entries:
            if not entry.embedding:
                continue
            sim = _cosine(query_embedding, entry.embedding)
            scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def validate_citations(self, citations: list[str]) -> list[CitationValidation]:
        """Check whether each citation resolves to a real element in the index."""
        results: list[CitationValidation] = []
        for raw in citations:
            # Check PID citation
            m = _PID_CITE_RE.match(raw)
            if m:
                element_id = m.group(3)
                valid = element_id in self._element_ids
                results.append(CitationValidation(
                    raw=raw,
                    valid=valid,
                    entry_id=element_id if valid else None,
                    reason="" if valid else f"Element '{element_id}' not found in index",
                ))
                continue

            # Check delta citation
            m = _DELTA_CITE_RE.match(raw)
            if m:
                idx = int(m.group(1))
                valid = idx in self._delta_indices
                results.append(CitationValidation(
                    raw=raw,
                    valid=valid,
                    entry_id=f"delta-{idx}" if valid else None,
                    reason="" if valid else f"Delta entry {idx} not found in index",
                ))
                continue

            # Unknown format
            results.append(CitationValidation(
                raw=raw, valid=False, reason=f"Unrecognized citation format: {raw}",
            ))

        return results


def _cosine(a: list[float], b: list[float]) -> float:
    """Simple cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
