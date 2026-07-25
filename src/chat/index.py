"""Retrieval index over canonical document elements and delta entries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.canonical.model import CanonicalDocument, Element
from src.delta.engine import DeltaEntry


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
    return IndexEntry(
        id=f"delta-{idx}",
        text=f"{citation} {entry.kind.value}: {old_part} {new_part}".strip(),
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

    @property
    def entries(self) -> list[IndexEntry]:
        return list(self._entries)

    def add_document(self, doc: CanonicalDocument, label: str) -> None:
        for el in doc.elements:
            self._entries.append(_element_to_entry(el, label))

    def add_delta_entries(self, entries: list[DeltaEntry]) -> None:
        for idx, entry in enumerate(entries):
            self._entries.append(_delta_to_entry(idx, entry))

    def set_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Set pre-computed embeddings by entry id."""
        for entry in self._entries:
            if entry.id in embeddings:
                entry.embedding = embeddings[entry.id]

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


def _cosine(a: list[float], b: list[float]) -> float:
    """Simple cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
