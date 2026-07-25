"""Element alignment between two canonical documents.

Strategy:
1. Match by stable key (tag, line spec, note number) — deterministic.
2. Fuzzy text similarity fallback for remaining unmatched elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.canonical.model import CanonicalDocument, Element
from src.config import DeltaConfig


_NUMERIC_ONLY_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*(?:[A-Za-z°%]+)?\s*$")


def _center(el: Element) -> tuple[float, float] | None:
    if el.bbox is None:
        return None
    return ((el.bbox.x0 + el.bbox.x1) / 2.0, (el.bbox.y0 + el.bbox.y1) / 2.0)


def _distance(a: Element, b: Element) -> float:
    ac = _center(a)
    bc = _center(b)
    if ac is None or bc is None:
        return float("inf")
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5


def _is_plain_numeric(el: Element) -> bool:
    return bool(_NUMERIC_ONLY_RE.match(el.raw_text.strip()))


def _to_context_label(el: Element) -> str:
    if el.tag_number:
        return f"{el.tag_number} value"
    if el.line_spec:
        return f"Line spec {el.line_spec} value"
    if el.note_number is not None:
        return f"Note {el.note_number} value"

    text = el.raw_text.strip()
    if len(text) > 48:
        text = text[:48].rstrip() + "..."
    return text


def _enrich_document_context(doc: CanonicalDocument) -> None:
    """Populate missing grid cells and label numeric values with nearby context."""
    for el in doc.elements:
        if el.grid_cell is None and el.bbox is not None:
            el.grid_cell = el.bbox.calculate_grid_cell()

    for el in doc.elements:
        if not _is_plain_numeric(el):
            continue
        if el.metadata.get("context_label"):
            continue

        best_label: str | None = None
        best_distance = float("inf")

        for candidate in doc.elements:
            if candidate.id == el.id:
                continue
            if candidate.page != el.page:
                continue
            if _is_plain_numeric(candidate):
                continue

            dist = _distance(el, candidate)
            if dist < best_distance:
                best_distance = dist
                best_label = _to_context_label(candidate)

        if best_label and best_distance <= 180.0:
            el.metadata["context_label"] = best_label


@dataclass
class Alignment:
    """Result of aligning two document element sets."""
    # (old_element, new_element) pairs
    matched: list[tuple[Element, Element]] = field(default_factory=list)
    # Elements only in old (removed)
    removed: list[Element] = field(default_factory=list)
    # Elements only in new (added)
    added: list[Element] = field(default_factory=list)


def align_documents(
    old: CanonicalDocument,
    new: CanonicalDocument,
    config: DeltaConfig | None = None,
) -> Alignment:
    """Produce an Alignment between two revisions of a document."""
    cfg = config or DeltaConfig()
    alignment = Alignment()

    _enrich_document_context(old)
    _enrich_document_context(new)

    old_by_key = old.stable_keyed_elements()
    new_by_key = new.stable_keyed_elements()

    matched_old_ids: set[str] = set()
    matched_new_ids: set[str] = set()

    # ---- Phase 1: Stable-key matching ----
    all_keys = set(old_by_key) | set(new_by_key)
    for key in all_keys:
        old_els = old_by_key.get(key, [])
        new_els = new_by_key.get(key, [])
        # Match 1-to-1 in order (most common: exactly 1 each)
        for o, n in zip(old_els, new_els):
            alignment.matched.append((o, n))
            matched_old_ids.add(o.id)
            matched_new_ids.add(n.id)
        # Extras go to added/removed
        for o in old_els[len(new_els) :]:
            alignment.removed.append(o)
            matched_old_ids.add(o.id)
        for n in new_els[len(old_els) :]:
            alignment.added.append(n)
            matched_new_ids.add(n.id)

    # ---- Phase 2: Fuzzy fallback ----
    unmatched_old = [e for e in old.elements if e.id not in matched_old_ids]
    unmatched_new = [e for e in new.elements if e.id not in matched_new_ids]

    used_new: set[str] = set()
    for o in unmatched_old:
        best_score: float = 0
        best_new: Element | None = None
        for n in unmatched_new:
            if n.id in used_new:
                continue
            score = fuzz.ratio(o.raw_text, n.raw_text)
            if score > best_score:
                best_score = score
                best_new = n
        if best_new is not None and best_score >= cfg.fuzzy_threshold:
            alignment.matched.append((o, best_new))
            matched_old_ids.add(o.id)
            matched_new_ids.add(best_new.id)
            used_new.add(best_new.id)
        else:
            alignment.removed.append(o)

    for n in unmatched_new:
        if n.id not in matched_new_ids:
            alignment.added.append(n)

    return alignment
