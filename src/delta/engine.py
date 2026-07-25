"""Delta engine — classifies element-level changes from an alignment."""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.canonical.model import ChangeKind, Element
from src.delta.align import Alignment


@dataclass
class DeltaEntry:
    """One classified change between two revisions."""
    kind: ChangeKind
    old: Element | None = None
    new: Element | None = None
    similarity: float = 1.0  # 0-1; 1.0 for stable-key matches, fuzzy otherwise
    confidence: float = 1.0  # calibrated confidence in this classification
    description: str = ""    # human-readable: "Setpoint changed from 100 to 200"
    reasons: list[str] = field(default_factory=list)

    @property
    def element_id_old(self) -> str | None:
        return self.old.id if self.old else None

    @property
    def element_id_new(self) -> str | None:
        return self.new.id if self.new else None

    @property
    def page(self) -> int:
        if self.new:
            return self.new.page
        if self.old:
            return self.old.page
        return 0


def _text_similarity(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0


def _describe_change(old: Element, new: Element, reasons: list[str]) -> str:
    """Produce a human-readable description of what changed between two elements."""
    parts: list[str] = []

    # Setpoint change
    if (old.setpoint_value is not None and new.setpoint_value is not None
            and old.setpoint_value != new.setpoint_value):
        unit = new.setpoint_unit or old.setpoint_unit or ""
        parts.append(f"Setpoint changed from {old.setpoint_value} {unit} to {new.setpoint_value} {unit}".rstrip())

    # Tag number label change (same tag, different surrounding text)
    if old.tag_number and new.tag_number and old.tag_number == new.tag_number and old.raw_text != new.raw_text:
        parts.append(f"Label text changed for {old.tag_number}")

    # Note text change
    if (old.note_number is not None and new.note_number is not None
            and old.note_number == new.note_number and old.raw_text != new.raw_text):
        parts.append(f"Note {old.note_number} text revised")

    # Line spec change
    if old.line_spec and new.line_spec and old.line_spec == new.line_spec and old.raw_text != new.raw_text:
        parts.append(f"Line spec {old.line_spec} details changed")

    # Generic text change
    if not parts:
        sim = _text_similarity(old.raw_text, new.raw_text)
        if sim >= 0.95:
            parts.append("Minor text edit")
        elif sim >= 0.5:
            parts.append("Significant text change")
        else:
            parts.append("Substantially rewritten")

    return "; ".join(parts)


def _compute_confidence(kind: ChangeKind, old: Element, new: Element | None, similarity: float) -> float:
    """Compute a calibrated confidence score for this classification.

    - Stable-key matches (tag, line_spec, note): high confidence (0.95)
    - Fuzzy matches: scaled by similarity score
    - Added/removed (no pair): 1.0 (we're certain it's absent from one side)
    """
    if kind in (ChangeKind.ADDED, ChangeKind.REMOVED):
        return 1.0

    # Modified: confidence based on how we matched
    has_stable_key = bool(
        (old.tag_number and new and new.tag_number == old.tag_number)
        or (old.line_spec and new and new.line_spec == old.line_spec)
        or (old.note_number is not None and new and new.note_number == old.note_number)
    )
    if has_stable_key:
        return 0.95
    # Fuzzy match: confidence scales with similarity
    return max(0.5, similarity)


def _classify_pair(old: Element, new: Element) -> DeltaEntry | None:
    """Classify a matched pair of elements.

    Returns None when the pair is a genuine no-op (identical text).
    """
    reasons: list[str] = []

    # Defensive no-op check: ignore whitespace-only differences.
    if old.raw_text.strip() == new.raw_text.strip():
        return None

    # Check structured fields first
    if old.tag_number and new.tag_number and old.tag_number == new.tag_number:
        reasons.append("tag_value_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        desc = _describe_change(old, new, reasons)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim,
                          confidence=0.95, description=desc, reasons=reasons)

    if old.line_spec and new.line_spec and old.line_spec == new.line_spec:
        reasons.append("line_spec_value_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        desc = _describe_change(old, new, reasons)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim,
                          confidence=0.95, description=desc, reasons=reasons)

    if old.note_number is not None and new.note_number is not None and old.note_number == new.note_number:
        reasons.append("note_text_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        desc = _describe_change(old, new, reasons)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim,
                          confidence=0.95, description=desc, reasons=reasons)

    # General text comparison
    sim = _text_similarity(old.raw_text, new.raw_text)
    if sim >= 1.0:
        return None  # identical text, no-op
    desc = _describe_change(old, new, reasons)
    if sim >= 0.95:
        reasons.append("near_identical_text")
    else:
        reasons.append("text_changed")
    conf = max(0.5, sim)
    return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim,
                      confidence=conf, description=desc, reasons=reasons)


def compute_delta(alignment: Alignment) -> list[DeltaEntry]:
    """Turn an Alignment into a list of classified DeltaEntry items."""
    entries: list[DeltaEntry] = []

    for old, new in alignment.matched:
        entry = _classify_pair(old, new)
        if entry is not None:
            entries.append(entry)

    for el in alignment.removed:
        desc = f"Removed: {el.raw_text[:80]}"
        entries.append(DeltaEntry(kind=ChangeKind.REMOVED, old=el, similarity=1.0,
                                  confidence=1.0, description=desc, reasons=["not_in_new"]))

    for el in alignment.added:
        desc = f"Added: {el.raw_text[:80]}"
        entries.append(DeltaEntry(kind=ChangeKind.ADDED, new=el, similarity=1.0,
                                  confidence=1.0, description=desc, reasons=["not_in_old"]))

    # Sort by page then by kind priority
    kind_order = {ChangeKind.REMOVED: 0, ChangeKind.ADDED: 1, ChangeKind.MODIFIED: 2}
    entries.sort(key=lambda e: (e.page, kind_order[e.kind]))
    return entries
