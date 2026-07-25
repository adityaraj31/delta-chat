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


def _classify_pair(old: Element, new: Element) -> DeltaEntry:
    """Classify a matched pair of elements."""
    reasons: list[str] = []

    # Check structured fields first
    if old.tag_number and new.tag_number and old.tag_number == new.tag_number:
        if old.raw_text == new.raw_text:
            reasons.append("identical_tag_text")
            return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=1.0, reasons=reasons)
        reasons.append("tag_value_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim, reasons=reasons)

    if old.line_spec and new.line_spec and old.line_spec == new.line_spec:
        if old.raw_text == new.raw_text:
            reasons.append("identical_line_spec_text")
            return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=1.0, reasons=reasons)
        reasons.append("line_spec_value_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim, reasons=reasons)

    if old.note_number is not None and new.note_number is not None and old.note_number == new.note_number:
        if old.raw_text == new.raw_text:
            reasons.append("identical_note_text")
            return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=1.0, reasons=reasons)
        reasons.append("note_text_changed")
        sim = _text_similarity(old.raw_text, new.raw_text)
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim, reasons=reasons)

    # General text comparison
    sim = _text_similarity(old.raw_text, new.raw_text)
    if sim >= 0.95:
        reasons.append("near_identical_text")
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim, reasons=reasons)
    else:
        reasons.append("text_changed")
        return DeltaEntry(kind=ChangeKind.MODIFIED, old=old, new=new, similarity=sim, reasons=reasons)


def compute_delta(alignment: Alignment) -> list[DeltaEntry]:
    """Turn an Alignment into a list of classified DeltaEntry items."""
    entries: list[DeltaEntry] = []

    for old, new in alignment.matched:
        entries.append(_classify_pair(old, new))

    for el in alignment.removed:
        entries.append(DeltaEntry(kind=ChangeKind.REMOVED, old=el, similarity=1.0, reasons=["not_in_new"]))

    for el in alignment.added:
        entries.append(DeltaEntry(kind=ChangeKind.ADDED, new=el, similarity=1.0, reasons=["not_in_old"]))

    # Sort by page then by kind priority
    kind_order = {ChangeKind.REMOVED: 0, ChangeKind.ADDED: 1, ChangeKind.MODIFIED: 2}
    entries.sort(key=lambda e: (e.page, kind_order[e.kind]))
    return entries
