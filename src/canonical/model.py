"""Canonical document model — the single shared schema all adapters normalise into."""

from __future__ import annotations

import enum
import hashlib
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ElementType(str, enum.Enum):
    """Discriminant for the kind of extracted element."""
    TEXT = "text"
    TAG = "tag"
    LINE_SPEC = "line_spec"
    NOTE = "note"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    SYMBOL = "symbol"
    TITLE_BLOCK = "title_block"


class ChangeKind(str, enum.Enum):
    """Classification of a single element-level change."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


# ---------------------------------------------------------------------------
# Canonical elements
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Axis-aligned bounding box on a page (all values in points, 72 dpi)."""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class Element(BaseModel):
    """One atomic piece of content extracted from a PID revision."""
    id: str
    type: ElementType
    raw_text: str
    bbox: BBox | None = None
    page: int = 0
    confidence: float = 1.0  # 0-1, useful for OCR / heuristic parses

    # Structured fields populated by type-specific extractors
    tag_number: str | None = None  # e.g. "PI-101-01"
    line_spec: str | None = None   # e.g. "4"-XX-NN-NNNN"
    note_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def stable_key(self) -> str | None:
        """Return the best stable identifier for alignment, or None."""
        if self.tag_number:
            return f"tag:{self.tag_number}"
        if self.line_spec:
            return f"line:{self.line_spec}"
        if self.note_number is not None:
            return f"note:{self.note_number}"
        return None


class TitleBlock(BaseModel):
    """Standard PID title-block metadata."""
    project: str = ""
    document_number: str = ""
    revision: str = ""
    title: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical document
# ---------------------------------------------------------------------------

class CanonicalDocument(BaseModel):
    """Format-agnostic representation of one PID revision."""
    source_path: str
    format_adapter: str  # e.g. "pdf_native", "pdf_scanned", "dwg"
    page_count: int = 0
    title_block: TitleBlock = Field(default_factory=TitleBlock)
    elements: list[Element] = Field(default_factory=list)

    @property
    def element_index(self) -> dict[str, Element]:
        """Lazily-built lookup by element.id."""
        return {e.id: e for e in self.elements}

    def elements_by_type(self, etype: ElementType) -> list[Element]:
        return [e for e in self.elements if e.type == etype]

    def stable_keyed_elements(self) -> dict[str, list[Element]]:
        """Group elements that have a stable key."""
        groups: dict[str, list[Element]] = {}
        for e in self.elements:
            key = e.stable_key()
            if key is not None:
                groups.setdefault(key, []).append(e)
        return groups


def make_element_id(source_path: str, page: int, index: int, prefix: str = "el") -> str:
    """Deterministic element ID derived from source location + position."""
    raw = f"{source_path}:{page}:{index}"
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
