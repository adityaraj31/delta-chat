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
    SETPOINT = "setpoint"
    TAG = "tag"
    LINE_SPEC = "line_spec"
    NOTE = "note"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    SYMBOL = "symbol"
    TITLE_BLOCK = "title_block"
    GEOMETRY = "geometry"


class ChangeKind(str, enum.Enum):
    """Classification of a single element-level change."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


# ---------------------------------------------------------------------------
# Canonical elements
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Axis-aligned bounding box on a page (all values normalized or in points)."""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 1

    def calculate_grid_cell(self, page_width: float = 842.0, page_height: float = 595.0) -> str:
        """Calculate drawing grid reference (A-H vertically, 1-12 horizontally)."""
        # Standard A0/A1 P&ID grid layout calculation
        cx = (self.x0 + self.x1) / 2.0
        cy = (self.y0 + self.y1) / 2.0

        # Horizontal grid (1 to 12)
        col = max(1, min(12, int((cx / page_width) * 12) + 1))
        
        # Vertical grid (A to H)
        row_idx = max(0, min(7, int((cy / page_height) * 8)))
        row = chr(ord('A') + row_idx)

        return f"Grid {row}-{col}"


class Element(BaseModel):
    """One atomic piece of content extracted from a PID revision."""
    id: str
    type: ElementType
    raw_text: str
    bbox: BBox | None = None
    page: int = 1
    confidence: float = 1.0  # 0-1, useful for OCR / heuristic parses

    # Structured fields populated by type-specific extractors
    tag_number: str | None = None   # e.g. "26-KA-901"
    line_spec: str | None = None    # e.g. '2"-WC-40-9014-AC21-00'
    note_number: int | None = None  # e.g. 11
    grid_cell: str | None = None    # e.g. "Grid E-5"
    setpoint_value: float | None = None  # e.g. 100.0
    setpoint_unit: str | None = None     # e.g. "PSI", "bar"
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

    def human_readable_label(self) -> str:
        """Convert this element into a clear human-readable string for Delta Reports & Chat."""
        location = f" ({self.grid_cell})" if self.grid_cell else f" (Page {self.page})"
        
        if self.type == ElementType.TAG or self.tag_number:
            return f"Equipment Tag '{self.tag_number or self.raw_text}'{location}"
        elif self.type == ElementType.LINE_SPEC or self.line_spec:
            return f"Line Spec '{self.line_spec or self.raw_text}'{location}"
        elif self.type == ElementType.NOTE or self.note_number is not None:
            return f"Note #{self.note_number}{location}: \"{self.raw_text}\""
        elif self.type == ElementType.SETPOINT or self.setpoint_value is not None:
            value = self.setpoint_value if self.setpoint_value is not None else self.raw_text
            unit = f" {self.setpoint_unit}" if self.setpoint_unit else ""
            return f"Setpoint '{value}{unit}'{location}"
        elif self.type == ElementType.TABLE_CELL:
            return f"Table Value '{self.raw_text}'{location}"
        else:
            return f"Text '{self.raw_text}'{location}"


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
    page_count: int = 1
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