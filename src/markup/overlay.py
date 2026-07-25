"""Markup / overlay — draw delta highlights directly on PDF pages.

Uses PyMuPDF to render colored rectangles and labels over elements
that were added, removed, or modified between revisions.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-untyped]

from src.canonical.model import CanonicalDocument
from src.delta.engine import DeltaEntry

# Colors: (R, G, B) with alpha
_COLORS = {
    "added": (0, 180, 0),       # green
    "removed": (220, 0, 0),     # red
    "modified": (200, 160, 0),  # amber
}


def overlay_deltas(
    new_doc: CanonicalDocument,
    entries: list[DeltaEntry],
    output_path: Path,
) -> Path:
    """Write a new PDF with delta overlays on the new revision.

    - Added elements: green highlight
    - Removed elements: red highlight with strikethrough text
    - Modified elements: amber highlight

    Returns the path to the written PDF.
    """
    doc = fitz.open(new_doc.source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup: element_id -> list of DeltaEntry
    el_entries: dict[str, list[DeltaEntry]] = {}
    for entry in entries:
        if entry.new and entry.kind.value != "removed":
            el_entries.setdefault(entry.new.id, []).append(entry)
        if entry.old and entry.kind.value != "added":
            el_entries.setdefault(entry.old.id, []).append(entry)

    for page in doc:
        for el in new_doc.elements:
            if el.page != page.number + 1:
                continue
            if el.id not in el_entries:
                continue
            if not el.bbox:
                continue

            entry_list = el_entries[el.id]
            kind = entry_list[0].kind.value
            color = _COLORS.get(kind, (150, 150, 150))

            # Draw highlight rectangle
            rect = fitz.Rect(el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1)
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(
                color=color,
                fill=color,
                fill_opacity=0.15,
                width=1.5,
            )
            shape.commit()

            # Add small label
            label = kind.upper()
            page.insert_text(
                fitz.Point(el.bbox.x1 + 4, el.bbox.y0 + 10),
                label,
                fontsize=7,
                fontname="helv",
                color=color,
            )

    doc.save(str(output_path))
    doc.close()
    return output_path
