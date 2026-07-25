"""PDF-native adapter — extracts structured elements from machine-readable PDFs."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # type: ignore[import-untyped]  # PyMuPDF

from src.canonical.model import (
    BBox,
    CanonicalDocument,
    Element,
    ElementType,
    TitleBlock,
    make_element_id,
)
from src.config import IngestConfig
from src.ingest.base import FormatAdapter, register_adapter

# ---------------------------------------------------------------------------
# Regex patterns for PID domain entities
# ---------------------------------------------------------------------------

# Instrument tags: PI-101-01, TI-202-01, FV-10-01, etc.
_TAG_RE = re.compile(r"\b([A-Z]{1,4}-\d{1,5}(?:-\d{1,3})?)\b")

# Line spec: 4"-XX-NN-NNNN-XXXXX-NN or similar
_LINE_SPEC_RE = re.compile(
    r"""(\d+(?:\.\d+)?\s*"\s*-\s*[A-Z0-9]{2}\s*-\s*"""
    r"""[A-Z0-9]{2}\s*-\s*[A-Z0-9]{4}\s*-\s*"""
    r"""[A-Z0-9]{5}\s*-\s*[A-Z0-9]{2})""",
    re.IGNORECASE,
)

# Numbered notes: "1." "2." etc. at start of a line or after whitespace
_NOTE_NUM_RE = re.compile(r"^\s*(\d{1,3})\.\s+", re.MULTILINE)

# Title block fields (common P&ID title block keys)
_TB_KEYS = {
    "project": re.compile(r"project\s*[:=]\s*(.+)", re.IGNORECASE),
    "document": re.compile(r"document\s*(?:no|number)?\s*[:=]\s*(.+)", re.IGNORECASE),
    "rev(?:ision)?": re.compile(r"rev(?:ision)?\s*[:=]\s*(.+)", re.IGNORECASE),
    "title": re.compile(r"title\s*[:=]\s*(.+)", re.IGNORECASE),
}


def _parse_title_block(full_text: str) -> TitleBlock:
    tb = TitleBlock()
    for field_name, pat in _TB_KEYS.items():
        m = pat.search(full_text)
        if m:
            value = m.group(1).strip()
            if field_name == "project":
                tb.project = value
            elif field_name == "document":
                tb.document_number = value
            elif field_name.startswith("rev"):
                tb.revision = value
            elif field_name == "title":
                tb.title = value
    return tb


def _extract_elements(page: fitz.Page, page_num: int, source_path: str) -> list[Element]:
    """Extract elements from a single PDF page."""
    elements: list[Element] = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    idx = 0
    for block in blocks:
        if block["type"] != 0:  # text block
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # Combine spans into one logical text line
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            bbox_raw = line["bbox"]
            bbox = BBox(x0=bbox_raw[0], y0=bbox_raw[1], x1=bbox_raw[2], y1=bbox_raw[3], page=page_num)
            avg_size = sum(s["size"] for s in spans) / len(spans)

            etype, tag_number, line_spec, note_number, confidence = _classify_line(text)

            el = Element(
                id=make_element_id(source_path, page_num, idx),
                type=etype,
                raw_text=text,
                bbox=bbox,
                page=page_num,
                confidence=confidence,
                tag_number=tag_number,
                line_spec=line_spec,
                note_number=note_number,
                metadata={"font_size": round(avg_size, 1)},
            )
            elements.append(el)
            idx += 1

    return elements


def _classify_line(text: str) -> tuple[ElementType, str | None, str | None, int | None, float]:
    """Best-effort classification of a single text line."""
    # Line spec (check before tags because line specs contain quotes/letters)
    m = _LINE_SPEC_RE.search(text)
    if m:
        return ElementType.LINE_SPEC, None, m.group(1).replace(" ", ""), None, 0.9

    # Tag number
    m = _TAG_RE.search(text)
    if m:
        return ElementType.TAG, m.group(1), None, None, 0.9

    # Numbered note
    m = _NOTE_NUM_RE.match(text)
    if m:
        return ElementType.NOTE, None, None, int(m.group(1)), 0.85

    return ElementType.TEXT, None, None, None, 1.0


@register_adapter
class PdfNativeAdapter(FormatAdapter):
    """Extracts text + entities from machine-readable (non-scanned) PDFs."""

    name = "pdf_native"

    def __init__(self, config: IngestConfig | None = None) -> None:
        self._config = config or IngestConfig()

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        # Heuristic: if the PDF contains extractable text, treat as native.
        try:
            doc = fitz.open(str(path))
            has_text = any(page.get_text().strip() for page in doc)
            doc.close()
            return has_text
        except Exception:  # noqa: BLE001
            return False

    def ingest(self, path: Path) -> CanonicalDocument:
        doc = fitz.open(str(path))
        source = str(path)
        all_elements: list[Element] = []
        title_block = TitleBlock()

        for page_num, page in enumerate(doc, start=1):
            page_elements = _extract_elements(page, page_num, source)
            all_elements.extend(page_elements)

            # Try to parse title block from first page only
            if page_num == 1:
                full_text = page.get_text()
                title_block = _parse_title_block(full_text)

        doc.close()

        return CanonicalDocument(
            source_path=source,
            format_adapter=self.name,
            page_count=page_num,
            title_block=title_block,
            elements=all_elements,
        )
