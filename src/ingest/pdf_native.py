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

# Instrument tags: PI-101-01, TI-202-01, FV-101-01, etc.
# Require >= 2 letters and >= 2 digits to reject single-letter grid labels (D-1, B-2).
_TAG_RE = re.compile(r"\b([A-Z]{2,4}-\d{2,5}(?:-\d{1,3})?)\b")

# Line spec: 4"-XX-NN-NNNN-XXXXX-NN or similar
_LINE_SPEC_RE = re.compile(
    r"""(\d+(?:\.\d+)?\s*"\s*-\s*[A-Z0-9]{2}\s*-\s*"""
    r"""[A-Z0-9]{2}\s*-\s*[A-Z0-9]{4}\s*-\s*"""
    r"""[A-Z0-9]{5}\s*-\s*[A-Z0-9]{2})""",
    re.IGNORECASE,
)

# Numbered notes: "1." "2." etc. at start of a line or after whitespace
_NOTE_NUM_RE = re.compile(r"^\s*(\d{1,3})\.\s+", re.MULTILINE)

# Setpoint values: "SP: 100 PSI", "Setpoint = 200.5 °F", "100 PSIG"
_SETPOINT_RE = re.compile(
    r"(?:sp|setpoint|set\s*point)\s*[:=]\s*([\d.]+)\s*([A-Za-z°%]+)?",
    re.IGNORECASE,
)
_SETPOINT_VALUE_RE = re.compile(
    r"\b([\d.]+)\s*(PSI[G]?|BARG?|KPA|MPA|[°]?F|[°]?C|RPM|LPM|GPM|MH[₂Z]|%)\b",
    re.IGNORECASE,
)
_NUMERIC_ONLY_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*(?:[A-Za-z°%]+)?\s*$")

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

            etype, tag_number, line_spec, note_number, confidence, sp_val, sp_unit = _classify_line(text)

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
                setpoint_value=sp_val,
                setpoint_unit=sp_unit,
                metadata={"font_size": round(avg_size, 1)},
            )
            elements.append(el)
            idx += 1

    return elements


def _classify_line(
    text: str,
) -> tuple[ElementType, str | None, str | None, int | None, float, float | None, str | None]:
    """Best-effort classification of a single text line.

    Returns: (type, tag_number, line_spec, note_number, confidence, setpoint_value, setpoint_unit)
    """
    # Line spec (check before tags because line specs contain quotes/letters)
    m = _LINE_SPEC_RE.search(text)
    if m:
        return ElementType.LINE_SPEC, None, m.group(1).replace(" ", ""), None, 0.9, None, None

    # Tag number
    m = _TAG_RE.search(text)
    if m:
        # Check for setpoint on same line as tag
        sp_val, sp_unit = _extract_setpoint(text)
        return ElementType.TAG, m.group(1), None, None, 0.9, sp_val, sp_unit

    # Numbered note
    m = _NOTE_NUM_RE.match(text)
    if m:
        return ElementType.NOTE, None, None, int(m.group(1)), 0.85, None, None

    # Standalone setpoint line
    sp_val, sp_unit = _extract_setpoint(text)
    if sp_val is not None:
        return ElementType.SETPOINT, None, None, None, 0.88, sp_val, sp_unit

    # Numeric-only lines are often instrument/setpoint values in P&IDs.
    if _NUMERIC_ONLY_RE.match(text):
        try:
            numeric_match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            value = float(numeric_match.group(0)) if numeric_match else None
        except ValueError:
            value = None
        unit_match = re.search(r"[A-Za-z°%]+\s*$", text)
        unit = unit_match.group(0) if unit_match else None
        return ElementType.SETPOINT, None, None, None, 0.72, value, unit

    return ElementType.TEXT, None, None, None, 1.0, None, None


def _extract_setpoint(text: str) -> tuple[float | None, str | None]:
    """Try to extract a setpoint value and unit from text."""
    # Explicit setpoint: "SP: 100 PSI"
    m = _SETPOINT_RE.search(text)
    if m:
        try:
            return float(m.group(1)), m.group(2)
        except ValueError:
            pass
    # Implicit: "100 PSIG", "200 °F"
    m = _SETPOINT_VALUE_RE.search(text)
    if m:
        try:
            return float(m.group(1)), m.group(2)
        except ValueError:
            pass
    return None, None


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
            raw = path.read_bytes()
            doc = fitz.open(stream=raw)
            has_text = any(page.get_text().strip() for page in doc)
            doc.close()
            return has_text
        except Exception:  # noqa: BLE001
            return False

    def resolve(self, path: Path) -> tuple[bytes, dict[str, object]]:
        raw = path.read_bytes()
        metadata: dict[str, object] = {
            "format": "pdf",
            "file_size": len(raw),
            "source_path": str(path),
        }
        return raw, metadata

    def parse(self, raw: bytes, metadata: dict[str, object]) -> CanonicalDocument:
        source = str(metadata.get("source_path", "unknown"))
        doc = fitz.open(stream=raw)
        all_elements: list[Element] = []
        title_block = TitleBlock()
        page_count = 0

        for page_num, page in enumerate(doc, start=1):
            page_elements = _extract_elements(page, page_num, source)
            all_elements.extend(page_elements)
            page_count = page_num

            # Try to parse title block from first page only
            if page_num == 1:
                full_text = page.get_text()
                title_block = _parse_title_block(full_text)

        doc.close()

        return CanonicalDocument(
            source_path=source,
            format_adapter=self.name,
            page_count=page_count,
            title_block=title_block,
            elements=all_elements,
        )
