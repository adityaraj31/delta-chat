"""PDF-scanned (OCR) adapter — extracts text from image-only PDFs via Tesseract."""

from __future__ import annotations

from pathlib import Path

from src.canonical.model import CanonicalDocument, Element, ElementType, TitleBlock, make_element_id
from src.config import IngestConfig
from src.ingest.base import FormatAdapter, register_adapter


@register_adapter
class PdfScannedAdapter(FormatAdapter):
    """Extracts text from scanned/image-only PDFs using Tesseract OCR.

    Requires the ``tesseract`` binary to be installed on the system.
    Install: ``sudo apt install tesseract-ocr`` (Debian/Ubuntu)
             ``brew install tesseract`` (macOS)
    """

    name = "pdf_scanned"

    def __init__(self, config: IngestConfig | None = None) -> None:
        self._config = config or IngestConfig()

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        # Check if Tesseract is available
        try:
            import subprocess
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        # Check if PDF has little or no extractable text (i.e. it's scanned)
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(str(path))
            has_text = any(page.get_text().strip() for page in doc)
            doc.close()
            return not has_text  # If no text, it's likely scanned
        except Exception:  # noqa: BLE001
            return False

    def ingest(self, path: Path) -> CanonicalDocument:
        try:
            import fitz
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "OCR adapter requires pytesseract and Pillow. "
                "Install with: pip install pytesseract Pillow"
            ) from exc

        doc = fitz.open(str(path))
        source = str(path)
        all_elements: list[Element] = []
        title_block = TitleBlock()

        for page_num, page in enumerate(doc, start=1):
            # Render page to image
            pix = page.get_pixmap(dpi=self._config.pdf_dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # OCR with layout analysis
            ocr_data = pytesseract.image_to_data(
                img,
                lang=self._config.ocr_lang,
                output_type=pytesseract.Output.DICT,
            )

            # Group words into lines by block_num + line_num
            lines: dict[tuple[int, int], list[dict[str, str | int | float]]] = {}
            for i, word in enumerate(ocr_data["text"]):
                if word.strip():
                    key = (ocr_data["block_num"][i], ocr_data["line_num"][i])
                    lines.setdefault(key, []).append({
                        "text": word,
                        "left": ocr_data["left"][i],
                        "top": ocr_data["top"][i],
                        "width": ocr_data["width"][i],
                        "height": ocr_data["height"][i],
                        "conf": ocr_data["conf"][i],
                    })

            idx = 0
            for _key, words in sorted(lines.items()):
                text = " ".join(str(w["text"]) for w in words)
                if not text.strip():
                    continue

                # Bounding box from first/last word
                x0 = int(words[0]["left"])
                y0 = min(int(w["top"]) for w in words)
                x1 = int(words[-1]["left"]) + int(words[-1]["width"])
                y1 = max(int(w["top"]) + int(w["height"]) for w in words)
                avg_conf = sum(float(w["conf"]) for w in words) / len(words) / 100.0

                from src.canonical.model import BBox
                bbox = BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), page=page_num)

                el = Element(
                    id=make_element_id(source, page_num, idx, prefix="ocr"),
                    type=ElementType.TEXT,
                    raw_text=text,
                    bbox=bbox,
                    page=page_num,
                    confidence=max(0.0, min(1.0, avg_conf)),
                )
                all_elements.append(el)
                idx += 1

        page_count = page_num if doc else 0
        doc.close()

        return CanonicalDocument(
            source_path=source,
            format_adapter=self.name,
            page_count=page_count,
            title_block=title_block,
            elements=all_elements,
        )
