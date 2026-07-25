"""Image OCR adapter — extracts text from image files via Tesseract."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.canonical.model import BBox, CanonicalDocument, Element, TitleBlock, make_element_id
from src.config import IngestConfig
from src.ingest.base import FormatAdapter, register_adapter
from src.ingest.pdf_native import _classify_line

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _has_tesseract() -> bool:
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@register_adapter
class ImageOcrAdapter(FormatAdapter):
    """Extract text + entities from standalone image files using OCR."""

    name = "image_ocr"

    def __init__(self, config: IngestConfig | None = None) -> None:
        self._config = config or IngestConfig()

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in _IMAGE_EXTS

    def resolve(self, path: Path) -> tuple[bytes, dict[str, object]]:
        raw = path.read_bytes()
        metadata: dict[str, object] = {
            "format": "image",
            "file_size": len(raw),
            "source_path": str(path),
            "suffix": path.suffix.lower(),
        }
        return raw, metadata

    def parse(self, raw: bytes, metadata: dict[str, object]) -> CanonicalDocument:
        if not _has_tesseract():
            raise RuntimeError(
                "Image OCR requires the system 'tesseract' binary. "
                "Install with 'sudo apt install tesseract-ocr' (Linux) or 'brew install tesseract' (macOS)."
            )

        try:
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Image OCR requires pytesseract and Pillow. "
                "Install with: pip install pytesseract Pillow"
            ) from exc

        import io

        source = str(metadata.get("source_path", "unknown"))
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        ocr_data = pytesseract.image_to_data(
            image,
            lang=self._config.ocr_lang,
            output_type=pytesseract.Output.DICT,
        )

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

        elements: list[Element] = []
        for idx, words in enumerate([lines[k] for k in sorted(lines.keys())]):
            text = " ".join(str(w["text"]) for w in words)
            if not text.strip():
                continue

            x0 = int(words[0]["left"])
            y0 = min(int(w["top"]) for w in words)
            x1 = int(words[-1]["left"]) + int(words[-1]["width"])
            y1 = max(int(w["top"]) + int(w["height"]) for w in words)
            avg_conf = sum(float(w["conf"]) for w in words) / len(words) / 100.0

            bbox = BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), page=1)

            etype, tag_number, line_spec, note_number, cls_conf, sp_val, sp_unit = _classify_line(text)

            elements.append(
                Element(
                    id=make_element_id(source, 1, idx, prefix="img"),
                    type=etype,
                    raw_text=text,
                    bbox=bbox,
                    page=1,
                    confidence=max(0.0, min(1.0, (avg_conf + cls_conf) / 2.0)),
                    tag_number=tag_number,
                    line_spec=line_spec,
                    note_number=note_number,
                    setpoint_value=sp_val,
                    setpoint_unit=sp_unit,
                )
            )

        return CanonicalDocument(
            source_path=source,
            format_adapter=self.name,
            page_count=1,
            title_block=TitleBlock(),
            elements=elements,
        )