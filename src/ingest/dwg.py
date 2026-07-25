"""DWG adapter — stub behind the FormatAdapter interface.

Real implementation requires ODA File Converter or a commercial library.
This stub demonstrates the adapter seam and raises a clear error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter, register_adapter


@register_adapter
class DwgAdapter(FormatAdapter):
    """Stub adapter for AutoCAD DWG files.

    To implement fully, integrate with:
    - ODA File Converter (free, requires registration)
    - ezdxf (open-source DXF parser, DWG → DXF conversion needed)
    - LibreDWG (GNU C library)

    The adapter interface is identical to PDF adapters — swap in a real
    implementation and everything downstream (alignment, delta, chat)
    works without changes.
    """

    name = "dwg"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".dwg"

    def resolve(self, path: Path) -> tuple[bytes, dict[str, Any]]:
        raise NotImplementedError(
            f"DWG adapter is a stub. Cannot resolve {path.name}. "
            "To implement: integrate ODA File Converter or ezdxf, "
            "then normalise output into CanonicalDocument elements."
        )

    def parse(self, raw: bytes, metadata: dict[str, Any]) -> CanonicalDocument:
        raise NotImplementedError(
            "DWG adapter is a stub. Cannot parse DWG data. "
            "To implement: integrate ODA File Converter or ezdxf, "
            "then normalise output into CanonicalDocument elements."
        )
