"""Tests for automatic adapter selection."""

from pathlib import Path

from src.ingest.base import auto_adapter


def test_auto_adapter_selects_image_ocr_for_png() -> None:
    adapter = auto_adapter(Path("example.png"))
    assert adapter.name == "image_ocr"
