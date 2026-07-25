"""Abstract base class for all format adapters."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from langfuse import observe

from src.canonical.model import CanonicalDocument


class FormatAdapter(abc.ABC):
    """Every format adapter must implement this interface.

    The two-step interface separates I/O from parsing:
    - ``resolve(path)`` reads raw bytes + metadata (format, file size, etc.)
    - ``parse(raw, metadata)`` normalises bytes into a CanonicalDocument

    The high-level ``ingest(path)`` calls resolve then parse by default.
    Adapters may override ``ingest`` for backward compatibility, but new
    adapters should implement ``resolve`` + ``parse`` instead.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``pdf_native``."""

    @abc.abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this adapter can ingest the given file."""

    @abc.abstractmethod
    def resolve(self, path: Path) -> tuple[bytes, dict[str, Any]]:
        """Read *path* and return raw bytes + metadata dict.

        Metadata should include at minimum:
        - ``"format"``: the format name (e.g. ``"pdf"``)
        - ``"file_size"``: byte length of the raw data
        """

    @abc.abstractmethod
    def parse(self, raw: bytes, metadata: dict[str, Any]) -> CanonicalDocument:
        """Parse raw bytes into a normalised canonical document."""

    @observe(as_type="span", name="ingest")
    def ingest(self, path: Path) -> CanonicalDocument:
        """High-level entry point: resolve then parse. Override if needed."""
        raw, metadata = self.resolve(path)
        return self.parse(raw, metadata)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# Registry ---------------------------------------------------------------

_registry: dict[str, type[FormatAdapter]] = {}


def register_adapter(cls: type[FormatAdapter]) -> type[FormatAdapter]:
    """Class decorator that registers an adapter by its name."""
    instance = cls.__new__(cls)
    _registry[instance.name] = cls
    return cls


def get_adapter(name: str) -> type[FormatAdapter]:
    return _registry[name]


def adapters_for_path(path: Path) -> list[type[FormatAdapter]]:
    """Return all registered adapters that claim they can handle *path*."""
    return [cls for cls in _registry.values() if cls.__new__(cls).can_handle(path)]


def auto_adapter(path: Path) -> FormatAdapter:
    """Pick the first matching adapter and return an instance."""
    # Ensure adapter decorators have run before selecting a handler.
    import src.ingest  # noqa: F401

    candidates = adapters_for_path(path)
    if not candidates:
        raise ValueError(f"No format adapter found for {path}")
    return candidates[0]()
