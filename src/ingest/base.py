"""Abstract base class for all format adapters."""

from __future__ import annotations

import abc
from pathlib import Path

from src.canonical.model import CanonicalDocument


class FormatAdapter(abc.ABC):
    """Every format adapter must implement this interface."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``pdf_native``."""

    @abc.abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this adapter can ingest the given file."""

    @abc.abstractmethod
    def ingest(self, path: Path) -> CanonicalDocument:
        """Parse *path* and return a normalised canonical document."""

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
    candidates = adapters_for_path(path)
    if not candidates:
        raise ValueError(f"No format adapter found for {path}")
    return candidates[0]()
