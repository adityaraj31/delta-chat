# Tests for alignment and delta engine
"""Tests for src/delta/align.py and src/delta/engine.py."""

from src.canonical.model import CanonicalDocument, Element, ElementType
from src.delta.align import align_documents
from src.delta.engine import compute_delta


def _make_doc(elements: list[Element], path: str = "test.pdf") -> CanonicalDocument:
    return CanonicalDocument(source_path=path, format_adapter="pdf_native", elements=elements)


def test_align_identical_docs():
    elements = [
        Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="b", type=ElementType.TEXT, raw_text="Pump A"),
    ]
    old = _make_doc(elements, "old.pdf")
    new = _make_doc([Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
                     Element(id="b2", type=ElementType.TEXT, raw_text="Pump A")], "new.pdf")
    alignment = align_documents(old, new)
    assert len(alignment.matched) == 2
    assert len(alignment.removed) == 0
    assert len(alignment.added) == 0


def test_align_added_element():
    old = _make_doc([Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01")])
    new = _make_doc([
        Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="b", type=ElementType.TAG, raw_text="TI-200-01", tag_number="TI-200-01"),
    ])
    alignment = align_documents(old, new)
    assert len(alignment.matched) == 1
    assert len(alignment.added) == 1
    assert alignment.added[0].tag_number == "TI-200-01"


def test_align_removed_element():
    old = _make_doc([
        Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="b", type=ElementType.TAG, raw_text="TI-200-01", tag_number="TI-200-01"),
    ])
    new = _make_doc([Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01")])
    alignment = align_documents(old, new)
    assert len(alignment.matched) == 1
    assert len(alignment.removed) == 1
    assert alignment.removed[0].tag_number == "TI-200-01"


def test_align_fuzzy_fallback():
    old = _make_doc([Element(id="a", type=ElementType.TEXT, raw_text="This is a label")])
    new = _make_doc([Element(id="b", type=ElementType.TEXT, raw_text="This is a lable")])  # typo
    alignment = align_documents(old, new, config=None)
    # Should fuzzy-match despite typo
    assert len(alignment.matched) == 1


def test_delta_added_and_removed():
    old = _make_doc([
        Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="b", type=ElementType.TAG, raw_text="TI-200-01", tag_number="TI-200-01"),
    ])
    new = _make_doc([
        Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="c", type=ElementType.TAG, raw_text="FV-300-01", tag_number="FV-300-01"),
    ])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    kinds = {e.kind.value for e in entries}
    assert "removed" in kinds
    assert "added" in kinds


def test_delta_modified_text():
    old = _make_doc([Element(id="a", type=ElementType.TAG, raw_text="PI-101-01 setpoint 100", tag_number="PI-101-01")])
    new = _make_doc([Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01 setpoint 200", tag_number="PI-101-01")])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    assert len(entries) == 1
    assert entries[0].kind.value == "modified"


def test_delta_empty_docs():
    old = _make_doc([])
    new = _make_doc([])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    assert len(entries) == 0


def test_delta_noop_identical_tag_not_emitted():
    """Identical elements matched by stable key must NOT produce a MODIFIED entry."""
    old = _make_doc([Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01")])
    new = _make_doc([Element(id="a2", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01")])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    assert len(entries) == 0, f"Expected 0 entries for identical pair, got {len(entries)}"


def test_delta_noop_identical_note_not_emitted():
    """Identical note elements must not produce a MODIFIED entry."""
    old = _make_doc([Element(id="n1", type=ElementType.NOTE, raw_text="1. Refer to datasheet", note_number=1)])
    new = _make_doc([Element(id="n2", type=ElementType.NOTE, raw_text="1. Refer to datasheet", note_number=1)])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    assert len(entries) == 0


def test_delta_noop_identical_line_spec_not_emitted():
    """Identical line-spec elements must not produce a MODIFIED entry."""
    old = _make_doc([Element(id="l1", type=ElementType.LINE_SPEC, raw_text='4"-CS-12-1234-12345-12', line_spec='4"CS1212341234512')])
    new = _make_doc([Element(id="l2", type=ElementType.LINE_SPEC, raw_text='4"-CS-12-1234-12345-12', line_spec='4"CS1212341234512')])
    alignment = align_documents(old, new)
    entries = compute_delta(alignment)
    assert len(entries) == 0
