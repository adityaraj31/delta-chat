# Tests for canonical model
"""Tests for src/canonical/model.py."""

from src.canonical.model import (
    CanonicalDocument,
    Element,
    ElementType,
    TitleBlock,
    make_element_id,
)


def test_make_element_id_deterministic():
    id1 = make_element_id("test.pdf", 1, 0)
    id2 = make_element_id("test.pdf", 1, 0)
    assert id1 == id2


def test_make_element_id_unique_per_position():
    id1 = make_element_id("test.pdf", 1, 0)
    id2 = make_element_id("test.pdf", 1, 1)
    assert id1 != id2


def test_element_stable_key_tag():
    el = Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01")
    assert el.stable_key() == "tag:PI-101-01"


def test_element_stable_key_line_spec():
    el = Element(id="b", type=ElementType.LINE_SPEC, raw_text='4"-XX-NN-NNNN', line_spec='4"-XX-NN-NNNN')
    assert el.stable_key() == 'line:4"-XX-NN-NNNN'


def test_element_stable_key_note():
    el = Element(id="c", type=ElementType.NOTE, raw_text="1. Some note", note_number=1)
    assert el.stable_key() == "note:1"


def test_element_stable_key_none_for_text():
    el = Element(id="d", type=ElementType.TEXT, raw_text="Some label")
    assert el.stable_key() is None


def test_canonical_document_element_index():
    el1 = Element(id="e1", type=ElementType.TAG, raw_text="TI-200-01", tag_number="TI-200-01")
    el2 = Element(id="e2", type=ElementType.TEXT, raw_text="Pump")
    doc = CanonicalDocument(source_path="test.pdf", format_adapter="pdf_native", elements=[el1, el2])
    idx = doc.element_index
    assert "e1" in idx
    assert "e2" in idx
    assert idx["e1"].tag_number == "TI-200-01"


def test_canonical_document_elements_by_type():
    elements = [
        Element(id="a", type=ElementType.TAG, raw_text="PI-101-01"),
        Element(id="b", type=ElementType.TEXT, raw_text="label"),
        Element(id="c", type=ElementType.TAG, raw_text="TI-200-01"),
    ]
    doc = CanonicalDocument(source_path="x.pdf", format_adapter="pdf_native", elements=elements)
    tags = doc.elements_by_type(ElementType.TAG)
    assert len(tags) == 2


def test_canonical_document_stable_keyed_elements():
    elements = [
        Element(id="a", type=ElementType.TAG, raw_text="PI-101-01", tag_number="PI-101-01"),
        Element(id="b", type=ElementType.TEXT, raw_text="no key"),
        Element(id="c", type=ElementType.NOTE, raw_text="1. Note", note_number=1),
    ]
    doc = CanonicalDocument(source_path="x.pdf", format_adapter="pdf_native", elements=elements)
    keyed = doc.stable_keyed_elements()
    assert "tag:PI-101-01" in keyed
    assert "note:1" in keyed
    assert len(keyed) == 2


def test_title_block_defaults():
    tb = TitleBlock()
    assert tb.project == ""
    assert tb.document_number == ""
    assert tb.extra == {}
