"""Chunker tests: metadata, multi-chunk, overlap, empty pages, determinism."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_document, count_tokens
from tests.conftest import make_extracted_document

LONG_PAGE = (
    "Algebra introduces variables that represent unknown quantities. " * 200
)  # comfortably exceeds a few hundred tokens


def test_single_page_one_chunk() -> None:
    doc = make_extracted_document(["A short page about linear equations."])
    chunks, warnings = chunk_document(doc)
    assert len(chunks) == 1
    assert warnings == []
    assert chunks[0].page_number == 1
    assert chunks[0].page_chunk_index == 0


def test_multipage_metadata_one_based() -> None:
    doc = make_extracted_document(
        ["Page one algebra.", "Page two equations.", "Page three parabolas."]
    )
    chunks, _ = chunk_document(doc)
    assert [c.page_number for c in chunks] == [1, 2, 3]
    for c in chunks:
        assert c.document_id == doc.document_id
        assert c.filename == doc.filename
        assert c.page_chunk_index == 0


def test_long_page_multiple_chunks() -> None:
    doc = make_extracted_document([LONG_PAGE])
    chunks, _ = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # All chunks trace to page 1 with increasing 0-based indices.
    assert all(c.page_number == 1 for c in chunks)
    assert [c.page_chunk_index for c in chunks] == list(range(len(chunks)))


def test_short_page_single_chunk() -> None:
    doc = make_extracted_document(["Just a sentence."])
    chunks, _ = chunk_document(doc, chunk_size_tokens=700, overlap_tokens=100)
    assert len(chunks) == 1


def test_empty_page_skipped_with_warning() -> None:
    doc = make_extracted_document(["Real content here.", "   ", "More content."])
    chunks, warnings = chunk_document(doc)
    assert [c.page_number for c in chunks] == [1, 3]  # page 2 skipped
    assert any("Page 2" in w for w in warnings)


def test_no_empty_chunks_emitted() -> None:
    doc = make_extracted_document([LONG_PAGE])
    chunks, _ = chunk_document(doc, chunk_size_tokens=80, overlap_tokens=10)
    assert all(c.text.strip() for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_deterministic_chunk_ids() -> None:
    doc = make_extracted_document([LONG_PAGE, "Second page content."])
    a, _ = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=20)
    b, _ = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]
    # IDs are unique and formatted document/page/index.
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id == f"{doc.document_id}_p1_c0"


def test_configurable_overlap_changes_output() -> None:
    doc = make_extracted_document([LONG_PAGE])
    few, _ = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=0)
    many, _ = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=50)
    # More overlap => more chunks for the same page.
    assert len(many) >= len(few)


def test_invalid_chunk_config_rejected() -> None:
    doc = make_extracted_document(["content"])
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_tokens=0, overlap_tokens=0)


def test_token_count_matches_encoder() -> None:
    doc = make_extracted_document(["Algebra equations and parabolas everywhere."])
    chunks, _ = chunk_document(doc)
    assert chunks[0].token_count == count_tokens(chunks[0].text)
