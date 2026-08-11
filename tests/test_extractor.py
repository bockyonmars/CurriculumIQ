"""Extractor tests: multi-page extraction, page numbers, counts, edge cases."""

from __future__ import annotations

import pytest

from src.ingestion.extractor import (
    ExtractionError,
    extract_document,
    normalize_whitespace,
)


def test_multipage_extraction(multipage_pdf: bytes) -> None:
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    assert doc.page_count == 3
    assert len(doc.pages) == 3
    assert "Algebra" in doc.pages[0].text
    assert "quadratic" in doc.pages[2].text


def test_one_based_page_numbers(multipage_pdf: bytes) -> None:
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    assert [p.page_number for p in doc.pages] == [1, 2, 3]


def test_page_count_matches(multipage_pdf: bytes) -> None:
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    assert doc.page_count == len(doc.pages) == 3


def test_word_and_char_counts(multipage_pdf: bytes) -> None:
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    for page in doc.pages:
        assert page.word_count == len(page.text.split())
        assert page.character_count == len(page.text)
    assert doc.total_word_count == sum(p.word_count for p in doc.pages)
    assert doc.total_character_count == sum(p.character_count for p in doc.pages)
    assert doc.total_word_count > 0


def test_metadata(multipage_pdf: bytes) -> None:
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    assert doc.filename == "curriculum.pdf"
    assert doc.file_size_bytes == len(multipage_pdf)


def test_empty_text_pdf_warns(blank_pdf: bytes) -> None:
    doc = extract_document(blank_pdf, "blank.pdf")
    assert doc.page_count == 2
    assert doc.total_word_count == 0
    assert doc.extraction_warnings  # per-page + document-level warnings present
    assert any("No extractable text" in w for w in doc.extraction_warnings)


def test_corrupted_pdf_raises(corrupted_pdf: bytes) -> None:
    with pytest.raises(ExtractionError):
        extract_document(corrupted_pdf, "corrupt.pdf")


def test_normalize_whitespace_preserves_paragraphs() -> None:
    raw = "Line one\nwrapped here.\n\n\nSecond   paragraph\twith   spaces."
    out = normalize_whitespace(raw)
    assert out == "Line one wrapped here.\n\nSecond paragraph with spaces."
