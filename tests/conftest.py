"""Shared test fixtures: small PDFs generated in memory (no external files)."""

from __future__ import annotations

from typing import List

import fitz  # PyMuPDF
import pytest

from src.models import DocumentPage, ExtractedDocument


def make_extracted_document(
    pages_text: List[str],
    filename: str = "curriculum.pdf",
    document_id: str = "doc_test0000000000",
) -> ExtractedDocument:
    """Build an ExtractedDocument directly (no PDF) for chunk/index/search tests."""
    pages = [
        DocumentPage(
            page_number=i + 1,
            text=text,
            character_count=len(text),
            word_count=len(text.split()),
        )
        for i, text in enumerate(pages_text)
    ]
    return ExtractedDocument(
        filename=filename,
        file_size_bytes=sum(len(t) for t in pages_text),
        page_count=len(pages),
        total_character_count=sum(p.character_count for p in pages),
        total_word_count=sum(p.word_count for p in pages),
        pages=pages,
        document_id=document_id,
    )


def make_text_pdf(pages_text: List[str]) -> bytes:
    """Build a PDF with one page per string, each rendering that text."""
    doc = fitz.open()
    for content in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), content, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def make_blank_pdf(page_count: int = 1) -> bytes:
    """Build a PDF with blank pages (no text to extract)."""
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def multipage_pdf() -> bytes:
    return make_text_pdf(
        [
            "Introduction to Algebra. Variables represent unknown numbers.",
            "Chapter Two covers linear equations and their solutions.",
            "Chapter Three introduces quadratic functions and parabolas.",
        ]
    )


@pytest.fixture
def blank_pdf() -> bytes:
    return make_blank_pdf(2)


@pytest.fixture
def corrupted_pdf() -> bytes:
    """Has a PDF signature but a broken body PyMuPDF cannot parse."""
    return b"%PDF-1.4\n%broken\nthis is not a real pdf body at all"
