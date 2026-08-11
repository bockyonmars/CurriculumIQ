"""PyMuPDF text extraction — UI-independent, reusable by the RAG pipeline.

Extracts text page by page into an :class:`ExtractedDocument`. Whitespace is
normalized without destroying paragraph breaks, and pages with little or no
extractable text (e.g. scanned images) are recorded as warnings.
"""

from __future__ import annotations

import logging
import re

import fitz  # PyMuPDF

from src.config import MIN_CHARS_PER_PAGE
from src.models import DocumentPage, ExtractedDocument, compute_document_id

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when a validated PDF still cannot be extracted. User-facing."""


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excess blank lines, keep paragraphs.

    Single newlines within a paragraph become spaces; two or more newlines
    (a paragraph break) collapse to exactly one blank line.
    """
    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Protect paragraph breaks (2+ newlines) with a sentinel.
    text = re.sub(r"\n[ \t]*\n[ \t\n]*", "\x00", text)
    # Remaining single newlines join wrapped lines within a paragraph.
    text = text.replace("\n", " ")
    # Restore paragraph breaks.
    text = text.replace("\x00", "\n\n")
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Trim trailing/leading whitespace on each line and overall.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def extract_document(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """Extract text page by page from a validated PDF.

    Assumes :func:`validate_pdf` has already passed; still guards the open()
    call so a race or edge case surfaces as a clean :class:`ExtractionError`.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.exception("Failed to open PDF %r for extraction", filename)
        raise ExtractionError(
            "This PDF could not be read for extraction. It may be corrupted."
        ) from exc

    pages: list[DocumentPage] = []
    warnings: list[str] = []

    try:
        for index, page in enumerate(doc):
            page_number = index + 1  # one-based
            raw = page.get_text("text")
            text = normalize_whitespace(raw)
            char_count = len(text)
            words = _word_count(text)

            if char_count < MIN_CHARS_PER_PAGE:
                warnings.append(
                    f"Page {page_number} contains little or no extractable text "
                    "(it may be a scanned image)."
                )

            pages.append(
                DocumentPage(
                    page_number=page_number,
                    text=text,
                    character_count=char_count,
                    word_count=words,
                )
            )
    finally:
        doc.close()

    if pages and all(p.character_count < MIN_CHARS_PER_PAGE for p in pages):
        warnings.append(
            "No extractable text was found in this document. It may be a "
            "scanned PDF that needs OCR (not supported in this milestone)."
        )

    return ExtractedDocument(
        filename=filename,
        file_size_bytes=len(file_bytes),
        page_count=len(pages),
        total_character_count=sum(p.character_count for p in pages),
        total_word_count=sum(p.word_count for p in pages),
        pages=pages,
        extraction_warnings=warnings,
        document_id=compute_document_id(file_bytes),
    )
