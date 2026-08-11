"""Validator tests: valid, empty, oversized, non-PDF, bad signature."""

from __future__ import annotations

import pytest

from src.config import MAX_FILE_SIZE_BYTES
from src.ingestion import validator as validator_mod
from src.ingestion.validator import ValidationError, validate_pdf


def test_valid_pdf_passes(multipage_pdf: bytes) -> None:
    # Should not raise.
    assert validate_pdf(multipage_pdf, "curriculum.pdf") is None


def test_too_many_pages_rejected(multipage_pdf: bytes, monkeypatch) -> None:
    # multipage_pdf has 3 pages; lower the limit to force rejection.
    monkeypatch.setattr(validator_mod, "MAX_PAGE_COUNT", 2)
    with pytest.raises(ValidationError, match="exceeds"):
        validate_pdf(multipage_pdf, "curriculum.pdf")


def test_empty_file_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_pdf(b"", "curriculum.pdf")


def test_oversized_file_rejected() -> None:
    big = b"%PDF-" + b"0" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(ValidationError, match="too large"):
        validate_pdf(big, "curriculum.pdf")


def test_non_pdf_extension_rejected(multipage_pdf: bytes) -> None:
    with pytest.raises(ValidationError, match="Only PDF"):
        validate_pdf(multipage_pdf, "notes.txt")


def test_invalid_signature_rejected() -> None:
    with pytest.raises(ValidationError, match="signature"):
        validate_pdf(b"This is plain text, not a PDF at all.", "curriculum.pdf")


def test_corrupted_pdf_rejected(corrupted_pdf: bytes) -> None:
    with pytest.raises(ValidationError):
        validate_pdf(corrupted_pdf, "curriculum.pdf")
