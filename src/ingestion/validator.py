"""PDF validation — runs before extraction.

Every failure raises ``ValidationError`` carrying a short, user-friendly
message safe to show in the UI. Technical detail goes to the logger, never to
the message. Validation works on in-memory bytes; no temp files are written.
"""

from __future__ import annotations

import logging

import fitz  # PyMuPDF

from src.config import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, MAX_PAGE_COUNT

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"


class ValidationError(Exception):
    """Raised when an uploaded file is not a usable PDF.

    The message is user-facing and must not contain stack traces or paths.
    """


def validate_pdf(file_bytes: bytes, filename: str) -> None:
    """Validate an uploaded PDF. Returns ``None`` on success, else raises.

    Checks: extension, non-empty, size limit, PDF signature, that PyMuPDF can
    open it, and that it is not password protected.
    """
    if not filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files are supported. Please upload a .pdf file.")

    size = len(file_bytes)
    if size == 0:
        raise ValidationError("This file is empty (0 bytes). Please upload a valid PDF.")

    if size > MAX_FILE_SIZE_BYTES:
        actual_mb = size / (1024 * 1024)
        raise ValidationError(
            f"File is too large ({actual_mb:.1f} MB). "
            f"The maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    if not file_bytes.startswith(PDF_SIGNATURE):
        raise ValidationError(
            "This does not appear to be a valid PDF (missing PDF signature). "
            "The file may be renamed or corrupted."
        )

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValidationError(
                    "This PDF is password protected. Please upload an unprotected copy."
                )
            if doc.page_count == 0:
                raise ValidationError("This PDF has no pages.")
            if doc.page_count > MAX_PAGE_COUNT:
                raise ValidationError(
                    f"This PDF has {doc.page_count} pages, which exceeds the "
                    f"{MAX_PAGE_COUNT}-page limit. Please split it into smaller files."
                )
    except ValidationError:
        raise
    except Exception as exc:  # PyMuPDF raises a variety of low-level errors
        logger.exception("Failed to open PDF %r during validation", filename)
        raise ValidationError(
            "This PDF could not be opened. It may be corrupted or malformed."
        ) from exc
