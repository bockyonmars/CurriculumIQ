"""Token-aware, page-preserving chunking. UI-independent, no API calls.

Each page is split independently so a chunk always carries exactly one source
page number for later citation. Splitting is token-based (tiktoken) and
deterministic for identical input and configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    RAG_CHUNK_OVERLAP_TOKENS,
    RAG_CHUNK_SIZE_TOKENS,
    validate_chunk_config,
)
from src.models import DocumentChunk, ExtractedDocument

# cl100k_base is the encoding used by text-embedding-3-* and gpt-4o models.
_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder() -> "tiktoken.Encoding":
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@lru_cache(maxsize=8)
def _splitter(chunk_size_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    # Measures length in tokens while splitting on paragraph/line/word
    # boundaries first, preserving readability where it can.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING_NAME,
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    document: ExtractedDocument,
    chunk_size_tokens: int = RAG_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = RAG_CHUNK_OVERLAP_TOKENS,
) -> Tuple[List[DocumentChunk], List[str]]:
    """Split a document into page-scoped chunks.

    Returns ``(chunks, warnings)``. Empty/whitespace pages are skipped and
    recorded as warnings; empty chunks are never emitted. Deterministic for
    the same document and configuration.
    """
    validate_chunk_config(chunk_size_tokens, overlap_tokens)

    document_id = document.document_id or ""
    splitter = _splitter(chunk_size_tokens, overlap_tokens)

    chunks: List[DocumentChunk] = []
    warnings: List[str] = []

    for page in document.pages:
        if not page.text.strip():
            warnings.append(f"Page {page.page_number} has no text and produced no chunks.")
            continue

        pieces = [p for p in splitter.split_text(page.text) if p.strip()]
        if not pieces:
            warnings.append(f"Page {page.page_number} produced no chunks.")
            continue

        for idx, piece in enumerate(pieces):
            chunk_id = f"{document_id}_p{page.page_number}_c{idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    page_chunk_index=idx,
                    text=piece,
                    token_count=count_tokens(piece),
                )
            )

    return chunks, warnings
