"""Typed data models for CurriculumIQ.

Kept free of any Streamlit or I/O dependency so the ingestion and RAG pipelines
can import and reuse these directly.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Chroma metadata values must be primitives.
MetadataValue = Union[str, int, float, bool]


def compute_document_id(file_bytes: bytes) -> str:
    """Deterministic content identifier from a file's checksum.

    Same bytes -> same id, so re-uploading an identical PDF is detectable.
    """
    return "doc_" + hashlib.sha256(file_bytes).hexdigest()[:16]


class DocumentPage(BaseModel):
    """A single extracted page, with one-based numbering."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str
    character_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)


class ExtractedDocument(BaseModel):
    """The full result of extracting a curriculum PDF."""

    filename: str
    file_size_bytes: int = Field(..., ge=0)
    page_count: int = Field(..., ge=0)
    total_character_count: int = Field(..., ge=0)
    total_word_count: int = Field(..., ge=0)
    pages: List[DocumentPage] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    # Content-derived id; empty only if constructed without source bytes.
    document_id: str = ""

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


class DocumentChunk(BaseModel):
    """A token-bounded, page-traceable slice of a document, ready to embed."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(..., ge=1, description="1-based source page")
    page_chunk_index: int = Field(..., ge=0, description="0-based index within its page")
    text: str
    token_count: int = Field(..., ge=0)
    metadata: Dict[str, MetadataValue] = Field(default_factory=dict)

    def chroma_metadata(self) -> Dict[str, MetadataValue]:
        """Chroma-safe metadata (primitives only) for storage/filtering."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "page_chunk_index": self.page_chunk_index,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
        }


class IndexingResult(BaseModel):
    """Outcome of indexing one document into the vector store."""

    document_id: str
    filename: str
    pages_processed: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    embedding_model: str
    duration_seconds: float = Field(..., ge=0)
    warnings: List[str] = Field(default_factory=list)
    # "indexed" | "reindexed" | "duplicate" | "no_chunks"
    status: str


class RetrievalResult(BaseModel):
    """A single search hit. Lower ``distance`` means a closer (better) match."""

    rank: int = Field(..., ge=1)
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    text: str
    distance: Optional[float] = None


class SourceCitation(BaseModel):
    """A retrieved source given a stable ``source_id`` (e.g. ``S1``).

    Displayed filename/page ALWAYS come from these trusted retrieval fields —
    never from model-generated text.
    """

    source_id: str  # "S1", "S2", ... assigned in Python before generation
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    passage: str
    distance: Optional[float] = None
    rank: int = Field(..., ge=1)


class TutorAnswer(BaseModel):
    """A grounded tutor answer with validated citations (Milestone 3)."""

    answer_id: str
    question: str
    answer_text: str
    citations: List[SourceCitation] = Field(default_factory=list)
    retrieved_sources: List[SourceCitation] = Field(default_factory=list)
    abstained: bool = False
    model: str = ""
    retrieval_seconds: float = Field(0.0, ge=0)
    generation_seconds: float = Field(0.0, ge=0)
    latency_seconds: float = Field(0.0, ge=0)
    usage: Optional[Dict[str, int]] = None
    warnings: List[str] = Field(default_factory=list)
