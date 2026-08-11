"""Indexing service: chunk -> embed -> store, with safe duplicate/re-index.

Coordinates chunking, embedding, and Chroma insertion into one typed result.
Never reports success unless every intended chunk was actually stored.
"""

from __future__ import annotations

import logging
import time
from typing import List

from src.config import RAG_CHUNK_OVERLAP_TOKENS, RAG_CHUNK_SIZE_TOKENS
from src.ingestion.chunker import chunk_document
from src.models import DocumentChunk, ExtractedDocument, IndexingResult
from src.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from src.retrieval.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Safe, user-facing indexing failure.

    Carries a safe ``category`` propagated from the underlying cause (e.g. an
    embedding quota/auth error) for accurate, non-sensitive reporting.
    """

    def __init__(self, message: str, category: str = "other") -> None:
        super().__init__(message)
        self.category = category


class IndexingService:
    """Turns an ExtractedDocument into stored, searchable chunks."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunk_size_tokens: int = RAG_CHUNK_SIZE_TOKENS,
        overlap_tokens: int = RAG_CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_provider
        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens

    def is_indexed(self, document_id: str) -> bool:
        return self._store.has_document(document_id)

    def index_document(
        self, document: ExtractedDocument, reindex: bool = False
    ) -> IndexingResult:
        start = time.perf_counter()

        chunks, warnings = chunk_document(
            document, self._chunk_size, self._overlap
        )

        if not chunks:
            warnings.append("No chunks were produced; nothing was indexed.")
            return IndexingResult(
                document_id=document.document_id,
                filename=document.filename,
                pages_processed=len(document.pages),
                chunks_created=0,
                embedding_model=self._embedder.model_name,
                duration_seconds=round(time.perf_counter() - start, 3),
                warnings=warnings,
                status="no_chunks",
            )

        already = self._store.has_document(document.document_id)
        if already and not reindex:
            return IndexingResult(
                document_id=document.document_id,
                filename=document.filename,
                pages_processed=len(document.pages),
                chunks_created=0,
                embedding_model=self._embedder.model_name,
                duration_seconds=round(time.perf_counter() - start, 3),
                warnings=[
                    "This document is already indexed. Use re-index to replace it."
                ],
                status="duplicate",
            )

        # Embed first, so a failure never leaves a half-deleted re-index.
        try:
            embeddings = self._embed(chunks)
        except EmbeddingError as exc:
            raise IndexingError(str(exc), category=getattr(exc, "category", "other")) from exc

        try:
            if already and reindex:
                self._store.delete_document(document.document_id)
            self._store.add_chunks(chunks, embeddings)
        except VectorStoreError as exc:
            raise IndexingError(str(exc)) from exc

        # Verify every intended chunk landed before claiming success.
        stored = self._store.count(document.document_id)
        if stored != len(chunks):
            logger.error(
                "Index verification mismatch for %s: expected %d, stored %d",
                document.document_id, len(chunks), stored,
            )
            raise IndexingError(
                "Indexing did not store all chunks. The index may be incomplete."
            )

        return IndexingResult(
            document_id=document.document_id,
            filename=document.filename,
            pages_processed=len(document.pages),
            chunks_created=len(chunks),
            embedding_model=self._embedder.model_name,
            duration_seconds=round(time.perf_counter() - start, 3),
            warnings=warnings,
            status="reindexed" if (already and reindex) else "indexed",
        )

    def _embed(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        # Single batched call; provider reuses one client internally.
        return self._embedder.embed_documents([c.text for c in chunks])
