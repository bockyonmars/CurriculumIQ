"""Semantic retrieval service. Returns ranked chunks — never generates answers.

Scores are Chroma cosine distances: LOWER means a closer (better) match. If the
store returns no distance, ``distance`` is ``None`` (never fabricated).
"""

from __future__ import annotations

from typing import List, Optional

from src.config import RAG_TOP_K
from src.models import RetrievalResult
from src.retrieval.embeddings import EmbeddingProvider
from src.retrieval.vector_store import VectorStore


class RetrievalService:
    """Turns a natural-language query into ranked, page-traceable results."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        default_top_k: int = RAG_TOP_K,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_provider
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        k = self._default_top_k if top_k is None else int(top_k)
        if k < 1:
            k = 1

        query_embedding = self._embedder.embed_query(query)
        hits = self._store.query(query_embedding, top_k=k, document_id=document_id)

        results: List[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            md = hit.get("metadata") or {}
            results.append(
                RetrievalResult(
                    rank=rank,
                    chunk_id=hit.get("chunk_id", md.get("chunk_id", "")),
                    document_id=md.get("document_id", ""),
                    filename=md.get("filename", ""),
                    page_number=int(md.get("page_number", 1)),
                    text=hit.get("text", ""),
                    distance=hit.get("distance"),
                )
            )
        return results
