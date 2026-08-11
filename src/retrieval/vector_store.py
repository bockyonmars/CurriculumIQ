"""Chroma-backed vector store. Streamlit-independent.

Embeddings are always supplied by the caller (via an EmbeddingProvider), so
this module never talks to OpenAI. Distances use cosine space: lower is closer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Safe, user-facing error for vector-store failures."""


class VectorStore:
    """Persistent Chroma collection for CurriculumIQ chunks."""

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Failed to initialize Chroma at %r", persist_directory)
            raise VectorStoreError(
                "Could not initialize the local vector database (Chroma)."
            ) from exc

    # --- writes ---------------------------------------------------------
    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts do not match.")
        try:
            self._collection.add(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[c.chroma_metadata() for c in chunks],
            )
        except Exception as exc:
            logger.exception("Chroma add failed for %d chunks", len(chunks))
            raise VectorStoreError("Failed to store chunks in the vector database.") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            logger.exception("Chroma delete failed for document %s", document_id)
            raise VectorStoreError("Failed to delete existing document chunks.") from exc

    def clear(self) -> None:
        """Drop and recreate the collection (dev/reset operation)."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Chroma clear failed")
            raise VectorStoreError("Failed to clear the vector database.") from exc

    # --- reads ----------------------------------------------------------
    def count(self, document_id: Optional[str] = None) -> int:
        try:
            if document_id is None:
                return self._collection.count()
            got = self._collection.get(where={"document_id": document_id}, include=[])
            return len(got.get("ids", []))
        except Exception as exc:
            logger.exception("Chroma count failed")
            raise VectorStoreError("Failed to read from the vector database.") from exc

    def has_document(self, document_id: str) -> bool:
        return self.count(document_id) > 0

    def list_document_ids(self) -> List[str]:
        try:
            got = self._collection.get(include=["metadatas"])
        except Exception as exc:
            logger.exception("Chroma list failed")
            raise VectorStoreError("Failed to list indexed documents.") from exc
        seen: List[str] = []
        for md in got.get("metadatas", []) or []:
            doc_id = md.get("document_id")
            if doc_id and doc_id not in seen:
                seen.append(doc_id)
        return seen

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        document_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return up to ``top_k`` nearest chunks as dicts, closest first."""
        if self.count() == 0:
            return []
        where = {"document_id": document_id} if document_id else None
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("Chroma query failed")
            raise VectorStoreError("Failed to search the vector database.") from exc

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[Dict] = []
        for i, chunk_id in enumerate(ids):
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out
