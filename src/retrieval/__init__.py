"""Retrieval: embeddings, Chroma vector store, indexing, and search.

All OpenAI access is isolated behind an injectable ``EmbeddingProvider`` so
tests can run offline with deterministic fake embeddings.
"""

from src.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from src.retrieval.indexer import IndexingError, IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore, VectorStoreError

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "IndexingError",
    "IndexingService",
    "RetrievalService",
    "VectorStore",
    "VectorStoreError",
]
