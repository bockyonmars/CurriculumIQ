"""Retriever tests: ordering, top-k, doc filtering, empty query/collection."""

from __future__ import annotations

import pytest

from src.config import CHROMA_COLLECTION_NAME
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.indexer import IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore
from tests.conftest import make_extracted_document


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"), CHROMA_COLLECTION_NAME)


@pytest.fixture
def indexed(store, embedder):
    """Index a small, semantically distinct corpus (one chunk per page)."""
    svc = IndexingService(store, embedder, chunk_size_tokens=200, overlap_tokens=20)
    doc = make_extracted_document(
        [
            "Photosynthesis converts sunlight into chemical energy in plant leaves.",
            "The quadratic formula solves quadratic equations using coefficients.",
            "Newton's laws describe motion, force, mass, and acceleration in physics.",
        ],
        filename="science.pdf",
        document_id="doc_corpus01",
    )
    svc.index_document(doc)
    return store, embedder


def test_relevant_result_ranked_first(indexed):
    store, embedder = indexed
    retriever = RetrievalService(store, embedder, default_top_k=3)
    results = retriever.search("quadratic equations and coefficients")
    assert results
    # The quadratic page (page 2) should be the closest match.
    assert results[0].page_number == 2
    assert results[0].rank == 1
    # Distances are non-decreasing (closest first).
    dists = [r.distance for r in results if r.distance is not None]
    assert dists == sorted(dists)


def test_top_k_respected(indexed):
    store, embedder = indexed
    retriever = RetrievalService(store, embedder, default_top_k=5)
    results = retriever.search("energy", top_k=2)
    assert len(results) <= 2


def test_document_filter(indexed):
    store, embedder = indexed
    # Add a second document; filter must exclude it.
    svc = IndexingService(store, embedder, chunk_size_tokens=200, overlap_tokens=20)
    other = make_extracted_document(
        ["Photosynthesis and sunlight in a totally different document."],
        filename="other.pdf",
        document_id="doc_other01",
    )
    svc.index_document(other)

    retriever = RetrievalService(store, embedder, default_top_k=10)
    results = retriever.search("photosynthesis", document_id="doc_corpus01")
    assert results
    assert all(r.document_id == "doc_corpus01" for r in results)


def test_metadata_returned(indexed):
    store, embedder = indexed
    retriever = RetrievalService(store, embedder, default_top_k=1)
    results = retriever.search("Newton force acceleration")
    r = results[0]
    assert r.filename == "science.pdf"
    assert r.page_number == 3
    assert r.chunk_id
    assert r.text.strip()


def test_blank_query_rejected(indexed):
    store, embedder = indexed
    retriever = RetrievalService(store, embedder)
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_empty_collection_returns_empty(store, embedder):
    retriever = RetrievalService(store, embedder, default_top_k=5)
    assert retriever.search("anything at all") == []
