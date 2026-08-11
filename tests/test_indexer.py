"""Indexer tests: success, duplicate, re-index, verification, metadata.

All embeddings are the deterministic FakeEmbeddingProvider; Chroma writes to a
per-test temporary directory. No network, no OpenAI.
"""

from __future__ import annotations

import pytest

from src.config import CHROMA_COLLECTION_NAME
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.indexer import IndexingService
from src.retrieval.vector_store import VectorStore
from tests.conftest import make_extracted_document


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"), CHROMA_COLLECTION_NAME)


@pytest.fixture
def service(store):
    return IndexingService(store, FakeEmbeddingProvider(), chunk_size_tokens=60, overlap_tokens=10)


def test_successful_indexing(service, store):
    doc = make_extracted_document(
        ["Algebra variables and equations.", "Parabolas and quadratic functions."],
        document_id="doc_success01",
    )
    result = service.index_document(doc)
    assert result.status == "indexed"
    assert result.chunks_created > 0
    assert result.pages_processed == 2
    assert result.embedding_model == "fake-embedding"
    assert store.count("doc_success01") == result.chunks_created


def test_duplicate_detection(service):
    doc = make_extracted_document(["Content about algebra."], document_id="doc_dup01")
    first = service.index_document(doc)
    assert first.status == "indexed"
    second = service.index_document(doc)  # no reindex
    assert second.status == "duplicate"
    assert second.chunks_created == 0
    assert any("already indexed" in w.lower() for w in second.warnings)


def test_reindex_replaces_chunks(service, store):
    doc = make_extracted_document(["Original algebra content."], document_id="doc_re01")
    service.index_document(doc)
    count_before = store.count("doc_re01")

    updated = make_extracted_document(
        ["Completely different content about biology and cells now, much longer text "
         "to change the chunk layout entirely across the page."],
        document_id="doc_re01",  # same id => same document, re-indexed
    )
    result = service.index_document(updated, reindex=True)
    assert result.status == "reindexed"
    # Old chunks were deleted first; only the new chunks remain.
    assert store.count("doc_re01") == result.chunks_created
    # No orphaned duplicates from the first index.
    assert store.count("doc_re01") >= 1
    assert count_before >= 1


def test_no_chunks_for_empty_document(service, store):
    doc = make_extracted_document(["   ", ""], document_id="doc_empty01")
    result = service.index_document(doc)
    assert result.status == "no_chunks"
    assert result.chunks_created == 0
    assert store.count("doc_empty01") == 0


def test_stored_metadata_is_correct(service, store):
    doc = make_extracted_document(
        ["Page one about algebra.", "Page two about geometry."],
        filename="math.pdf",
        document_id="doc_meta01",
    )
    service.index_document(doc)
    got = store._collection.get(where={"document_id": "doc_meta01"}, include=["metadatas"])
    metas = got["metadatas"]
    assert metas
    for md in metas:
        assert md["document_id"] == "doc_meta01"
        assert md["filename"] == "math.pdf"
        assert md["page_number"] in (1, 2)
        assert isinstance(md["page_number"], int)
