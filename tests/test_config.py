"""Config tests: defaults, chunk-config validation, key-optional extraction."""

from __future__ import annotations

import importlib

import pytest

from src import config
from src.ingestion.extractor import extract_document


def test_defaults_load_correctly():
    assert config.RAG_CHUNK_SIZE_TOKENS == 700
    assert config.RAG_CHUNK_OVERLAP_TOKENS == 100
    assert config.RAG_TOP_K == 5
    assert config.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert config.CHROMA_PERSIST_DIRECTORY  # non-empty


def test_generation_defaults_load_correctly():
    # Chat model is configurable via env/.env; just require it to be set.
    assert isinstance(config.OPENAI_CHAT_MODEL, str) and config.OPENAI_CHAT_MODEL.strip()
    assert config.RAG_MAX_CONTEXT_CHUNKS == 5
    assert config.RAG_MAX_CONTEXT_TOKENS == 2000
    assert config.RAG_MAX_DISTANCE == 0.7
    assert config.RAG_HISTORY_MESSAGE_LIMIT == 6
    assert config.RAG_MAX_QUESTION_LENGTH == 1000


def test_valid_generation_config_passes():
    assert config.validate_generation_config(5, 2000, 0.7, 6, 1000) is None


def test_invalid_generation_config_rejected():
    with pytest.raises(ValueError):
        config.validate_generation_config(0, 2000, 0.7, 6, 1000)  # chunks
    with pytest.raises(ValueError):
        config.validate_generation_config(5, 0, 0.7, 6, 1000)  # tokens
    with pytest.raises(ValueError):
        config.validate_generation_config(5, 2000, 0.0, 6, 1000)  # distance
    with pytest.raises(ValueError):
        config.validate_generation_config(5, 2000, 0.7, -1, 1000)  # history
    with pytest.raises(ValueError):
        config.validate_generation_config(5, 2000, 0.7, 6, 0)  # question length


def test_generation_enabled_requires_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    assert config.generation_enabled() is False
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(config, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
    assert config.generation_enabled() is True


def test_valid_chunk_config_passes():
    assert config.validate_chunk_config(700, 100) is None


def test_invalid_chunk_config_rejected():
    with pytest.raises(ValueError):
        config.validate_chunk_config(0, 0)
    with pytest.raises(ValueError):
        config.validate_chunk_config(100, 100)  # overlap == size
    with pytest.raises(ValueError):
        config.validate_chunk_config(100, -1)


def test_has_openai_key_reflects_env(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    assert config.has_openai_key() is False
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-not-a-real-key")
    assert config.has_openai_key() is True


def test_extraction_works_without_api_key(monkeypatch, multipage_pdf):
    # Simulate a missing key; extraction must still succeed and not crash.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    importlib.reload(config)
    assert config.has_openai_key() is False
    doc = extract_document(multipage_pdf, "curriculum.pdf")
    assert doc.page_count == 3
    assert doc.document_id.startswith("doc_")
