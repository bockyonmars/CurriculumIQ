"""Injectable providers/store for the API, so tests can swap in offline fakes.

Production wires the real OpenAI providers and the persistent Chroma store;
tests override these via FastAPI ``dependency_overrides`` with fakes and a temp
store, so the automated suite never makes a live OpenAI call.
"""

from __future__ import annotations

from functools import lru_cache

from src import config
from src.generation.provider import OpenAIAnswerProvider
from src.retrieval.embeddings import OpenAIEmbeddingProvider
from src.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Persistent Chroma store (one instance reused across requests)."""
    return VectorStore(config.CHROMA_PERSIST_DIRECTORY, config.CHROMA_COLLECTION_NAME)


@lru_cache(maxsize=4)
def get_embedder():
    return OpenAIEmbeddingProvider(config.OPENAI_EMBEDDING_MODEL, config.OPENAI_API_KEY)


@lru_cache(maxsize=4)
def get_answerer():
    return OpenAIAnswerProvider(config.OPENAI_CHAT_MODEL, config.OPENAI_API_KEY)
