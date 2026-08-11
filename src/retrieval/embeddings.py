"""Embedding providers behind one injectable interface.

Production uses OpenAI via the langchain-openai integration; tests use a
deterministic, offline fake. A single client is created per provider instance
and reused for every batch — never one client per chunk.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import List

from src.openai_safe import classify_openai_error

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Safe, user-facing error for embedding failures (no keys, no traces).

    ``category`` is a safe code (auth/quota/rate_limit/timeout/...) for reporting.
    """

    def __init__(self, message: str, category: str = "other") -> None:
        super().__init__(message)
        self.category = category


class EmbeddingProvider(ABC):
    """Minimal embedding interface used by indexing and retrieval."""

    model_name: str

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts. Rejects empty input."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Rejects blank input."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings via langchain-openai. Model name comes from config."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise EmbeddingError("OpenAI API key is not configured.")
        # Imported lazily so the app runs without langchain-openai loaded until
        # a real embedding is actually requested.
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model
        # One client per provider instance, reused across all batches.
        self._client = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise EmbeddingError("No texts to embed.")
        if any(not t.strip() for t in texts):
            raise EmbeddingError("Refusing to embed empty text.")
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:  # network / auth / rate-limit / etc.
            logger.error("OpenAI embed_documents failed: %s", type(exc).__name__)
            raise EmbeddingError(
                "Failed to generate embeddings from OpenAI. Check your API key, "
                "network connection, and quota.",
                category=classify_openai_error(exc),
            ) from exc

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        try:
            return self._client.embed_query(text)
        except Exception as exc:
            logger.error("OpenAI embed_query failed: %s", type(exc).__name__)
            raise EmbeddingError(
                "Failed to generate a query embedding from OpenAI. Check your "
                "API key, network connection, and quota.",
                category=classify_openai_error(exc),
            ) from exc


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing-bag-of-words embeddings for tests. No network.

    Vectors are L2-normalized token-hash histograms over content words (common
    English stopwords are ignored), so cosine similarity tracks meaningful
    lexical overlap — enough for reproducible retrieval ordering, and enough to
    separate on-topic from off-topic text, without any API call.
    """

    # Ignoring these keeps unrelated questions from matching via filler words.
    _STOPWORDS = frozenset(
        "a an the of to in on at for and or is are was were be been being do does "
        "did what which who whom whose how why when where that this these those "
        "with from by as it its into you your i we they he she them his her our "
        "can could will would should may might must have has had not no yes if "
        "about over under between each any some all more most such than then".split()
    )

    def __init__(self, dim: int = 256, model_name: str = "fake-embedding") -> None:
        self.dim = dim
        self.model_name = model_name

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % dim

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in self._STOPWORDS:
                continue
            v[self._bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            # No alphanumeric tokens; return a stable unit vector.
            v[0] = 1.0
            return v
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise EmbeddingError("No texts to embed.")
        if any(not t.strip() for t in texts):
            raise EmbeddingError("Refusing to embed empty text.")
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._vec(text)
