"""TutorService: the grounded RAG answer pipeline (Milestone 3).

Flow: validate question -> retrieve -> filter by quality/budget -> assign source
IDs -> (abstain locally if no evidence) -> build prompt -> generate -> validate
citations -> assemble a typed TutorAnswer. Every factual question retrieves;
prior turns are context only, never evidence.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional, Tuple

from src.config import (
    RAG_HISTORY_MESSAGE_LIMIT,
    RAG_MAX_CONTEXT_CHUNKS,
    RAG_MAX_CONTEXT_TOKENS,
    RAG_MAX_DISTANCE,
    RAG_MAX_QUESTION_LENGTH,
)
from src.generation.citations import assign_source_ids, validate_citations
from src.generation.prompts import (
    FALLBACK_ANSWER,
    SYSTEM_INSTRUCTIONS,
    build_user_prompt,
)
from src.generation.provider import AnswerGenerationError, AnswerProvider
from src.ingestion.chunker import count_tokens
from src.models import RetrievalResult, SourceCitation, TutorAnswer
from src.retrieval.embeddings import EmbeddingError
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

# History is a list of (role, content) pairs; role is "user" or "assistant".
History = List[Tuple[str, str]]


class TutorError(Exception):
    """Safe, user-facing tutor failure.

    Carries a safe ``category`` propagated from the underlying cause (retrieval
    or generation error) for accurate, non-sensitive reporting.
    """

    def __init__(self, message: str, category: str = "other") -> None:
        super().__init__(message)
        self.category = category


def bound_history(history: Optional[History], limit: int) -> History:
    """Keep only the most recent ``limit`` messages (never unbounded)."""
    if not history or limit <= 0:
        return []
    return list(history)[-limit:]


class TutorService:
    """Coordinates retrieval, grounding, generation, and citation validation."""

    def __init__(
        self,
        retriever: RetrievalService,
        answer_provider: AnswerProvider,
        *,
        max_context_chunks: int = RAG_MAX_CONTEXT_CHUNKS,
        max_context_tokens: int = RAG_MAX_CONTEXT_TOKENS,
        max_distance: float = RAG_MAX_DISTANCE,
        max_question_length: int = RAG_MAX_QUESTION_LENGTH,
        history_message_limit: int = RAG_HISTORY_MESSAGE_LIMIT,
    ) -> None:
        self._retriever = retriever
        self._provider = answer_provider
        self._max_chunks = max_context_chunks
        self._max_tokens = max_context_tokens
        self._max_distance = max_distance
        self._max_question_length = max_question_length
        self._history_limit = history_message_limit

    def answer(
        self,
        question: str,
        document_id: Optional[str] = None,
        history: Optional[History] = None,
    ) -> TutorAnswer:
        started = time.perf_counter()
        question = (question or "").strip()

        # 1. Validate the question.
        if not question:
            raise TutorError("Please enter a question.")
        if len(question) > self._max_question_length:
            raise TutorError(
                f"Your question is too long (max {self._max_question_length} "
                "characters). Please shorten it."
            )

        # 2. Retrieve — always, even for follow-ups.
        t_ret = time.perf_counter()
        try:
            results = self._retriever.search(
                question, top_k=self._max_chunks, document_id=document_id
            )
        except (EmbeddingError, VectorStoreError) as exc:
            raise TutorError(str(exc), category=getattr(exc, "category", "other")) from exc
        retrieval_seconds = time.perf_counter() - t_ret

        # 3-6. Filter empties, apply the distance gate, then the token budget.
        kept = [
            r
            for r in results
            if r.text.strip()
            and (r.distance is None or r.distance <= self._max_distance)
        ]
        kept = self._apply_token_budget(kept)
        sources = assign_source_ids(kept)

        # 7. No acceptable evidence -> abstain locally, never call the model.
        if not sources:
            return self._abstain(question, started, retrieval_seconds, sources=[])

        # 8. Build prompt (prior turns are context only) and generate.
        bounded = bound_history(history, self._history_limit)
        user_prompt = build_user_prompt(question, sources, history=bounded)

        t_gen = time.perf_counter()
        try:
            generated = self._provider.generate(SYSTEM_INSTRUCTIONS, user_prompt)
        except AnswerGenerationError as exc:
            raise TutorError(str(exc), category=getattr(exc, "category", "other")) from exc
        generation_seconds = time.perf_counter() - t_gen

        raw_text = (generated.text or "").strip()

        # Model produced nothing usable -> treat as abstention, not a fake answer.
        if not raw_text:
            ans = self._abstain(question, started, retrieval_seconds, sources)
            ans.warnings.append("The model returned an empty response.")
            ans.generation_seconds = generation_seconds
            return ans

        # 9. Validate citations against the trusted, supplied source IDs.
        cleaned_text, cited, warnings = validate_citations(raw_text, sources)

        abstained = cleaned_text.strip() == FALLBACK_ANSWER
        if not abstained and not cited:
            warnings.append("Answer contained no valid citations to curriculum sources.")

        return TutorAnswer(
            answer_id=uuid.uuid4().hex,
            question=question,
            answer_text=cleaned_text,
            citations=cited,
            retrieved_sources=sources,
            abstained=abstained,
            model=self._provider.model_name,
            retrieval_seconds=round(retrieval_seconds, 3),
            generation_seconds=round(generation_seconds, 3),
            latency_seconds=round(time.perf_counter() - started, 3),
            usage=generated.usage,
            warnings=warnings,
        )

    def _apply_token_budget(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Keep results in order until the context token budget is reached.

        Always keeps at least the first result so a single long chunk can still
        ground an answer.
        """
        kept = []
        total = 0
        for r in results:
            tokens = count_tokens(r.text)
            if kept and total + tokens > self._max_tokens:
                break
            kept.append(r)
            total += tokens
        return kept

    def _abstain(
        self,
        question: str,
        started: float,
        retrieval_seconds: float,
        sources: List[SourceCitation],
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_id=uuid.uuid4().hex,
            question=question,
            answer_text=FALLBACK_ANSWER,
            citations=[],
            retrieved_sources=sources,
            abstained=True,
            model=self._provider.model_name,
            retrieval_seconds=round(retrieval_seconds, 3),
            generation_seconds=0.0,
            latency_seconds=round(time.perf_counter() - started, 3),
            usage=None,
            warnings=[],
        )
