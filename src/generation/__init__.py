"""Answer generation: grounded, cited tutor responses (Milestone 3).

All OpenAI access is isolated behind an injectable ``AnswerProvider`` so tests
run fully offline. Displayed citations are built from trusted retrieval
metadata, never from model-generated text.
"""

from src.generation.citations import (
    assign_source_ids,
    extract_citation_ids,
    validate_citations,
)
from src.generation.prompts import (
    FALLBACK_ANSWER,
    SYSTEM_INSTRUCTIONS,
    build_user_prompt,
)
from src.generation.provider import (
    AnswerGenerationError,
    AnswerProvider,
    FakeAnswerProvider,
    GeneratedAnswer,
    OpenAIAnswerProvider,
)
from src.generation.tutor import TutorError, TutorService, bound_history

__all__ = [
    "assign_source_ids",
    "extract_citation_ids",
    "validate_citations",
    "FALLBACK_ANSWER",
    "SYSTEM_INSTRUCTIONS",
    "build_user_prompt",
    "AnswerGenerationError",
    "AnswerProvider",
    "FakeAnswerProvider",
    "GeneratedAnswer",
    "OpenAIAnswerProvider",
    "TutorError",
    "TutorService",
    "bound_history",
]
