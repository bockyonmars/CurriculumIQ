"""Request/response schemas for the CurriculumIQ Python AI API.

These are the service's public contract. They deliberately expose only safe,
student-facing fields — never API keys, prompts, internal paths, distances,
chunk IDs, or stack traces.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from src import config


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "curriculumiq-python"


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages: int = Field(..., ge=0)
    chunks: int = Field(..., ge=0)
    skipped_pages: List[int] = Field(default_factory=list)
    status: str = "ready"


class QuestionRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=config.RAG_MAX_QUESTION_LENGTH)


class Citation(BaseModel):
    source_id: str
    filename: str
    page: int = Field(..., ge=1)
    passage: str


class QuestionResponse(BaseModel):
    answer: str
    abstained: bool = False
    citations: List[Citation] = Field(default_factory=list)
