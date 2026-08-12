"""CurriculumIQ Python AI API — a thin FastAPI adapter over existing services.

It reuses the domain logic under ``src/`` (validation, extraction, chunking,
indexing, retrieval, grounded generation, citations, abstention) without
duplicating any of it. Errors are converted to safe HTTP responses; request
timing is logged without ever logging document contents.
"""

from __future__ import annotations

import logging
import time

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src import config
from src.generation.provider import AnswerGenerationError, AnswerProvider
from src.generation.tutor import TutorError, TutorService
from src.ingestion.chunker import chunk_document
from src.ingestion.extractor import ExtractionError, extract_document
from src.ingestion.validator import ValidationError, validate_pdf
from src.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from src.retrieval.indexer import IndexingError, IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore, VectorStoreError
from python_service import deps
from python_service.schemas import (
    Citation,
    DocumentResponse,
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
)

logger = logging.getLogger("curriculumiq.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CurriculumIQ Python AI API", version="1.0.0")

MAX_UPLOAD_BYTES = config.MAX_FILE_SIZE_BYTES


@app.middleware("http")
async def _timing(request: Request, call_next):
    """Log method, path, status, and duration — never the request body."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def _unhandled(_request: Request, _exc: Exception) -> JSONResponse:
    # Last-resort guard: never leak a stack trace to the client.
    logger.exception("Unhandled error in API")
    return JSONResponse(status_code=500, content={"detail": "Internal service error."})


def _skipped_pages(doc) -> list:
    return [p.page_number for p in doc.pages
            if len(p.text.strip()) < config.MIN_CHARS_PER_PAGE]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/documents", response_model=DocumentResponse)
async def create_document(
    file: UploadFile = File(...),
    store: VectorStore = Depends(deps.get_store),
    embedder: EmbeddingProvider = Depends(deps.get_embedder),
) -> DocumentResponse:
    filename = file.filename or "upload.pdf"
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=(
            f"File is too large. The maximum size is {config.MAX_FILE_SIZE_MB} MB."))

    try:
        validate_pdf(file_bytes, filename)
        doc = extract_document(file_bytes, filename)
        chunks, _warnings = chunk_document(doc)
    except (ValidationError, ExtractionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=422, detail=(
            "No readable text was found. This looks like a scanned or image-only "
            "PDF, which isn't supported."))

    try:
        already = store.has_document(doc.document_id)
        result = IndexingService(store, embedder).index_document(doc, reindex=already)
    except (IndexingError, EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DocumentResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        pages=doc.page_count,
        chunks=result.chunks_created,
        skipped_pages=_skipped_pages(doc),
        status="ready",
    )


@app.post("/api/questions", response_model=QuestionResponse)
def ask_question(
    payload: QuestionRequest,
    store: VectorStore = Depends(deps.get_store),
    embedder: EmbeddingProvider = Depends(deps.get_embedder),
    answerer: AnswerProvider = Depends(deps.get_answerer),
) -> QuestionResponse:
    tutor = TutorService(RetrievalService(store, embedder), answerer)
    try:
        answer = tutor.answer(payload.question, document_id=payload.document_id)
    except TutorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (EmbeddingError, AnswerGenerationError, VectorStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QuestionResponse(
        answer=answer.answer_text,
        abstained=answer.abstained,
        citations=[
            Citation(source_id=c.source_id, filename=c.filename,
                     page=c.page_number, passage=c.passage)
            for c in answer.citations
        ],
    )
