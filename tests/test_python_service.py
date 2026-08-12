"""API tests for the FastAPI adapter — offline only (fake providers, temp store).

No live OpenAI calls: the real providers are replaced via dependency_overrides.
"""

from __future__ import annotations

import pytest

# The FastAPI adapter is an optional extra (installed in the python_service
# image / dev venv). Skip these API tests cleanly if FastAPI isn't present.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from python_service import deps  # noqa: E402
from python_service.main import app  # noqa: E402
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.generation.provider import FakeAnswerProvider
from tests.conftest import make_text_pdf


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"), "curriculumiq_test")


@pytest.fixture
def client(store):
    app.dependency_overrides[deps.get_store] = lambda: store
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[deps.get_answerer] = lambda: FakeAnswerProvider(
        response_text="Grounded answer [S1]."
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, text_pages, name="doc.pdf"):
    pdf = make_text_pdf(text_pages)
    return client.post("/api/documents", files={"file": (name, pdf, "application/pdf")})


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "curriculumiq-python"}


def test_create_document_indexes_and_returns_summary(client):
    r = _upload(client, [
        "Photosynthesis converts sunlight into chemical energy in plant leaves.",
        "The quadratic formula solves quadratic equations using coefficients.",
    ], name="science.pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "science.pdf"
    assert body["pages"] == 2
    assert body["chunks"] >= 1
    assert body["status"] == "ready"
    assert body["skipped_pages"] == []
    assert body["document_id"].startswith("doc_")


def test_non_pdf_rejected_safely(client):
    r = client.post("/api/documents",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    assert "PDF" in r.json()["detail"]
    assert "Traceback" not in r.text


def test_scanned_pdf_no_text_rejected(client):
    # A PDF with a blank page (no extractable text).
    import fitz
    doc = fitz.open(); doc.new_page()
    blank = doc.tobytes(); doc.close()
    r = client.post("/api/documents",
                    files={"file": ("scan.pdf", blank, "application/pdf")})
    assert r.status_code == 422
    assert "scanned" in r.json()["detail"].lower()


def test_ask_supported_question_returns_citation(client):
    up = _upload(client, [
        "Photosynthesis converts sunlight into chemical energy in plant leaves.",
    ], name="bio.pdf")
    doc_id = up.json()["document_id"]
    r = client.post("/api/questions", json={
        "document_id": doc_id,
        "question": "Photosynthesis converts sunlight into chemical energy",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstained"] is False
    assert len(body["citations"]) >= 1
    c = body["citations"][0]
    assert c["source_id"] == "S1"
    assert c["filename"] == "bio.pdf"
    assert c["page"] == 1
    # Safe payload only — no internal fields leak.
    assert "distance" not in c and "chunk_id" not in c and "document_id" not in c


def test_ask_unsupported_question_abstains(client):
    up = _upload(client, [
        "Photosynthesis converts sunlight into chemical energy in plant leaves.",
    ], name="bio.pdf")
    doc_id = up.json()["document_id"]
    r = client.post("/api/questions", json={
        "document_id": doc_id,
        "question": "Who won the 1998 football world cup final?",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_question_length_validation(client):
    r = client.post("/api/questions", json={
        "document_id": "doc_whatever",
        "question": "x" * 5000,
    })
    assert r.status_code == 422  # schema rejects over-long questions
