"""UI state tests via Streamlit AppTest — no real OpenAI calls, no network.

Covers the redesigned onboarding/tutor flow: access gate, empty state,
ready/tutor state, sources, abstention, question limit, aggregated warnings,
and developer-detail gating.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.models import SourceCitation, TutorAnswer
from tests.conftest import make_extracted_document

APP = "app.py"


def _texts(at) -> str:
    out = []
    for name in ("title", "header", "subheader", "markdown", "caption", "text",
                 "info", "warning", "error", "success"):
        for e in getattr(at, name, []):
            v = getattr(e, "value", None)
            if v is not None:
                out.append(str(v))
    for b in at.button:
        out.append(str(b.label))
    for e in at.get("expander"):
        out.append(str(e.label))
    for ti in at.text_input:
        out.append(str(getattr(ti, "label", "")))
    for ci in at.chat_input:
        out.append(str(getattr(ci, "placeholder", "")))
    return "\n".join(out)


def _open_gate(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENAI_CHAT_MODEL", "gpt-test")
    monkeypatch.setattr(config, "SHOW_DEVELOPER_DETAILS", False)


def _answer(citations, abstained=False):
    return TutorAnswer(
        answer_id="a1", question="q",
        answer_text=("I could not find enough information in the available curriculum materials."
                     if abstained else "Here is the answer [S1]."),
        citations=citations, retrieved_sources=citations, abstained=abstained,
        model="gpt-test", retrieval_seconds=0.11, generation_seconds=0.22, latency_seconds=0.33,
    )


def _cite(sid, page):
    return SourceCitation(source_id=sid, chunk_id=f"chunk_{sid}", document_id="doc_x",
                          filename="intro_to_algebra.pdf", page_number=page,
                          passage=f"passage for {sid}", distance=0.3, rank=int(sid[1:]))


def _ready(monkeypatch, *, chat_history=None, questions_asked=0, indexed=True,
           dev=False, pages=None):
    _open_gate(monkeypatch)
    monkeypatch.setattr(config, "SHOW_DEVELOPER_DETAILS", dev)
    doc = make_extracted_document(pages or ["Algebra intro", "Linear equations", "Quadratic"])
    at = AppTest.from_file(APP)
    at.session_state["document"] = doc
    at.session_state["chunks"] = [0] * doc.page_count
    at.session_state["indexed"] = indexed
    at.session_state["questions_asked"] = questions_asked
    if chat_history is not None:
        at.session_state["chat_history"] = chat_history
    at.run(timeout=30)
    return at


# --- access gate ------------------------------------------------------------
def test_access_gate_shown_and_secret_safe(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "super-secret-code")
    at = AppTest.from_file(APP)
    at.run(timeout=30)
    text = _texts(at)
    assert "Demo access code" in text
    assert "Open demo" in text
    assert "source-grounded AI tutor" in text
    assert "Private portfolio demo" in text
    # The gate blocks the product and never reveals the code.
    assert "Choose your curriculum" not in text
    assert "super-secret-code" not in text


def test_incorrect_access_code_shows_inline_error(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "the-real-code")
    at = AppTest.from_file(APP)
    at.run(timeout=30)
    at.text_input[0].set_value("wrong-code")
    at.button[0].click()
    at.run(timeout=30)
    # Still gated (product not shown) and a clear inline error is present.
    assert "Choose your curriculum" not in _texts(at)
    assert any("isn't correct" in str(e.value) for e in at.error)


def test_correct_access_code_opens_demo(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "the-real-code")
    at = AppTest.from_file(APP)
    at.run(timeout=30)
    at.text_input[0].set_value("the-real-code")
    at.button[0].click()
    at.run(timeout=30)
    assert at.session_state["access_granted"] is True
    assert "Choose your curriculum" in _texts(at)


# --- empty state ------------------------------------------------------------
def test_empty_state_hides_technical_sections(monkeypatch):
    _open_gate(monkeypatch)
    at = AppTest.from_file(APP)
    at.run(timeout=30)
    text = _texts(at)
    assert "Choose your curriculum" in text
    # Stage labels are present (direct, not "Step 1").
    for label in ("Choose PDF", "Prepare curriculum", "Ask questions"):
        assert label in text
    # No engineering surfaces in the empty state.
    for forbidden in ("Ask your curriculum", "Advanced tools", "Document details",
                      "Semantic Search", "Prepared sections", "Indexing", "Chunking summary"):
        assert forbidden not in text
    # Prepare is disabled until a PDF is chosen.
    prep = [b for b in at.button if b.label == "Prepare curriculum"][0]
    assert prep.disabled is True


def test_validation_failure_shows_error(monkeypatch):
    _open_gate(monkeypatch)
    at = AppTest.from_file(APP)
    at.session_state["error"] = "Only PDF files are supported."
    at.run(timeout=30)
    assert any("Only PDF files are supported." in str(e.value) for e in at.error)


# --- ready / tutor ----------------------------------------------------------
def test_ready_state_makes_tutor_primary(monkeypatch):
    at = _ready(monkeypatch)
    text = _texts(at)
    assert "Ask your curriculum" in text
    assert "Your curriculum is ready" in text
    # Three example-question buttons + a chat input.
    assert sum(1 for b in at.button if b.label in config_example_labels()) == 3
    assert len(at.chat_input) == 1


def config_example_labels():
    import app  # noqa
    return set(app.EXAMPLE_QUESTIONS)


def test_answer_with_one_source(monkeypatch):
    hist = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "answer": _answer([_cite("S1", 5)])}]
    at = _ready(monkeypatch, chat_history=hist)
    assert any(e.label == "Sources (1)" for e in at.get("expander"))


def test_answer_with_multiple_sources(monkeypatch):
    hist = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a",
             "answer": _answer([_cite("S1", 2), _cite("S2", 4)])}]
    at = _ready(monkeypatch, chat_history=hist)
    assert any(e.label == "Sources (2)" for e in at.get("expander"))


def test_abstained_answer_is_clear_and_sourceless(monkeypatch):
    hist = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "answer": _answer([], abstained=True)}]
    at = _ready(monkeypatch, chat_history=hist)
    text = _texts(at)
    assert "couldn't find enough" in text
    assert not any(str(e.label).startswith("Sources") for e in at.get("expander"))


def test_question_limit_reached(monkeypatch):
    at = _ready(monkeypatch, questions_asked=config.MAX_QUESTIONS_PER_SESSION)
    assert "Session limit reached" in _texts(at)
    assert at.chat_input[0].disabled is True


def test_developer_details_hidden_in_ready_state(monkeypatch):
    hist = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "answer": _answer([_cite("S1", 5)])}]
    at = _ready(monkeypatch, chat_history=hist, dev=False)
    text = _texts(at)
    assert "Developer details" not in text
    assert "distance" not in text          # cosine distance hidden
    assert "gpt-test" not in text          # model name hidden
    assert "Chunk ID" not in text          # raw chunk id hidden
    assert "⏱" not in text                 # latency hidden


def test_skipped_pages_warning_is_aggregated(monkeypatch):
    # 3 empty/near-empty pages among 6 → exactly ONE warning, not one per page.
    pages = ["Real algebra content", "", "Linear equations", "  ", "Quadratic", "x"]
    at = _ready(monkeypatch, pages=pages)
    assert len(at.warning) == 1
    assert "3 pages contained no readable text" in str(at.warning[0].value)
    assert any(e.label == "View skipped pages" for e in at.get("expander"))


def test_no_whats_next_anywhere(monkeypatch):
    _open_gate(monkeypatch)
    empty = AppTest.from_file(APP); empty.run(timeout=30)
    ready = _ready(monkeypatch)
    assert "What's next" not in _texts(empty)
    assert "What's next" not in _texts(ready)


def test_gateway_mode_renders_from_normalized_view(monkeypatch):
    # In gateway mode, the ready state is built from the gateway's JSON response
    # (no in-process ExtractedDocument, no local vector store call).
    _open_gate(monkeypatch)
    monkeypatch.setattr(config, "SERVICE_MODE", "gateway")
    at = AppTest.from_file(APP)
    at.session_state["gateway_doc"] = {
        "document_id": "doc_gw", "filename": "gateway_doc.pdf", "pages": 6,
        "chunks": 6, "skipped_pages": [], "status": "ready",
    }
    at.session_state["indexed"] = True
    at.run(timeout=30)
    text = _texts(at)
    assert "Your curriculum is ready" in text
    assert "gateway_doc.pdf" in text
    assert "Ask your curriculum" in text
    # Passage search is direct-mode only; it must not appear in gateway mode.
    assert "Advanced tools" not in text
