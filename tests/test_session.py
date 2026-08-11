"""Session-helper tests: bounded history, feedback storage, new-doc reset."""

from __future__ import annotations

from src.generation.session import (
    bound_history,
    record_feedback,
    should_reset_chat,
)


def test_bounded_history():
    hist = [("user", str(i)) for i in range(8)]
    assert bound_history(hist, 3) == hist[-3:]


def test_record_feedback_stores_in_memory():
    store: dict = {}
    record_feedback(store, "ans1", helpful=True)
    record_feedback(store, "ans2", helpful=False, reason="  wrong page ")
    assert store["ans1"] == {"helpful": True, "reason": ""}
    assert store["ans2"] == {"helpful": False, "reason": "wrong page"}


def test_should_reset_chat_on_document_change():
    assert should_reset_chat("doc_a", "doc_b") is True
    assert should_reset_chat("doc_a", "doc_a") is False
    assert should_reset_chat(None, "doc_a") is False   # first document, nothing to reset
    assert should_reset_chat("doc_a", None) is False   # cleared document
