"""Evaluation dataset: loads, validates, ground-truth verified, rejects junk."""

from __future__ import annotations

import pathlib

import pytest

from src.evaluation.schema import EvalDataset, EvalItem, load_dataset
from src.ingestion.extractor import extract_document

SAMPLE = pathlib.Path("data/sample_documents/intro_to_algebra.pdf")


def test_dataset_loads_and_has_required_breakdown():
    ds = load_dataset()
    assert len(ds.items) >= 20
    assert len(ds.answerable) >= 16
    assert len(ds.unsupported) >= 4
    counts = ds.category_counts()
    assert counts.get("paraphrase", 0) >= 4
    assert counts.get("multi_chunk", 0) >= 2
    assert counts.get("unsupported", 0) >= 4


def test_ground_truth_keywords_are_on_expected_pages():
    """Every expected keyword must actually appear on an expected page."""
    ds = load_dataset()
    ex = extract_document(SAMPLE.read_bytes(), SAMPLE.name)
    page_text = {p.page_number: p.text.lower() for p in ex.pages}
    misses = []
    for item in ds.answerable:
        assert item.expected_document == SAMPLE.name
        for kw in item.expected_keywords:
            if not any(kw.lower() in page_text.get(p, "") for p in item.expected_pages):
                misses.append((item.id, kw))
    assert misses == [], f"unverified ground truth: {misses}"


def test_multi_chunk_items_span_multiple_pages():
    ds = load_dataset()
    multi = [i for i in ds.items if i.category == "multi_chunk"]
    assert len(multi) >= 2
    for i in multi:
        assert len(i.expected_pages) >= 2


def test_answerable_item_requires_pages_and_keywords():
    with pytest.raises(ValueError):
        EvalItem(id="x", question="q?", answerable=True, category="factual",
                 expected_document="d.pdf", expected_pages=[], expected_keywords=["k"])
    with pytest.raises(ValueError):
        EvalItem(id="x", question="q?", answerable=True, category="factual",
                 expected_document="d.pdf", expected_pages=[1], expected_keywords=[])


def test_unsupported_item_must_not_carry_ground_truth():
    with pytest.raises(ValueError):
        EvalItem(id="u", question="q?", answerable=False, category="unsupported",
                 expected_pages=[1])


def test_duplicate_ids_rejected():
    good = EvalItem(id="a", question="q?", answerable=False, category="unsupported")
    dup = EvalItem(id="a", question="q2?", answerable=False, category="unsupported")
    with pytest.raises(ValueError):
        EvalDataset(items=[good, dup])
