"""Evaluation dataset schema, loader, and pass thresholds.

Ground truth is authored and verified against the sample curriculum PDFs — see
``tests/test_eval_dataset.py`` which asserts every expected keyword appears on
its expected page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

# Categories are free-form labels but constrained to a known set for sanity.
ANSWERABLE_CATEGORIES = {"factual", "paraphrase", "multi_chunk"}
UNSUPPORTED_CATEGORY = "unsupported"


class EvalItem(BaseModel):
    """One evaluation question with verified ground truth."""

    id: str
    question: str
    answerable: bool
    category: str
    # For answerable items only (empty/None for unsupported):
    expected_document: Optional[str] = None
    expected_pages: List[int] = Field(default_factory=list)
    expected_keywords: List[str] = Field(default_factory=list)
    multi_chunk: bool = False
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "EvalItem":
        if not self.id or not self.question.strip():
            raise ValueError("Eval item needs a non-empty id and question.")
        if self.answerable:
            if self.category not in ANSWERABLE_CATEGORIES:
                raise ValueError(
                    f"Answerable item {self.id} has invalid category {self.category!r}."
                )
            if not self.expected_document:
                raise ValueError(f"Answerable item {self.id} needs expected_document.")
            if not self.expected_pages:
                raise ValueError(f"Answerable item {self.id} needs expected_pages.")
            if any(p < 1 for p in self.expected_pages):
                raise ValueError(f"Item {self.id} has a non-one-based page.")
            if not self.expected_keywords:
                raise ValueError(f"Answerable item {self.id} needs expected_keywords.")
        else:
            if self.category != UNSUPPORTED_CATEGORY:
                raise ValueError(
                    f"Unsupported item {self.id} must use category 'unsupported'."
                )
            if self.expected_pages or self.expected_keywords:
                raise ValueError(
                    f"Unsupported item {self.id} must not carry expected pages/keywords."
                )
        return self


class EvalDataset(BaseModel):
    """A collection of eval items with unique IDs."""

    items: List[EvalItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalDataset":
        ids = [it.id for it in self.items]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate eval item IDs: {sorted(dupes)}")
        return self

    # --- convenience breakdowns ---
    @property
    def answerable(self) -> List[EvalItem]:
        return [i for i in self.items if i.answerable]

    @property
    def unsupported(self) -> List[EvalItem]:
        return [i for i in self.items if not i.answerable]

    def category_counts(self) -> dict:
        counts: dict = {}
        for i in self.items:
            counts[i.category] = counts.get(i.category, 0) + 1
        return counts


class Thresholds(BaseModel):
    """Minimum metric values for an overall PASS. All in [0, 1]."""

    retrieval_hit_rate: float = 0.80
    expected_page_accuracy: float = 0.70
    citation_validity: float = 0.95
    citation_page_accuracy: float = 0.60
    abstention_accuracy: float = 0.90
    keyword_coverage: float = 0.55
    item_pass_rate: float = 0.75


DEFAULT_THRESHOLDS = Thresholds()

# Default dataset location (version-controlled).
DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "curriculum_eval.json"


def load_dataset(path: Optional[Path] = None) -> EvalDataset:
    """Load and validate the evaluation dataset. Raises on malformed data."""
    p = Path(path) if path else DATASET_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list of items.")
    return EvalDataset(items=[EvalItem(**item) for item in raw])
