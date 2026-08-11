"""Deterministic evaluation harness for CurriculumIQ (Milestone 4).

Measures retrieval and grounded-answer quality against a version-controlled
dataset of verified questions. No LLM is used as a judge — scoring is based on
expected pages, keywords, citations, and abstention labels. Offline mode uses
deterministic fakes; live mode is opt-in.
"""

from src.evaluation.schema import (
    DEFAULT_THRESHOLDS,
    EvalDataset,
    EvalItem,
    Thresholds,
    load_dataset,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "EvalDataset",
    "EvalItem",
    "Thresholds",
    "load_dataset",
]
