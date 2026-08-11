"""Deterministic scoring for evaluation. No LLM judge — expected pages,
keywords, citations, and abstention labels only.

All functions are pure so they can be unit-tested without any retrieval or API.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.evaluation.schema import EvalItem, Thresholds
from src.models import TutorAnswer


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def keyword_coverage(answer_text: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present (case/space-insensitive substring)."""
    if not keywords:
        return 1.0
    hay = _normalize(answer_text)
    hits = sum(1 for kw in keywords if _normalize(kw) in hay)
    return hits / len(keywords)


def page_coverage(expected_pages: List[int], retrieved_pages: List[int]) -> float:
    """Fraction of expected pages that appear in the retrieved pages."""
    if not expected_pages:
        return 1.0
    found = sum(1 for p in expected_pages if p in retrieved_pages)
    return found / len(expected_pages)


class ItemResult(BaseModel):
    """Per-question scored result."""

    id: str
    category: str
    answerable: bool
    error: Optional[str] = None

    # retrieval (from raw top-K search)
    retrieval_hit: bool = False
    page_coverage: float = 0.0

    # answer (from the tutor pipeline)
    abstained: bool = False
    answered: bool = False
    citation_valid: bool = False
    citation_page_hit: bool = False
    keyword_coverage: float = 0.0

    # latency
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    latency_seconds: float = 0.0

    passed: bool = False
    reason: str = ""


def evaluate_item(
    item: EvalItem,
    retrieved_pages: List[int],
    answer: Optional[TutorAnswer],
    error: Optional[str] = None,
) -> ItemResult:
    """Score one item from raw retrieved pages and the tutor's answer."""
    res = ItemResult(id=item.id, category=item.category, answerable=item.answerable)

    if error is not None:
        res.error = error
        res.reason = f"error: {error}"
        return res

    if answer is not None:
        res.abstained = answer.abstained
        res.answered = not answer.abstained
        res.retrieval_seconds = answer.retrieval_seconds
        res.generation_seconds = answer.generation_seconds
        res.latency_seconds = answer.latency_seconds
        cited_ids = {c.source_id for c in answer.citations}
        retrieved_ids = {s.source_id for s in answer.retrieved_sources}
        cited_pages = [c.page_number for c in answer.citations]

    if item.answerable:
        res.retrieval_hit = any(p in retrieved_pages for p in item.expected_pages)
        res.page_coverage = page_coverage(item.expected_pages, retrieved_pages)
        if answer is not None and res.answered:
            res.citation_valid = bool(cited_ids) and cited_ids.issubset(retrieved_ids)
            res.citation_page_hit = any(p in item.expected_pages for p in cited_pages)
            res.keyword_coverage = keyword_coverage(answer.answer_text, item.expected_keywords)

        # Per-item pass criteria for answerable questions.
        reasons = []
        if not res.retrieval_hit:
            reasons.append("expected page not retrieved")
        if answer is not None and res.abstained:
            reasons.append("abstained on an answerable question")
        if res.answered and not res.citation_valid:
            reasons.append("answer had no valid citation")
        if res.answered and not res.citation_page_hit:
            reasons.append("no citation on an expected page")
        res.passed = not reasons
        res.reason = "; ".join(reasons)
    else:
        # Unsupported: must abstain.
        if answer is not None:
            res.passed = res.abstained
            res.reason = "" if res.abstained else "answered an unsupported question"
    return res


class AggregateMetrics(BaseModel):
    """Dataset-level metrics and the overall verdict."""

    n_items: int
    n_answerable: int
    n_unsupported: int

    retrieval_hit_rate: float = 0.0
    expected_page_accuracy: float = 0.0
    citation_validity: float = 0.0
    citation_page_accuracy: float = 0.0
    abstention_accuracy: float = 0.0
    keyword_coverage: float = 0.0
    item_pass_rate: float = 0.0

    mean_retrieval_seconds: float = 0.0
    mean_generation_seconds: float = 0.0
    mean_latency_seconds: float = 0.0

    error_count: int = 0
    error_categories: dict = Field(default_factory=dict)

    failed_ids: List[str] = Field(default_factory=list)
    verdict: str = "FAIL"
    thresholds_met: dict = Field(default_factory=dict)


def _mean(xs: List[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(results: List[ItemResult], thresholds: Thresholds) -> AggregateMetrics:
    """Roll up per-item results into dataset metrics and a PASS/PARTIAL/FAIL verdict."""
    answerable = [r for r in results if r.answerable and r.error is None]
    unsupported = [r for r in results if not r.answerable and r.error is None]
    answered = [r for r in answerable if r.answered]

    error_results = [r for r in results if r.error is not None]
    error_cats: dict = {}
    for r in error_results:
        key = (r.error or "unknown").split(":")[0]
        error_cats[key] = error_cats.get(key, 0) + 1

    metrics = AggregateMetrics(
        n_items=len(results),
        n_answerable=len([r for r in results if r.answerable]),
        n_unsupported=len([r for r in results if not r.answerable]),
        retrieval_hit_rate=_mean([1.0 if r.retrieval_hit else 0.0 for r in answerable]),
        expected_page_accuracy=_mean([r.page_coverage for r in answerable]),
        citation_validity=_mean([1.0 if r.citation_valid else 0.0 for r in answered]),
        citation_page_accuracy=_mean([1.0 if r.citation_page_hit else 0.0 for r in answered]),
        abstention_accuracy=_mean([1.0 if r.abstained else 0.0 for r in unsupported]),
        keyword_coverage=_mean([r.keyword_coverage for r in answered]),
        item_pass_rate=_mean([1.0 if r.passed else 0.0 for r in results]),
        mean_retrieval_seconds=_mean([r.retrieval_seconds for r in results if r.error is None]),
        mean_generation_seconds=_mean([r.generation_seconds for r in results if r.error is None]),
        mean_latency_seconds=_mean([r.latency_seconds for r in results if r.error is None]),
        error_count=len(error_results),
        error_categories=error_cats,
        failed_ids=[r.id for r in results if not r.passed],
    )

    checks = {
        "retrieval_hit_rate": metrics.retrieval_hit_rate >= thresholds.retrieval_hit_rate,
        "expected_page_accuracy": metrics.expected_page_accuracy >= thresholds.expected_page_accuracy,
        "citation_validity": metrics.citation_validity >= thresholds.citation_validity,
        "citation_page_accuracy": metrics.citation_page_accuracy >= thresholds.citation_page_accuracy,
        "abstention_accuracy": metrics.abstention_accuracy >= thresholds.abstention_accuracy,
        "keyword_coverage": metrics.keyword_coverage >= thresholds.keyword_coverage,
        "item_pass_rate": metrics.item_pass_rate >= thresholds.item_pass_rate,
    }
    metrics.thresholds_met = checks

    passed_checks = sum(1 for v in checks.values() if v)
    if metrics.error_count == 0 and passed_checks == len(checks):
        metrics.verdict = "PASS"
    elif passed_checks >= len(checks) - 1 and metrics.error_count == 0:
        metrics.verdict = "PARTIAL"
    elif passed_checks >= len(checks) // 2:
        metrics.verdict = "PARTIAL"
    else:
        metrics.verdict = "FAIL"
    return metrics
