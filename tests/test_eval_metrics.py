"""Metric calculations: coverage, per-item scoring, aggregation, verdict."""

from __future__ import annotations

from src.evaluation.metrics import (
    aggregate,
    evaluate_item,
    keyword_coverage,
    page_coverage,
)
from src.evaluation.schema import DEFAULT_THRESHOLDS, EvalItem
from src.models import SourceCitation, TutorAnswer


def _answer(text, cited_pages, retrieved_pages, abstained=False):
    retrieved = [
        SourceCitation(source_id=f"S{i+1}", chunk_id=f"c{i}", document_id="doc_x",
                       filename="intro_to_algebra.pdf", page_number=pg, passage="p", rank=i + 1)
        for i, pg in enumerate(retrieved_pages)
    ]
    cited = [s for s in retrieved if s.page_number in cited_pages]
    return TutorAnswer(
        answer_id="a", question="q", answer_text=text, citations=cited,
        retrieved_sources=retrieved, abstained=abstained, model="fake",
        retrieval_seconds=0.01, generation_seconds=0.02, latency_seconds=0.03,
    )


def _item(**kw):
    base = dict(id="f", question="q?", answerable=True, category="factual",
                expected_document="intro_to_algebra.pdf", expected_pages=[2],
                expected_keywords=["slope", "intercept"])
    base.update(kw)
    return EvalItem(**base)


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("The SLOPE and the Intercept.", ["slope", "intercept"]) == 1.0
    assert keyword_coverage("only slope here", ["slope", "intercept"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0


def test_page_coverage():
    assert page_coverage([2], [2, 4]) == 1.0
    assert page_coverage([2, 4], [2]) == 0.5
    assert page_coverage([2, 4], [1, 3]) == 0.0


def test_supported_question_scoring_pass():
    item = _item()
    ans = _answer("The slope and intercept [S1].", cited_pages=[2], retrieved_pages=[2, 3])
    res = evaluate_item(item, [2, 3], ans)
    assert res.retrieval_hit and res.answered
    assert res.citation_valid and res.citation_page_hit
    assert res.keyword_coverage == 1.0
    assert res.passed and res.reason == ""


def test_supported_question_wrong_page_fails():
    item = _item(expected_pages=[2])
    ans = _answer("Something [S1].", cited_pages=[3], retrieved_pages=[3, 5])
    res = evaluate_item(item, [3, 5], ans)
    assert not res.retrieval_hit
    assert not res.passed
    assert "expected page not retrieved" in res.reason


def test_supported_question_abstention_is_failure():
    item = _item()
    ans = _answer("I could not find enough information in the available curriculum materials.",
                  cited_pages=[], retrieved_pages=[2], abstained=True)
    res = evaluate_item(item, [2], ans)
    assert res.abstained and not res.passed
    assert "abstained" in res.reason


def test_unsupported_scoring():
    item = EvalItem(id="u", question="q?", answerable=False, category="unsupported")
    good = _answer("fallback", cited_pages=[], retrieved_pages=[], abstained=True)
    bad = _answer("here is an answer [S1]", cited_pages=[2], retrieved_pages=[2], abstained=False)
    assert evaluate_item(item, [], good).passed is True
    r = evaluate_item(item, [], bad)
    assert r.passed is False and "unsupported" in r.reason


def test_error_item_recorded():
    item = _item()
    res = evaluate_item(item, [], None, error="quota")
    assert res.error == "quota" and not res.passed


def test_aggregate_verdict_pass_and_thresholds():
    item = _item()
    results = [evaluate_item(item, [2, 3],
               _answer("slope intercept [S1]", [2], [2, 3])) for _ in range(4)]
    unsupported = EvalItem(id="u", question="q?", answerable=False, category="unsupported")
    results.append(evaluate_item(unsupported, [],
                   _answer("fallback", [], [], abstained=True)))
    m = aggregate(results, DEFAULT_THRESHOLDS)
    assert m.retrieval_hit_rate == 1.0
    assert m.abstention_accuracy == 1.0
    assert m.item_pass_rate == 1.0
    assert m.verdict == "PASS"
    assert all(m.thresholds_met.values())


def test_aggregate_counts_errors():
    item = _item()
    results = [
        evaluate_item(item, [2], _answer("slope intercept [S1]", [2], [2])),
        evaluate_item(item, [], None, error="quota"),
        evaluate_item(item, [], None, error="auth"),
    ]
    m = aggregate(results, DEFAULT_THRESHOLDS)
    assert m.error_count == 2
    assert m.error_categories == {"quota": 1, "auth": 1}
