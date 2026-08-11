"""Citation tests: deterministic IDs, marker parsing, unknown-ID rejection."""

from __future__ import annotations

from src.generation.citations import (
    assign_source_ids,
    extract_citation_ids,
    validate_citations,
)
from src.models import RetrievalResult


def _results(n=2):
    return [
        RetrievalResult(
            rank=i,
            chunk_id=f"doc_x_p{i}_c0",
            document_id="doc_x",
            filename="math.pdf",
            page_number=i,
            text=f"Passage {i}",
            distance=0.1 * i,
        )
        for i in range(1, n + 1)
    ]


def test_source_ids_are_deterministic_and_ordered():
    sources = assign_source_ids(_results(3))
    assert [s.source_id for s in sources] == ["S1", "S2", "S3"]
    # IDs map to trusted metadata, one-based pages preserved.
    assert sources[0].page_number == 1
    assert sources[1].filename == "math.pdf"


def test_extract_citation_ids_first_appearance_order():
    assert extract_citation_ids("A [S2] then [S1] then [S2] again") == ["S2", "S1"]


def test_valid_citations_map_to_metadata():
    sources = assign_source_ids(_results(2))
    cleaned, cited, warnings = validate_citations("Variables are symbols [S1].", sources)
    assert cleaned == "Variables are symbols [S1]."
    assert [c.source_id for c in cited] == ["S1"]
    assert cited[0].page_number == 1
    assert warnings == []


def test_unknown_citation_rejected_and_warned():
    sources = assign_source_ids(_results(1))  # only S1 exists
    cleaned, cited, warnings = validate_citations("Claim [S1] and bad [S9].", sources)
    assert "[S9]" not in cleaned
    assert [c.source_id for c in cited] == ["S1"]
    assert any("S9" in w for w in warnings)


def test_answer_with_no_citation_yields_empty_cited():
    sources = assign_source_ids(_results(2))
    cleaned, cited, warnings = validate_citations("No markers here at all.", sources)
    assert cited == []
    assert cleaned == "No markers here at all."
