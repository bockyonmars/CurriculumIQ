"""Deterministic source-ID assignment and citation validation.

Source IDs (S1, S2, …) are assigned in Python BEFORE generation and mapped to
trusted retrieval metadata. After generation, only IDs that were actually
supplied are accepted; unknown markers are stripped and warned about. Filenames
and page numbers shown to the user always come from these trusted records —
never from model output.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.models import RetrievalResult, SourceCitation

# Matches citation markers like [S1] or [S12].
_CITATION_RE = re.compile(r"\[(S\d+)\]")


def assign_source_ids(results: List[RetrievalResult]) -> List[SourceCitation]:
    """Turn retrieval results into S-numbered citations, in result order."""
    citations: List[SourceCitation] = []
    for i, r in enumerate(results, start=1):
        citations.append(
            SourceCitation(
                source_id=f"S{i}",
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                page_number=r.page_number,
                passage=r.text,
                distance=r.distance,
                rank=r.rank,
            )
        )
    return citations


def extract_citation_ids(text: str) -> List[str]:
    """All citation IDs appearing in ``text``, in first-appearance order."""
    seen: List[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def validate_citations(
    answer_text: str, sources: List[SourceCitation]
) -> Tuple[str, List[SourceCitation], List[str]]:
    """Validate markers in ``answer_text`` against the supplied ``sources``.

    Returns ``(cleaned_text, cited_sources, warnings)``:
    - unknown ``[Sx]`` markers are removed from the text and warned about;
    - ``cited_sources`` are the trusted records for the valid IDs, in
      first-appearance order.
    """
    by_id = {s.source_id: s for s in sources}
    used_ids = extract_citation_ids(answer_text)

    valid_ids = [cid for cid in used_ids if cid in by_id]
    unknown_ids = [cid for cid in used_ids if cid not in by_id]

    cleaned = answer_text
    warnings: List[str] = []
    if unknown_ids:
        warnings.append(
            "Removed unknown citation marker(s): " + ", ".join(unknown_ids)
        )
        for cid in unknown_ids:
            # Remove the invalid marker; tidy any doubled spaces it leaves.
            cleaned = cleaned.replace(f"[{cid}]", "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    cited = [by_id[cid] for cid in valid_ids]
    return cleaned, cited, warnings
