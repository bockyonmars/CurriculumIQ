"""Pure session helpers for the chat UI (Streamlit-independent, so testable).

The app stores conversation history and per-answer feedback in Streamlit session
state; these functions hold the bounded-history and feedback logic so it can be
unit-tested without a running Streamlit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Re-exported for convenience; the canonical implementation lives in tutor.py.
from src.generation.tutor import bound_history

History = List[Tuple[str, str]]

__all__ = ["bound_history", "record_feedback", "should_reset_chat"]


def record_feedback(
    store: Dict[str, dict],
    answer_id: str,
    helpful: bool,
    reason: Optional[str] = None,
) -> Dict[str, dict]:
    """Record thumbs feedback for one answer (in-memory only). Returns ``store``."""
    store[answer_id] = {"helpful": helpful, "reason": (reason or "").strip()}
    return store


def should_reset_chat(previous_document_id: Optional[str], new_document_id: Optional[str]) -> bool:
    """True when switching to a different document should clear stale chat state."""
    if new_document_id is None:
        return False
    return previous_document_id is not None and previous_document_id != new_document_id
