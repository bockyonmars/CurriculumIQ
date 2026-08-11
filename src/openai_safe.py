"""Classify OpenAI SDK exceptions into safe, non-sensitive category codes.

Used to label failures (auth / quota / rate_limit / timeout / model_unavailable
/ other) without exposing messages, keys, or stack traces to users or reports.
"""

from __future__ import annotations


def classify_openai_error(exc: BaseException) -> str:
    """Return a safe category string for an OpenAI/wrapped error."""
    name = type(exc).__name__.lower()
    code = str(getattr(exc, "code", "") or "")
    etype = str(getattr(exc, "type", "") or "")
    blob = f"{name} {code} {etype}".lower()

    if "authentication" in name or "permission" in name or "auth" in blob:
        return "auth"
    if "insufficient_quota" in blob or "credit" in blob or "quota" in blob:
        return "quota"
    if "ratelimit" in name or "rate_limit" in blob:
        return "rate_limit"
    if "timeout" in blob or "connection" in name or "apiconnection" in name:
        return "timeout"
    if "notfound" in name or "model" in blob:
        return "model_unavailable"
    return "other"


def category_from_chain(exc: BaseException) -> str:
    """Best safe category for ``exc``, preferring an attached ``category`` and
    walking the ``__cause__`` chain (so a wrapped app error still reports the
    real underlying reason)."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        attached = getattr(cur, "category", None)
        if attached:
            return str(attached)
        cat = classify_openai_error(cur)
        if cat != "other":
            return cat
        cur = cur.__cause__
    return "other"
