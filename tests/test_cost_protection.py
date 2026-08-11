"""Cost-protection tests: per-session question cap and access-code gate."""

from __future__ import annotations

from src import config


# --- session question limit ---
def test_question_limit_reached(monkeypatch):
    monkeypatch.setattr(config, "MAX_QUESTIONS_PER_SESSION", 3)
    assert config.question_limit_reached(0) is False
    assert config.question_limit_reached(2) is False
    assert config.question_limit_reached(3) is True
    assert config.question_limit_reached(4) is True


def test_question_limit_zero_means_unlimited(monkeypatch):
    monkeypatch.setattr(config, "MAX_QUESTIONS_PER_SESSION", 0)
    assert config.question_limit_reached(0) is False
    assert config.question_limit_reached(9999) is False


# --- access gate ---
def test_access_not_required_when_unset(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "")
    assert config.access_required() is False
    # Local dev stays open: any candidate (even empty) passes.
    assert config.verify_access_code("") is True
    assert config.verify_access_code("whatever") is True


def test_access_required_and_verified(monkeypatch):
    monkeypatch.setattr(config, "APP_ACCESS_CODE", "s3cret-code")
    assert config.access_required() is True
    assert config.verify_access_code("s3cret-code") is True
    assert config.verify_access_code(" s3cret-code ") is True  # trimmed
    assert config.verify_access_code("wrong") is False
    assert config.verify_access_code("") is False
    assert config.verify_access_code(None) is False  # type: ignore[arg-type]
