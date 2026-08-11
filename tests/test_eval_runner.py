"""Runner: deterministic offline execution, report generation, error mapping,
and live-mode being opt-in (off by default)."""

from __future__ import annotations

import json

import pytest

from src.evaluation import runner as runner_mod
from src.evaluation.runner import (
    EXIT_LIVE_BLOCKED,
    EXIT_LIVE_CONFIG_MISSING,
    EXIT_OK,
    EchoAnswerProvider,
    _safe_error_category,
    main,
    run_evaluation,
    write_reports,
)
from src.evaluation.schema import DEFAULT_THRESHOLDS, load_dataset
from src.retrieval.indexer import IndexingError


def test_offline_evaluation_is_deterministic_and_grounded():
    ds = load_dataset()
    m1, r1, meta1 = run_evaluation(ds, mode="offline", k=5, thresholds=DEFAULT_THRESHOLDS)
    m2, r2, meta2 = run_evaluation(ds, mode="offline", k=5, thresholds=DEFAULT_THRESHOLDS)

    # No network was used, and scoring repeats exactly (timing fields excluded).
    assert meta1["mode"] == "offline"
    assert meta1["chat_model"] == "offline-echo"
    timing = {"mean_retrieval_seconds", "mean_generation_seconds", "mean_latency_seconds"}
    d1 = {k: v for k, v in m1.model_dump().items() if k not in timing}
    d2 = {k: v for k, v in m2.model_dump().items() if k not in timing}
    assert d1 == d2
    # Per-item scoring (non-timing) is identical too.
    strip = lambda r: {k: v for k, v in r.model_dump().items()
                       if k not in {"retrieval_seconds", "generation_seconds", "latency_seconds"}}
    assert [strip(r) for r in r1] == [strip(r) for r in r2]

    assert m1.n_items == len(ds.items)
    assert m1.error_count == 0
    # Deterministic offline guarantees: every unsupported abstains; every
    # answerable expected page is retrievable in top-5.
    assert m1.abstention_accuracy == 1.0
    assert m1.retrieval_hit_rate == 1.0
    assert m1.verdict in {"PASS", "PARTIAL"}


def test_live_mode_is_off_by_default():
    ds = load_dataset()
    _, _, meta = run_evaluation(ds, mode="offline", k=5, thresholds=DEFAULT_THRESHOLDS)
    assert "skipped" in meta["live_validation"]
    assert meta["embedding_model"] == "fake-embedding"


def test_report_generation(tmp_path):
    ds = load_dataset()
    metrics, results, meta = run_evaluation(ds, mode="offline", k=5, thresholds=DEFAULT_THRESHOLDS)
    write_reports(tmp_path, metrics, results, meta, DEFAULT_THRESHOLDS)

    for name in ("latest.json", "latest.csv", "latest.md"):
        assert (tmp_path / name).exists()

    payload = json.loads((tmp_path / "latest.json").read_text())
    assert payload["metrics"]["verdict"] == metrics.verdict
    assert payload["run"]["mode"] == "offline"
    assert len(payload["items"]) == len(ds.items)

    md = (tmp_path / "latest.md").read_text()
    assert "Evaluation Report" in md and "Overall:" in md


def test_echo_provider_grounds_in_prompt_sources():
    prompt = '<SOURCES>\n<SOURCE id="S1" filename="d.pdf" page="2">slope intercept</SOURCE>\n</SOURCES>'
    out = EchoAnswerProvider().generate("instructions", prompt)
    assert "slope intercept" in out.text
    assert "[S1]" in out.text


def test_safe_error_category_mapping():
    class RateLimitError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class NotFoundError(Exception):
        pass

    assert _safe_error_category(RateLimitError()) == "rate_limit"
    assert _safe_error_category(AuthenticationError()) == "auth"
    assert _safe_error_category(APITimeoutError()) == "timeout"
    assert _safe_error_category(NotFoundError()) == "model_unavailable"
    assert _safe_error_category(ValueError("boom")) == "other"


class _FakeQuotaError(Exception):
    """Mimics openai.RateLimitError(code='credit_balance_exhausted')."""

    def __init__(self):
        super().__init__("You have no credits remaining.")
        self.code = "credit_balance_exhausted"
        self.status_code = 429


def test_credit_exhausted_live_produces_no_traceback(monkeypatch, tmp_path, capsys):
    """The exact reported failure: --live with credit_balance_exhausted must
    exit cleanly (code 3), print one safe line, no traceback, no reports."""
    # Make a setup failure surface as it does in production: an IndexingError
    # whose category was classified from the underlying quota error.
    def boom(mode):
        try:
            raise _FakeQuotaError()
        except _FakeQuotaError as cause:
            from src.openai_safe import classify_openai_error
            raise IndexingError("Failed to store chunks.",
                                category=classify_openai_error(cause)) from cause

    monkeypatch.setattr(runner_mod.config, "generation_enabled", lambda: True)
    monkeypatch.setattr(runner_mod, "build_pipeline", boom)

    code = main(["--live", "--out-dir", str(tmp_path)])
    out = capsys.readouterr()

    assert code == EXIT_LIVE_BLOCKED
    combined = out.out + out.err
    assert "Traceback" not in combined
    # No raw provider message / status leaked (the safe action text may mention
    # "billing credits" — that's fine; the provider's own wording must not appear).
    assert "no credits remaining" not in combined.lower()
    assert "429" not in combined
    assert "quota" in combined.lower()
    assert "BLOCKED" in combined
    # No reports written for a blocked run.
    assert not list(tmp_path.glob("latest.*"))


def test_blocked_live_does_not_overwrite_existing_reports(monkeypatch, tmp_path):
    # Seed an existing (prior good) report.
    (tmp_path / "latest.json").write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr(runner_mod.config, "generation_enabled", lambda: True)
    monkeypatch.setattr(runner_mod, "build_pipeline",
                        lambda mode: (_ for _ in ()).throw(IndexingError("x", category="quota")))
    code = main(["--live", "--out-dir", str(tmp_path)])
    assert code == EXIT_LIVE_BLOCKED
    assert (tmp_path / "latest.json").read_text() == '{"ok": true}'  # untouched


def test_live_config_missing_returns_documented_code(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_mod.config, "generation_enabled", lambda: False)
    code = main(["--live", "--out-dir", str(tmp_path)])
    assert code == EXIT_LIVE_CONFIG_MISSING


def test_offline_main_writes_reports_and_exits_ok(tmp_path):
    code = main(["--out-dir", str(tmp_path)])
    assert code == EXIT_OK
    assert (tmp_path / "latest.json").exists()


def test_default_out_dir_separates_modes():
    from src.evaluation.runner import default_out_dir
    assert default_out_dir("offline").name == "offline"
    assert default_out_dir("live").name == "live"
    assert default_out_dir("offline").parent == default_out_dir("live").parent


def test_offline_default_writes_only_to_offline_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "REPORTS_BASE", tmp_path)
    code = main([])  # offline, no --out-dir
    assert code == EXIT_OK
    assert (tmp_path / "offline" / "latest.json").exists()
    assert not (tmp_path / "live").exists()  # live untouched


def test_live_default_writes_only_to_live_subdir(tmp_path, monkeypatch):
    # Run the "live" code path with fakes so no API is called, and confirm it
    # writes only under live/ — proving offline and live reports are separated.
    from src.evaluation.runner import EchoAnswerProvider
    from src.generation.tutor import TutorService
    from src.ingestion.extractor import extract_document
    from src.retrieval.embeddings import FakeEmbeddingProvider
    from src.retrieval.indexer import IndexingService
    from src.retrieval.retriever import RetrievalService
    from src.retrieval.vector_store import VectorStore
    import pathlib, tempfile

    def fake_pipeline(mode):
        store = VectorStore(tempfile.mkdtemp(), "sep")
        doc = extract_document(
            pathlib.Path("data/sample_documents/intro_to_algebra.pdf").read_bytes(),
            "intro_to_algebra.pdf")
        IndexingService(store, FakeEmbeddingProvider()).index_document(doc)
        retr = RetrievalService(store, FakeEmbeddingProvider())
        return retr, TutorService(retr, EchoAnswerProvider()), doc.filename

    monkeypatch.setattr(runner_mod, "REPORTS_BASE", tmp_path)
    monkeypatch.setattr(runner_mod.config, "generation_enabled", lambda: True)
    monkeypatch.setattr(runner_mod, "build_pipeline", fake_pipeline)

    code = main(["--live"])
    assert code == EXIT_OK
    assert (tmp_path / "live" / "latest.json").exists()
    assert not (tmp_path / "offline").exists()  # offline untouched


def test_blocked_live_preserves_prior_live_report(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    live_dir.mkdir(parents=True)
    (live_dir / "latest.json").write_text('{"good": "prior live PASS"}', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "REPORTS_BASE", tmp_path)
    monkeypatch.setattr(runner_mod.config, "generation_enabled", lambda: True)
    monkeypatch.setattr(runner_mod, "build_pipeline",
                        lambda mode: (_ for _ in ()).throw(IndexingError("x", category="quota")))
    code = main(["--live"])  # default out-dir -> tmp/live
    assert code == EXIT_LIVE_BLOCKED
    # The prior successful live report is untouched.
    assert (live_dir / "latest.json").read_text() == '{"good": "prior live PASS"}'
