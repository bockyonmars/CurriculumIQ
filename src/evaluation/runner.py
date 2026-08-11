"""Evaluation runner: offline (deterministic) or live (opt-in) end-to-end.

    python -m src.evaluation.runner            # offline, deterministic, no API
    python -m src.evaluation.runner --live     # uses configured OpenAI models

Writes JSON, CSV, and Markdown reports under reports/evaluation/.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from src import config
from src.evaluation.metrics import AggregateMetrics, ItemResult, aggregate, evaluate_item
from src.evaluation.schema import (
    DEFAULT_THRESHOLDS,
    EvalDataset,
    Thresholds,
    load_dataset,
)
from src.generation.provider import AnswerGenerationError, AnswerProvider, GeneratedAnswer
from src.generation.tutor import TutorError, TutorService
from src.ingestion.extractor import extract_document
from src.openai_safe import category_from_chain
from src.retrieval.embeddings import EmbeddingError, FakeEmbeddingProvider
from src.retrieval.indexer import IndexingError, IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DOC = REPO_ROOT / "data" / "sample_documents" / "intro_to_algebra.pdf"
REPORTS_BASE = REPO_ROOT / "reports" / "evaluation"


def default_out_dir(mode: str) -> Path:
    """Reports are separated by mode so live and offline never overwrite."""
    return REPORTS_BASE / ("live" if mode == "live" else "offline")

_SOURCE_RE = re.compile(r'<SOURCE id="(S\d+)"[^>]*>(.*?)</SOURCE>', re.DOTALL)


class EchoAnswerProvider(AnswerProvider):
    """Deterministic offline provider: grounds its answer in the prompt's sources.

    It returns the retrieved passages, each tagged with its supplied source ID —
    exactly what a faithful grounded model would cite — so offline scoring
    exercises the full citation/keyword path without any API call.
    """

    model_name = "offline-echo"

    def generate(self, instructions: str, user_prompt: str) -> GeneratedAnswer:
        parts = []
        for sid, text in _SOURCE_RE.findall(user_prompt):
            parts.append(f"{html.unescape(text.strip())} [{sid}]")
        return GeneratedAnswer(text=" ".join(parts) if parts else "", usage=None)


# Documented CLI exit codes.
EXIT_OK = 0
EXIT_LIVE_CONFIG_MISSING = 2   # --live requested but key/model not configured
EXIT_LIVE_BLOCKED = 3          # live evaluation blocked by a provider error

# Expected, already-safe error types surfaced at the CLI boundary. Their
# messages never contain keys, prompts, or document text.
_EXPECTED_ERRORS = (
    EmbeddingError,
    IndexingError,
    AnswerGenerationError,
    VectorStoreError,
    TutorError,
)

_ACTIONS = {
    "quota": "Add OpenAI billing credits, then rerun.",
    "auth": "Check OPENAI_API_KEY, then rerun.",
    "rate_limit": "Wait and rerun, or reduce request volume.",
    "timeout": "Check your network connection, then rerun.",
    "model_unavailable": "Set OPENAI_CHAT_MODEL to a model your account can access.",
    "other": "See application logs for details.",
}


def _safe_error_category(exc: BaseException) -> str:
    """Safe category code for an error (delegates to the shared classifier)."""
    return category_from_chain(exc)


def build_pipeline(mode: str) -> Tuple[RetrievalService, TutorService, str]:
    """Create retriever + tutor for the given mode, indexing the eval document."""
    if mode == "live":
        from src.generation.provider import OpenAIAnswerProvider
        from src.retrieval.embeddings import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(config.OPENAI_EMBEDDING_MODEL, config.OPENAI_API_KEY)
        answerer: AnswerProvider = OpenAIAnswerProvider(config.OPENAI_CHAT_MODEL, config.OPENAI_API_KEY)
    else:
        embedder = FakeEmbeddingProvider()
        answerer = EchoAnswerProvider()

    store = VectorStore(tempfile.mkdtemp(prefix="curriculumiq_eval_"), "eval")
    doc = extract_document(EVAL_DOC.read_bytes(), EVAL_DOC.name)
    IndexingService(store, embedder).index_document(doc)

    retriever = RetrievalService(store, embedder)
    tutor = TutorService(retriever, answerer)
    return retriever, tutor, doc.filename


def run_evaluation(
    dataset: EvalDataset,
    mode: str,
    k: int,
    thresholds: Thresholds,
) -> Tuple[AggregateMetrics, List[ItemResult], dict]:
    """Run every dataset item through the pipeline and score it.

    A *setup* failure (e.g. live embedding/indexing hitting a quota or auth
    error) propagates to the caller so the CLI can report it safely without
    writing a report for a run that never actually happened. Per-item failures
    during the loop are recorded as errored items in a completed run.
    """
    retriever, tutor, doc_name = build_pipeline(mode)

    results: List[ItemResult] = []
    for item in dataset.items:
        # Raw retrieval (for retrieval metrics), independent of the abstain gate.
        retrieved_pages: List[int] = []
        try:
            hits = retriever.search(item.question, top_k=k)
            retrieved_pages = [h.page_number for h in hits]
        except Exception as exc:  # noqa: BLE001
            results.append(evaluate_item(item, [], None, error=_safe_error_category(exc)))
            continue

        # Full tutor pipeline (for answer/abstention/citation metrics).
        try:
            answer = tutor.answer(item.question)
            results.append(evaluate_item(item, retrieved_pages, answer))
        except (TutorError, Exception) as exc:  # noqa: BLE001
            results.append(evaluate_item(item, retrieved_pages, None, error=_safe_error_category(exc)))

    metrics = aggregate(results, thresholds)
    run_meta = _run_meta(mode, k, dataset, doc_name)
    run_meta["live_validation"] = (
        "completed" if mode == "live" and metrics.error_count == 0
        else ("attempted-with-errors" if mode == "live" else "skipped (offline mode)")
    )
    return metrics, results, run_meta


def _run_meta(mode: str, k: int, dataset: EvalDataset, doc_name: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "provider": "real OpenAI (live product evaluation)" if mode == "live"
        else "deterministic fakes (offline pipeline validation — not model-quality evidence)",
        "embedding_model": config.OPENAI_EMBEDDING_MODEL if mode == "live" else "fake-embedding",
        "chat_model": config.OPENAI_CHAT_MODEL if mode == "live" else "offline-echo",
        "k": k,
        "dataset_size": len(dataset.items),
        "category_counts": dataset.category_counts(),
        "eval_document": doc_name,
    }


# --- reporting --------------------------------------------------------------
def write_reports(
    out_dir: Path,
    metrics: AggregateMetrics,
    results: List[ItemResult],
    run_meta: dict,
    thresholds: Thresholds,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run": run_meta,
        "thresholds": thresholds.model_dump(),
        "metrics": metrics.model_dump(),
        "items": [r.model_dump() for r in results],
    }
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # CSV: one row per item.
    with (out_dir / "latest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "id", "category", "answerable", "passed", "retrieval_hit", "page_coverage",
            "answered", "abstained", "citation_valid", "citation_page_hit",
            "keyword_coverage", "latency_seconds", "error", "reason",
        ])
        for r in results:
            writer.writerow([
                r.id, r.category, r.answerable, r.passed, r.retrieval_hit,
                f"{r.page_coverage:.3f}", r.answered, r.abstained, r.citation_valid,
                r.citation_page_hit, f"{r.keyword_coverage:.3f}",
                f"{r.latency_seconds:.4f}", r.error or "", r.reason,
            ])

    (out_dir / "latest.md").write_text(
        _markdown_report(metrics, results, run_meta, thresholds), encoding="utf-8"
    )


def _markdown_report(
    metrics: AggregateMetrics, results: List[ItemResult], run_meta: dict, thresholds: Thresholds
) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    lines = [
        "# CurriculumIQ — Evaluation Report",
        "",
        f"- **Timestamp:** {run_meta['timestamp']}",
        f"- **Mode:** `{run_meta['mode']}`",
        f"- **Evaluation type:** "
        + ("**Live product evaluation** (real embeddings + real generation) — a "
           "valid basis for product-quality claims." if run_meta["mode"] == "live"
           else "**Offline pipeline validation** (deterministic fakes) — software "
           "verification only, NOT real model-quality evidence."),
        f"- **Provider:** {run_meta.get('provider', '')}",
        f"- **Embedding model:** `{run_meta['embedding_model']}`",
        f"- **Chat model:** `{run_meta['chat_model']}`",
        f"- **Top-K:** {run_meta['k']}",
        f"- **Live validation:** {run_meta['live_validation']}",
        "",
        f"## Overall: {metrics.verdict}",
        "",
    ]
    if run_meta["mode"] != "live":
        lines += [
            "> ⚠️ **Offline mode** — deterministic fakes, no API calls. These "
            "numbers validate the harness and retrieval plumbing; they are **not** "
            "a substitute for live validation of the real OpenAI models.",
            "",
        ]
    else:
        lines += [
            "> ✅ **Live mode** — real OpenAI embeddings and generation against the "
            "configured models. These metrics are a valid basis for product-quality "
            "claims (on the synthetic demo curriculum).",
            "",
        ]

    lines += [
        "## Dataset",
        "",
        f"- Items: **{metrics.n_items}** "
        f"(answerable {metrics.n_answerable}, unsupported {metrics.n_unsupported})",
        f"- Category breakdown: {run_meta['category_counts']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Threshold | Met |",
        "|---|---|---|---|",
    ]
    rows = [
        ("Retrieval hit rate @K", metrics.retrieval_hit_rate, thresholds.retrieval_hit_rate, "retrieval_hit_rate"),
        ("Expected-page accuracy", metrics.expected_page_accuracy, thresholds.expected_page_accuracy, "expected_page_accuracy"),
        ("Citation validity", metrics.citation_validity, thresholds.citation_validity, "citation_validity"),
        ("Citation page accuracy", metrics.citation_page_accuracy, thresholds.citation_page_accuracy, "citation_page_accuracy"),
        ("Abstention accuracy", metrics.abstention_accuracy, thresholds.abstention_accuracy, "abstention_accuracy"),
        ("Keyword coverage", metrics.keyword_coverage, thresholds.keyword_coverage, "keyword_coverage"),
        ("Per-item pass rate", metrics.item_pass_rate, thresholds.item_pass_rate, "item_pass_rate"),
    ]
    for label, val, thr, key in rows:
        met = "✅" if metrics.thresholds_met.get(key) else "❌"
        lines.append(f"| {label} | {pct(val)} | {pct(thr)} | {met} |")

    lines += [
        "",
        "## Latency (mean)",
        "",
        f"- Retrieval: {metrics.mean_retrieval_seconds * 1000:.1f} ms",
        f"- Generation: {metrics.mean_generation_seconds * 1000:.1f} ms",
        f"- End-to-end: {metrics.mean_latency_seconds * 1000:.1f} ms",
        "",
        f"## Errors: {metrics.error_count}",
        "",
    ]
    if metrics.error_categories:
        for cat, n in sorted(metrics.error_categories.items()):
            lines.append(f"- `{cat}`: {n}")
        lines.append("")

    failed = [r for r in results if not r.passed]
    lines += [f"## Failed questions: {len(failed)}", ""]
    if failed:
        lines.append("| ID | Category | Reason |")
        lines.append("|---|---|---|")
        for r in failed:
            lines.append(f"| {r.id} | {r.category} | {r.reason or r.error or 'unknown'} |")
        lines.append("")

    lines += ["## Remaining risks", ""]
    if run_meta["mode"] == "live":
        lines += [
            "- Ground truth is scoped to the **synthetic demo curriculum** "
            "(`intro_to_algebra.pdf`); these results validate real model quality on "
            "that document, not on real-world course material.",
            "- Single borderline paraphrase abstentions are expected near the "
            "distance gate; tune `RAG_MAX_DISTANCE` if abstention is too eager.",
        ]
    else:
        lines += [
            "- Offline metrics use lexical fake embeddings; paraphrase/semantic "
            "quality is only truly measured in `--live` mode.",
            "- Ground truth is scoped to the bundled synthetic sample curriculum; "
            "broaden the dataset before drawing product-wide conclusions.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CurriculumIQ evaluation.")
    parser.add_argument("--live", action="store_true",
                        help="Use the configured OpenAI models (opt-in; spends credits).")
    parser.add_argument("--k", type=int, default=config.RAG_TOP_K, help="Top-K for retrieval metrics.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the dataset JSON.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Report output directory (default: reports/evaluation/<mode>/).")
    args = parser.parse_args(argv)

    mode = "live" if args.live else "offline"
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir(mode)
    dataset = load_dataset(args.dataset)
    thresholds = DEFAULT_THRESHOLDS

    if mode == "live" and not config.generation_enabled():
        print("[live] BLOCKED — phase=config; error=config_missing. "
              "Set OPENAI_API_KEY and OPENAI_CHAT_MODEL, then rerun.", file=sys.stderr)
        return EXIT_LIVE_CONFIG_MISSING

    # CLI boundary: turn an expected provider/embedding/indexing failure into one
    # concise, safe line. No traceback, no reports overwritten, documented exit
    # code. Unexpected (programming) errors are left to propagate for diagnosis.
    try:
        metrics, results, run_meta = run_evaluation(dataset, mode, args.k, thresholds)
    except _EXPECTED_ERRORS as exc:
        category = _safe_error_category(exc)
        action = _ACTIONS.get(category, _ACTIONS["other"])
        print(f"[{mode}] evaluation BLOCKED — phase=model-setup (embedding/indexing); "
              f"error={category}. {action}", file=sys.stderr)
        print("No reports were written; existing reports were left untouched.", file=sys.stderr)
        return EXIT_LIVE_BLOCKED

    write_reports(out_dir, metrics, results, run_meta, thresholds)

    print(f"[{mode}] verdict={metrics.verdict} "
          f"hit@{args.k}={metrics.retrieval_hit_rate:.2f} "
          f"page_acc={metrics.expected_page_accuracy:.2f} "
          f"cite_valid={metrics.citation_validity:.2f} "
          f"abstain={metrics.abstention_accuracy:.2f} "
          f"keyword={metrics.keyword_coverage:.2f} "
          f"errors={metrics.error_count}")
    print(f"Reports written to {out_dir}/latest.{{json,csv,md}}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
