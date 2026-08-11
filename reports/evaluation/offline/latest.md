# CurriculumIQ — Evaluation Report

- **Timestamp:** 2026-08-11T13:55:56.409214+00:00
- **Mode:** `offline`
- **Evaluation type:** **Offline pipeline validation** (deterministic fakes) — software verification only, NOT real model-quality evidence.
- **Provider:** deterministic fakes (offline pipeline validation — not model-quality evidence)
- **Embedding model:** `fake-embedding`
- **Chat model:** `offline-echo`
- **Top-K:** 5
- **Live validation:** skipped (offline mode)

## Overall: PARTIAL

> ⚠️ **Offline mode** — deterministic fakes, no API calls. These numbers validate the harness and retrieval plumbing; they are **not** a substitute for live validation of the real OpenAI models.

## Dataset

- Items: **24** (answerable 20, unsupported 4)
- Category breakdown: {'factual': 14, 'paraphrase': 4, 'multi_chunk': 2, 'unsupported': 4}

## Metrics

| Metric | Value | Threshold | Met |
|---|---|---|---|
| Retrieval hit rate @K | 100.0% | 80.0% | ✅ |
| Expected-page accuracy | 100.0% | 70.0% | ✅ |
| Citation validity | 100.0% | 95.0% | ✅ |
| Citation page accuracy | 86.7% | 60.0% | ✅ |
| Abstention accuracy | 100.0% | 90.0% | ✅ |
| Keyword coverage | 87.8% | 55.0% | ✅ |
| Per-item pass rate | 70.8% | 75.0% | ❌ |

## Latency (mean)

- Retrieval: 2.0 ms
- Generation: 0.0 ms
- End-to-end: 2.2 ms

## Errors: 0

## Failed questions: 7

| ID | Category | Reason |
|---|---|---|
| f01 | factual | abstained on an answerable question |
| f02 | factual | no citation on an expected page |
| f03 | factual | no citation on an expected page |
| p02 | paraphrase | abstained on an answerable question |
| p03 | paraphrase | abstained on an answerable question |
| p04 | paraphrase | abstained on an answerable question |
| m01 | multi_chunk | abstained on an answerable question |

## Remaining risks

- Offline metrics use lexical fake embeddings; paraphrase/semantic quality is only truly measured in `--live` mode.
- Ground truth is scoped to the bundled synthetic sample curriculum; broaden the dataset before drawing product-wide conclusions.
