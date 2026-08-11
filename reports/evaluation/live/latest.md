# CurriculumIQ — Evaluation Report

- **Timestamp:** 2026-08-11T13:32:14.261546+00:00
- **Mode:** `live`
- **Evaluation type:** **Live product evaluation** (real embeddings + real generation) — a valid basis for product-quality claims.
- **Provider:** real OpenAI (live product evaluation)
- **Embedding model:** `text-embedding-3-small`
- **Chat model:** `gpt-5.6`
- **Top-K:** 5
- **Live validation:** completed

## Overall: PASS

> ✅ **Live mode** — real OpenAI embeddings and generation against the configured models. These metrics are a valid basis for product-quality claims (on the synthetic demo curriculum).

## Dataset

- Items: **24** (answerable 20, unsupported 4)
- Category breakdown: {'factual': 14, 'paraphrase': 4, 'multi_chunk': 2, 'unsupported': 4}

## Metrics

| Metric | Value | Threshold | Met |
|---|---|---|---|
| Retrieval hit rate @K | 100.0% | 80.0% | ✅ |
| Expected-page accuracy | 100.0% | 70.0% | ✅ |
| Citation validity | 100.0% | 95.0% | ✅ |
| Citation page accuracy | 100.0% | 60.0% | ✅ |
| Abstention accuracy | 100.0% | 90.0% | ✅ |
| Keyword coverage | 97.5% | 55.0% | ✅ |
| Per-item pass rate | 100.0% | 75.0% | ✅ |

## Latency (mean)

- Retrieval: 293.5 ms
- Generation: 1880.5 ms
- End-to-end: 2174.3 ms

## Errors: 0

## Failed questions: 0

## Remaining risks

- Ground truth is scoped to the **synthetic demo curriculum** (`intro_to_algebra.pdf`); these results validate real model quality on that document, not on real-world course material.
- Single borderline paraphrase abstentions are expected near the distance gate; tune `RAG_MAX_DISTANCE` if abstention is too eager.
