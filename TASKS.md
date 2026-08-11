# CurriculumIQ — Tasks & Roadmap

## Milestone 1 — Foundation & PDF extraction ✅ (verified)

> Re-verified at the start of Milestone 2: all 14 M1 tests pass on Python 3.11;
> extraction interface unchanged (an optional `document_id` field was added to
> `ExtractedDocument`, backward compatible).

### Completed Day 1 tasks

- [x] Modular project structure (`src/`, `tests/`, `data/`, `screenshots/`).
- [x] Pinned Day-1 dependencies in `requirements.txt` (no RAG libs yet).
- [x] Typed models: `DocumentPage`, `ExtractedDocument` (Pydantic v2).
- [x] Config module reading env vars (limits + OpenAI settings, unused Day 1).
- [x] PDF validation: extension, empty, size (15 MB), signature, encryption,
      corruption — with safe user-facing messages and logged technical detail.
- [x] PyMuPDF extraction: page-by-page, one-based page numbers, whitespace
      normalization, per-page and document totals, no-text warnings.
- [x] Extraction/validation independent of Streamlit (unit-testable).
- [x] Streamlit UI: upload, process, summary, per-page previews, warnings,
      clear/reset, privacy notice, next-milestone section.
- [x] Session-state behavior: results persist across reruns; new upload clears
      stale results; reset clears document; no duplicate processing.
- [x] Test suite for validator and extractor using in-memory generated PDFs.
- [x] Docs: README, PROJECT_SPEC, TASKS, `.env.example`, `.gitignore`.

### Acceptance criteria (Milestone 1)

- App starts with `streamlit run app.py`. ✅
- A valid digital PDF uploads and processes. ✅
- Text extracted page by page with correct one-based numbers/stats. ✅
- Invalid/empty/oversized/corrupted files → safe, clear errors. ✅
- Uploaded files are not persisted. ✅
- Extraction logic is UI-independent. ✅
- `pytest` passes. ✅ (see Current test status)

## Milestone 2 — Chunking, embeddings & Chroma ✅ (complete)

### Completed tasks

- [x] `document_id` derived from file checksum (`compute_document_id`); added to
      `ExtractedDocument` (backward compatible).
- [x] `DocumentChunk`, `IndexingResult`, `RetrievalResult` models (Pydantic v2).
- [x] Token-aware, page-preserving chunking (`chunker.py`, tiktoken via
      langchain splitter); deterministic IDs; skips empty pages; no API calls.
- [x] Injectable `EmbeddingProvider`: `OpenAIEmbeddingProvider` (production) and
      deterministic `FakeEmbeddingProvider` (tests). One client per provider,
      reused across batches; empty input rejected; failures → safe errors.
- [x] Chroma `VectorStore` (cosine): add, query, list document ids, has_document,
      delete-by-document, clear; local persist directory (git-ignored).
- [x] `IndexingService`: chunk → embed (batched) → store, with duplicate
      detection, safe re-index (embed → delete old → insert), and a
      stored-count verification before reporting success.
- [x] `RetrievalService`: query embedding → ranked results with filename/page,
      top-k, optional document filter, blank-query rejection, empty-collection
      handling. No answer generation; distances documented (lower = closer).
- [x] Streamlit: chunking summary, index/re-index, semantic search (top-k slider,
      current-document filter), developer details panel; safe without a key.
- [x] Session state: extraction/chunks survive reruns; new upload/clear resets
      derived state and search; indexing not repeated accidentally.
- [x] Tests: chunker, indexer, retriever, config — fake embeddings, temp Chroma
      dirs, deterministic, self-cleaning. Milestone 1 tests still pass.
- [x] Dependencies added and version-resolved on Python 3.11 (see requirements).
- [x] Docs updated: README (architecture, env, indexing/retrieval, troubleshooting),
      `.env.example`, `.gitignore` (`.curriculumiq_data/`).

### Acceptance criteria (Milestone 2)

- All Milestone 1 tests still pass. ✅
- Page-aware token chunking works; chunk IDs deterministic. ✅
- OpenAI embeddings isolated behind an injectable provider. ✅
- Chroma stores and retrieves chunks; duplicates handled safely. ✅
- Search returns filename, page, and passage metadata. ✅
- Automated tests use fake embeddings (offline). ✅
- App works safely without an API key. ✅
- No secret committed; docs accurate; app starts. ✅
- **Real OpenAI indexing/search:** verified offline with fake embeddings against
  the real sample PDF. Real-API run confirmed **PARTIAL** during Milestone 3: the
  key authenticates, but the account has no credits (see blockers).

## Milestone 3 — Grounded AI tutor (answers + citations + chat) ✅ PARTIAL

> Delivers grounded answer generation, verified citations, and the student chat
> interface (the original plan's M3 + M4 scope). PASS on all offline criteria;
> PARTIAL only because the live account has no OpenAI credits.

### Completed tasks

- [x] Generation config: `RAG_MAX_CONTEXT_CHUNKS/TOKENS`, `RAG_MAX_DISTANCE`
      (cosine gate, lower=closer), `RAG_HISTORY_MESSAGE_LIMIT`,
      `RAG_MAX_QUESTION_LENGTH`; independent embedding/chat models;
      `validate_generation_config`, `has_chat_model`, `generation_enabled`.
- [x] `SourceCitation` and `TutorAnswer` models (Pydantic v2).
- [x] `src/generation/prompts.py`: version-controlled instructions (grounding,
      injection defense, exact fallback, cite-only-supplied-IDs) + delimited,
      escaped `<SOURCES>/<RECENT_CONVERSATION>/<QUESTION>` user prompt.
- [x] Injectable `AnswerProvider`: `OpenAIAnswerProvider` (Responses API,
      `client.responses.create`, `output_text`, one reused client, errors mapped
      to safe messages) and deterministic `FakeAnswerProvider` (offline).
- [x] `src/generation/citations.py`: deterministic S-ID assignment, marker
      parsing, unknown-ID rejection with warnings; trusted metadata only.
- [x] `TutorService`: validate → retrieve (always) → distance+token grounding
      gate → local abstain (no model call) → prompt → generate → citation
      validation → typed `TutorAnswer` with retrieval/generation latency.
      Retrieval/generation API failures become safe `TutorError`s.
- [x] Bounded history helpers (`bound_history`) + session helpers
      (`record_feedback`, `should_reset_chat`), Streamlit-independent + tested.
- [x] Streamlit **AI Tutor**: `st.chat_message`/`st.chat_input`, scope toggle,
      example questions, sources (filename/page/distance/passage), latency,
      abstention status, session-only feedback, clear-conversation, study-aid
      disclaimer; disabled safely without a key/model.
- [x] Tests: prompts, citations, tutor, session, generation config — offline
      fakes only. All prior tests still pass.

### Acceptance criteria (Milestone 3)

- All existing tests remain green. ✅ (73 passed)
- A question retrieves curriculum evidence. ✅
- Responses API generates an answer from that evidence. ✅ (offline fake + real
  code path; live call blocked by credits — see below)
- Source IDs assigned before generation. ✅
- Displayed filenames/pages come from trusted metadata. ✅
- Unknown citations rejected. ✅
- Unsupported questions abstain (locally, without calling the model). ✅
- Every factual question performs retrieval (incl. follow-ups). ✅
- Chat history is bounded. ✅
- UI supports a complete student conversation. ✅
- Tests work without network access. ✅
- No secret committed. ✅
- **Real API verification succeeds. ❌ — account has no credits (see blockers).**
  Per the milestone's own rule, this makes the milestone **PARTIAL**, not PASS.

## Milestone 4 — Live validation & evaluation ✅ PARTIAL (Day 4)

> Delivers the deterministic evaluation harness, dataset, reports, and product
> hardening. PASS on all offline criteria; PARTIAL only because the live smoke
> test is blocked by billing (no credits — see blockers).

### Completed tasks

- [x] Live smoke test wired through real provider code (embedding → index →
      retrieve → generate → abstain). Result: **blocked by billing** (HTTP 429
      `insufficient_quota`); recorded, not retried.
- [x] Verified evaluation dataset: `data/evaluation/curriculum_eval.json`, 24
      items (14 factual, 4 paraphrase, 2 multi-chunk, 4 unsupported), each
      answerable item ground-truth-checked against `intro_to_algebra.pdf`
      (a new 6-page sample authored for evaluation).
- [x] Evaluation package: `src/evaluation/{schema,metrics,runner}.py` with
      `python -m src.evaluation.runner` (offline default, `--live` opt-in).
- [x] Deterministic metrics (no LLM judge): retrieval hit@K, expected-page
      accuracy, citation validity, citation-page accuracy, abstention accuracy,
      keyword coverage, per-item pass rate, latencies, error counts/categories.
- [x] Reports: `reports/evaluation/latest.{json,csv,md}` with timestamp, model
      names (no secrets), category breakdown, metric table, latency, failed IDs
      + reasons, verdict, and remaining risks. Configurable thresholds.
- [x] Hardening: verified safe handling of no-doc, unindexed-doc (added a
      nudge), empty/long question, unsupported question, invalid citation, auth
      failure, insufficient quota (confirmed live), model unavailable, timeout,
      no-evidence — all safe messages, no traces/secrets.
- [x] Improved `FakeEmbeddingProvider` to ignore stopwords (better separation
      for offline eval; no regression — all prior tests still green).
- [x] Tests: dataset validation, metric calculations, offline execution,
      report generation, live-off-by-default, safe error mapping.

### Acceptance criteria (Milestone 4)

- Real API smoke test succeeds **or** exact blocker documented. ✅ (blocker
  documented: 429 insufficient_quota)
- Evaluation harness runs end to end. ✅
- ≥ 20 verified questions. ✅ (24)
- Offline evaluation is deterministic. ✅
- Live evaluation is opt-in. ✅
- JSON/CSV/MD reports generated. ✅
- All tests pass (93), pip check + compile pass. ✅
- No secret committed. ✅
- README documents evaluation commands. ✅
- **Live smoke test succeeds. ❌ — account has no credits.** → milestone PARTIAL.

## Milestone 5 — Production readiness & deployment ✅ PARTIAL (Day 5)

> Deployment configuration, hardening, docs, and a robust live-eval CLI are all
> in place. PARTIAL only because the account has no credits, so live product
> evaluation and a real public URL cannot be completed here.

### Completed tasks

- [x] **Live-eval CLI hardening (bug fix):** `--live` on
      `credit_balance_exhausted` now exits with a documented code (3), prints one
      safe line (mode, phase, category, action) with **no traceback**, leaks no
      provider message, and **does not overwrite** existing reports. Centralized
      safe error classifier (`src/openai_safe.py`); `category` propagated through
      `EmbeddingError`/`IndexingError`/`AnswerGenerationError`/`TutorError`;
      embedding layer no longer logs full tracebacks. Regression-tested.
- [x] Correct evaluation labeling: offline = deterministic pipeline validation
      (not model-quality); live = product evaluation. Synthetic sample PDFs
      labeled; real-document evaluation documented as a release limitation.
- [x] Hardening: added `MAX_PAGE_COUNT` limit; tutor input **disabled until a
      document is indexed**; verified no absolute paths, in-memory uploads, safe
      filenames (metadata only), safe messages for all API/PDF error cases.
- [x] Deployment config: `.streamlit/secrets.toml.example` (placeholders),
      `DEPLOYMENT.md`, `RELEASE_CHECKLIST.md`, `scripts/release_check.sh`;
      `.env`/`secrets.toml` git-ignored; local-`.env`-vs-hosted-secrets model.
- [x] README: problem/target user, deployment, demo walkthrough, screenshots
      placeholders, privacy/security, evaluation labeling.

### Acceptance criteria (Milestone 5)

- App reproducible from documentation; release gate green. ✅
- Deployment configured (Streamlit Cloud) and documented. ✅
- Safe, no-traceback live-eval error handling. ✅ (regression-tested)
- No secret committed; reports secret-free. ✅
- **App actually deployed to a public URL and tested. ❌** — requires funded
  credits + the user's Streamlit Cloud account. → milestone PARTIAL.

## Milestone 6 — Final deployment & production verification ✅ PARTIAL (Day 6)

> **Live evaluation now PASSES** against real OpenAI (billing enabled). Reports
> are separated, public-demo cost protection is in place, and the app is
> deploy-ready. PARTIAL only because the public deploy + URL test require the
> user's GitHub/Streamlit accounts (this dir isn't even a git repo yet).

### Completed tasks

- [x] **Live evaluation PASS** (`--live`) against real OpenAI
      (`text-embedding-3-small` / `gpt-5.6`): retrieval 100%, page 100%,
      citation validity 100%, citation-page 100%, abstention 100%, keyword
      ~0.97, 0 errors, 1 borderline paraphrase abstention. Report in
      `reports/evaluation/live/latest.*`.
- [x] **Report separation:** offline → `reports/evaluation/offline/`, live →
      `reports/evaluation/live/`; a blocked/failed run writes nothing and never
      overwrites the last good report; each report states its evaluation type.
      Regression-tested (default-dir separation + blocked-preserves-prior).
- [x] **Public-demo cost protection:** `MAX_QUESTIONS_PER_SESSION` cap with a
      visible remaining count + friendly limit message + disabled input;
      optional `APP_ACCESS_CODE` gate with constant-time `hmac.compare_digest`
      comparison (never logged/echoed); local dev open when unset. Placeholders
      in `.env.example` + `.streamlit/secrets.toml.example`. Tested.
- [x] Final checks: full suite, pip check, compile, secret scan, release gate,
      boot smoke, desktop (1280) + mobile (375) width (no overflow), no absolute
      paths; access gate verified rendering + blocking content.
- [x] Prepared commit scope; verified secrets excluded from tracked files.

### Acceptance criteria (Milestone 6)

- Live evaluation PASS with real models. ✅
- Offline + live reports preserved separately. ✅
- Cost protection (session cap + access gate). ✅
- No secret committed; reports secret-free. ✅
- **Deployed to a public URL and tested. ❌** — requires the user's accounts.
  → milestone PARTIAL.

## Known blockers

- **Public deployment requires your accounts.** This directory is not yet a git
  repository and has no remote; Streamlit Community Cloud is tied to your
  GitHub/Streamlit login. The final `git init`/push + deploy + public-URL test
  must be done by you (exact steps in `DEPLOYMENT.md` and the Day-6 response).
- OpenAI billing is now working (live eval PASS). Keep an eye on spend; the
  session cap + access gate limit public exposure.

## Current test status

- **106 passed** (`pytest`), all offline (fake embeddings + fake/echo answer
  providers, temp Chroma dirs):
  - 7 validator + 8 extractor (M1)
  - 10 chunker + 5 indexer + 6 retriever (M2)
  - 9 config + 6 prompts + 5 citations + 15 tutor + 3 session (M3)
  - 6 eval-dataset + 9 eval-metrics + 13 eval-runner (M4–M6; +4 report separation)
  - 4 cost-protection (M6)
