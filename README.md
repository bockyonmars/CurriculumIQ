# CurriculumIQ

**Ask questions about your curriculum and receive clear answers with verifiable sources.**

CurriculumIQ is an AI tutor with Retrieval-Augmented Generation (RAG). Upload
curriculum PDFs, index their content, ask questions, and receive grounded
answers with document and page citations.

**Problem:** students and educators work from dense course PDFs, and
general-purpose chatbots answer confidently without showing where the answer
came from — so answers can't be trusted or verified. **Target users:** students
revising from course material, educators/curriculum designers, and self-learners.
CurriculumIQ answers *only* from the user's own curriculum and shows the exact
document and page behind every answer (or abstains when the material doesn't
support one). See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full scope.

> **Status: Milestone 6 — final deployment & verification.** Grounded tutoring
> (OpenAI Responses API), verifiable `[S#]` citations, safe abstention, a
> deterministic evaluation harness, deployment configuration, and public-demo
> cost protection are all in place. **Live evaluation against real OpenAI models
> passes** (see `reports/evaluation/live/`). The only remaining step is the
> public deploy, which requires your GitHub/Streamlit accounts. See
> [TASKS.md](TASKS.md).

## Capabilities

**Milestone 1 (foundation):**
- Upload a curriculum PDF (up to 15 MB, text-based).
- Validation with safe errors for invalid, empty, oversized, password-protected,
  or corrupted files.
- Page-by-page extraction with correct one-based page numbers.
- Document summary: filename, size, pages, words, characters; per-page previews.

**Milestone 2 (retrieval foundation):**
- Token-aware, page-preserving chunking (tiktoken); every chunk traces to one
  document + one page.
- OpenAI embeddings behind an **injectable provider** (tests use a deterministic
  offline fake — no network, no cost).
- Chroma vector store with duplicate detection, safe re-index, and clear.
- Semantic search returning rank, filename, page, passage, and a cosine-distance
  score (**lower = closer**).
- Extraction and chunking work **without** an API key; indexing/search require one.

**Milestone 3 (AI tutor):**
- Grounded answers via the **OpenAI Responses API** (`client.responses.create`),
  behind the same injectable-provider pattern (tests use a deterministic fake).
- **Source IDs (`S1`, `S2`, …) assigned in Python before generation** and mapped
  to trusted chunk metadata; the model may only cite those IDs.
- Citation validation: unknown `[S#]` markers are stripped and warned about;
  displayed filenames/pages always come from retrieval, never from the model.
- **Grounding gate:** a configurable cosine-distance threshold plus a
  context-chunk / token budget. If no acceptable evidence remains, the tutor
  **abstains locally without calling the model** and returns the exact fallback.
- Bounded conversation history; every question re-runs retrieval (follow-ups
  never rely on prior answers as evidence).
- Streamlit chat UI (`st.chat_message` / `st.chat_input`) with sources, latency,
  abstention status, session-only feedback, and a study-aid disclaimer.

## Requirements

- **Python 3.11** (verified). Python 3.9 is not recommended for this milestone —
  the Chroma/embedding stack resolves cleanly on 3.11.
- Dependencies pinned in [requirements.txt](requirements.txt).

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment setup

Extraction and chunking need no key. Indexing, semantic search, and the AI tutor
call OpenAI (embeddings + chat), so they require a key.

```bash
cp .env.example .env
```

Then edit `.env` and set your key (the file is git-ignored — never commit it):

```
OPENAI_API_KEY=sk-...your key...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

Embedding and chat models are configured independently. Optional tuning
(defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_CHUNK_SIZE_TOKENS` | `700` | Target tokens per chunk. |
| `RAG_CHUNK_OVERLAP_TOKENS` | `100` | Token overlap between chunks. |
| `RAG_TOP_K` | `5` | Default search results. |
| `RAG_MAX_CONTEXT_CHUNKS` | `5` | Max chunks fed to the tutor as context. |
| `RAG_MAX_CONTEXT_TOKENS` | `2000` | Token budget for tutor context. |
| `RAG_MAX_DISTANCE` | `0.7` | **Max** cosine distance accepted as evidence. Lower is closer, so a *smaller* value abstains sooner. Tune per embedding model. |
| `RAG_HISTORY_MESSAGE_LIMIT` | `6` | Recent chat messages kept for follow-ups. |
| `RAG_MAX_QUESTION_LENGTH` | `1000` | Reject questions longer than this. |
| `MAX_FILE_SIZE_MB` | `15` | Max upload size. |
| `MAX_PAGE_COUNT` | `300` | Max pages per PDF (protects a hosted instance). |
| `CHROMA_PERSIST_DIRECTORY` | `.curriculumiq_data/chroma` | Local vector DB (git-ignored). |

If no `OPENAI_API_KEY` (or no chat model) is set, extraction/chunking still work;
indexing, search, and the tutor are disabled with a clear message.

## Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501).

## Run the tests

```bash
pytest
```

All tests use fake embeddings and temporary Chroma directories — **no network,
no OpenAI cost.**

## Indexing pipeline architecture

```
PDF bytes
  └─ validator.py     structural checks, safe errors
  └─ extractor.py     PyMuPDF → ExtractedDocument (+ content-checksum document_id)
       └─ chunker.py  per-page token-aware split → DocumentChunk[]  (no API calls)
            └─ retrieval/embeddings.py   EmbeddingProvider (OpenAI | Fake)
                 └─ retrieval/indexer.py      chunk → embed (batched) → store
                      └─ retrieval/vector_store.py   Chroma (cosine), per-document ops
                           └─ retrieval/retriever.py  query → ranked RetrievalResult[]
```

- **Embeddings are injectable.** Production injects `OpenAIEmbeddingProvider`;
  tests inject `FakeEmbeddingProvider`. Nothing else in the pipeline knows or
  cares which is used.
- Everything under `src/` is Streamlit-independent and unit-tested.

### How indexing works

1. Extract the PDF and derive a `document_id` from the file checksum.
2. Chunk each page independently (~700 tokens, 100 overlap), skipping empty pages.
3. Embed all chunks in a single batched call.
4. Store chunk text + embedding + metadata (document_id, filename, page number,
   chunk index) in Chroma.
5. Verify every intended chunk was stored before reporting success.
   Re-indexing embeds the new chunks, deletes the old ones, then inserts.

### How semantic retrieval works

The query is embedded with the same provider and compared against stored chunk
embeddings using **cosine distance** (Chroma `hnsw:space = cosine`). Results are
returned closest-first with rank, filename, page number, passage text, and the
distance score. **Lower distance = closer match.**

## End-to-end RAG flow (the AI tutor)

```
question
  └─ TutorService (src/generation/tutor.py)
       1. validate question (non-empty, length)
       2. retrieve  ← always, even for follow-ups
       3. filter empties, apply RAG_MAX_DISTANCE gate, apply token budget
       4. assign source IDs S1..Sn  ← in Python, before the model
       5. if no evidence → abstain locally (exact fallback), DO NOT call model
       6. build prompt: <SOURCES> + optional <RECENT_CONVERSATION> + <QUESTION>
       7. generate  ← generation/provider.py (OpenAI Responses API | Fake)
       8. validate citations against supplied IDs; strip unknown [S#]
       9. TutorAnswer(answer, citations, retrieved_sources, abstained, latency, …)
```

- **OpenAI Responses API.** The production provider calls
  `client.responses.create(model=…, instructions=…, input=…)` with one reused
  client, and reads `response.output_text`. The static tutor rules live in the
  `instructions` parameter; untrusted source text and the question go in `input`.
- **Grounding.** The prompt instructs the model to answer *only* from `<SOURCES>`
  and to treat everything inside the delimiters as untrusted data, never
  instructions (prompt-injection defense). If evidence is insufficient it returns
  the exact fallback: *"I could not find enough information in the available
  curriculum materials."*
- **Citations are trusted, not model-authored.** Source IDs are assigned to
  retrieved chunks in Python before generation. After generation, only IDs that
  were actually supplied are accepted; unknown markers are removed and a warning
  is recorded. The **filename and page shown always come from retrieval
  metadata** — the model can never invent them.
- **Abstention.** Two paths: (a) *local* abstention when the grounding gate finds
  no acceptable evidence (the model is never called), and (b) the model itself
  emitting the fallback. Both set `abstained=True` and attach no citations.
- **Conversation.** History is bounded (`RAG_HISTORY_MESSAGE_LIMIT`) and used only
  to interpret follow-ups; every question re-runs retrieval, and prior answers
  are never treated as curriculum evidence.

### Manual end-to-end check

With a key configured and credits available:

1. `streamlit run app.py`, upload `data/sample_documents/sample_curriculum.pdf`,
   click **Process document**, then **Index document**.
2. In **AI Tutor**, ask a supported question (e.g. *"Explain the quadratic
   formula"*) → expect an answer citing `sample_curriculum.pdf` page 3.
3. Ask a paraphrase (e.g. *"how do straight-line graphs work"*) → page 2.
4. Ask an unsupported question (e.g. *"medieval castle architecture"*) → the
   tutor abstains with the fallback and no citations.
5. Ask a follow-up → retrieval runs again; sources refresh.
6. Restart Streamlit → the indexed document persists (Chroma on disk).

### Clear or rebuild the local index

- **Clear from code:** `VectorStore(...).clear()` drops and recreates the
  collection.
- **Clear from disk:** delete the persist directory:
  ```bash
  rm -rf .curriculumiq_data
  ```
- **Rebuild:** re-run the app and index documents again, or use the app's
  **Re-index document** button (appears when a document is already indexed).

## Evaluation

A deterministic, non-LLM-judge evaluation harness scores retrieval and grounded
answers against a version-controlled dataset of **verified** questions
(`data/evaluation/curriculum_eval.json`, ground truth checked against
`data/sample_documents/intro_to_algebra.pdf`).

> **Synthetic demo data.** `intro_to_algebra.pdf` (and `sample_curriculum.pdf`)
> are **synthetic** curriculum documents authored for demos and pipeline
> testing — not real course material. Offline evaluation therefore validates the
> *software pipeline*, not real-world model accuracy. Evaluating against a real
> curriculum PDF is a **known release limitation** (no real document ships in
> this repo; add one and a matching dataset — without fabricating ground truth —
> to measure production quality).

```bash
# Offline (deterministic, no API, no cost) — the default:
python -m src.evaluation.runner

# Live (opt-in; uses the configured OpenAI models and spends credits):
python -m src.evaluation.runner --live
```

Reports are written per mode so live and offline never overwrite each other:
`reports/evaluation/offline/latest.{json,csv,md}` and
`reports/evaluation/live/latest.{json,csv,md}`.

**Metrics** (deterministic — expected pages, keywords, citations, abstention
labels; no model judges another model):

| Metric | Default threshold |
|---|---|
| Retrieval hit rate @K | ≥ 0.80 |
| Expected-page accuracy | ≥ 0.70 |
| Citation validity | ≥ 0.95 |
| Citation page accuracy | ≥ 0.60 |
| Abstention accuracy (unsupported → abstains) | ≥ 0.90 |
| Grounded-answer keyword coverage | ≥ 0.55 |
| Per-item pass rate | ≥ 0.75 |

Also reported: retrieval / generation / end-to-end latency and an error count by
category. Thresholds live in `src/evaluation/schema.py` (`Thresholds`).

- **Offline mode** uses the deterministic fake embedding + an echo answer
  provider. It validates the harness and retrieval plumbing but, because fake
  embeddings are lexical (not semantic), it can falsely abstain on some
  answerable questions — so offline typically reports **PARTIAL**, and the report
  says so. It is **not** a substitute for live validation.
- **Live mode** measures real semantic quality against the configured models.
  It is opt-in so the normal test suite and CI never spend credits.

## Current limitations

- Text-based (digital) PDFs only; scanned pages are flagged, not OCR'd.
- Indexing, search, and the tutor require an OpenAI API key, an account with
  available credits, and network access.
- Single-document-centric UI (the store can hold many; the UI focuses on the
  current upload).
- Conversation history is session-only and not persisted.
- Evaluation ground truth is scoped to the bundled sample curriculum.

## Troubleshooting

- **Missing API key / chat model** — the app shows a clear banner;
  extraction/chunking still work. Add `OPENAI_API_KEY` (and optionally
  `OPENAI_CHAT_MODEL`) to `.env` to enable indexing, search, and the tutor.
- **Authentication error (401)** — the key is wrong or revoked. Generate a new
  key and update `.env`. The app shows *"Authentication with OpenAI failed."*
- **Quota / no credits (429 `insufficient_quota`)** — the key is valid but the
  account has no remaining credits. Add billing credits; the app shows *"OpenAI
  rate limit or quota reached."* and never crashes.
- **Model access / not found** — the configured `OPENAI_CHAT_MODEL` isn't
  available to your account. Set `OPENAI_CHAT_MODEL` to a model you can access.
  The app does **not** silently change your configured model.
- **Chroma initialization error** — usually a permissions issue on the persist
  directory, or a stale/corrupt store. Delete `.curriculumiq_data` and retry.
- **Incompatible Python version** — use Python 3.11. On 3.9 the Chroma/embedding
  stack does not resolve cleanly. Recreate the venv with `python3.11 -m venv`.
- **Empty retrieval / tutor keeps abstaining** — confirm the document was
  **indexed** (search and the tutor use the vector store, not the raw PDF). If
  the tutor abstains on clearly relevant questions, your `RAG_MAX_DISTANCE` may
  be too strict (lower = stricter); raise it slightly. If filtered to the current
  document, confirm it's the one you indexed.
- **Duplicate document** — indexing the same file twice is detected via
  `document_id` and skipped; use **Re-index** to replace the existing chunks.
- **Live evaluation blocked** — `python -m src.evaluation.runner --live` exits
  with a documented non-zero code and one safe line (no traceback) on
  quota/auth/model errors; the offline report is left untouched.

## Deployment

Deploy to **Streamlit Community Cloud** (or any Streamlit host). Full steps,
the local-`.env`-vs-hosted-secrets configuration model, and the **ephemeral
storage caveat** are in **[DEPLOYMENT.md](DEPLOYMENT.md)**. Before deploying, run
the release gate:

```bash
bash scripts/release_check.sh
```

and the manual checks in **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)**.

## Demo walkthrough

1. `streamlit run app.py` and open the URL.
2. Upload `data/sample_documents/intro_to_algebra.pdf` → **Process document**
   (summary shows 6 pages, word/character counts, per-page previews).
3. **Index document** (needs a funded `OPENAI_API_KEY`).
4. **AI Tutor** → ask *"Explain the quadratic formula"* → grounded answer citing
   `intro_to_algebra.pdf`, page 5; expand **Sources** to inspect the passage.
5. Ask *"Who wrote Pride and Prejudice?"* → the tutor abstains (no citations).
6. **Clear conversation** resets the chat but keeps the indexed document.

## Screenshots

_Add images to `screenshots/` and reference them here before submission._

- `screenshots/01-upload-and-summary.png` — upload + document summary
- `screenshots/02-tutor-answer-citations.png` — grounded answer with `[S#]` sources
- `screenshots/03-abstention.png` — safe abstention on an unsupported question

## Access control & usage limits (public demos)

- **Session question cap:** `MAX_QUESTIONS_PER_SESSION` (default 20) limits tutor
  questions per browser session and shows a remaining count — protecting a public
  deployment from unbounded API cost. Set to `0` to disable.
- **Access code:** set `APP_ACCESS_CODE` (env or Streamlit secret) to gate the
  app behind a shared code (constant-time comparison; never logged or displayed).
  Leave it unset for open local development.

## Privacy & security

- No API key is ever entered through the UI — keys come only from the environment
  or host secrets. `.env` and `.streamlit/secrets.toml` are git-ignored.
- User-facing errors are safe messages: no stack traces, keys, prompts, or
  document text. Server logs contain exception type names only.
- Uploaded PDFs are processed in memory; on a hosted Streamlit instance local
  vector data and uploads are **ephemeral** (see DEPLOYMENT.md) — no durable
  persistence is implemented.
- Indexing/tutoring sends curriculum text to OpenAI to compute embeddings and
  answers; do not upload confidential material you are not permitted to share.

## Roadmap

Milestones 1–4 are complete (foundation, retrieval, grounded tutor, evaluation).
This is **Milestone 5 — production readiness & deployment**. See
[TASKS.md](TASKS.md) for status and remaining submission steps.
