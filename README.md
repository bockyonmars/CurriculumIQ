# CurriculumIQ

**Ask questions about your curriculum and receive clear answers with verifiable sources.**

CurriculumIQ is an AI tutor built on Retrieval-Augmented Generation (RAG). Upload
curriculum PDFs, index their content, and ask questions — every answer is drawn
*only* from your material and shows the exact document and page behind it, or
abstains when the material doesn't support an answer.

**Problem it solves:** students and educators work from dense course PDFs, and
general-purpose chatbots answer confidently without showing where the answer came
from — so answers can't be trusted or verified. **Who it's for:** students
revising from course material, educators and curriculum designers, and
self-learners.

> **Status: production deployment complete.** ✅
> - 🔗 **Live demo:** https://cewkymd8vqval4sjweyy8q.streamlit.app/ (access-code protected)
> - 💻 **Repository:** https://github.com/bockyonmars/CurriculumIQ
> - ✅ **109 tests passing** (offline, deterministic — no network, no API cost)
> - ✅ **Live evaluation PASS** against real OpenAI models
>   (`text-embedding-3-small` + `gpt-5.6`) — see [results](#live-evaluation-results)

## Try it

**Live app:** https://cewkymd8vqval4sjweyy8q.streamlit.app/ — the public demo is
**access-code protected** (to control API cost); request the code, or run it
locally (see [Local setup](#local-setup)). Then, in three steps:

1. **Upload** a text-based curriculum PDF (a synthetic sample,
   `data/sample_documents/intro_to_algebra.pdf`, is bundled) → **Process document**.
2. **Index document** — chunks are embedded and stored in a vector database.
3. **Ask** a question in the **AI Tutor** — get a grounded answer with `[S#]`
   citations you can expand to the exact source page.

## Features

- **Grounded answers only.** The tutor answers strictly from retrieved curriculum
  passages via the **OpenAI Responses API**; if the evidence is weak it **abstains
  locally without calling the model** and returns an exact fallback.
- **Trustworthy citations.** Source IDs (`S1`, `S2`, …) are assigned in Python
  *before* generation; only IDs actually supplied are accepted back, unknown
  markers are stripped, and **displayed filenames/pages always come from retrieval
  metadata — never from the model.**
- **Prompt-injection resistant.** Retrieved document text is treated as untrusted
  data, escaped and delimited, and never placed in developer instructions.
- **Page-accurate ingestion.** PyMuPDF extraction with correct one-based page
  numbers; token-aware, page-preserving chunking (tiktoken) so every chunk traces
  to one document and one page.
- **Semantic search** over a Chroma vector store (cosine distance) with duplicate
  detection, safe re-index, and clear.
- **Injectable providers.** All OpenAI access sits behind interfaces, so the full
  pipeline is unit-tested offline with deterministic fakes — no network, no cost.
- **Deterministic evaluation harness** (no LLM-as-judge) scoring retrieval,
  citations, abstention, and keyword coverage, with offline and live modes.
- **Production hardening:** file-size and page-count limits, safe error messages,
  a per-session question cap, an optional access-code gate, and a
  diagnostics panel hidden by default.

Robust with or without a key: **PDF extraction and chunking work without an API
key**; indexing, search, and the tutor require one.

## Live evaluation results

Run with real OpenAI models (`text-embedding-3-small`, `gpt-5.6`) over the bundled
evaluation set (`reports/evaluation/live/latest.md`):

| Metric | Result |
|---|---|
| Evaluation questions | 24 (20 answerable, 4 unsupported) |
| Retrieval hit@5 | **100%** |
| Expected-page accuracy | **100%** |
| Citation validity | **100%** |
| Citation page accuracy | **100%** |
| Abstention accuracy (unsupported → abstains) | **100%** |
| Grounded-answer keyword coverage | **97.5%** |
| Errors | **0** |
| Verdict | **PASS** |

> **Scope caveat.** This benchmark runs against a **synthetic demo curriculum**
> (`intro_to_algebra.pdf`) authored for demonstration, with human-verified ground
> truth. It validates the end-to-end product on that document — it is **not** a
> measure of broad real-world accuracy across arbitrary course material. Scoring
> is deterministic (expected pages, keywords, citations, abstention labels); no
> model judges another model.

## Architecture

```
PDF bytes
  └─ ingestion/validator.py   structural checks, safe errors
  └─ ingestion/extractor.py   PyMuPDF → ExtractedDocument (+ content-checksum document_id)
       └─ ingestion/chunker.py   per-page token-aware split → DocumentChunk[]  (no API calls)
            └─ retrieval/embeddings.py   EmbeddingProvider (OpenAI | Fake)
                 └─ retrieval/indexer.py      chunk → embed (batched) → store, with verify
                      └─ retrieval/vector_store.py   Chroma (cosine), per-document ops
                           └─ retrieval/retriever.py   query → ranked RetrievalResult[]
                                └─ generation/tutor.py   grounding gate → answer → citation check
```

Everything under `src/` is Streamlit-independent and unit-tested. Production
injects `OpenAIEmbeddingProvider` / `OpenAIAnswerProvider`; tests inject
deterministic fakes — nothing else in the pipeline knows which is used.

### End-to-end tutor flow

```
question
  1. validate (non-empty, length)
  2. retrieve  ← always, even for follow-ups
  3. filter empties, apply RAG_MAX_DISTANCE gate, apply token budget
  4. assign source IDs S1..Sn  ← in Python, before the model
  5. if no evidence → abstain locally (exact fallback), DO NOT call model
  6. build prompt: <SOURCES> + optional <RECENT_CONVERSATION> + <QUESTION>
  7. generate  ← OpenAI Responses API (client.responses.create) | Fake
  8. validate citations against supplied IDs; strip unknown [S#]
  9. TutorAnswer(answer, citations, retrieved_sources, abstained, latency, …)
```

- **Grounding.** The model is instructed to answer *only* from `<SOURCES>` and to
  treat delimited content as untrusted data. If evidence is insufficient it
  returns the exact fallback: *"I could not find enough information in the
  available curriculum materials."*
- **Retrieval every turn.** History is bounded (`RAG_HISTORY_MESSAGE_LIMIT`) and
  used only to interpret follow-ups; prior answers are never treated as evidence.
- **Distance semantics.** Chroma uses cosine distance — **lower = closer match.**

For milestone-by-milestone history and design rationale, see
[PROJECT_SPEC.md](PROJECT_SPEC.md) and [TASKS.md](TASKS.md).

## Local setup

**Requirements:** Python **3.11** (the Chroma/embedding stack resolves cleanly on
3.11; 3.9 is not recommended). Dependencies are pinned in
[requirements.txt](requirements.txt).

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Extraction and chunking need no key. Indexing, search, and the tutor call OpenAI
(embeddings + chat), so they require a key.

```bash
cp .env.example .env
```

Then edit `.env` (git-ignored — never commit it):

```
OPENAI_API_KEY=sk-...your key...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5.6          # deployed demo & live evaluation use this
# OPENAI_CHAT_MODEL=gpt-4o-mini    # lower-cost alternative for local development
```

**Models are configured independently and are never changed silently.** The
deployed demo and live evaluation run on **`gpt-5.6`**; the repository ships
`gpt-4o-mini` as a low-cost default for local development — set `OPENAI_CHAT_MODEL`
to whichever your account can access. Optional tuning (defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_CHUNK_SIZE_TOKENS` | `700` | Target tokens per chunk. |
| `RAG_CHUNK_OVERLAP_TOKENS` | `100` | Token overlap between chunks. |
| `RAG_TOP_K` | `5` | Default search results. |
| `RAG_MAX_CONTEXT_CHUNKS` | `5` | Max chunks fed to the tutor as context. |
| `RAG_MAX_CONTEXT_TOKENS` | `2000` | Token budget for tutor context. |
| `RAG_MAX_DISTANCE` | `0.7` | **Max** cosine distance accepted as evidence (lower = stricter; abstains sooner). |
| `RAG_HISTORY_MESSAGE_LIMIT` | `6` | Recent chat messages kept for follow-ups. |
| `RAG_MAX_QUESTION_LENGTH` | `1000` | Reject questions longer than this. |
| `MAX_FILE_SIZE_MB` | `15` | Max upload size. |
| `MAX_PAGE_COUNT` | `300` | Max pages per PDF (protects a hosted instance). |
| `MAX_QUESTIONS_PER_SESSION` | `20` | Per-session tutor question cap (`0` = unlimited). |
| `APP_ACCESS_CODE` | *(empty)* | Shared access code gating the app (empty = open). |
| `SHOW_DEVELOPER_DETAILS` | `false` | Show the internal diagnostics panel (local debugging only). |
| `CHROMA_PERSIST_DIRECTORY` | `.curriculumiq_data/chroma` | Local vector DB (git-ignored). |

If no key (or chat model) is set, extraction/chunking still work; indexing,
search, and the tutor are disabled with a clear message.

### Run and test

```bash
streamlit run app.py     # open the URL Streamlit prints (default :8501)
pytest                   # 109 tests, all offline (fake embeddings, temp Chroma) — no cost
```

### Demo walkthrough (bundled synthetic PDF)

1. `streamlit run app.py` and open the URL (enter the access code if one is set).
2. Upload `data/sample_documents/intro_to_algebra.pdf` → **Process document**
   (summary shows 6 pages, word/character counts, per-page previews).
3. **Index document** (needs a funded `OPENAI_API_KEY`).
4. **AI Tutor** → ask *"Explain the quadratic formula"* → grounded answer citing
   `intro_to_algebra.pdf`, **page 5**; expand **Sources** to inspect the passage.
5. Ask a paraphrase, *"how do straight-line graphs work"* → cites **page 2**.
6. Ask *"Who wrote Pride and Prejudice?"* → the tutor **abstains** (no citations).
7. **Clear conversation** resets the chat but keeps the indexed document. Locally,
   the Chroma index persists across restarts (on hosted Streamlit it is ephemeral).

## Evaluation

A deterministic, non-LLM-judge harness scores retrieval and grounded answers
against a version-controlled dataset of verified questions
(`data/evaluation/curriculum_eval.json`, ground truth checked against
`data/sample_documents/intro_to_algebra.pdf`).

```bash
# Offline (deterministic, no API, no cost) — the default:
python -m src.evaluation.runner

# Live (opt-in; uses the configured OpenAI models and spends credits):
python -m src.evaluation.runner --live
```

Reports are written per mode so live and offline never overwrite each other:
`reports/evaluation/offline/latest.{json,csv,md}` and
`reports/evaluation/live/latest.{json,csv,md}`. Default pass thresholds live in
`src/evaluation/schema.py` (`Thresholds`); also reported are per-phase latency and
an error count by category.

- **Offline mode** uses deterministic fake providers — it validates the harness
  and retrieval plumbing (software verification), **not** real model quality, and
  typically reports PARTIAL (the report says so).
- **Live mode** measures real semantic quality against the configured models
  (see [results](#live-evaluation-results)). It is opt-in so the test suite and CI
  never spend credits.

## Deployment

Deployed on **Streamlit Community Cloud**. Full steps, the
local-`.env`-vs-hosted-secrets model, cost-protection secrets, and the **ephemeral
storage caveat** are in **[DEPLOYMENT.md](DEPLOYMENT.md)**. Before deploying, run
the release gate and the manual checks:

```bash
bash scripts/release_check.sh
```

See **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)**.

## Access control & cost protection (public demos)

- **Session question cap:** `MAX_QUESTIONS_PER_SESSION` (default 20) limits tutor
  questions per browser session and shows a remaining count — protecting a public
  deployment from unbounded API cost.
- **Access code:** set `APP_ACCESS_CODE` (env or Streamlit secret) to gate the app
  behind a shared code (constant-time comparison; never logged or displayed).

## Security & privacy

- **No API key is ever entered through the UI** — keys come only from the
  environment or host secrets. `.env` and `.streamlit/secrets.toml` are
  git-ignored, and the diagnostics panel is off by default.
- **Error handling:** expected provider and configuration failures (auth, quota,
  timeout, model-access, invalid/empty/corrupt PDFs) are converted to safe,
  user-readable messages that **do not include secrets, prompts, or document
  contents**; server logs record exception *type names* only. Unexpected
  programming defects are not guaranteed to be caught here and should be monitored
  through deployment logs.
- **Data flow:** uploaded PDFs are processed in memory. Indexing/tutoring sends
  curriculum text to OpenAI to compute embeddings and answers — do not upload
  confidential material you are not permitted to share.
- **Hosted storage is ephemeral:** on Streamlit Community Cloud, local vector data
  and uploads may not survive a restart (no durable persistence is implemented).

## Limitations

- Text-based (digital) PDFs only; scanned/image pages are flagged, not read.
- Indexing, search, and the tutor require an OpenAI API key with available credits
  and network access.
- The UI focuses on the current uploaded document at a time.
- Conversation history is session-only (not persisted); hosted storage is
  ephemeral.
- The bundled evaluation ground truth is scoped to a synthetic demo curriculum.

## Future improvements

- **OCR** for scanned/image-only PDFs.
- **Persistent hosted storage** (external vector DB / object storage) so indexed
  documents survive restarts.
- **Multi-document UI** for querying across a whole library at once.
- **Authentication** and per-user workspaces (beyond the shared access code).
- **Real-course evaluation** — a dataset built from genuine curriculum material
  (without fabricated ground truth) to measure real-world accuracy.

## Troubleshooting

- **Missing API key / chat model** — the app shows a clear banner;
  extraction/chunking still work. Add `OPENAI_API_KEY` (and `OPENAI_CHAT_MODEL`).
- **Authentication error (401)** — the key is wrong or revoked; update `.env`.
- **Quota / no credits (429 `insufficient_quota`)** — add billing credits; the app
  shows *"OpenAI rate limit or quota reached."* and never crashes.
- **Model access / not found** — set `OPENAI_CHAT_MODEL` to a model your account
  can access. The app does **not** silently change your configured model.
- **Chroma initialization error** — delete `.curriculumiq_data` and retry.
- **Incompatible Python version** — use Python 3.11 (recreate the venv).
- **Tutor keeps abstaining** — confirm the document was **indexed**; if it abstains
  on clearly relevant questions, raise `RAG_MAX_DISTANCE` slightly (lower = stricter).
- **Live evaluation blocked** — `python -m src.evaluation.runner --live` exits with
  a documented non-zero code and one safe line (no traceback) on quota/auth/model
  errors; the offline report is left untouched.

## Project documentation

- [PROJECT_SPEC.md](PROJECT_SPEC.md) — product spec, scope, and architecture.
- [TASKS.md](TASKS.md) — milestone history and status (all milestones complete).
- [DEPLOYMENT.md](DEPLOYMENT.md) — deployment procedure and configuration.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) — pre-release verification.
