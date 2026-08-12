# CurriculumIQ — Performance

All figures below are **measured**, not estimated. Where a measurement was not
taken, that is stated explicitly rather than guessed.

## Test environment

- Machine: local dev laptop, macOS (Darwin), x86_64, 12 logical cores.
- Python 3.11.15; the pinned dependency set in `requirements.txt`.
- Sample document: `data/sample_documents/intro_to_algebra.pdf` — **6 pages,
  ~3.9 KB, 6 chunks** (a synthetic demo curriculum).
- Live model figures use `text-embedding-3-small` + `gpt-5.6`.

## Method

- **Offline pipeline** (validate/extract/chunk): timed in-process over 20
  iterations after a warm-up; medians reported (`python -c` micro-benchmark).
- **Live retrieval/generation latency**: taken from the committed live
  evaluation run over the 24-question dataset
  (`reports/evaluation/live/latest.json`), which records mean per-phase latency.
- **Docker**: image sizes from `docker images`; startup from container start to
  the first passing health check.

## Results — document preparation (offline, no API)

| Phase | Median | Notes |
|-------|--------|-------|
| Validate PDF | ~0.5 ms | signature, size, page-count, encryption checks |
| Extract text (PyMuPDF) | ~9.9 ms | 6 pages, one-based numbering |
| Chunk (token-aware) | ~1.4 ms | tiktoken; 6 chunks |
| **Prepare (validate+extract+chunk)** | **~13 ms** | excludes embedding/indexing |

Embedding + indexing add one **batched** OpenAI embeddings call plus a Chroma
write. It was not benchmarked in isolation here; the live per-call embedding
round-trip below (~0.3 s) is a reasonable proxy for the batched call at this
document size.

## Results — question answering (live, real OpenAI)

Mean per phase over 24 questions (from the live evaluation report):

| Phase | Mean |
|-------|------|
| Retrieval (query embed + Chroma search) | ~294 ms |
| Generation (OpenAI Responses API) | ~1880 ms |
| **End-to-end** | **~2174 ms** |

Generation dominates end-to-end latency. Locally-abstained questions (no
acceptable evidence) skip generation entirely and return in roughly the
retrieval time.

## Results — Docker

Measured on this machine with `docker compose`:

| Item | Measurement |
|------|-------------|
| Image — `python-ai-service` | ~1.41 GB (full RAG stack: chromadb/onnxruntime/pymupdf/langchain) |
| Image — `streamlit-app` | ~1.41 GB (same base Python stack) |
| Image — `spring-gateway` | ~502 MB (JRE + jar) |
| First `docker compose build` | ~10–15 min (dominated by the heavy Python `pip install`); cached rebuilds are fast |
| Cold start to **all three healthy** | **~19 s** (`docker compose up` → all health checks passing) |

The Python images are large because the RAG stack pulls ONNX Runtime and native
libraries. A future optimization is a slimmer image (e.g. drop unused optional
Chroma extras) — not done here to avoid destabilizing the working stack.

## Bottlenecks

1. **LLM generation (~1.9 s)** is the largest cost by far — inherent to the
   model call, not the app.
2. **Network round-trips to OpenAI** (embeddings + chat) dominate over local CPU
   work (extraction/chunking are single-digit milliseconds).
3. **First request per process** pays one-time client/model-encoder init; steady
   state is faster (why the benchmark warms up first).

## Optimization decisions already in place

- **Cached provider clients** — one OpenAI embeddings client and one chat client
  per process (`@st.cache_resource` / `@lru_cache`), reused across requests.
- **Batched embeddings** — all chunks of a document are embedded in a single API
  call, not one call per chunk.
- **Bounded context** — retrieval is capped by `RAG_MAX_CONTEXT_CHUNKS` and a
  `RAG_MAX_CONTEXT_TOKENS` budget, keeping prompt size (and cost/latency) bounded.
- **Local abstention gate** — weak-evidence questions abstain *without* calling
  the model, saving a full generation round-trip.
- **Bounded chat history** — `RAG_HISTORY_MESSAGE_LIMIT` caps follow-up context.
- **Document filtering** — retrieval is scoped to the active `document_id`.
- **Input limits** — `MAX_FILE_SIZE_MB` and `MAX_PAGE_COUNT` bound work per upload.
- **Session question cap** — `MAX_QUESTIONS_PER_SESSION` bounds cost per visitor.
- **Persistent Chroma volume** (Docker) — the vector store survives container
  restarts, so a document isn't re-embedded after a restart.
- **Health checks + timeouts** — the gateway uses connect/read timeouts and
  Compose gates start-up on health, so a slow/unavailable dependency fails fast
  and safely.

## Future scaling plan (not implemented)

- Cache embeddings/answers for repeated identical inputs.
- Stream tokens to the UI to reduce perceived generation latency.
- Move Chroma to a shared/managed vector store for multi-instance deployments.
- Horizontal scaling of the stateless Python service behind the gateway.

These are **future** directions. This project does **not** implement Redis,
Kubernetes, a message queue, distributed scaling, or production load testing.
