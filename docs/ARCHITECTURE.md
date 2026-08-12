# CurriculumIQ — Architecture

CurriculumIQ is a source-grounded RAG tutor. The AI/domain logic lives in one
place (`src/`) and is exposed two ways: directly (Streamlit imports it in
process) and over HTTP (a FastAPI service, fronted by a Spring Boot gateway).

## Components

```
                         ┌───────────────────────────────────────────────┐
   Browser  ──────────▶  │  Streamlit UI  (app.py)                        │
                         │  3-stage flow: Choose PDF → Prepare → Ask      │
                         └───────────────┬───────────────────────────────┘
                                         │
              SERVICE_MODE=direct        │        SERVICE_MODE=gateway
        (in-process Python calls) ◀──────┴──────▶ (HTTP)
                     │                                   │
                     │                          ┌────────▼─────────┐
                     │                          │ Spring Boot      │  Java 17
                     │                          │ gateway (:8080)  │  no OpenAI key
                     │                          │ REST proxy +     │
                     │                          │ health + timeouts│
                     │                          └────────┬─────────┘
                     │                                   │ REST
                     ▼                          ┌────────▼─────────┐
        ┌────────────────────────┐              │ Python AI API    │  FastAPI
        │   src/ domain logic     │◀────import──│ (:8000)          │  (owns AI ops)
        │  validate → extract →   │              └────────┬─────────┘
        │  chunk → embed → index  │                       │ import
        │  → retrieve → generate  │◀──────────────────────┘
        │  → cite / abstain       │
        └───────┬─────────┬───────┘
                │         │
        ┌───────▼──┐  ┌───▼──────────┐
        │ OpenAI   │  │ Chroma vector│
        │ API      │  │ store (disk) │
        └──────────┘  └──────────────┘
```

- **`src/`** — the single source of truth for all AI/domain logic (ingestion,
  retrieval, generation, citations, abstention). Both the Streamlit app and the
  FastAPI service import it; nothing is duplicated.
- **Python AI service (`python_service/`, FastAPI)** — a thin HTTP adapter over
  `src/`. It owns all OpenAI and Chroma access. Endpoints: `/health`,
  `POST /api/documents`, `POST /api/questions`.
- **Spring Boot gateway (`spring-gateway/`, Java 17)** — a small REST proxy in
  front of the Python service. It demonstrates Java/Spring, REST proxying,
  externalized config, connection/read timeouts, health checks (custom +
  Actuator), and global safe exception handling. **It never calls OpenAI and
  holds no OpenAI key.**
- **Streamlit UI (`app.py`)** — the student experience. `SERVICE_MODE` selects
  whether it calls `src/` directly or goes through the gateway; the visible
  three-stage UX is identical either way.

## Deployment modes

| Mode | How it runs | Used by |
|------|-------------|---------|
| **direct** (default) | Streamlit imports `src/` and calls OpenAI/Chroma in process | Streamlit Community Cloud (the public demo) |
| **gateway** | Streamlit → Spring gateway → Python AI service → `src/` | Docker Compose demo |

Streamlit Cloud runs **direct** mode (one process, simplest hosting). Docker
Compose runs the full **Streamlit → Spring → Python** architecture. The Python
service owns AI operations in both; Spring is a proxy only.

## Request flows

**Prepare a document** (`POST /api/documents`): validate PDF → extract text
(PyMuPDF, one-based pages) → token-aware chunk → embed (OpenAI, batched) → index
(Chroma). Returns `{document_id, filename, pages, chunks, skipped_pages,
status}`.

**Ask a question** (`POST /api/questions`): embed the query → retrieve nearest
chunks (Chroma cosine) → apply the distance/token grounding gate (abstain
locally if evidence is weak, *without* calling the model) → generate an answer
grounded in `<SOURCES>` (OpenAI Responses API) → validate citations against the
supplied source IDs. Returns `{answer, abstained, citations[]}` where each
citation carries only `source_id, filename, page, passage`.

## Security boundaries

- Only the Python service holds the OpenAI key (passed at runtime, never baked
  into an image). Spring and Streamlit-in-gateway-mode hold no key.
- Every service converts expected errors into safe messages — no stack traces,
  keys, prompts, internal paths, distances, or chunk IDs cross a boundary.
- Chroma data persists to a Docker named volume in the Compose stack; hosted
  Streamlit storage is ephemeral.

See [API.md](API.md) for the endpoint contract and [PERFORMANCE.md](PERFORMANCE.md)
for measurements.
