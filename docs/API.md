# CurriculumIQ — API Reference

Two HTTP surfaces share the same contract:

- **Python AI service** (FastAPI) — `http://localhost:8000` (internal in Docker).
- **Spring gateway** (proxy) — `http://localhost:8080`, paths under `/api`.

Responses expose only safe, student-facing fields. Errors return
`{"detail": "..."}` with an appropriate status — never a stack trace, key,
prompt, internal path, distance, or chunk ID.

## Health

**Python:** `GET /health`
```json
{ "status": "ok", "service": "curriculumiq-python" }
```

**Gateway:** `GET /api/health` (reports the downstream Python status too)
```json
{ "gateway": "ok", "pythonService": "up" }
```

The gateway also exposes Spring Actuator at `GET /actuator/health`.

## Prepare a document

`POST /api/documents` — `multipart/form-data`, field `file` = one PDF.

```bash
curl -F "file=@data/sample_documents/intro_to_algebra.pdf;type=application/pdf" \
     http://localhost:8080/api/documents
```

**200 OK**
```json
{
  "document_id": "doc_f34d92bc917887dd",
  "filename": "intro_to_algebra.pdf",
  "pages": 6,
  "chunks": 6,
  "skipped_pages": [],
  "status": "ready"
}
```

Errors: `422` (not a PDF / empty / no readable text), `413` (too large),
`502` (embedding/index failure), `503` (AI service unavailable).

## Ask a question

`POST /api/questions` — `application/json`.

```bash
curl -X POST http://localhost:8080/api/questions \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"doc_f34d92bc917887dd","question":"Explain the quadratic formula."}'
```

**200 OK — grounded answer**
```json
{
  "answer": "The quadratic formula gives the roots of a quadratic equation [S1].",
  "abstained": false,
  "citations": [
    { "source_id": "S1", "filename": "intro_to_algebra.pdf",
      "page": 5, "passage": "The quadratic formula gives the roots..." }
  ]
}
```

**200 OK — abstention** (question not supported by the curriculum)
```json
{ "answer": "I could not find enough information in the available curriculum materials.",
  "abstained": true, "citations": [] }
```

Validation: `question` must be 1–1000 characters (`422` otherwise);
`document_id` is required. Generation/retrieval failures return `502`; an
unreachable AI service returns `503`.

## Notes

- `document_id` is derived from the file checksum, so re-uploading the same PDF
  is idempotent (it re-indexes in place).
- The gateway forwards these requests verbatim; it adds timeouts, health
  reporting, and safe error mapping but no AI logic.
