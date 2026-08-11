#!/usr/bin/env bash
# CurriculumIQ release validation. Run from the repo root inside the venv:
#   bash scripts/release_check.sh
# Exits non-zero on the first failed gate. Does NOT make live API calls.
set -euo pipefail

PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

echo "==> [1/6] pip check"
$PY -m pip check

echo "==> [2/6] compile / import validation"
$PY -m py_compile app.py src/*.py src/ingestion/*.py src/retrieval/*.py \
    src/generation/*.py src/evaluation/*.py tests/*.py
$PY -c "import app"  # import the Streamlit entrypoint (bare mode)

echo "==> [3/6] test suite"
$PY -m pytest -q

echo "==> [4/6] secret scan of tracked-style files"
if grep -RIn -E "sk-[A-Za-z0-9_-]{20,}" \
      --exclude=".env" --exclude-dir=.venv --exclude-dir=.git \
      --exclude-dir=.curriculumiq_data . ; then
  echo "SECRET SCAN FAILED: possible API key found above." >&2
  exit 1
fi
echo "secret scan clean"

echo "==> [5/6] offline evaluation (deterministic, no API)"
$PY -m src.evaluation.runner

echo "==> [6/6] ingestion + indexing smoke (offline, fake embeddings)"
$PY - <<'PYEOF'
import tempfile, pathlib
from src.ingestion.extractor import extract_document
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.indexer import IndexingService
from src.retrieval.vector_store import VectorStore
pdf = pathlib.Path("data/sample_documents/intro_to_algebra.pdf").read_bytes()
doc = extract_document(pdf, "intro_to_algebra.pdf")
store = VectorStore(tempfile.mkdtemp(), "release_smoke")
res = IndexingService(store, FakeEmbeddingProvider()).index_document(doc)
assert res.status == "indexed" and res.chunks_created > 0, res
print(f"ingestion+indexing OK: {res.chunks_created} chunks from {res.pages_processed} pages")
PYEOF

echo ""
echo "ALL RELEASE CHECKS PASSED (offline). For live model validation run:"
echo "  python -m src.evaluation.runner --live"
