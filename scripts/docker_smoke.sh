#!/usr/bin/env bash
# Deterministic Docker smoke test for the CurriculumIQ stack.
#   Health checks require NO live API call and are always run.
#   The upload/question flow is OPTIONAL (costs OpenAI credit) — enable with:
#     SMOKE_LIVE=1 bash scripts/docker_smoke.sh
# Assumes `docker compose up --build -d` has been run (or run it first).
set -euo pipefail

GATEWAY="http://localhost:8080"
STREAMLIT="http://localhost:8501"
SAMPLE="data/sample_documents/intro_to_algebra.pdf"

pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; exit 1; }

echo "==> Health checks (no API cost)"

# Spring gateway health also reports the Python service availability.
gw=$(curl -fsS "$GATEWAY/api/health") || fail "Spring gateway /api/health unreachable"
echo "     gateway: $gw"
echo "$gw" | grep -q '"gateway":"ok"' || fail "gateway not ok"
echo "$gw" | grep -q '"pythonService":"up"' || fail "python service not up (via gateway)"
pass "Spring gateway healthy and Python service reachable"

# Python health directly (service is internal; reach it via the compose network).
docker compose exec -T python-ai-service curl -fsS http://localhost:8000/health >/dev/null \
    && pass "Python service /health ok" || fail "Python /health failed"

# Spring Actuator.
curl -fsS "$GATEWAY/actuator/health" | grep -q '"status":"UP"' \
    && pass "Spring actuator UP" || fail "Spring actuator not UP"

# Streamlit HTTP.
curl -fsS "$STREAMLIT/_stcore/health" >/dev/null \
    && pass "Streamlit responding" || fail "Streamlit not responding"

if [ "${SMOKE_LIVE:-0}" != "1" ]; then
    echo "==> Live upload/question flow SKIPPED (set SMOKE_LIVE=1 to run; costs API credit)."
    echo "All health checks passed."
    exit 0
fi

echo "==> Live flow through Spring gateway (uses OpenAI credit)"
[ -f "$SAMPLE" ] || fail "sample PDF not found: $SAMPLE"

doc=$(curl -fsS -F "file=@${SAMPLE};type=application/pdf" "$GATEWAY/api/documents")
echo "     prepared: $doc"
doc_id=$(echo "$doc" | sed -n 's/.*"document_id":"\([^"]*\)".*/\1/p')
[ -n "$doc_id" ] || fail "no document_id returned"
pass "Uploaded + indexed sample PDF (document_id=$doc_id)"

supported=$(curl -fsS -X POST "$GATEWAY/api/questions" -H 'Content-Type: application/json' \
    -d "{\"document_id\":\"$doc_id\",\"question\":\"What is a variable in algebra?\"}")
echo "$supported" | grep -q '"source_id"' || fail "supported answer had no citation"
pass "Supported question returned at least one citation"

unsupported=$(curl -fsS -X POST "$GATEWAY/api/questions" -H 'Content-Type: application/json' \
    -d "{\"document_id\":\"$doc_id\",\"question\":\"Who won the 1998 World Cup final?\"}")
echo "$unsupported" | grep -q '"abstained":true' || fail "unsupported question did not abstain"
pass "Unsupported question abstained"

echo "All smoke checks passed (including live flow)."
