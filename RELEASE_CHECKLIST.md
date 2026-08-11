# CurriculumIQ — Release Checklist

Run through this before tagging a release or deploying. The offline items are
automated by `bash scripts/release_check.sh`; the UI items are manual.

## Automated (scripts/release_check.sh)

- [ ] **Clean installation** — fresh venv: `python3.11 -m venv .venv && pip install -r requirements.txt`
- [ ] **Dependency validation** — `python -m pip check` → no broken requirements
- [ ] **Import validation** — `python -c "import app"` succeeds
- [ ] **Tests** — `pytest -q` all green
- [ ] **Secret scan** — no `sk-…` keys in tracked files
- [ ] **Evaluation report** — `python -m src.evaluation.runner` writes
      `reports/evaluation/offline/latest.{json,csv,md}` (offline = pipeline validation)
- [ ] **Sample PDF ingestion + indexing** — smoke step passes

## Manual UI walkthrough (`streamlit run app.py`)

- [ ] **Deployment boot** — app starts, no errors, privacy notice visible
- [ ] **Sample PDF ingestion** — upload `data/sample_documents/intro_to_algebra.pdf`,
      Process → summary shows 6 pages with correct counts
- [ ] **Indexing** — click **Index document** → success with chunk count
      (requires a funded `OPENAI_API_KEY`)
- [ ] **Supported question** — ask "Explain the quadratic formula" → grounded
      answer with `[S#]` citation
- [ ] **Citation opening/inspection** — expand **Sources**, open a supporting
      passage; filename + page come from retrieval metadata
- [ ] **Unsupported-question abstention** — ask "Who wrote Pride and Prejudice?"
      → the fallback, no citations
- [ ] **Clear-conversation** — clears chat but keeps the indexed document
- [ ] **Input gating** — before indexing, the tutor input is disabled with a nudge
- [ ] **Mobile-width check** — at ~375px width the page has no horizontal scroll
      and controls remain usable

## Live model validation (optional; spends credits)

- [ ] `python -m src.evaluation.runner --live` → separate live reports; inspect
      failed items; record embedding + chat models used
- [ ] On `credit_balance_exhausted` / auth / model errors: the CLI exits with a
      documented non-zero code and one safe line (no traceback) — status PARTIAL

## Deploy

- [ ] `.env` and `.streamlit/secrets.toml` are NOT committed (`git status`)
- [ ] Secrets set in the host dashboard (see `DEPLOYMENT.md`)
- [ ] Public URL opened and the walkthrough above re-run against it
