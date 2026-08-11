# Deploying CurriculumIQ

Target: **Streamlit Community Cloud** (the app is a single `streamlit run app.py`
entrypoint). Any host that can run Streamlit and set environment variables /
secrets works the same way.

## Configuration model

The app reads all configuration from environment variables (see `src/config.py`).
Two supported sources:

- **Local:** a `.env` file (loaded via `python-dotenv`) **or** a local
  `.streamlit/secrets.toml`. Both are git-ignored.
- **Deployed (Streamlit Cloud):** the app's **Secrets** box in the dashboard.
  Streamlit exposes top-level secrets as environment variables, which is how the
  app picks them up — no code change between local and deployed.

Required for indexing / search / tutor (extraction & chunking work without them):

| Key | Example | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sk-…` | Required for AI features. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | |
| `APP_ACCESS_CODE` | `choose-a-code` | **Recommended for public demos** — gates the app behind a shared code. Empty = open. |
| `MAX_QUESTIONS_PER_SESSION` | `20` | Per-session tutor question cap (cost control; `0` = unlimited). |

Optional tuning is documented in `README.md` and `.streamlit/secrets.toml.example`.
**Never expose these values** — set them only in the host's secrets UI.

**Startup validation:** if a key/model is absent, the app does not crash — it
shows a clear banner and disables indexing, search, and the tutor while keeping
PDF extraction and chunking available.

## Steps (Streamlit Community Cloud)

1. Push this repository to GitHub (ensure `.env` and `.streamlit/secrets.toml`
   are **not** committed — they are git-ignored; verify with
   `git status --porcelain`).
2. On https://share.streamlit.io, create an app pointing at this repo, branch,
   and `app.py`. Use Python 3.11.
3. Open **Advanced settings → Secrets** and paste the keys from
   `.streamlit/secrets.toml.example`, filling in your real `OPENAI_API_KEY`.
4. Deploy. Streamlit installs `requirements.txt` (pinned, verified on 3.11).
5. When the app boots, open the URL and run the demo walkthrough in `README.md`.

## Persistence caveat (important — do not overstate)

The vector store (Chroma) persists to a **local directory**
(`CHROMA_PERSIST_DIRECTORY`, default `.curriculumiq_data/chroma`) and uploaded
files are processed in memory. On Streamlit Community Cloud the filesystem is
**ephemeral**: the container can restart and wipe local data, so **indexed
documents and any local vector data may not survive a restart**. There is no
external/persistent storage backend implemented. Users may need to re-upload and
re-index after a restart. Do not claim durable persistence that does not exist.

## Pre-deploy validation

Run the offline release gate before every deploy:

```bash
bash scripts/release_check.sh
```

Optionally validate real model quality (spends credits):

```bash
python -m src.evaluation.runner --live
```

## Security

- No API key is ever entered through the app UI; keys come only from the
  environment / secrets.
- Errors shown to users are safe messages — no stack traces, keys, prompts, or
  document contents. Provider failures (auth / quota / timeout / model access)
  map to friendly text; technical detail stays in server logs (type names only).
- Generated evaluation reports contain model names but no secrets.
