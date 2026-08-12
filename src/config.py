"""Application configuration.

Values come from environment variables (loaded from a local .env if present).
No secrets live in source. Day 1 does not call any external API; the OpenAI
settings are read here only so later milestones can reuse this module.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent

# --- Ingestion limits (Day 1) ---
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# A page with fewer than this many characters is flagged as low/no text
# (scanned or image-only pages typically extract to near-empty strings).
MIN_CHARS_PER_PAGE: int = int(os.getenv("MIN_CHARS_PER_PAGE", "3"))

# Upper bound on pages per PDF — protects a hosted instance from a huge upload
# that would be slow/expensive to extract, chunk, and embed.
MAX_PAGE_COUNT: int = int(os.getenv("MAX_PAGE_COUNT", "300"))

# --- OpenAI config ---
# OPENAI_API_KEY is only needed for indexing/search (Milestone 2+). Its absence
# must never break PDF extraction or crash the app.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Service architecture (Module 13/14) ---
# "direct"  = Streamlit calls the Python domain services in-process (default;
#             this is what Streamlit Cloud uses).
# "gateway" = Streamlit calls the Spring gateway, which proxies the Python API
#             (used by the Docker Compose demo).
SERVICE_MODE: str = os.getenv("SERVICE_MODE", "direct").strip().lower()
SPRING_GATEWAY_URL: str = os.getenv("SPRING_GATEWAY_URL", "http://localhost:8080")


def gateway_mode() -> bool:
    return SERVICE_MODE == "gateway"

# --- RAG / retrieval config (Milestone 2) ---
# Local, untracked directory for the Chroma vector database (see .gitignore).
CHROMA_PERSIST_DIRECTORY: str = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", ".curriculumiq_data/chroma"
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "curriculumiq")

RAG_CHUNK_SIZE_TOKENS: int = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "700"))
RAG_CHUNK_OVERLAP_TOKENS: int = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "100"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
# Safe upper bound for the top-k selector in the UI.
RAG_MAX_TOP_K: int = int(os.getenv("RAG_MAX_TOP_K", "20"))

# --- Generation / tutor config (Milestone 3) ---
# Max retrieved chunks used as answer context, and the token budget for them.
RAG_MAX_CONTEXT_CHUNKS: int = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "5"))
RAG_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2000"))
# Retrieval-quality gate. Chroma uses COSINE DISTANCE, so LOWER is closer:
# a result is accepted only if its distance <= RAG_MAX_DISTANCE. Raising this
# value accepts weaker matches; lowering it makes the tutor abstain sooner.
# Distances vary by embedding model, so this is deliberately configurable.
RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.7"))
# Bounded chat history: number of recent messages kept for follow-up context.
RAG_HISTORY_MESSAGE_LIMIT: int = int(os.getenv("RAG_HISTORY_MESSAGE_LIMIT", "6"))
# Reject questions longer than this many characters.
RAG_MAX_QUESTION_LENGTH: int = int(os.getenv("RAG_MAX_QUESTION_LENGTH", "1000"))


# --- Public-demo cost protection (Milestone 6) ---
# Cap the number of tutor questions per browser session so a public deployment
# cannot run up unbounded API cost. Set to 0 to disable the cap.
MAX_QUESTIONS_PER_SESSION: int = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))
# Optional shared access code gating the app. Empty = no gate (local dev).
APP_ACCESS_CODE: str = os.getenv("APP_ACCESS_CODE", "")

# Internal diagnostics panel (document IDs, model config, vector-store state).
# Hidden by default; opt in for local debugging only. Never enable in public.
SHOW_DEVELOPER_DETAILS: bool = os.getenv("SHOW_DEVELOPER_DETAILS", "false").strip().lower() in (
    "1", "true", "yes", "on",
)


def has_openai_key() -> bool:
    """True if an OpenAI API key is configured. Never logs or returns the key."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())


def access_required() -> bool:
    """True if an access code is configured (gate enabled)."""
    return bool(APP_ACCESS_CODE and APP_ACCESS_CODE.strip())


def verify_access_code(candidate: str) -> bool:
    """Constant-time check of a submitted access code. Never logs the value.

    Returns True when no gate is configured (local dev stays open).
    """
    if not access_required():
        return True
    if not candidate:
        return False
    import hmac

    return hmac.compare_digest(candidate.strip(), APP_ACCESS_CODE.strip())


def question_limit_reached(asked: int) -> bool:
    """True if the per-session question cap has been hit (0 = unlimited)."""
    if MAX_QUESTIONS_PER_SESSION <= 0:
        return False
    return asked >= MAX_QUESTIONS_PER_SESSION


def has_chat_model() -> bool:
    """True if a chat model is configured for answer generation."""
    return bool(OPENAI_CHAT_MODEL and OPENAI_CHAT_MODEL.strip())


def generation_enabled() -> bool:
    """Generation needs both an API key and a configured chat model."""
    return has_openai_key() and has_chat_model()


def validate_generation_config(
    max_context_chunks: int,
    max_context_tokens: int,
    max_distance: float,
    history_message_limit: int,
    max_question_length: int,
) -> None:
    """Reject nonsensical tutor settings. Raises ``ValueError`` on bad input."""
    if max_context_chunks <= 0:
        raise ValueError("RAG_MAX_CONTEXT_CHUNKS must be a positive integer.")
    if max_context_tokens <= 0:
        raise ValueError("RAG_MAX_CONTEXT_TOKENS must be a positive integer.")
    if max_distance <= 0:
        raise ValueError("RAG_MAX_DISTANCE must be greater than 0.")
    if history_message_limit < 0:
        raise ValueError("RAG_HISTORY_MESSAGE_LIMIT must be zero or positive.")
    if max_question_length <= 0:
        raise ValueError("RAG_MAX_QUESTION_LENGTH must be a positive integer.")


def validate_chunk_config(chunk_size_tokens: int, overlap_tokens: int) -> None:
    """Reject nonsensical chunk settings. Raises ``ValueError`` on bad input."""
    if chunk_size_tokens <= 0:
        raise ValueError("RAG_CHUNK_SIZE_TOKENS must be a positive integer.")
    if overlap_tokens < 0:
        raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be zero or positive.")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            "RAG_CHUNK_OVERLAP_TOKENS must be smaller than RAG_CHUNK_SIZE_TOKENS."
        )
