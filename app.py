"""CurriculumIQ — Streamlit interface (Milestones 1–3).

Upload a curriculum PDF, extract text page by page, chunk it, index it into a
local Chroma vector store, run semantic search, and ask an AI tutor questions
that are answered strictly from your curriculum with verifiable [S#] citations.
Extraction and chunking work with or without an OpenAI key; indexing, search,
and the tutor require a key.
"""

from __future__ import annotations

import logging

import streamlit as st

from src import config
from src.generation.provider import OpenAIAnswerProvider
from src.generation.session import bound_history, record_feedback
from src.generation.tutor import TutorError, TutorService
from src.ingestion.chunker import chunk_document
from src.ingestion.extractor import ExtractionError, extract_document
from src.ingestion.validator import ValidationError, validate_pdf
from src.retrieval.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from src.retrieval.indexer import IndexingError, IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore, VectorStoreError

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="CurriculumIQ", page_icon="📘", layout="centered")


# --- Cached shared resources (created once, reused across reruns) -----------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore(config.CHROMA_PERSIST_DIRECTORY, config.CHROMA_COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_embedding_provider(model: str):
    # Cached per model name; one OpenAI client reused for all requests.
    return OpenAIEmbeddingProvider(model=model, api_key=config.OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_answer_provider(model: str):
    # Cached per model name; one OpenAI client reused for all generations.
    return OpenAIAnswerProvider(model=model, api_key=config.OPENAI_API_KEY)


def build_tutor() -> TutorService:
    retriever = RetrievalService(
        get_vector_store(), get_embedding_provider(config.OPENAI_EMBEDDING_MODEL)
    )
    provider = get_answer_provider(config.OPENAI_CHAT_MODEL)
    return TutorService(retriever, provider)


# --- Session helpers --------------------------------------------------------
def _clear_search() -> None:
    st.session_state.pop("search_results", None)
    st.session_state.pop("search_error", None)


def _clear_chat() -> None:
    """Clear conversation + feedback. Never touches indexed documents."""
    st.session_state.pop("chat_history", None)
    st.session_state.pop("feedback", None)


def _reset() -> None:
    """Clear the current document and all derived UI state (incl. chat)."""
    for key in ("document", "processed_name", "chunks", "chunk_warnings", "index_result"):
        st.session_state.pop(key, None)
    _clear_search()
    _clear_chat()


def _process(file_bytes: bytes, filename: str) -> None:
    """Validate, extract, and chunk; store results (or a safe error)."""
    try:
        validate_pdf(file_bytes, filename)
        doc = extract_document(file_bytes, filename)
        chunks, warnings = chunk_document(doc)
        _reset()
        st.session_state["document"] = doc
        st.session_state["processed_name"] = filename
        st.session_state["chunks"] = chunks
        st.session_state["chunk_warnings"] = warnings
        st.session_state.pop("error", None)
    except (ValidationError, ExtractionError) as exc:
        _reset()
        st.session_state["error"] = str(exc)


def _render_answer_extras(answer) -> None:
    """Render sources, latency, warnings, and feedback controls for one answer."""
    if answer is None:
        return
    if answer.abstained:
        st.caption("⚠️ The tutor abstained — the curriculum did not contain enough "
                   "evidence to answer.")
    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")

    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})"):
            for c in answer.citations:
                score = f" · distance {c.distance:.4f}" if c.distance is not None else ""
                st.markdown(f"**[{c.source_id}] {c.filename}, page {c.page_number}**{score}")
                # Non-expander presentation: Streamlit forbids nesting an
                # expander inside the outer "Sources" expander.
                with st.container(border=True):
                    st.caption("Supporting passage")
                    st.text(c.passage)
                    st.caption(f"Chunk ID: `{c.chunk_id}`")

    meta = f"⏱ {answer.latency_seconds:.2f}s (retrieval {answer.retrieval_seconds:.2f}s, "
    meta += f"generation {answer.generation_seconds:.2f}s) · model `{answer.model}`"
    st.caption(meta)

    # Feedback (session-only for now).
    fb_store = st.session_state.setdefault("feedback", {})
    existing = fb_store.get(answer.answer_id)
    fc1, fc2, fc3 = st.columns([1, 1, 3])
    if fc1.button("👍 Helpful", key=f"fb_up_{answer.answer_id}"):
        record_feedback(fb_store, answer.answer_id, True,
                        st.session_state.get(f"fb_reason_{answer.answer_id}", ""))
        st.rerun()
    if fc2.button("👎 Not helpful", key=f"fb_down_{answer.answer_id}"):
        record_feedback(fb_store, answer.answer_id, False,
                        st.session_state.get(f"fb_reason_{answer.answer_id}", ""))
        st.rerun()
    fc3.text_input("Optional reason", key=f"fb_reason_{answer.answer_id}",
                   label_visibility="collapsed", placeholder="Optional feedback reason")
    if existing is not None:
        verdict = "helpful" if existing["helpful"] else "not helpful"
        st.caption(f"Feedback recorded: {verdict}." + (f" “{existing['reason']}”" if existing['reason'] else ""))


def _handle_question(question: str, document, only_this_doc: bool) -> None:
    """Run one tutor turn and append user + assistant messages to history."""
    history = st.session_state.setdefault("chat_history", [])

    # Public-demo cost protection: cap tutor questions per session.
    asked = st.session_state.get("questions_asked", 0)
    if config.question_limit_reached(asked):
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": (
                f"⚠️ You've reached this session's limit of "
                f"{config.MAX_QUESTIONS_PER_SESSION} questions. Refresh the page to "
                "start a new session."
            ),
            "answer": None,
        })
        return

    prior = [(m["role"], m["content"]) for m in history]
    history.append({"role": "user", "content": question})
    bounded = bound_history(prior, config.RAG_HISTORY_MESSAGE_LIMIT)
    doc_filter = document.document_id if only_this_doc else None
    try:
        answer = build_tutor().answer(question, document_id=doc_filter, history=bounded)
        history.append({"role": "assistant", "content": answer.answer_text, "answer": answer})
        st.session_state["questions_asked"] = asked + 1
    except TutorError as exc:
        # A failed call doesn't count against the session budget.
        history.append({"role": "assistant", "content": f"⚠️ {exc}", "answer": None})


# --- Header -----------------------------------------------------------------
st.title("📘 CurriculumIQ")
st.caption("Ask questions about your curriculum and receive clear answers with verifiable sources.")

# --- Access gate (optional; enabled only when APP_ACCESS_CODE is configured) -
if config.access_required() and not st.session_state.get("access_granted"):
    st.info("This demo is access-protected. Enter the access code to continue.", icon="🔒")
    code = st.text_input("Access code", type="password", key="access_code_input")
    if st.button("Enter", type="primary"):
        if config.verify_access_code(code):
            st.session_state["access_granted"] = True
            st.session_state.pop("access_code_input", None)
            st.rerun()
        else:
            st.error("Incorrect access code.", icon="🚫")
    st.stop()  # block the rest of the app until access is granted

st.info(
    "🔒 **Privacy:** Your PDF is processed locally in memory. When you index a "
    "document, its text chunks are sent to OpenAI to compute embeddings and are "
    "stored only in a local Chroma database on this machine.",
    icon=None,
)

key_present = config.has_openai_key()
if not key_present:
    st.warning(
        "No `OPENAI_API_KEY` is configured. **PDF extraction and chunking work "
        "normally**, but indexing and semantic search are disabled. Add a key to "
        "`.env` (see README) to enable them.",
        icon="🔑",
    )

# --- Upload -----------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a curriculum PDF",
    type=["pdf"],
    help=f"Digital (text-based) PDFs up to {config.MAX_FILE_SIZE_MB} MB.",
)
st.caption(f"Maximum file size: {config.MAX_FILE_SIZE_MB} MB. Text-based PDFs only.")

# A new upload invalidates any previously processed document + search state.
if uploaded is not None and st.session_state.get("processed_name") not in (None, uploaded.name):
    _reset()
    st.session_state.pop("error", None)

col_process, col_clear = st.columns(2)
with col_process:
    process_clicked = st.button(
        "Process document",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )
with col_clear:
    if st.button("Clear document", use_container_width=True):
        _reset()
        st.session_state.pop("error", None)
        st.rerun()

if process_clicked and uploaded is not None:
    with st.spinner("Processing document…"):
        _process(uploaded.getvalue(), uploaded.name)

# --- Error state ------------------------------------------------------------
if st.session_state.get("error"):
    st.error(st.session_state["error"], icon="🚫")

# --- Results ----------------------------------------------------------------
doc = st.session_state.get("document")
if doc is not None:
    chunks = st.session_state.get("chunks", [])
    chunk_warnings = st.session_state.get("chunk_warnings", [])

    st.success(f"Processed **{doc.filename}** successfully.", icon="✅")

    st.subheader("Document summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", f"{doc.page_count:,}")
    c2.metric("Words", f"{doc.total_word_count:,}")
    c3.metric("Characters", f"{doc.total_character_count:,}")
    c4, c5 = st.columns(2)
    c4.metric("File size", f"{doc.file_size_mb:.2f} MB")
    c5.metric("Filename", doc.filename)

    if doc.extraction_warnings:
        for warning in doc.extraction_warnings:
            st.warning(warning, icon="⚠️")

    # --- Chunking summary ---------------------------------------------------
    st.subheader("Chunking summary")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Chunks", f"{len(chunks):,}")
    cc2.metric("Chunk size (tokens)", f"{config.RAG_CHUNK_SIZE_TOKENS}")
    cc3.metric("Overlap (tokens)", f"{config.RAG_CHUNK_OVERLAP_TOKENS}")
    for warning in chunk_warnings:
        st.warning(warning, icon="⚠️")

    # --- Indexing -----------------------------------------------------------
    st.subheader("Indexing")
    if not key_present:
        st.info("Indexing is disabled until an `OPENAI_API_KEY` is configured.", icon="🔑")
    elif not chunks:
        st.info("This document produced no chunks, so there is nothing to index.")
    else:
        try:
            store = get_vector_store()
            already = store.has_document(doc.document_id)
        except VectorStoreError as exc:
            store = None
            already = False
            st.error(str(exc), icon="🚫")

        if store is not None:
            if already:
                st.warning(
                    "This document is already indexed. Re-indexing replaces its "
                    "existing chunks.",
                    icon="♻️",
                )
            label = "Re-index document" if already else "Index document"
            if st.button(label, type="primary"):
                with st.spinner("Generating embeddings and indexing…"):
                    try:
                        provider = get_embedding_provider(config.OPENAI_EMBEDDING_MODEL)
                        service = IndexingService(store, provider)
                        result = service.index_document(doc, reindex=already)
                        st.session_state["index_result"] = result
                        _clear_search()
                    except (IndexingError, EmbeddingError, VectorStoreError) as exc:
                        st.session_state["index_result"] = None
                        st.session_state["index_error"] = str(exc)

            if st.session_state.get("index_error"):
                st.error(st.session_state.pop("index_error"), icon="🚫")

            result = st.session_state.get("index_result")
            if result is not None and result.document_id == doc.document_id:
                if result.status in ("indexed", "reindexed"):
                    st.success(
                        f"{result.status.capitalize()} **{result.chunks_created}** "
                        f"chunks from {result.pages_processed} page(s) in "
                        f"{result.duration_seconds:.2f}s "
                        f"(model: `{result.embedding_model}`).",
                        icon="✅",
                    )
                elif result.status == "duplicate":
                    st.info("Document already indexed — use Re-index to replace it.", icon="ℹ️")
                for warning in result.warnings:
                    st.caption(f"⚠️ {warning}")

    # --- Semantic search ----------------------------------------------------
    st.subheader("Semantic Search")
    if not key_present:
        st.info("Semantic search is disabled until an `OPENAI_API_KEY` is configured.", icon="🔑")
    else:
        query = st.text_input("Search your curriculum", placeholder="e.g. What is a quadratic function?")
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            top_k = st.slider("Results (top-k)", 1, config.RAG_MAX_TOP_K, config.RAG_TOP_K)
        with sc2:
            only_this = st.checkbox("Search only this document", value=True)
        st.caption("Scores are cosine distances — **lower means a closer match**.")

        if st.button("Search", disabled=not query.strip()):
            _clear_search()
            try:
                store = get_vector_store()
                provider = get_embedding_provider(config.OPENAI_EMBEDDING_MODEL)
                retriever = RetrievalService(store, provider)
                doc_filter = doc.document_id if only_this else None
                st.session_state["search_results"] = retriever.search(
                    query, top_k=top_k, document_id=doc_filter
                )
            except (ValueError, EmbeddingError, VectorStoreError) as exc:
                st.session_state["search_error"] = str(exc)

        if st.session_state.get("search_error"):
            st.error(st.session_state["search_error"], icon="🚫")

        results = st.session_state.get("search_results")
        if results is not None:
            if not results:
                st.info("No matching passages found. Try indexing the document first, "
                        "or rephrasing your query.")
            for r in results:
                score = f" · distance {r.distance:.4f}" if r.distance is not None else ""
                st.markdown(f"**#{r.rank} — {r.filename}, page {r.page_number}**{score}")
                preview = r.text[:300] + ("…" if len(r.text) > 300 else "")
                st.write(preview)
                with st.expander("Full chunk text"):
                    st.text(r.text)
                    st.caption(f"Chunk ID: `{r.chunk_id}`")
                st.divider()

    # --- AI Tutor -----------------------------------------------------------
    st.subheader("🎓 AI Tutor")
    st.caption(
        "Answers are grounded strictly in your **indexed** curriculum and cite "
        "sources as [S1], [S2], … This is a study aid, not an authority — verify "
        "high-stakes information against the cited pages or your lecturer."
    )
    if not config.generation_enabled():
        st.info(
            "The AI Tutor needs an `OPENAI_API_KEY` and a chat model "
            "(`OPENAI_CHAT_MODEL`). Extraction and search still work without them.",
            icon="🔑",
        )
    else:
        tsc1, tsc2 = st.columns([3, 2])
        with tsc1:
            only_this_doc = st.checkbox("Answer only from this document", value=True, key="tutor_scope")
        with tsc2:
            if st.button("Clear conversation", use_container_width=True):
                _clear_chat()
                st.rerun()
        st.caption(f"Scope: **{doc.filename}**" if only_this_doc else "Scope: **all indexed documents**")

        # The tutor can only answer from indexed content. Gate the input on
        # index status so students aren't met with confusing abstentions.
        can_ask = True
        try:
            store = get_vector_store()
            if only_this_doc:
                can_ask = store.has_document(doc.document_id)
            else:
                can_ask = store.count() > 0
        except VectorStoreError:
            can_ask = False  # tutor path still surfaces any real store error safely

        if not can_ask:
            st.info("Index a document first (**Index document** above) so the tutor "
                    "has curriculum to draw from.", icon="ℹ️")

        with st.expander("Example questions"):
            st.markdown(
                "- What is a variable?\n"
                "- Explain the quadratic formula.\n"
                "- How does a linear equation graph?"
            )

        # Public-demo question budget (visible remaining count).
        limit = config.MAX_QUESTIONS_PER_SESSION
        asked = st.session_state.get("questions_asked", 0)
        remaining = max(limit - asked, 0) if limit > 0 else None
        at_limit = config.question_limit_reached(asked)
        if remaining is not None:
            if at_limit:
                st.warning(f"Session question limit reached ({limit}). Refresh to start "
                           "a new session.", icon="🚦")
            else:
                st.caption(f"Questions remaining this session: **{remaining}** of {limit}")

        # Render conversation so far.
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                _render_answer_extras(msg.get("answer"))

        input_enabled = can_ask and not at_limit
        question = st.chat_input(
            "Ask about your curriculum…" if input_enabled
            else ("Session limit reached — refresh to continue…" if at_limit
                  else "Index a document to enable the tutor…"),
            disabled=not input_enabled,
        )
        if question:
            with st.spinner("Thinking…"):
                _handle_question(question, doc, only_this_doc)
            st.rerun()

    # --- Developer details --------------------------------------------------
    with st.expander("🛠 Developer details"):
        st.write({
            "document_id": doc.document_id,
            "chunk_count": len(chunks),
            "embedding_model": config.OPENAI_EMBEDDING_MODEL,
            "chat_model": config.OPENAI_CHAT_MODEL,
            "chunk_size_tokens": config.RAG_CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.RAG_CHUNK_OVERLAP_TOKENS,
            "top_k_default": config.RAG_TOP_K,
            "max_context_chunks": config.RAG_MAX_CONTEXT_CHUNKS,
            "max_context_tokens": config.RAG_MAX_CONTEXT_TOKENS,
            "max_distance (lower=closer)": config.RAG_MAX_DISTANCE,
            "history_message_limit": config.RAG_HISTORY_MESSAGE_LIMIT,
            "api_key_configured": key_present,
            "generation_enabled": config.generation_enabled(),
        })
        if key_present:
            try:
                store = get_vector_store()
                st.write({
                    "chroma_collection": config.CHROMA_COLLECTION_NAME,
                    "chroma_total_chunks": store.count(),
                    "this_document_indexed": store.has_document(doc.document_id),
                    "indexed_document_ids": store.list_document_ids(),
                })
            except VectorStoreError as exc:
                st.caption(f"Chroma status unavailable: {exc}")

# --- Roadmap ----------------------------------------------------------------
st.divider()
with st.expander("What's next?"):
    st.markdown(
        "This is **Milestone 3: Grounded AI tutor**. You can now ask questions "
        "and get answers drawn strictly from your indexed curriculum, with "
        "verifiable [S#] citations and safe abstention when evidence is missing.\n\n"
        "Coming next:\n\n"
        "- A citation-quality **evaluation harness** over a question set\n"
        "- Hardening, cost/performance controls, and deployment\n\n"
        "Every answer already carries trusted filename/page citations and "
        "retrieval/generation metadata — the foundation the evaluation layer uses."
    )
