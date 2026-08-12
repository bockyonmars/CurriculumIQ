"""CurriculumIQ — a calm, source-grounded study tutor (Streamlit UI).

The interface guides a first-time user through three stages —
**Choose PDF → Prepare curriculum → Ask questions** — while the RAG backend
(validation, extraction, chunking, embeddings, Chroma indexing, retrieval,
grounded generation, citations, abstention) is unchanged. Implementation detail
(embeddings, indexing, top-k, cosine distance, chunk/document IDs, model names,
latency) is kept out of the main journey and only shown under
``SHOW_DEVELOPER_DETAILS``.
"""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

import streamlit as st

from src import config
from src.generation.provider import OpenAIAnswerProvider
from src.generation.session import bound_history, record_feedback
from src.generation.tutor import TutorError, TutorService
from src.ingestion.chunker import chunk_document
from src.ingestion.extractor import ExtractionError, extract_document
from src.ingestion.validator import ValidationError, validate_pdf
from src.models import SourceCitation, TutorAnswer
from src.retrieval.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from src.retrieval.indexer import IndexingError, IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore, VectorStoreError
from src.service_client import GatewayClient, GatewayError

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="CurriculumIQ", page_icon="📘", layout="centered")

EXAMPLE_QUESTIONS = [
    "Summarize the main topics in this document.",
    "What are the key concepts and definitions?",
    "Explain the first topic in more detail.",
]


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


@st.cache_resource(show_spinner=False)
def get_gateway_client() -> GatewayClient:
    return GatewayClient(config.SPRING_GATEWAY_URL)


def _current_view():
    """Normalized view of the prepared document for both service modes.

    In ``gateway`` mode the fields come from the gateway's JSON response; in
    ``direct`` mode they come from the in-process ExtractedDocument. Returns
    ``None`` when no document is prepared.
    """
    gw = st.session_state.get("gateway_doc")
    if gw is not None:
        return SimpleNamespace(
            document_id=gw.get("document_id", ""),
            filename=gw.get("filename", "document.pdf"),
            page_count=int(gw.get("pages", 0)),
            chunk_count=int(gw.get("chunks", 0)),
            skipped_pages=list(gw.get("skipped_pages", [])),
        )
    doc = st.session_state.get("document")
    if doc is not None:
        return SimpleNamespace(
            document_id=doc.document_id,
            filename=doc.filename,
            page_count=doc.page_count,
            chunk_count=len(st.session_state.get("chunks", [])),
            skipped_pages=_skipped_pages(doc),
        )
    return None


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
    for key in ("document", "processed_name", "chunks", "chunk_warnings", "indexed",
                "gateway_doc"):
        st.session_state.pop(key, None)
    _clear_search()
    _clear_chat()


def _skipped_pages(doc) -> list:
    """Page numbers with no readable text (aggregated for a single warning)."""
    return [p.page_number for p in doc.pages if len(p.text.strip()) < config.MIN_CHARS_PER_PAGE]


def _run_preparation(file_bytes: bytes, filename: str) -> None:
    """One user action: validate → extract → chunk → (embed + index).

    Reuses the existing backend functions and their safe errors; presents the
    sequence as a single friendly progress flow. On success, advances to the
    ready state; on any handled error, stays put with a clear message.
    In ``gateway`` mode the same flow is proxied through the Spring gateway.
    """
    if config.gateway_mode():
        _run_preparation_gateway(file_bytes, filename)
        return
    try:
        with st.status("Preparing your curriculum…", expanded=True) as status:
            status.update(label="Reading PDF")
            validate_pdf(file_bytes, filename)
            doc = extract_document(file_bytes, filename)

            status.update(label="Organizing content")
            chunks, warnings = chunk_document(doc)
            if not chunks:
                raise ExtractionError(
                    "No readable text was found. This looks like a scanned or "
                    "image-only PDF, which isn't supported yet. Please upload a "
                    "text-based PDF."
                )

            indexed = False
            if config.generation_enabled():
                status.update(label="Preparing your tutor")
                store = get_vector_store()
                provider = get_embedding_provider(config.OPENAI_EMBEDDING_MODEL)
                already = store.has_document(doc.document_id)
                IndexingService(store, provider).index_document(doc, reindex=already)
                indexed = True

            status.update(label="Ready", state="complete")

        _reset()
        st.session_state["document"] = doc
        st.session_state["processed_name"] = filename
        st.session_state["chunks"] = chunks
        st.session_state["chunk_warnings"] = warnings
        st.session_state["indexed"] = indexed
        st.session_state.pop("error", None)
        st.rerun()
    except (ValidationError, ExtractionError) as exc:
        _reset()
        st.session_state["error"] = str(exc)
    except (IndexingError, EmbeddingError, VectorStoreError) as exc:
        _reset()
        st.session_state["error"] = str(exc)


def _run_preparation_gateway(file_bytes: bytes, filename: str) -> None:
    """Prepare the document by proxying through the Spring gateway."""
    try:
        with st.status("Preparing your curriculum…", expanded=True) as status:
            status.update(label="Reading PDF")
            status.update(label="Preparing your tutor")
            data = get_gateway_client().prepare_document(file_bytes, filename)
            status.update(label="Ready", state="complete")
        _reset()
        st.session_state["gateway_doc"] = data
        st.session_state["indexed"] = True
        st.session_state.pop("error", None)
        st.rerun()
    except GatewayError as exc:
        _reset()
        st.session_state["error"] = str(exc)


def _answer_from_gateway(question: str, document_id: str, data: dict) -> TutorAnswer:
    """Adapt a gateway /api/questions response into a TutorAnswer for rendering."""
    cites = [
        SourceCitation(
            source_id=c.get("source_id", f"S{i + 1}"), chunk_id="", document_id=document_id,
            filename=c.get("filename", ""), page_number=int(c.get("page", 1)),
            passage=c.get("passage", ""), distance=None, rank=i + 1,
        )
        for i, c in enumerate(data.get("citations", []))
    ]
    return TutorAnswer(
        answer_id=uuid.uuid4().hex, question=question, answer_text=data.get("answer", ""),
        citations=cites, retrieved_sources=cites, abstained=bool(data.get("abstained", False)),
        model="", retrieval_seconds=0.0, generation_seconds=0.0, latency_seconds=0.0,
    )


def _handle_question(question: str, document_id: str) -> None:
    """Run one tutor turn and append user + assistant messages to history."""
    history = st.session_state.setdefault("chat_history", [])

    # Public-demo cost protection: cap tutor questions per session.
    asked = st.session_state.get("questions_asked", 0)
    if config.question_limit_reached(asked):
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": (
                f"You've reached this session's limit of "
                f"{config.MAX_QUESTIONS_PER_SESSION} questions. Refresh the page to "
                "start a new session."
            ),
            "answer": None,
        })
        return

    prior = [(m["role"], m["content"]) for m in history]
    history.append({"role": "user", "content": question})

    if config.gateway_mode():
        try:
            data = get_gateway_client().ask(document_id, question)
            answer = _answer_from_gateway(question, document_id, data)
            history.append({"role": "assistant", "content": answer.answer_text, "answer": answer})
            st.session_state["questions_asked"] = asked + 1
        except GatewayError as exc:
            history.append({"role": "assistant", "content": f"Sorry — {exc}", "answer": None})
        return

    bounded = bound_history(prior, config.RAG_HISTORY_MESSAGE_LIMIT)
    try:
        answer = build_tutor().answer(question, document_id=document_id, history=bounded)
        history.append({"role": "assistant", "content": answer.answer_text, "answer": answer})
        st.session_state["questions_asked"] = asked + 1
    except TutorError as exc:
        # A failed call doesn't count against the session budget.
        history.append({"role": "assistant", "content": f"Sorry — {exc}", "answer": None})


# --- Small UI pieces --------------------------------------------------------
def _stepper(active: str) -> None:
    """Compact, non-color-only progress indicator with direct stage labels."""
    stages = [("choose", "Choose PDF"), ("prepare", "Prepare curriculum"), ("ask", "Ask questions")]
    order = [s[0] for s in stages]
    ai = order.index(active)
    parts = []
    for i, (_key, label) in enumerate(stages):
        if i < ai:
            parts.append(f"✓ {label}")
        elif i == ai:
            parts.append(f"**{label}**")
        else:
            parts.append(label)
    st.markdown("&nbsp;&nbsp;→&nbsp;&nbsp;".join(parts))


def _privacy_note() -> None:
    st.caption(
        "Your PDF is used to prepare this study session. Text is sent to OpenAI "
        "for search and answer generation."
    )
    with st.expander("Privacy and data use"):
        st.markdown(
            "- Your PDF is read in the browser session to extract and organize its "
            "text; the file itself is not shown to anyone else.\n"
            "- To enable search and the tutor, passages are sent to OpenAI to "
            "compute embeddings and generate answers.\n"
            "- Prepared text is stored in a vector database **for this session**. "
            "On the hosted demo this storage is temporary and may be cleared when "
            "the app restarts — it is not guaranteed to be private or permanent.\n"
            "- Please don't upload confidential material you aren't permitted to share."
        )


def _render_skipped_warning(skipped) -> None:
    """Aggregate empty-page warnings into one calm message + optional detail."""
    if not skipped:
        return
    st.warning(
        f"{len(skipped)} page{'s' if len(skipped) != 1 else ''} contained no "
        "readable text and were skipped.",
        icon="⚠️",
    )
    with st.expander("View skipped pages"):
        st.write("Pages: " + ", ".join(str(p) for p in skipped))


def _render_answer_extras(answer) -> None:
    """Render sources and (secondary) feedback for one answer.

    Latency, model names, retrieval timing, and cosine distance are hidden from
    normal users and only shown when SHOW_DEVELOPER_DETAILS is enabled.
    """
    if answer is None:
        return
    if answer.abstained:
        st.info("I couldn't find enough in your curriculum to answer that "
                "confidently, so I didn't guess.")

    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})"):
            for c in answer.citations:
                dev = f" · distance {c.distance:.4f}" if (
                    config.SHOW_DEVELOPER_DETAILS and c.distance is not None) else ""
                st.markdown(f"**[{c.source_id}] {c.filename} — page {c.page_number}**{dev}")
                with st.container(border=True):
                    st.caption("Supporting passage")
                    st.text(c.passage)
                    if config.SHOW_DEVELOPER_DETAILS:
                        st.caption(f"Chunk ID: `{c.chunk_id}`")

    if config.SHOW_DEVELOPER_DETAILS:
        st.caption(
            f"⏱ {answer.latency_seconds:.2f}s (retrieval {answer.retrieval_seconds:.2f}s, "
            f"generation {answer.generation_seconds:.2f}s) · model `{answer.model}`"
        )

    # Feedback — kept, but visually secondary.
    fb_store = st.session_state.setdefault("feedback", {})
    existing = fb_store.get(answer.answer_id)
    fc1, fc2, _sp = st.columns([1, 1, 4])
    if fc1.button("👍", key=f"fb_up_{answer.answer_id}", help="Helpful"):
        record_feedback(fb_store, answer.answer_id, True)
        st.rerun()
    if fc2.button("👎", key=f"fb_down_{answer.answer_id}", help="Not helpful"):
        record_feedback(fb_store, answer.answer_id, False)
        st.rerun()
    if existing is not None:
        st.caption("Thanks for the feedback." if existing["helpful"]
                   else "Thanks — noted that this wasn't helpful.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📘 CurriculumIQ")
st.caption("Turn your curriculum into a source-grounded AI tutor.")


# ---------------------------------------------------------------------------
# Access screen (welcome gate; only when APP_ACCESS_CODE is configured)
# ---------------------------------------------------------------------------
if config.access_required() and not st.session_state.get("access_granted"):
    _left, mid, _right = st.columns([1, 3, 1])
    with mid:
        with st.container(border=True):
            st.markdown("### Turn your curriculum into a source-grounded AI tutor.")
            st.write(
                "Upload a PDF and get clear answers with page citations you can "
                "verify against the original material."
            )
            st.caption("🔒 Private portfolio demo")
            st.markdown(
                "Access is limited to protect shared AI usage. "
                "Use the demo code provided with this link."
            )
            with st.form("access_form"):
                code = st.text_input("Demo access code", type="password")
                submitted = st.form_submit_button("Open demo", type="primary",
                                                  use_container_width=True)
            if submitted:
                if config.verify_access_code(code):
                    st.session_state["access_granted"] = True
                    st.rerun()
                else:
                    st.error("That access code isn't correct. Please try again.", icon="🚫")
    st.stop()  # block the rest of the app until access is granted


# ---------------------------------------------------------------------------
# Product flow
# ---------------------------------------------------------------------------
view = _current_view()
indexed = bool(st.session_state.get("indexed"))

# ===== STATE A — no document =====
if view is None:
    _stepper("choose")
    st.divider()
    st.subheader("Choose your curriculum")
    st.write("Upload a course PDF and CurriculumIQ will prepare it so you can ask "
             "questions and get answers grounded in the exact pages.")

    _privacy_note()

    uploaded = st.file_uploader("Curriculum PDF", type=["pdf"])
    st.caption(f"Text-based PDFs up to {config.MAX_FILE_SIZE_MB} MB.")

    if st.button("Prepare curriculum", type="primary",
                 disabled=uploaded is None, use_container_width=True):
        _run_preparation(uploaded.getvalue(), uploaded.name)

    if st.session_state.get("error"):
        st.error(st.session_state["error"], icon="🚫")

# ===== STATE C — document ready (or prepared without a key) =====
else:
    _stepper("ask")
    st.divider()

    # Compact success card.
    with st.container(border=True):
        st.markdown("### ✓ Your curriculum is ready")
        st.write(f"**{view.filename}** · {view.page_count} "
                 f"page{'s' if view.page_count != 1 else ''}")

    _render_skipped_warning(view.skipped_pages)

    # Collapsed technical summary — out of the main journey.
    with st.expander("Document details"):
        details = (
            f"- **File:** {view.filename}\n"
            f"- **Pages:** {view.page_count:,}\n"
            f"- **Prepared sections:** {view.chunk_count:,}\n"
            f"- **Skipped (empty) pages:** "
            + (", ".join(str(p) for p in view.skipped_pages) or "none")
        )
        st.markdown(details)

    # Secondary action: start over / replace PDF.
    if st.button("Replace PDF / Start over", use_container_width=False):
        _reset()
        st.session_state.pop("error", None)
        st.rerun()

    st.divider()

    # ----- Tutor: the main destination -----
    if not indexed:
        st.subheader("Ask your curriculum")
        st.info("The AI Tutor needs an OpenAI API key to answer questions. Your "
                "document was read and organized, but answering is disabled until "
                "a key is configured.")
    else:
        st.subheader("Ask your curriculum")
        st.write("Answers come from your uploaded PDF and include page citations "
                 "you can open and verify.")

        # Example questions — clickable, stacked (mobile-friendly), secondary.
        st.caption("Try an example:")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                with st.spinner("Thinking…"):
                    _handle_question(example, view.document_id)
                st.rerun()

        # Conversation so far.
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                _render_answer_extras(msg.get("answer"))

        # Quiet session budget + secondary clear action.
        limit = config.MAX_QUESTIONS_PER_SESSION
        asked = st.session_state.get("questions_asked", 0)
        at_limit = config.question_limit_reached(asked)
        info_col, clear_col = st.columns([3, 1])
        with info_col:
            if limit > 0:
                if at_limit:
                    st.caption(f"Session limit reached ({limit} questions). Refresh to continue.")
                else:
                    st.caption(f"{max(limit - asked, 0)} of {limit} questions left this session.")
        with clear_col:
            if st.session_state.get("chat_history"):
                if st.button("Clear conversation", use_container_width=True):
                    _clear_chat()
                    st.rerun()

        question = st.chat_input(
            "Ask a question about your curriculum…" if not at_limit
            else "Session limit reached — refresh to continue…",
            disabled=at_limit,
        )
        if question:
            with st.spinner("Thinking…"):
                _handle_question(question, view.document_id)
            st.rerun()

        # ----- Advanced tools (optional; must not compete with the tutor) -----
        # Direct-mode only: passage search uses the in-process vector store.
        if not config.gateway_mode():
          with st.expander("Advanced tools — passage search"):
            st.caption("Search the exact passages in your curriculum.")
            query = st.text_input("Search your curriculum")
            if st.button("Search", disabled=not query.strip()):
                _clear_search()
                try:
                    retriever = RetrievalService(
                        get_vector_store(),
                        get_embedding_provider(config.OPENAI_EMBEDDING_MODEL),
                    )
                    st.session_state["search_results"] = retriever.search(
                        query, top_k=config.RAG_TOP_K, document_id=view.document_id
                    )
                except (ValueError, EmbeddingError, VectorStoreError) as exc:
                    st.session_state["search_error"] = str(exc)

            if st.session_state.get("search_error"):
                st.error(st.session_state["search_error"], icon="🚫")

            results = st.session_state.get("search_results")
            if results is not None:
                if not results:
                    st.info("No matching passages found. Try rephrasing your query.")
                for r in results:
                    dev = f" · distance {r.distance:.4f}" if (
                        config.SHOW_DEVELOPER_DETAILS and r.distance is not None) else ""
                    st.markdown(f"**{r.filename} — page {r.page_number}**{dev}")
                    with st.container(border=True):
                        st.text(r.text)
                        if config.SHOW_DEVELOPER_DETAILS:
                            st.caption(f"Chunk ID: `{r.chunk_id}`")

    # ----- Developer details (hidden by default; opt-in for local debugging) -
    if config.SHOW_DEVELOPER_DETAILS:
        with st.expander("🛠 Developer details"):
            st.write({
                "service_mode": config.SERVICE_MODE,
                "document_id": view.document_id,
                "chunk_count": view.chunk_count,
                "embedding_model": config.OPENAI_EMBEDDING_MODEL,
                "chat_model": config.OPENAI_CHAT_MODEL,
                "chunk_size_tokens": config.RAG_CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": config.RAG_CHUNK_OVERLAP_TOKENS,
                "top_k_default": config.RAG_TOP_K,
                "max_context_chunks": config.RAG_MAX_CONTEXT_CHUNKS,
                "max_context_tokens": config.RAG_MAX_CONTEXT_TOKENS,
                "max_distance (lower=closer)": config.RAG_MAX_DISTANCE,
                "history_message_limit": config.RAG_HISTORY_MESSAGE_LIMIT,
                "indexed": indexed,
                "generation_enabled": config.generation_enabled(),
            })
            if not config.gateway_mode():
                try:
                    store = get_vector_store()
                    st.write({
                        "chroma_collection": config.CHROMA_COLLECTION_NAME,
                        "chroma_total_chunks": store.count(),
                        "this_document_indexed": store.has_document(view.document_id),
                        "indexed_document_ids": store.list_document_ids(),
                    })
                except VectorStoreError as exc:
                    st.caption(f"Vector store status unavailable: {exc}")
