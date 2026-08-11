# CurriculumIQ — Project Specification

## Product overview

CurriculumIQ is an AI tutor built on Retrieval-Augmented Generation (RAG).
Users upload curriculum PDFs; the system indexes their content and answers
natural-language questions with clear, grounded answers that cite the source
document and page number.

**Tagline:** Ask questions about your curriculum and receive clear answers with
verifiable sources.

## Problem statement

Students and educators work from dense, multi-document curricula. Finding the
passage that answers a specific question is slow, and general-purpose chatbots
answer confidently without showing where the answer came from — so answers
cannot be trusted or verified. CurriculumIQ answers questions *from the user's
own curriculum* and always shows the document and page behind each answer.

## Target users

- **Students** revising from course PDFs who want fast, source-backed answers.
- **Educators / curriculum designers** checking coverage and locating material.
- **Self-learners** working through textbooks and syllabi.

## Final MVP scope

- Upload one or more curriculum PDFs.
- Validate and extract text page by page.
- Chunk, embed, and index content in a vector store.
- Retrieve relevant passages for a user question.
- Generate a grounded answer with document + page citations.
- A clean, restrained academic interface for asking questions and reading
  answers with sources.

## Final technology stack

- **Python** — core language.
- **Streamlit** — user interface.
- **PyMuPDF** — PDF text extraction.
- **LangChain** — RAG orchestration.
- **Chroma** — vector store.
- **OpenAI API** — embeddings and chat completions.
- **Pytest** — automated testing.
- **Pandas** — tabular summaries / evaluation.

## Explicit non-goals

- No user authentication or multi-tenant accounts.
- No permanent storage of user documents beyond what indexing requires.
- No OCR of scanned/image-only PDFs (detected and flagged, not processed).
- No general web search or answering outside the uploaded curriculum.
- No dashboards, analytics, or content-authoring tools.
- No fine-tuning or model training.
- Not a certified assessment or grading system.

## Planned milestones

1. **Foundation & PDF extraction** *(this milestone)* — Streamlit app,
   validation, page-by-page extraction, document summary, tests, docs.
2. **Chunking, embeddings & Chroma** — split extracted text into chunks, embed
   with OpenAI, persist in a Chroma vector store.
3. **Retrieval & grounded answers** — retrieve relevant chunks and generate
   answers grounded strictly in retrieved context.
4. **Citations & student interface** — surface document + page citations and a
   polished question/answer experience.
5. **Evaluation & hardening** — evaluation harness, error handling, edge cases,
   performance and cost controls.
6. **Deployment & submission** — packaging, deployment, documentation, and
   final capstone submission.

## Architecture principle

Extraction and retrieval logic live in `src/` and are independent of Streamlit,
so they are unit-testable and reusable by the RAG pipeline. The Streamlit layer
stays thin: it collects input, calls services, and renders results.

**External services are isolated behind injectable providers.** OpenAI access
goes through two interfaces — `EmbeddingProvider` (`src/retrieval/embeddings.py`)
and `AnswerProvider` (`src/generation/provider.py`). Production injects the
OpenAI-backed implementations; tests inject deterministic offline fakes — so the
entire index → retrieve → generate pipeline is testable without network access
or API cost. The vector store (Chroma) is a standalone service; retrieval scores
are cosine distances where **lower means a closer match**.

**Answer generation (Milestone 3)** uses the **OpenAI Responses API**
(`client.responses.create`) directly rather than another framework layer.
Grounding is enforced in two places: (1) a pre-generation gate in `TutorService`
that filters retrieval by a configurable distance threshold and a context-token
budget, abstaining locally (without any model call) when evidence is
insufficient; and (2) prompt rules that confine the model to the supplied
`<SOURCES>` and forbid outside knowledge. **Citations are trust-anchored in
Python:** source IDs are assigned to retrieved chunks before generation, only
those IDs are accepted back from the model, and displayed filenames/pages come
from retrieval metadata — never from model output. Retrieved document text is
treated as untrusted data (escaped, delimited, and never placed in developer
instructions) to resist prompt injection.

**Runtime:** verified on **Python 3.11**. The Chroma / embedding dependency set
resolves cleanly there; Python 3.9 is not recommended for the RAG milestones.
