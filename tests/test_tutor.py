"""TutorService tests: grounding, citations, abstention, filtering, errors.

Uses real retrieval (fake embeddings + temp Chroma) and a fake answer provider.
No network, no OpenAI.
"""

from __future__ import annotations

import pytest

from src.config import CHROMA_COLLECTION_NAME
from src.generation.prompts import FALLBACK_ANSWER
from src.generation.provider import AnswerGenerationError, FakeAnswerProvider
from src.generation.tutor import TutorError, TutorService, bound_history
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.indexer import IndexingService
from src.retrieval.retriever import RetrievalService
from src.retrieval.vector_store import VectorStore
from tests.conftest import make_extracted_document


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def retriever(tmp_path, embedder):
    store = VectorStore(str(tmp_path / "chroma"), CHROMA_COLLECTION_NAME)
    svc = IndexingService(store, embedder, chunk_size_tokens=200, overlap_tokens=20)
    doc = make_extracted_document(
        [
            "Photosynthesis converts sunlight into chemical energy in plant leaves.",
            "The quadratic formula solves quadratic equations using coefficients.",
            "Newton's laws describe motion, force, mass, and acceleration in physics.",
        ],
        filename="science.pdf",
        document_id="doc_corpus01",
    )
    svc.index_document(doc)
    return RetrievalService(store, embedder, default_top_k=5)


def _tutor(retriever, provider, **kw):
    kw.setdefault("max_distance", 2.0)  # lenient by default (cosine max is 2)
    return TutorService(retriever, provider, **kw)


def test_supported_question_produces_answer(retriever):
    provider = FakeAnswerProvider(response_text="A quadratic uses coefficients [S1].")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("quadratic formula coefficients")
    assert not ans.abstained
    assert ans.answer_text == "A quadratic uses coefficients [S1]."
    assert ans.model == "fake-chat"
    assert provider.call_count == 1


def test_retrieved_context_reaches_provider(retriever):
    provider = FakeAnswerProvider(response_text="ok [S1]")
    tutor = _tutor(retriever, provider)
    tutor.answer("quadratic formula coefficients")
    assert "<SOURCES>" in provider.last_user_prompt
    assert "quadratic" in provider.last_user_prompt.lower()


def test_source_ids_assigned_and_valid_citations_map(retriever):
    provider = FakeAnswerProvider(response_text="Grounded [S1].")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("quadratic formula coefficients")
    assert ans.retrieved_sources[0].source_id == "S1"
    assert [c.source_id for c in ans.citations] == ["S1"]
    # Displayed metadata comes from trusted retrieval, not the model.
    assert ans.citations[0].filename == "science.pdf"
    assert ans.citations[0].page_number == 2  # quadratic is page 2


def test_unknown_citation_rejected(retriever):
    provider = FakeAnswerProvider(response_text="Claim [S1] and fake [S99].")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("quadratic formula coefficients")
    assert "[S99]" not in ans.answer_text
    assert all(c.source_id != "S99" for c in ans.citations)
    assert any("S99" in w for w in ans.warnings)


def test_answer_without_citation_warns(retriever):
    provider = FakeAnswerProvider(response_text="An answer with no markers.")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("quadratic formula coefficients")
    assert ans.citations == []
    assert any("no valid citations" in w.lower() for w in ans.warnings)


def test_empty_retrieval_abstains_without_provider(tmp_path, embedder):
    empty_store = VectorStore(str(tmp_path / "empty"), CHROMA_COLLECTION_NAME)
    retriever = RetrievalService(empty_store, embedder, default_top_k=5)
    provider = FakeAnswerProvider(response_text="should not be used [S1]")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("anything")
    assert ans.abstained
    assert ans.answer_text == FALLBACK_ANSWER
    assert ans.citations == []
    assert provider.call_count == 0


def test_weak_retrieval_abstains_without_provider(retriever):
    provider = FakeAnswerProvider(response_text="should not be used [S1]")
    # Impossibly strict threshold filters every result out.
    tutor = _tutor(retriever, provider, max_distance=0.00001)
    ans = tutor.answer("quadratic formula coefficients")
    assert ans.abstained
    assert provider.call_count == 0


def test_document_filter_respected(tmp_path, embedder):
    store = VectorStore(str(tmp_path / "chroma"), CHROMA_COLLECTION_NAME)
    idx = IndexingService(store, embedder, chunk_size_tokens=200, overlap_tokens=20)
    idx.index_document(make_extracted_document(
        ["Photosynthesis in leaves."], filename="a.pdf", document_id="doc_a"))
    idx.index_document(make_extracted_document(
        ["Photosynthesis in a different doc."], filename="b.pdf", document_id="doc_b"))
    retriever = RetrievalService(store, embedder, default_top_k=10)
    provider = FakeAnswerProvider(response_text="ok [S1]")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("photosynthesis", document_id="doc_a")
    assert ans.retrieved_sources
    assert all(s.document_id == "doc_a" for s in ans.retrieved_sources)


def test_blank_question_rejected(retriever):
    tutor = _tutor(retriever, FakeAnswerProvider())
    with pytest.raises(TutorError):
        tutor.answer("   ")


def test_long_question_rejected(retriever):
    tutor = _tutor(retriever, FakeAnswerProvider(), max_question_length=10)
    with pytest.raises(TutorError):
        tutor.answer("this question is definitely longer than ten characters")


def test_provider_failure_becomes_safe_error(retriever):
    provider = FakeAnswerProvider(raise_error=AnswerGenerationError("boom"))
    tutor = _tutor(retriever, provider)
    with pytest.raises(TutorError):
        tutor.answer("quadratic formula coefficients")


def test_retrieval_failure_becomes_safe_error():
    from src.retrieval.embeddings import EmbeddingError

    class _BoomRetriever:
        def search(self, *a, **k):
            raise EmbeddingError("quota exhausted")

    provider = FakeAnswerProvider(response_text="unused [S1]")
    tutor = _tutor(_BoomRetriever(), provider)
    with pytest.raises(TutorError):
        tutor.answer("anything")
    assert provider.call_count == 0  # never reached generation


def test_followup_still_triggers_retrieval(retriever):
    provider = FakeAnswerProvider(response_text="ok [S1]")
    tutor = _tutor(retriever, provider)
    tutor.answer("photosynthesis energy")
    first_prompt = provider.last_user_prompt
    tutor.answer("newton force acceleration", history=[("user", "photosynthesis energy")])
    second_prompt = provider.last_user_prompt
    # Each call rebuilt a fresh SOURCES block from retrieval.
    assert "<SOURCES>" in first_prompt and "<SOURCES>" in second_prompt
    assert provider.call_count == 2
    assert "newton" in second_prompt.lower()


def test_empty_model_output_abstains(retriever):
    provider = FakeAnswerProvider(response_text="   ")
    tutor = _tutor(retriever, provider)
    ans = tutor.answer("quadratic formula coefficients")
    assert ans.abstained
    assert any("empty response" in w.lower() for w in ans.warnings)


def test_bound_history_limits_messages():
    hist = [("user", str(i)) for i in range(10)]
    assert bound_history(hist, 4) == hist[-4:]
    assert bound_history(hist, 0) == []
    assert bound_history(None, 5) == []
