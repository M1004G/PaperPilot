"""Tests for backend/rag_agent.py (fake_embedding_models fixture is autouse via
conftest -- no test needs a real sentence-transformers/cross-encoder download)."""
from backend.ingestion_agent import IngestedPaper, Section
from backend.rag_agent import chunk_section, chunk_paper, RAGIndex, _bounded_history
from backend import config


class TestChunking:
    def test_chunk_gets_single_precise_page_not_a_range(self):
        """Each chunk is built from one page-segment at a time (via LangChain's
        RecursiveCharacterTextSplitter, split per-segment), so page_start ==
        page_end always -- a stricter, simpler guarantee than the old sentence-
        aware chunker's occasional page *range* when a chunk straddled a break."""
        section = Section(
            heading="Introduction", text="", page_start=1, page_end=3,
            page_segments=[
                (1, "Page one content here. More page one text."),
                (2, "Page two content here. More page two text."),
                (3, "Page three content here. More page three text."),
            ],
        )
        chunks = chunk_section(section, chunk_size=60, overlap=10)
        assert any(c["page_start"] == c["page_end"] == 1 for c in chunks)
        assert any(c["page_start"] == c["page_end"] == 3 for c in chunks)
        assert all(c["page_start"] == c["page_end"] for c in chunks)

    def test_long_section_produces_multiple_chunks(self):
        section = Section(
            heading="Introduction", text="First sentence here. " * 30,
            page_start=1, page_end=1,
            page_segments=[(1, "First sentence here. " * 30)],
        )
        chunks = chunk_section(section, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(c["heading"] == "Introduction" for c in chunks)

    def test_chunk_paper_falls_back_to_full_text_when_no_sections(self):
        paper = IngestedPaper(title="T", abstract="A", sections=[], full_text="Some fallback text here. More text.", num_pages=1)
        chunks = chunk_paper(paper)
        assert len(chunks) >= 1
        assert chunks[0]["heading"] == "Full Text"

    def test_chunk_paper_skips_references_section(self):
        sections = [
            Section(heading="Introduction", text="Real content here.", page_start=1, page_end=1,
                    page_segments=[(1, "Real content here.")]),
            Section(heading="References", text="Citation one. Citation two.", page_start=2, page_end=2,
                    page_segments=[(2, "Citation one. Citation two.")]),
        ]
        paper = IngestedPaper(title="T", abstract="A", sections=sections, full_text="dummy", num_pages=2)
        chunks = chunk_paper(paper)
        assert all(c["heading"] != "References" for c in chunks)

    def test_empty_page_segment_produces_no_chunks(self):
        section = Section(
            heading="Introduction", text="", page_start=1, page_end=1,
            page_segments=[(1, "   ")],
        )
        assert chunk_section(section) == []


class TestRAGIndex:
    def _chunks(self):
        return [
            {"text": "The model achieves 95 percent accuracy on the benchmark.", "heading": "Results", "page_start": 4, "page_end": 4},
            {"text": "We did not evaluate on non-English datasets.", "heading": "Limitations", "page_start": 9, "page_end": 9},
            {"text": "This paper studies a new attention mechanism.", "heading": "Introduction", "page_start": 1, "page_end": 1},
        ]

    def test_retrieve_returns_requested_k(self):
        index = RAGIndex(self._chunks())
        results = index.retrieve("What accuracy did the model achieve?", k=2)
        assert len(results) == 2

    def test_retrieve_result_carries_metadata(self):
        index = RAGIndex(self._chunks())
        results = index.retrieve("accuracy", k=1)
        r = results[0]
        assert "heading" in r and "page_start" in r and "page_end" in r and "score" in r

    def test_retrieve_k_larger_than_corpus_does_not_error(self):
        index = RAGIndex(self._chunks())
        results = index.retrieve("anything", k=100)
        assert len(results) == len(self._chunks())

    def test_chunk_ids_map_back_to_original_chunks_list(self):
        chunks = self._chunks()
        index = RAGIndex(chunks)
        results = index.retrieve("accuracy", k=3)
        for r in results:
            assert chunks[r["chunk_id"]]["text"] == index.chunks[r["chunk_id"]]["text"]


class TestAbstractPinning:
    def _chunks_with_abstract(self):
        return [
            {"text": "This paper proposes a novel method for X.", "heading": "Abstract", "page_start": 1, "page_end": 1},
            {"text": "Completely unrelated filler about datasets and hyperparameters.", "heading": "Appendix", "page_start": 10, "page_end": 10},
        ]

    def test_abstract_always_present_in_results(self):
        """Regression test: even if the abstract doesn't naturally score in the
        top-k for a query, it must still appear in retrieve()'s results."""
        index = RAGIndex(self._chunks_with_abstract())
        results = index.retrieve("some completely unrelated query about nothing in particular", k=1)
        assert any(r["heading"] == "Abstract" for r in results)

    def test_no_abstract_section_does_not_error(self):
        chunks = [{"text": "Just a regular section.", "heading": "Methods", "page_start": 1, "page_end": 1}]
        index = RAGIndex(chunks)
        results = index.retrieve("anything", k=1)
        assert len(results) == 1


class TestBoundedHistory:
    def test_keeps_at_least_one_turn(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_HISTORY_CHARS", 5)
        history = [{"role": "user", "content": "x" * 50}]
        result = _bounded_history(history)
        assert len(result) == 1  # always keeps at least the most recent turn

    def test_trims_older_turns_over_budget(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_HISTORY_CHARS", 30)
        history = [
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
        ]
        result = _bounded_history(history)
        assert len(result) < len(history)
        assert result[-1]["content"] == "c" * 20


class TestChromaPersistence:
    """The actual reason for switching from FAISS to Chroma: a doc_id-keyed
    collection persists real embedded vectors to disk, so a second RAGIndex
    built with the same doc_id and chunk count reuses them instead of
    re-embedding -- unlike the previous FAISS setup, which held vectors in
    memory only and re-embedded from scratch on every load."""

    def _chunks(self):
        return [
            {"text": "The model achieves 95 percent accuracy on the benchmark.", "heading": "Results", "page_start": 4, "page_end": 4},
            {"text": "This paper studies a new attention mechanism.", "heading": "Introduction", "page_start": 1, "page_end": 1},
        ]

    def test_second_index_with_same_doc_id_reuses_existing_collection(self, monkeypatch):
        from backend import rag_agent
        from tests.conftest import FakeEmbedder

        embed_calls = {"n": 0}

        class CountingEmbedder(FakeEmbedder):
            def embed_documents(self, texts):
                embed_calls["n"] += 1
                return super().embed_documents(texts)

        monkeypatch.setattr(rag_agent, "_get_embeddings", lambda: CountingEmbedder())

        doc_id = "test-persist-doc"
        chunks = self._chunks()
        try:
            index1 = rag_agent.RAGIndex(chunks, doc_id)
            first_embed_calls = embed_calls["n"]
            assert first_embed_calls >= 1  # first build must embed

            index2 = rag_agent.RAGIndex(chunks, doc_id)  # same doc_id, same chunks
            assert embed_calls["n"] == first_embed_calls  # no new embedding calls on reuse

            results = index2.retrieve("accuracy", k=1)
            assert len(results) == 1
        finally:
            index1.vectorstore.delete_collection()

    def test_mismatched_chunk_count_triggers_reset_not_reuse(self):
        from backend import rag_agent
        doc_id = "test-mismatch-doc"
        chunks = self._chunks()
        try:
            index1 = rag_agent.RAGIndex(chunks, doc_id)
            assert index1.vectorstore._collection.count() == 2

            # Fewer chunks under the same doc_id (e.g. re-ingested with different
            # chunking) must NOT silently reuse the stale 2-chunk collection.
            fewer_chunks = chunks[:1]
            index2 = rag_agent.RAGIndex(fewer_chunks, doc_id)
            assert index2.vectorstore._collection.count() == 1
        finally:
            index2.vectorstore.delete_collection()

    def test_doc_id_none_uses_an_ephemeral_unpersisted_collection(self):
        from backend import rag_agent
        index = rag_agent.RAGIndex(self._chunks(), doc_id=None)
        assert index.doc_id is None
        results = index.retrieve("accuracy", k=1)
        assert len(results) == 1
