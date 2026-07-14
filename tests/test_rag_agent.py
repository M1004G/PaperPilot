"""Tests for backend/rag_agent.py (fake_embedding_models fixture is autouse via conftest)"""
from backend.ingestion_agent import IngestedPaper, Section
from backend.rag_agent import chunk_section, chunk_paper, RAGIndex, _bounded_history, _split_sentences
from backend import config


class TestSentenceSplitting:
    def test_splits_on_sentence_boundaries(self):
        text = "This is one. This is two! Is this three? Yes it is."
        sentences = _split_sentences(text)
        assert len(sentences) == 4

    def test_handles_empty_text(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []


class TestChunking:
    def test_chunks_never_split_mid_sentence(self):
        section = Section(
            heading="Introduction",
            text="First sentence here. " * 30,
            page_start=1, page_end=1,
            page_segments=[(1, "First sentence here. " * 30)],
        )
        chunks = chunk_section(section, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert c["text"][-1] in ".!?"

    def test_chunk_gets_precise_page_not_whole_section_range(self):
        section = Section(
            heading="Introduction", text="", page_start=1, page_end=3,
            page_segments=[
                (1, "Page one content here. More page one text."),
                (2, "Page two content here. More page two text."),
                (3, "Page three content here. More page three text."),
            ],
        )
        chunks = chunk_section(section, chunk_size=60, overlap=10)
        # At least one chunk should be scoped to a single page, not the full 1-3 range
        assert any(c["page_start"] == c["page_end"] == 1 for c in chunks)
        assert any(c["page_start"] == c["page_end"] == 3 for c in chunks)

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
        # Most recent turn must be kept
        assert result[-1]["content"] == "c" * 20
