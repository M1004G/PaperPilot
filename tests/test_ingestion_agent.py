"""Tests for backend/ingestion_agent.py"""
import pytest

from backend.ingestion_agent import ingest, IngestionError, prioritized_excerpt, IngestedPaper, Section
from tests.conftest import make_test_pdf


class TestIngest:
    def test_ingests_basic_multi_section_pdf(self, tmp_path):
        path = make_test_pdf(tmp_path, {
            "Abstract": "This paper studies an interesting problem.",
            "Introduction": "Background and motivation for the work.",
            "Results": "Our method achieves strong results.",
        })
        paper = ingest(path)
        assert isinstance(paper, IngestedPaper)
        assert paper.num_pages == 3
        headings = [s.heading for s in paper.sections]
        assert "Abstract" in headings
        assert "Introduction" in headings
        assert "Results" in headings

    def test_abstract_section_used_when_present(self, tmp_path):
        path = make_test_pdf(tmp_path, {
            "Abstract": "This is the real abstract content.",
            "Introduction": "Some intro text.",
        })
        paper = ingest(path)
        assert "real abstract content" in paper.abstract

    def test_section_tracks_page_segments(self, tmp_path):
        path = make_test_pdf(tmp_path, {
            "Results": "Our method achieves strong results on the benchmark.",
        })
        paper = ingest(path)
        results_section = next(s for s in paper.sections if s.heading == "Results")
        assert results_section.page_start == 1
        assert results_section.page_end == 1
        assert len(results_section.page_segments) == 1
        assert results_section.page_segments[0][0] == 1

    def test_rejects_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.pdf"
        path.write_bytes(b"this is not a valid pdf file")
        with pytest.raises(IngestionError):
            ingest(str(path))

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        with pytest.raises(IngestionError):
            ingest(str(path))

    def test_rejects_encrypted_pdf(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "Secret content")
        path = str(tmp_path / "encrypted.pdf")
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        doc.close()
        with pytest.raises(IngestionError, match="password"):
            ingest(path)


class TestPrioritizedExcerpt:
    def _paper(self):
        sections = [
            Section(heading="Introduction", text="Intro sentence. " * 50, page_start=1, page_end=2),
            Section(heading="Results", text="Result sentence. " * 50, page_start=3, page_end=4),
            Section(heading="Limitations", text="This is a key limitation.", page_start=9, page_end=9),
        ]
        return IngestedPaper(title="T", abstract="Abstract text.", sections=sections, full_text="dummy", num_pages=9)

    def test_limitations_survive_small_budget_despite_being_last(self):
        paper = self._paper()
        excerpt = prioritized_excerpt(paper, limit=200)
        assert "Limitations" in excerpt
        # Limitations (last in the paper) should appear before Introduction (first
        # in the paper) in the excerpt, since it's prioritized ahead of it.
        if "Introduction" in excerpt:
            assert excerpt.index("Limitations") < excerpt.index("Introduction")

    def test_respects_character_limit(self):
        paper = self._paper()
        excerpt = prioritized_excerpt(paper, limit=100)
        assert len(excerpt) <= 100 + 20  # small slack for the "[Heading]\n" labels

    def test_abstract_always_included_first(self):
        paper = self._paper()
        excerpt = prioritized_excerpt(paper, limit=5000)
        assert excerpt.startswith("Abstract text.")
