"""Tests for backend/persistence.py, using a temp SQLite DB (temp_data_dir fixture)."""
from backend.ingestion_agent import IngestedPaper, Section
from backend import persistence


def _sample_paper():
    section = Section(
        heading="Results", text="The model works well.", page_start=3, page_end=3,
        page_segments=[(3, "The model works well.")],
    )
    return IngestedPaper(title="Test Paper", abstract="Abstract text.", sections=[section],
                          full_text="dummy", num_pages=5)


class TestPersistence:
    def test_save_and_reload_new_session(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        chunks = [{"text": "chunk text", "heading": "Results", "page_start": 3, "page_end": 3}]

        persistence.save_new_session("abc123", paper, chunks)
        rows = persistence.load_all_sessions()

        assert len(rows) == 1
        row = rows[0]
        assert row["doc_id"] == "abc123"
        assert row["paper"].title == "Test Paper"
        assert row["paper"].sections[0].heading == "Results"
        assert row["chunks"] == chunks
        assert row["chat_history"] == []
        assert row["tldr"] is None

    def test_update_summary_cache_persists(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        persistence.update_summary_cache("doc1", "A tldr.", [{"heading": "Results", "summary": "Good."}], ["Finding 1"])

        rows = persistence.load_all_sessions()
        row = rows[0]
        assert row["tldr"] == "A tldr."
        assert row["section_summaries"] == [{"heading": "Results", "summary": "Good."}]
        assert row["key_findings"] == ["Finding 1"]

    def test_update_gaps_cache_persists(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        gaps = {"author_acknowledged": ["gap1"], "inferred": []}
        persistence.update_gaps_cache("doc1", gaps)

        rows = persistence.load_all_sessions()
        assert rows[0]["gaps"] == gaps

    def test_update_repro_cache_persists(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        repro = {"repo_url": "https://github.com/foo/bar", "score": 80, "checks": [], "claims": []}
        persistence.update_repro_cache("doc1", repro, "https://github.com/foo/bar")

        rows = persistence.load_all_sessions()
        assert rows[0]["repro"] == repro
        assert rows[0]["repro_url_used"] == "https://github.com/foo/bar"

    def test_repro_defaults_to_none_when_not_set(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        rows = persistence.load_all_sessions()
        assert rows[0]["repro"] is None
        assert rows[0]["repro_url_used"] is None

    def test_update_chat_history_persists(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        persistence.update_chat_history("doc1", history)

        rows = persistence.load_all_sessions()
        assert rows[0]["chat_history"] == history

    def test_multiple_sessions_load_independently(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("doc1", paper, [])
        persistence.save_new_session("doc2", paper, [])

        rows = persistence.load_all_sessions()
        doc_ids = {r["doc_id"] for r in rows}
        assert doc_ids == {"doc1", "doc2"}

    def test_corrupt_row_does_not_crash_loading_other_sessions(self, temp_data_dir):
        persistence.init_db()
        paper = _sample_paper()
        persistence.save_new_session("good_doc", paper, [])

        # Manually insert a row with invalid JSON to simulate corruption
        conn = persistence._get_conn()
        conn.execute(
            "INSERT INTO sessions (doc_id, paper_json, chunks_json, created_at) VALUES (?, ?, ?, ?)",
            ("corrupt_doc", "not valid json{{{", "[]", "2026-01-01T00:00:00"),
        )
        conn.commit()

        rows = persistence.load_all_sessions()
        doc_ids = {r["doc_id"] for r in rows}
        assert "good_doc" in doc_ids
        assert "corrupt_doc" not in doc_ids  # skipped, not crashed
