"""Tests for discovery_agent.py. All requests.get and llm_client calls are
monkeypatched -- no real Semantic Scholar API or LLM calls."""
import json
import os

import pytest

from backend import discovery_agent as da
from backend.discovery_agent import DiscoveryError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self._chunks = chunks or []

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=65536):
        yield from self._chunks


GOOD_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "p1", "title": "Attention Is All You Need", "abstract": "We propose a transformer.",
            "year": 2017, "citationCount": 90000,
            "externalIds": {"ArXiv": "1706.03762", "DOI": "10.1/abc"},
            "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762.pdf"},
        },
        {
            "paperId": "p2", "title": "Unrelated Paper", "abstract": "About something else entirely.",
            "year": 2020, "citationCount": 5,
            "externalIds": {"ArXiv": None, "DOI": "10.1/xyz"},
            "openAccessPdf": None,
        },
        {
            "paperId": "p3", "title": "No Abstract Paper", "abstract": None,
            "year": 2021, "citationCount": 1,
            "externalIds": {}, "openAccessPdf": None,
        },
    ]
}


class TestSearchPapers:
    def test_parses_candidates_and_drops_missing_abstract(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(200, GOOD_SEARCH_RESPONSE))
        results = da.search_papers("attention mechanisms")
        assert len(results) == 2  # the no-abstract paper is dropped
        assert results[0]["title"] == "Attention Is All You Need"
        assert results[0]["arxiv_id"] == "1706.03762"
        assert results[0]["pdf_url"] == "https://arxiv.org/pdf/1706.03762.pdf"

    def test_retries_once_on_429(self, monkeypatch):
        calls = []

        def fake_get(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse(429)
            return FakeResponse(200, GOOD_SEARCH_RESPONSE)

        monkeypatch.setattr(da.requests, "get", fake_get)
        monkeypatch.setattr(da.time, "sleep", lambda s: None)
        results = da.search_papers("attention")
        assert len(calls) == 2
        assert len(results) == 2

    def test_persistent_429_raises(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(429))
        monkeypatch.setattr(da.time, "sleep", lambda s: None)
        with pytest.raises(DiscoveryError, match="rate limit"):
            da.search_papers("attention")

    def test_server_error_raises(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(500))
        with pytest.raises(DiscoveryError):
            da.search_papers("attention")

    def test_network_error_raises(self, monkeypatch):
        import requests as real_requests

        def fake_get(*a, **k):
            raise real_requests.RequestException("boom")

        monkeypatch.setattr(da.requests, "get", fake_get)
        with pytest.raises(DiscoveryError, match="Network error"):
            da.search_papers("attention")


class TestFilterRelevant:
    def _candidates(self):
        return [
            {"title": "A", "abstract": "About A", "citation_count": 10},
            {"title": "B", "abstract": "About B", "citation_count": 100},
            {"title": "C", "abstract": "About C", "citation_count": 1},
        ]

    def test_empty_candidates_returns_empty(self, fake_llm):
        assert da.filter_relevant("topic", [], 5) == []

    def test_parses_valid_selection(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"selected_indices": [1, 0]}))
        result = da.filter_relevant("topic", self._candidates(), max_results=5)
        assert [c["title"] for c in result] == ["B", "A"]

    def test_respects_max_results_cap(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"selected_indices": [0, 1, 2]}))
        result = da.filter_relevant("topic", self._candidates(), max_results=1)
        assert len(result) == 1

    def test_invalid_json_falls_back_to_citation_sort(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        result = da.filter_relevant("topic", self._candidates(), max_results=2)
        assert [c["title"] for c in result] == ["B", "A"]  # sorted by citation_count desc

    def test_out_of_range_indices_ignored(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"selected_indices": [0, 99, -1]}))
        result = da.filter_relevant("topic", self._candidates(), max_results=5)
        assert [c["title"] for c in result] == ["A"]


class TestDownloadPdf:
    def test_successful_download_returns_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(
            200, headers={"content-type": "application/pdf"}, chunks=[b"%PDF-1.4 fake content"],
        ))
        path = da.download_pdf("https://example.com/paper.pdf")
        assert path is not None
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read().startswith(b"%PDF")
        os.unlink(path)

    def test_bad_status_returns_none(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(404))
        assert da.download_pdf("https://example.com/missing.pdf") is None

    def test_wrong_content_type_returns_none(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(
            200, headers={"content-type": "text/html"}, chunks=[b"<html>not a pdf</html>"],
        ))
        assert da.download_pdf("https://example.com/paywall") is None

    def test_pdf_extension_accepted_without_content_type_header(self, monkeypatch):
        monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(
            200, headers={}, chunks=[b"%PDF-1.4"],
        ))
        path = da.download_pdf("https://example.com/paper.pdf")
        assert path is not None
        os.unlink(path)

    def test_oversized_download_returns_none(self, monkeypatch):
        from backend import config
        original = config.DISCOVERY_PDF_MAX_SIZE_MB
        config.DISCOVERY_PDF_MAX_SIZE_MB = 0  # anything at all exceeds 0 MB
        try:
            monkeypatch.setattr(da.requests, "get", lambda *a, **k: FakeResponse(
                200, headers={"content-type": "application/pdf"}, chunks=[b"x" * 1000],
            ))
            assert da.download_pdf("https://example.com/huge.pdf") is None
        finally:
            config.DISCOVERY_PDF_MAX_SIZE_MB = original

    def test_network_error_returns_none(self, monkeypatch):
        import requests as real_requests

        def fake_get(*a, **k):
            raise real_requests.RequestException("boom")

        monkeypatch.setattr(da.requests, "get", fake_get)
        assert da.download_pdf("https://example.com/paper.pdf") is None


class TestDiscover:
    def test_full_pipeline_downloads_selected_papers(self, fake_llm, monkeypatch):
        monkeypatch.setattr(da, "search_papers", lambda topic, limit=None: [
            {"paper_id": "p1", "title": "A", "abstract": "x", "year": 2020, "citation_count": 5, "arxiv_id": "1111.1111", "doi": None, "pdf_url": "https://x.com/a.pdf"},
        ])
        monkeypatch.setattr(da, "filter_relevant", lambda topic, candidates, max_results: candidates)
        monkeypatch.setattr(da, "download_pdf", lambda url: "/tmp/fake_a.pdf")

        result = da.discover("attention mechanisms", max_results=5)
        assert result["candidates_found"] == 1
        assert len(result["downloaded"]) == 1
        assert result["downloaded"][0]["local_path"] == "/tmp/fake_a.pdf"
        assert result["skipped_no_pdf"] == []
        assert result["skipped_duplicate"] == []

    def test_no_pdf_url_is_skipped_not_downloaded(self, fake_llm, monkeypatch):
        monkeypatch.setattr(da, "search_papers", lambda topic, limit=None: [
            {"paper_id": "p2", "title": "B", "abstract": "x", "year": 2020, "citation_count": 5, "arxiv_id": None, "doi": "10.1/x", "pdf_url": None},
        ])
        monkeypatch.setattr(da, "filter_relevant", lambda topic, candidates, max_results: candidates)
        result = da.discover("topic", max_results=5)
        assert result["downloaded"] == []
        assert len(result["skipped_no_pdf"]) == 1

    def test_duplicate_arxiv_id_is_skipped(self, fake_llm, monkeypatch):
        monkeypatch.setattr(da, "search_papers", lambda topic, limit=None: [
            {"paper_id": "p1", "title": "A", "abstract": "x", "year": 2020, "citation_count": 5, "arxiv_id": "1706.03762", "doi": None, "pdf_url": "https://x.com/a.pdf"},
        ])
        monkeypatch.setattr(da, "filter_relevant", lambda topic, candidates, max_results: candidates)
        result = da.discover("topic", max_results=5, existing_arxiv_ids={"1706.03762"})
        assert result["downloaded"] == []
        assert len(result["skipped_duplicate"]) == 1

    def test_failed_download_is_reported_not_fatal(self, fake_llm, monkeypatch):
        monkeypatch.setattr(da, "search_papers", lambda topic, limit=None: [
            {"paper_id": "p1", "title": "A", "abstract": "x", "year": 2020, "citation_count": 5, "arxiv_id": None, "doi": None, "pdf_url": "https://x.com/broken.pdf"},
        ])
        monkeypatch.setattr(da, "filter_relevant", lambda topic, candidates, max_results: candidates)
        monkeypatch.setattr(da, "download_pdf", lambda url: None)
        result = da.discover("topic", max_results=5)
        assert result["downloaded"] == []
        assert len(result["skipped_download_failed"]) == 1
