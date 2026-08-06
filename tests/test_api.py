"""End-to-end API tests via FastAPI's TestClient. LLM calls are mocked; a real
(small, generated) PDF is used for upload so ingestion runs for real."""
import pytest
from fastapi.testclient import TestClient

from backend.ingestion_agent import IngestionError
from backend.llm_client import LLMProviderError
from tests.conftest import make_test_pdf


@pytest.fixture
def client(temp_data_dir, fake_llm, monkeypatch):
    """Fresh app + fresh Orchestrator per test, pointed at an isolated temp DB."""
    import backend.main as main
    from backend.orchestrator import Orchestrator
    fresh_orchestrator = Orchestrator()
    monkeypatch.setattr(main, "orchestrator", fresh_orchestrator)
    return TestClient(main.app)


class TestHealthAndRouting:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_response_carries_request_id_header(self, client):
        r = client.get("/health")
        assert "X-Request-ID" in r.headers


class TestUploadValidation:
    def test_rejects_non_pdf_extension(self, client):
        r = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_rejects_oversized_file(self, client, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "MAX_PDF_SIZE_MB", 1)
        big_content = b"0" * (2 * 1024 * 1024)
        r = client.post("/upload", files={"file": ("big.pdf", big_content, "application/pdf")})
        assert r.status_code == 400

    def test_rejects_empty_file(self, client):
        r = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
        assert r.status_code == 400

    def test_rejects_corrupt_pdf(self, client):
        r = client.post("/upload", files={"file": ("fake.pdf", b"not a real pdf", "application/pdf")})
        assert r.status_code == 400

    def test_accepts_valid_pdf(self, client, tmp_path):
        path = make_test_pdf(tmp_path, {"Abstract": "This paper studies something interesting."})
        with open(path, "rb") as f:
            r = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        assert r.status_code == 200
        data = r.json()
        assert "doc_id" in data
        assert data["num_pages"] == 1


class TestFullFlow:
    def test_upload_then_summary_then_gaps_then_chat(self, client, tmp_path):
        path = make_test_pdf(tmp_path, {
            "Abstract": "This paper studies an interesting problem in machine learning.",
            "Results": "Our method achieves strong results on the benchmark dataset.",
        })
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        summary_resp = client.get(f"/summary/{doc_id}")
        assert summary_resp.status_code == 200
        assert "tldr" in summary_resp.json()

        gaps_resp = client.get(f"/gaps/{doc_id}")
        assert gaps_resp.status_code == 200

        chat_resp = client.post("/chat", json={"doc_id": doc_id, "query": "What were the results?"})
        assert chat_resp.status_code == 200
        assert "answer" in chat_resp.json()

        report_resp = client.get(f"/report/{doc_id}")
        assert report_resp.status_code == 200


class TestReproducibilityEndpoint:
    def test_returns_note_when_no_repo_detected(self, client, tmp_path):
        path = make_test_pdf(tmp_path, {"Abstract": "This paper studies something with no code link."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        r = client.get(f"/reproducibility/{doc_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["repo_metadata"] is None
        assert data["note"]

    def test_explicit_repo_url_query_param_is_used(self, client, tmp_path, monkeypatch):
        path = make_test_pdf(tmp_path, {"Abstract": "This paper studies something interesting."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import repro_agent
        monkeypatch.setattr(
            repro_agent, "analyze",
            lambda paper, repo_url=None: {
                "repo_url": repo_url, "repo_metadata": {"full_name": "foo/bar"},
                "checks": [], "score": 90, "verdict": "Likely reproducible", "claims": [], "note": None,
            },
        )
        r = client.get(f"/reproducibility/{doc_id}", params={"repo_url": "https://github.com/foo/bar"})
        assert r.status_code == 200
        assert r.json()["repo_url"] == "https://github.com/foo/bar"

    def test_unknown_doc_id_returns_404(self, client):
        r = client.get("/reproducibility/does-not-exist")
        assert r.status_code == 404

    def test_report_includes_reproducibility_section(self, client, tmp_path):
        path = make_test_pdf(tmp_path, {
            "Abstract": "This paper studies something interesting.",
            "Methodology": "We use a novel transformer-based approach.",
        })
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        r = client.get(f"/report/{doc_id}")
        assert r.status_code == 200
        assert "## Code Reproducibility" in r.text


class TestErrorMapping:
    def test_unknown_doc_id_returns_404(self, client):
        r = client.get("/summary/does-not-exist")
        assert r.status_code == 404

    def test_ingestion_error_maps_to_400(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(
            main.orchestrator, "ingest_paper",
            lambda path: (_ for _ in ()).throw(IngestionError("This PDF is password-protected."))
        )
        r = client.post("/upload", files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")})
        assert r.status_code == 400

    def test_llm_provider_error_maps_to_502(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(
            main.orchestrator, "get_summary",
            lambda doc_id: (_ for _ in ()).throw(LLMProviderError("Groq failed after retries"))
        )
        r = client.get("/summary/some-doc-id")
        assert r.status_code == 502

    def test_unexpected_error_maps_to_500(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(
            main.orchestrator, "get_gaps",
            lambda doc_id: (_ for _ in ()).throw(ValueError("something unexpected"))
        )
        r = client.get("/gaps/some-doc-id")
        assert r.status_code == 500
