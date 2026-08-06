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
    def test_no_repo_triggers_code_generation(self, client, tmp_path, fake_llm, monkeypatch):
        path = make_test_pdf(tmp_path, {
            "Abstract": "This paper studies something with no code link.",
            "Methodology": "We use a transformer encoder trained with Adam.",
        })
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import codegen_agent
        monkeypatch.setattr(codegen_agent, "analyze", lambda paper: {
            "paper_info": {"title": "T", "gaps": ["LR not specified"]},
            "files": {"model.py": "import torch", "README.md": "# Generated"},
            "gap_report": "## Extraction Gaps\n- LR not specified",
        })

        r = client.get(f"/reproducibility/{doc_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "generated"
        assert data["files"] == {"model.py": "import torch", "README.md": "# Generated"}
        assert data["paper_info"]["gaps"] == ["LR not specified"]
        assert isinstance(data["score"], int)

    def test_explicit_repo_url_query_param_triggers_repo_check(self, client, tmp_path, monkeypatch):
        path = make_test_pdf(tmp_path, {"Abstract": "This paper studies something interesting."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import repo_fetch, repro_check_agent
        monkeypatch.setattr(repo_fetch, "fetch_repo_metadata", lambda o, r: {
            "full_name": "foo/bar", "default_branch": "main", "archived": False,
            "license_spdx_id": "mit", "license_name": "MIT", "html_url": "https://github.com/foo/bar",
            "stargazers_count": 1, "pushed_at": "2026-01-01",
        })
        monkeypatch.setattr(repo_fetch, "fetch_file_tree", lambda *a, **k: ["README.md"])
        monkeypatch.setattr(repo_fetch, "fetch_file_content", lambda *a, **k: "# Bar")
        monkeypatch.setattr(repro_check_agent, "verify_claims", lambda *a, **k: [])

        r = client.get(f"/reproducibility/{doc_id}", params={"repo_url": "https://github.com/foo/bar"})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "repo_check"
        assert data["repo_url"] == "https://github.com/foo/bar"
        assert data["files"] == {}

    def test_unknown_doc_id_returns_404(self, client):
        r = client.get("/reproducibility/does-not-exist")
        assert r.status_code == 404

    def test_report_includes_reproducibility_section(self, client, tmp_path, fake_llm):
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


class TestReproducibilityDownload:
    def test_returns_zip_for_generated_mode(self, client, tmp_path, fake_llm, monkeypatch):
        path = make_test_pdf(tmp_path, {"Abstract": "Abstract.", "Methodology": "We use Adam."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import codegen_agent
        monkeypatch.setattr(codegen_agent, "analyze", lambda paper: {
            "paper_info": {"title": "T", "gaps": []},
            "files": {"model.py": "import torch"},
            "gap_report": "## Extraction Gaps\n_No gaps identified._",
        })

        # Download reuses whatever GET /reproducibility already computed --
        # it must be called first to populate the cache.
        assert client.get(f"/reproducibility/{doc_id}").json()["mode"] == "generated"

        r = client.get(f"/reproducibility/{doc_id}/download")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"

        import io, zipfile
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert zf.namelist() == ["model.py"]
        assert zf.read("model.py").decode() == "import torch"

    def test_404_when_nothing_computed_yet(self, client, tmp_path):
        path = make_test_pdf(tmp_path, {"Abstract": "Abstract."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]
        r = client.get(f"/reproducibility/{doc_id}/download")
        assert r.status_code == 404

    def test_unsafe_filenames_from_cached_data_are_dropped_not_written(self, client, tmp_path, fake_llm, monkeypatch):
        """Defense in depth: even if a filename in cached/persisted repro data
        were somehow unsafe (bypassing plan_files()'s own sanitization --
        e.g. an older cache entry from before this guard existed), the
        download route must not write it into the zip."""
        path = make_test_pdf(tmp_path, {"Abstract": "Abstract.", "Methodology": "We use Adam."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import codegen_agent
        monkeypatch.setattr(codegen_agent, "analyze", lambda paper: {
            "paper_info": {"title": "T", "gaps": []},
            "files": {"model.py": "import torch", "../../etc/passwd": "malicious"},
            "gap_report": "## Extraction Gaps\n_No gaps identified._",
        })
        assert client.get(f"/reproducibility/{doc_id}").json()["mode"] == "generated"

        r = client.get(f"/reproducibility/{doc_id}/download")
        assert r.status_code == 200

        import io, zipfile
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert zf.namelist() == ["model.py"]  # the unsafe entry never made it into the zip

    def test_404_when_only_unsafe_filenames_present(self, client, tmp_path, fake_llm, monkeypatch):
        path = make_test_pdf(tmp_path, {"Abstract": "Abstract.", "Methodology": "We use Adam."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import codegen_agent
        monkeypatch.setattr(codegen_agent, "analyze", lambda paper: {
            "paper_info": {"title": "T", "gaps": []},
            "files": {"../../etc/passwd": "malicious"},
            "gap_report": "## Extraction Gaps\n_No gaps identified._",
        })
        assert client.get(f"/reproducibility/{doc_id}").json()["mode"] == "generated"

        r = client.get(f"/reproducibility/{doc_id}/download")
        assert r.status_code == 404

    def test_404_when_repo_check_mode(self, client, tmp_path, monkeypatch):
        path = make_test_pdf(tmp_path, {"Abstract": "Abstract."})
        with open(path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("paper.pdf", f, "application/pdf")})
        doc_id = upload_resp.json()["doc_id"]

        from backend import repo_fetch, repro_check_agent
        monkeypatch.setattr(repo_fetch, "fetch_repo_metadata", lambda o, r: {
            "full_name": "foo/bar", "default_branch": "main", "archived": False,
            "license_spdx_id": None, "license_name": None, "html_url": "https://github.com/foo/bar",
            "stargazers_count": 0, "pushed_at": None,
        })
        monkeypatch.setattr(repo_fetch, "fetch_file_tree", lambda *a, **k: [])
        monkeypatch.setattr(repo_fetch, "fetch_file_content", lambda *a, **k: None)
        monkeypatch.setattr(repro_check_agent, "verify_claims", lambda *a, **k: [])

        r = client.get(f"/reproducibility/{doc_id}", params={"repo_url": "https://github.com/foo/bar"})
        assert r.json()["mode"] == "repo_check"

        r2 = client.get(f"/reproducibility/{doc_id}/download")
        assert r2.status_code == 404


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
