"""Tests for repo_fetch.py. All requests.get calls are monkeypatched --
no real GitHub API calls in the test suite."""
import pytest

from backend import repo_fetch
from backend.repo_fetch import RepoFetchError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", raw_bytes=b""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.raw = _FakeRaw(raw_bytes)

    def json(self):
        return self._json_data


class _FakeRaw:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n, decode_content=True):
        return self._data[:n]


class TestExtractCodeUrl:
    def test_finds_github_url_in_text(self):
        text = "Some abstract text. Code is available at https://github.com/foo/bar for reproduction."
        assert repo_fetch.extract_code_url(text) == "https://github.com/foo/bar"

    def test_returns_none_when_no_url(self):
        assert repo_fetch.extract_code_url("No links here at all.") is None

    def test_ignores_non_github_urls(self):
        text = "See https://gitlab.com/foo/bar or https://example.com for details."
        assert repo_fetch.extract_code_url(text) is None


class TestParseGithubUrl:
    def test_parses_plain_url(self):
        assert repo_fetch.parse_github_url("https://github.com/foo/bar") == ("foo", "bar")

    def test_parses_url_with_git_suffix(self):
        assert repo_fetch.parse_github_url("https://github.com/foo/bar.git") == ("foo", "bar")

    def test_parses_url_with_trailing_path(self):
        assert repo_fetch.parse_github_url("https://github.com/foo/bar/tree/main") == ("foo", "bar")

    def test_rejects_non_github_host(self):
        with pytest.raises(RepoFetchError):
            repo_fetch.parse_github_url("https://gitlab.com/foo/bar")

    def test_rejects_garbage_url(self):
        with pytest.raises(RepoFetchError):
            repo_fetch.parse_github_url("not a url")


class TestFetchRepoMetadata:
    def test_parses_expected_fields(self, monkeypatch):
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(
            200, json_data={
                "full_name": "foo/bar", "description": "A repo", "default_branch": "main",
                "archived": False, "stargazers_count": 42,
                "license": {"spdx_id": "MIT", "name": "MIT License"},
                "html_url": "https://github.com/foo/bar", "pushed_at": "2026-01-01T00:00:00Z",
            }
        ))
        meta = repo_fetch.fetch_repo_metadata("foo", "bar")
        assert meta["full_name"] == "foo/bar"
        assert meta["license_spdx_id"] == "MIT"
        assert meta["archived"] is False
        assert meta["default_branch"] == "main"

    def test_missing_license_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(
            200, json_data={"full_name": "foo/bar", "default_branch": "main"}
        ))
        meta = repo_fetch.fetch_repo_metadata("foo", "bar")
        assert meta["license_spdx_id"] is None

    def test_404_raises_repo_fetch_error(self, monkeypatch):
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(404, text="Not Found"))
        with pytest.raises(RepoFetchError):
            repo_fetch.fetch_repo_metadata("foo", "doesnotexist")

    def test_rate_limit_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(
            repo_fetch.requests, "get",
            lambda url, **kw: FakeResponse(403, text="API rate limit exceeded for..."),
        )
        with pytest.raises(RepoFetchError, match="rate limit"):
            repo_fetch.fetch_repo_metadata("foo", "bar")


class TestFetchFileTree:
    def test_returns_blob_paths_only(self, monkeypatch):
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(
            200, json_data={"tree": [
                {"path": "README.md", "type": "blob"},
                {"path": "src", "type": "tree"},
                {"path": "src/main.py", "type": "blob"},
            ], "truncated": False}
        ))
        tree = repo_fetch.fetch_file_tree("foo", "bar", "main")
        assert tree == ["README.md", "src/main.py"]

    def test_caps_entries_at_configured_max(self, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "REPRO_MAX_TREE_ENTRIES", 2)
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(
            200, json_data={"tree": [
                {"path": f"file{i}.py", "type": "blob"} for i in range(10)
            ], "truncated": False}
        ))
        tree = repo_fetch.fetch_file_tree("foo", "bar", "main")
        assert len(tree) == 2


class TestFetchFileContent:
    def test_returns_decoded_text(self, monkeypatch):
        monkeypatch.setattr(
            repo_fetch.requests, "get",
            lambda url, **kw: FakeResponse(200, raw_bytes=b"# Hello\nInstall with pip."),
        )
        content = repo_fetch.fetch_file_content("foo", "bar", "main", "README.md")
        assert "Hello" in content

    def test_returns_none_on_404(self, monkeypatch):
        monkeypatch.setattr(repo_fetch.requests, "get", lambda url, **kw: FakeResponse(404))
        assert repo_fetch.fetch_file_content("foo", "bar", "main", "MISSING.md") is None

    def test_truncates_to_max_bytes(self, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "REPRO_MAX_FILE_BYTES", 10)
        monkeypatch.setattr(
            repo_fetch.requests, "get",
            lambda url, **kw: FakeResponse(200, raw_bytes=b"0123456789ABCDEFGHIJ"),
        )
        content = repo_fetch.fetch_file_content("foo", "bar", "main", "big.txt")
        assert len(content) <= 10
