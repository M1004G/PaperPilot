"""Tests for repro_agent.py. repo_fetch and llm_client are monkeypatched --
no real GitHub API or LLM calls."""
import json

import pytest

from backend import repro_agent
from backend.ingestion_agent import IngestedPaper, Section
from backend.repo_fetch import RepoFetchError


def make_paper(full_text="", method_text=""):
    sections = []
    if method_text:
        sections.append(Section(heading="Methodology", text=method_text))
    return IngestedPaper(title="Test Paper", abstract="An abstract.", sections=sections, full_text=full_text)


GOOD_METADATA = {
    "full_name": "foo/bar", "default_branch": "main", "archived": False,
    "license_spdx_id": "mit", "license_name": "MIT License",
    "html_url": "https://github.com/foo/bar", "stargazers_count": 10, "pushed_at": "2026-01-01",
}


class TestFindReadmeAndManifest:
    def test_finds_top_level_readme(self):
        assert repro_agent._find_readme_path(["README.md", "src/main.py"]) == "README.md"

    def test_ignores_nested_readme(self):
        assert repro_agent._find_readme_path(["docs/README.md", "src/main.py"]) is None

    def test_no_readme_returns_none(self):
        assert repro_agent._find_readme_path(["src/main.py"]) is None

    def test_finds_root_requirements_over_nested(self):
        tree = ["examples/requirements.txt", "requirements.txt"]
        assert repro_agent._find_manifest_path(tree) == "requirements.txt"

    def test_no_manifest_returns_none(self):
        assert repro_agent._find_manifest_path(["src/main.py"]) is None


class TestDependenciesPinnedRatio:
    def test_requirements_txt_fully_pinned(self):
        content = "numpy==1.26.4\npandas==2.2.2\n"
        assert repro_agent._dependencies_pinned_ratio("requirements.txt", content) == 1.0

    def test_requirements_txt_unpinned(self):
        content = "numpy\npandas\n"
        assert repro_agent._dependencies_pinned_ratio("requirements.txt", content) == 0.0

    def test_requirements_txt_ignores_comments_and_blank_lines(self):
        content = "# comment\n\nnumpy==1.26.4\n"
        assert repro_agent._dependencies_pinned_ratio("requirements.txt", content) == 1.0

    def test_package_json_pinned(self):
        content = json.dumps({"dependencies": {"react": "18.2.0"}})
        assert repro_agent._dependencies_pinned_ratio("package.json", content) == 1.0

    def test_package_json_caret_is_unpinned(self):
        content = json.dumps({"dependencies": {"react": "^18.2.0"}})
        assert repro_agent._dependencies_pinned_ratio("package.json", content) == 0.0

    def test_pyproject_toml_not_evaluated(self):
        assert repro_agent._dependencies_pinned_ratio("pyproject.toml", "[project]\nname='x'") is None


class TestScore:
    def test_all_pass_scores_100(self):
        checks = [
            repro_agent.CheckResult("a", "A", "pass", "", 60),
            repro_agent.CheckResult("b", "B", "pass", "", 40),
        ]
        score, verdict = repro_agent._score(checks)
        assert score == 100
        assert verdict == "Likely reproducible"

    def test_all_fail_scores_0(self):
        checks = [repro_agent.CheckResult("a", "A", "fail", "", 100)]
        score, verdict = repro_agent._score(checks)
        assert score == 0
        assert verdict == "Reproducibility at risk"

    def test_warn_counts_half(self):
        checks = [repro_agent.CheckResult("a", "A", "warn", "", 100)]
        score, _ = repro_agent._score(checks)
        assert score == 50

    def test_na_checks_excluded_from_denominator(self):
        checks = [
            repro_agent.CheckResult("a", "A", "pass", "", 50),
            repro_agent.CheckResult("b", "B", "na", "", 50),
        ]
        score, _ = repro_agent._score(checks)
        assert score == 100  # only the applicable (non-na) weight counts


class TestRunStaticChecks:
    def test_missing_everything_fails_most_checks(self, monkeypatch):
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_content", lambda *a, **k: None)
        metadata = {**GOOD_METADATA, "license_spdx_id": None, "license_name": None}
        checks = repro_agent._run_static_checks(metadata, tree=["src/main.py"], owner="foo", repo="bar", branch="main")
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "fail"
        assert by_id["has_license"].status == "fail"
        assert by_id["has_dependency_manifest"].status == "fail"
        assert by_id["has_tests"].status == "fail"
        assert by_id["has_ci"].status == "fail"

    def test_well_maintained_repo_passes_most_checks(self, monkeypatch):
        def fake_fetch(owner, repo, branch, path):
            if path == "README.md":
                return "# Bar\n\n## Install\n```pip install bar```"
            if path == "requirements.txt":
                return "numpy==1.26.4\n"
            return None

        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_content", fake_fetch)
        tree = [
            "README.md", "LICENSE", "requirements.txt", "Dockerfile", "CITATION.cff",
            ".github/workflows/ci.yml", "tests/test_main.py", "src/main.py",
        ]
        checks = repro_agent._run_static_checks(GOOD_METADATA, tree, owner="foo", repo="bar", branch="main")
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "pass"
        assert by_id["has_license"].status == "pass"
        assert by_id["license_is_recognized"].status == "pass"
        assert by_id["has_dependency_manifest"].status == "pass"
        assert by_id["dependencies_pinned"].status == "pass"
        assert by_id["has_tests"].status == "pass"
        assert by_id["has_ci"].status == "pass"
        assert by_id["has_dockerfile_or_env_spec"].status == "pass"
        assert by_id["readme_has_usage_instructions"].status == "pass"
        assert by_id["repo_not_archived"].status == "pass"
        assert by_id["has_citation_file"].status == "pass"

    def test_archived_repo_fails_maintenance_check(self, monkeypatch):
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_content", lambda *a, **k: None)
        metadata = {**GOOD_METADATA, "archived": True}
        checks = repro_agent._run_static_checks(metadata, tree=[], owner="foo", repo="bar", branch="main")
        by_id = {c.id: c for c in checks}
        assert by_id["repo_not_archived"].status == "fail"


class TestVerifyClaims:
    def test_no_method_text_returns_empty(self, fake_llm):
        paper = make_paper()
        assert repro_agent.verify_claims(paper, tree=["src/main.py"], readme_content=None) == []

    def test_parses_valid_llm_response(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder trained with Adam.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "claims": [
                {"claim": "Uses a transformer encoder", "verdict": "matches", "evidence": "src/model.py defines a Transformer class"},
                {"claim": "Releases pretrained checkpoints", "verdict": "not_evident", "evidence": "no supporting file/README evidence"},
            ]
        }))
        claims = repro_agent.verify_claims(paper, tree=["src/model.py"], readme_content="# Bar")
        assert len(claims) == 2
        assert claims[0]["verdict"] == "matches"

    def test_drops_malformed_items(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "claims": [
                {"claim": "", "verdict": "matches", "evidence": "x"},  # empty claim
                {"claim": "ok claim", "verdict": "bogus_verdict", "evidence": "x"},  # bad verdict
                {"claim": "good one", "verdict": "unclear", "evidence": "some reason"},
            ]
        }))
        claims = repro_agent.verify_claims(paper, tree=["src/model.py"], readme_content="# Bar")
        assert len(claims) == 1
        assert claims[0]["claim"] == "good one"

    def test_handles_invalid_json_gracefully(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        assert repro_agent.verify_claims(paper, tree=[], readme_content=None) == []


class TestAnalyze:
    def test_no_repo_url_found_or_supplied(self, fake_llm):
        paper = make_paper(full_text="No code link anywhere in this text.")
        result = repro_agent.analyze(paper)
        assert result["repo_url"] is None
        assert result["repo_metadata"] is None
        assert "no github repository url" in result["note"].lower() or "none was supplied" in result["note"].lower()

    def test_repo_fetch_error_is_captured_in_note(self, fake_llm, monkeypatch):
        paper = make_paper(full_text="Code at https://github.com/foo/bar.")

        def raise_error(owner, repo):
            raise RepoFetchError("Repository (or path) not found -- it may be private, moved, or deleted.")

        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_repo_metadata", raise_error)
        result = repro_agent.analyze(paper)
        assert result["repo_metadata"] is None
        assert "not found" in result["note"].lower()

    def test_explicit_repo_url_overrides_paper_text(self, fake_llm, monkeypatch):
        paper = make_paper(full_text="Code at https://github.com/wrong/repo.")
        seen = {}

        def fake_metadata(owner, repo):
            seen["owner_repo"] = (owner, repo)
            return GOOD_METADATA

        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_repo_metadata", fake_metadata)
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_tree", lambda *a, **k: ["README.md"])
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_content", lambda *a, **k: "# Right Repo")
        monkeypatch.setattr(repro_agent, "verify_claims", lambda *a, **k: [])

        result = repro_agent.analyze(paper, repo_url="https://github.com/right/repo")
        assert seen["owner_repo"] == ("right", "repo")
        assert result["repo_metadata"]["full_name"] == "foo/bar"

    def test_full_success_path_produces_score_and_claims(self, fake_llm, monkeypatch):
        paper = make_paper(
            full_text="Code at https://github.com/foo/bar.",
            method_text="We use a transformer encoder.",
        )
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_repo_metadata", lambda o, r: GOOD_METADATA)
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_tree", lambda *a, **k: [
            "README.md", "LICENSE", "requirements.txt", "tests/test_main.py", ".github/workflows/ci.yml",
        ])
        monkeypatch.setattr(
            repro_agent.repo_fetch, "fetch_file_content",
            lambda owner, repo, branch, path: {
                "README.md": "# Bar\n\n## Install\n```pip install bar```",
                "requirements.txt": "numpy==1.26.4\n",
            }.get(path),
        )
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "claims": [{"claim": "Uses a transformer", "verdict": "matches", "evidence": "seen in file tree"}]
        }))

        result = repro_agent.analyze(paper)
        assert result["repo_metadata"]["full_name"] == "foo/bar"
        assert isinstance(result["score"], int)
        assert result["verdict"]
        assert len(result["claims"]) == 1

    def test_llm_claims_disabled_skips_claim_check(self, fake_llm, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "REPRO_LLM_CLAIMS_ENABLED", False)
        paper = make_paper(full_text="Code at https://github.com/foo/bar.", method_text="Some method.")
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_repo_metadata", lambda o, r: GOOD_METADATA)
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_tree", lambda *a, **k: ["README.md"])
        monkeypatch.setattr(repro_agent.repo_fetch, "fetch_file_content", lambda *a, **k: "# Bar")

        called = {"claims": False}
        monkeypatch.setattr(repro_agent, "verify_claims", lambda *a, **k: called.__setitem__("claims", True) or [])

        result = repro_agent.analyze(paper)
        assert called["claims"] is False
        assert result["claims"] == []
