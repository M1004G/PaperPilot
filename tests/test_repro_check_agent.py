"""Tests for repro_check_agent.py. Source-agnostic -- no GitHub or generator
specifics here, just (tree, get_content, metadata) -> checks."""
import json

from backend import repro_check_agent as rca
from backend.ingestion_agent import IngestedPaper, Section


def make_paper(method_text=""):
    sections = [Section(heading="Methodology", text=method_text)] if method_text else []
    return IngestedPaper(title="Test Paper", abstract="An abstract.", sections=sections, full_text="")


GOOD_METADATA = {
    "full_name": "foo/bar", "default_branch": "main", "archived": False,
    "license_spdx_id": "mit", "license_name": "MIT License",
    "html_url": "https://github.com/foo/bar", "stargazers_count": 10, "pushed_at": "2026-01-01",
}


class TestFindReadmeAndManifest:
    def test_finds_top_level_readme(self):
        assert rca.find_readme_path(["README.md", "src/main.py"]) == "README.md"

    def test_ignores_nested_readme(self):
        assert rca.find_readme_path(["docs/README.md", "src/main.py"]) is None

    def test_finds_root_requirements_over_nested(self):
        assert rca.find_manifest_path(["examples/requirements.txt", "requirements.txt"]) == "requirements.txt"

    def test_no_manifest_returns_none(self):
        assert rca.find_manifest_path(["src/main.py"]) is None


class TestDependenciesPinnedRatio:
    def test_requirements_txt_fully_pinned(self):
        assert rca._dependencies_pinned_ratio("requirements.txt", "numpy==1.26.4\npandas==2.2.2\n") == 1.0

    def test_requirements_txt_unpinned(self):
        assert rca._dependencies_pinned_ratio("requirements.txt", "numpy\npandas\n") == 0.0

    def test_package_json_caret_is_unpinned(self):
        content = json.dumps({"dependencies": {"react": "^18.2.0"}})
        assert rca._dependencies_pinned_ratio("package.json", content) == 0.0


class TestScore:
    def test_all_pass_scores_100(self):
        checks = [rca.CheckResult("a", "A", "pass", "", 60), rca.CheckResult("b", "B", "pass", "", 40)]
        pct, verdict = rca.score(checks)
        assert pct == 100
        assert verdict == "Likely reproducible"

    def test_all_fail_scores_0(self):
        pct, verdict = rca.score([rca.CheckResult("a", "A", "fail", "", 100)])
        assert pct == 0
        assert verdict == "Reproducibility at risk"

    def test_na_checks_excluded_from_denominator(self):
        checks = [rca.CheckResult("a", "A", "pass", "", 50), rca.CheckResult("b", "B", "na", "", 50)]
        pct, _ = rca.score(checks)
        assert pct == 100


class TestRunChecksSourceAgnostic:
    """The whole point of the refactor: run_checks doesn't care whether
    `get_content` is backed by a GitHub API call or a plain dict.get."""

    def test_works_against_a_github_style_source(self):
        files = {"README.md": "# Bar\n\n## Install\n```pip install bar```", "requirements.txt": "numpy==1.26.4\n"}
        tree = ["README.md", "LICENSE", "requirements.txt", "tests/test_main.py", ".github/workflows/ci.yml"]
        checks = rca.run_checks(tree, files.get, GOOD_METADATA)
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "pass"
        assert by_id["has_license"].status == "pass"
        assert by_id["dependencies_pinned"].status == "pass"
        assert by_id["has_tests"].status == "pass"
        assert by_id["has_ci"].status == "pass"

    def test_works_against_a_generated_code_dict_with_no_metadata(self):
        files = {
            "README.md": "# Generated\n\n## Usage\n```python train.py```",
            "requirements.txt": "torch==2.1.0\n",
            "train.py": "print('train')",
        }
        tree = sorted(files.keys())
        checks = rca.run_checks(tree, files.get, metadata=None)
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "pass"
        # No metadata at all -> license/archived checks come back "na", not "fail"
        assert by_id["has_license"].status == "na"
        assert by_id["repo_not_archived"].status == "na"
        assert by_id["has_tests"].status == "fail"  # generator didn't produce tests

    def test_missing_everything_fails_most_checks(self):
        checks = rca.run_checks(["src/main.py"], lambda p: None, metadata={**GOOD_METADATA, "license_spdx_id": None, "license_name": None})
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "fail"
        assert by_id["has_license"].status == "fail"
        assert by_id["has_tests"].status == "fail"

    def test_archived_repo_fails_maintenance_check(self):
        checks = rca.run_checks([], lambda p: None, metadata={**GOOD_METADATA, "archived": True})
        by_id = {c.id: c for c in checks}
        assert by_id["repo_not_archived"].status == "fail"


class TestVerifyClaims:
    def test_no_method_text_returns_empty(self, fake_llm):
        assert rca.verify_claims(make_paper(), tree=["src/main.py"], readme_content=None) == []

    def test_parses_valid_llm_response(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder trained with Adam.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "claims": [{"claim": "Uses a transformer encoder", "verdict": "matches", "evidence": "model.py defines a Transformer class"}]
        }))
        claims = rca.verify_claims(paper, tree=["model.py"], readme_content="# Bar")
        assert len(claims) == 1
        assert claims[0]["verdict"] == "matches"

    def test_handles_invalid_json_gracefully(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        assert rca.verify_claims(paper, tree=[], readme_content=None) == []


class TestEvaluate:
    def test_returns_unified_shape(self, fake_llm, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        paper = make_paper(method_text="We use a transformer.")
        files = {"README.md": "# X", "requirements.txt": "numpy==1.0.0"}
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=None)
        assert set(result.keys()) == {"checks", "score", "verdict", "claims"}
        assert isinstance(result["score"], int)
