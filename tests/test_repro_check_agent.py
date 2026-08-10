"""Tests for repro_check_agent.py. Source-agnostic -- no GitHub or generator
specifics here, just (tree, get_content, metadata) -> checks."""
import json

import pytest

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

WELL_MAINTAINED_FILES = {
    "README.md": "# Bar\n\n## Install\n```pip install -r requirements.txt```",
    "LICENSE": "MIT", "requirements.txt": "torch==2.1.0\n",
    "tests/test_x.py": "def test(): pass", ".github/workflows/ci.yml": "name: ci",
}
SPARSE_BUT_REAL_FILES = {"README.md": "# nanoGPT\n\n## install\npip install torch numpy", "LICENSE": "MIT"}


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
        checks = [rca.CheckResult("a", "A", "pass", "", 60, "correctness"), rca.CheckResult("b", "B", "pass", "", 40, "correctness")]
        pct, verdict = rca.score(checks)
        assert pct == 100
        assert verdict == "Likely reproducible"

    def test_all_fail_scores_0(self):
        pct, verdict = rca.score([rca.CheckResult("a", "A", "fail", "", 100, "correctness")])
        assert pct == 0
        assert verdict == "Reproducibility at risk"

    def test_na_checks_excluded_from_denominator(self):
        checks = [rca.CheckResult("a", "A", "pass", "", 50, "correctness"), rca.CheckResult("b", "B", "na", "", 50, "correctness")]
        pct, _ = rca.score(checks)
        assert pct == 100

    def test_warning_category_excluded_from_score_entirely(self):
        checks = [
            rca.CheckResult("a", "A", "pass", "", 100, "correctness"),
            rca.CheckResult("b", "B", "fail", "", 999, "warning"),  # huge weight -- must not affect score
        ]
        pct, _ = rca.score(checks)
        assert pct == 100


class TestCategoryScores:
    def test_breaks_down_by_category(self):
        checks = [
            rca.CheckResult("a", "A", "pass", "", 100, "documentation"),
            rca.CheckResult("b", "B", "fail", "", 100, "hygiene"),
        ]
        result = rca.category_scores(checks)
        assert result["documentation"] == 100
        assert result["hygiene"] == 0

    def test_category_with_no_checks_is_none(self):
        checks = [rca.CheckResult("a", "A", "pass", "", 100, "documentation")]
        result = rca.category_scores(checks)
        assert result["code_quality"] is None
        assert result["correctness"] is None

    def test_all_checks_covers_all_four_categories(self):
        result = rca.category_scores([])
        assert set(result.keys()) == set(rca.CATEGORIES)


class TestRunChecksSourceAgnostic:
    def test_works_against_a_github_style_source(self):
        checks = rca.run_checks(sorted(WELL_MAINTAINED_FILES.keys()), WELL_MAINTAINED_FILES.get, GOOD_METADATA, profile="repo")
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "pass"
        assert by_id["has_license"].status == "pass"
        assert by_id["dependencies_pinned"].status == "pass"
        assert by_id["has_tests"].status == "pass"
        assert by_id["has_ci"].status == "pass"

    def test_missing_everything_fails_most_checks(self):
        checks = rca.run_checks(["src/main.py"], lambda p: None, metadata={**GOOD_METADATA, "license_spdx_id": None, "license_name": None}, profile="repo")
        by_id = {c.id: c for c in checks}
        assert by_id["has_readme"].status == "fail"
        assert by_id["has_license"].status == "fail"
        assert by_id["has_tests"].status == "fail"

    def test_archived_repo_fails_maintenance_check(self):
        checks = rca.run_checks([], lambda p: None, metadata={**GOOD_METADATA, "archived": True}, profile="repo")
        by_id = {c.id: c for c in checks}
        assert by_id["repo_not_archived"].status == "fail"


class TestWeightProfiles:
    def test_both_profiles_sum_to_100_scored_weight(self):
        # warning-category checks are weight=0 by construction -- only
        # scored categories need to sum to 100
        scored_repo = {k: v for k, v in rca._REPO_CHECK_WEIGHTS.items() if rca._CHECK_CATEGORY[k] != "warning"}
        scored_generated = {k: v for k, v in rca._GENERATED_WEIGHTS.items() if rca._CHECK_CATEGORY[k] != "warning"}
        assert sum(scored_repo.values()) == 100
        assert sum(scored_generated.values()) == 100

    def test_generated_profile_excludes_repo_only_hygiene_checks(self):
        files = {"README.md": "# X\n\n## Usage\n```run it```", "model.py": "class Model: pass"}
        checks = rca.run_checks(sorted(files.keys()), files.get, metadata=None, profile="generated")
        ids = {c.id for c in checks}
        assert "has_license" not in ids
        assert "license_is_recognized" not in ids
        assert "has_ci" not in ids
        assert "repo_not_archived" not in ids
        assert "has_citation_file" not in ids
        assert "static_analysis_clean" in ids

    def test_repo_profile_still_includes_hygiene_checks(self):
        files = {"README.md": "# X"}
        checks = rca.run_checks(sorted(files.keys()), files.get, metadata=GOOD_METADATA, profile="repo")
        ids = {c.id for c in checks}
        assert "has_license" in ids
        assert "has_ci" in ids
        assert "repo_not_archived" in ids
        assert "has_citation_file" in ids
        assert "static_analysis_clean" not in ids

    def test_dockerfile_and_citation_checks_are_warning_category(self):
        checks = rca.run_checks(["README.md"], lambda p: "# X", metadata=GOOD_METADATA, profile="repo")
        by_id = {c.id: c for c in checks}
        assert by_id["has_dockerfile_or_env_spec"].category == "warning"
        assert by_id["has_citation_file"].category == "warning"


class TestRepoScoringIsCalibrated:
    """The actual complaint this fixes: a real, well-known-but-minimal repo
    (no manifest/tests/CI) should score moderately -- reflecting a genuine
    gap -- not artificially high or artificially crushed; a genuinely
    well-maintained repo should score near 100."""

    def test_sparse_but_real_repo_scores_moderately_not_crushed(self, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        checks = rca.run_checks(sorted(SPARSE_BUT_REAL_FILES.keys()), SPARSE_BUT_REAL_FILES.get, GOOD_METADATA, profile="repo")
        scored = [c for c in checks if c.category != "warning"]
        pct, _ = rca.score(scored)
        assert 40 <= pct <= 65  # real gaps (no manifest/tests/CI) genuinely cost points, but not devastatingly

    def test_well_maintained_repo_scores_near_100(self, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        checks = rca.run_checks(sorted(WELL_MAINTAINED_FILES.keys()), WELL_MAINTAINED_FILES.get, GOOD_METADATA, profile="repo")
        scored = [c for c in checks if c.category != "warning"]
        pct, _ = rca.score(scored)
        assert pct == 100

    def test_generated_profile_not_penalized_for_missing_license_ci_citation(self, fake_llm, monkeypatch):
        """The bug directly reported: generated code (which will never have a
        LICENSE/CI/CITATION file) shouldn't be scored down for lacking them."""
        monkeypatch.setattr(rca, "_static_analysis_check", lambda tree, get_content, weight: rca.CheckResult(
            "static_analysis_clean", "Passes static analysis (ruff)", "pass", "No issues found.", weight, "code_quality",
        ))
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"coverage_score": 100, "findings": []}))
        files = {
            "README.md": "# Model\n\n## Install\n```pip install -r requirements.txt```",
            "requirements.txt": "torch==2.1.0\n",
            "model.py": "class Model: pass",
            "tests/test_model.py": "def test_model(): pass",
        }
        paper = make_paper(method_text="We use a transformer.")
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=None, profile="generated")
        assert result["score"] == 100
        assert result["verdict"] == "Likely reproducible"
        ids = {c["id"] for c in result["checks"]}
        assert "has_license" not in ids
        assert "has_ci" not in ids


class TestStaticAnalysisCheck:
    def test_no_python_files_is_na(self):
        result = rca._static_analysis_check(["README.md", "requirements.txt"], lambda p: "content", weight=25)
        assert result.status == "na"

    def test_ruff_unavailable_is_na(self, monkeypatch):
        monkeypatch.setattr(rca, "_run_ruff_on_file", lambda content, timeout: None)
        result = rca._static_analysis_check(["model.py"], lambda p: "class Model: pass", weight=25)
        assert result.status == "na"

    def test_clean_code_passes(self, monkeypatch):
        monkeypatch.setattr(rca, "_run_ruff_on_file", lambda content, timeout: 0)
        result = rca._static_analysis_check(["model.py"], lambda p: "class Model: pass", weight=25)
        assert result.status == "pass"

    def test_few_issues_warns(self, monkeypatch):
        monkeypatch.setattr(rca, "_run_ruff_on_file", lambda content, timeout: 2)
        result = rca._static_analysis_check(["model.py"], lambda p: "class Model: pass", weight=25)
        assert result.status == "warn"

    def test_many_issues_fails(self, monkeypatch):
        monkeypatch.setattr(rca, "_run_ruff_on_file", lambda content, timeout: 10)
        result = rca._static_analysis_check(["model.py"], lambda p: "class Model: pass", weight=25)
        assert result.status == "fail"


class TestRunRuffOnFileIntegration:
    @pytest.fixture(autouse=True)
    def _require_ruff(self):
        import shutil
        if shutil.which("ruff") is None:
            pytest.skip("ruff not installed in this environment")

    def test_clean_code_returns_zero(self):
        assert rca._run_ruff_on_file("def add(a, b):\n    return a + b\n", timeout=10) == 0

    def test_undefined_name_is_detected(self):
        count = rca._run_ruff_on_file("def foo():\n    return totally_undefined_name\n", timeout=10)
        assert count is not None and count >= 1

    def test_cosmetic_import_order_is_not_flagged(self):
        content = "import sys\nimport os\n\ndef foo():\n    return os.path.join(sys.prefix, 'x')\n"
        assert rca._run_ruff_on_file(content, timeout=10) == 0


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


class TestSemanticReview:
    def test_no_method_text_returns_none_score(self, fake_llm):
        result = rca.semantic_review(make_paper(), {"model.py": "class Model: pass"})
        assert result["coverage_score"] is None
        assert result["findings"] == []

    def test_no_py_files_returns_none_score(self, fake_llm):
        paper = make_paper(method_text="We use a transformer.")
        result = rca.semantic_review(paper, {})
        assert result["coverage_score"] is None

    def test_parses_valid_response(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer encoder with self-attention.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "coverage_score": 65,
            "findings": [{"severity": "high", "issue": "No self-attention mechanism found", "location": "model.py"}],
        }))
        result = rca.semantic_review(paper, {"model.py": "class Model: pass"})
        assert result["coverage_score"] == 65
        assert len(result["findings"]) == 1
        assert result["findings"][0]["severity"] == "high"

    def test_invalid_json_returns_none_score(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        result = rca.semantic_review(paper, {"model.py": "x = 1"})
        assert result["coverage_score"] is None

    def test_out_of_range_score_is_dropped(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"coverage_score": 150, "findings": []}))
        result = rca.semantic_review(paper, {"model.py": "x = 1"})
        assert result["coverage_score"] is None

    def test_malformed_findings_are_dropped(self, fake_llm, monkeypatch):
        paper = make_paper(method_text="We use a transformer.")
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "coverage_score": 80,
            "findings": [
                {"severity": "bogus", "issue": "x", "location": "a.py"},  # bad severity
                {"severity": "low", "issue": "", "location": "a.py"},  # empty issue
                {"severity": "medium", "issue": "real issue", "location": "b.py"},
            ],
        }))
        result = rca.semantic_review(paper, {"model.py": "x = 1"})
        assert len(result["findings"]) == 1
        assert result["findings"][0]["issue"] == "real issue"


class TestEvaluate:
    def test_returns_unified_shape(self, fake_llm, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        paper = make_paper(method_text="We use a transformer.")
        files = {"README.md": "# X", "requirements.txt": "numpy==1.0.0"}
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=GOOD_METADATA, profile="repo")
        assert set(result.keys()) == {"checks", "warnings", "score", "category_scores", "verdict", "claims", "semantic_findings"}
        assert isinstance(result["score"], int)

    def test_repo_profile_has_no_semantic_findings(self, fake_llm, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        paper = make_paper()
        result = rca.evaluate(paper, ["README.md"], lambda p: "# X", metadata=GOOD_METADATA, profile="repo")
        assert result["semantic_findings"] == []
        assert "semantic_review" not in {c["id"] for c in result["checks"]}

    def test_generated_profile_runs_semantic_review(self, fake_llm, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        monkeypatch.setattr(rca, "_static_analysis_check", lambda tree, get_content, weight: rca.CheckResult(
            "static_analysis_clean", "Passes static analysis (ruff)", "pass", "ok", weight, "code_quality",
        ))
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"coverage_score": 70, "findings": [{"severity": "medium", "issue": "x", "location": "model.py"}]}))
        paper = make_paper(method_text="We use a transformer.")
        files = {"model.py": "class Model: pass"}
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=None, profile="generated")
        assert any(c["id"] == "semantic_review" for c in result["checks"])
        assert len(result["semantic_findings"]) == 1

    def test_semantic_review_disabled_skips_it(self, fake_llm, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        monkeypatch.setattr(rca, "_static_analysis_check", lambda tree, get_content, weight: rca.CheckResult(
            "static_analysis_clean", "Passes static analysis (ruff)", "pass", "ok", weight, "code_quality",
        ))
        paper = make_paper(method_text="We use a transformer.")
        files = {"model.py": "class Model: pass"}
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=None, profile="generated")
        assert "semantic_review" not in {c["id"] for c in result["checks"]}
        assert result["semantic_findings"] == []

    def test_warnings_are_separated_from_checks(self, fake_llm, monkeypatch):
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        paper = make_paper()
        files = {"README.md": "# X"}
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=GOOD_METADATA, profile="repo")
        check_ids = {c["id"] for c in result["checks"]}
        warning_ids = {w["id"] for w in result["warnings"]}
        assert "has_dockerfile_or_env_spec" not in check_ids
        assert "has_citation_file" not in check_ids
        assert "has_dockerfile_or_env_spec" in warning_ids
        assert "has_citation_file" in warning_ids
