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

    def test_code_excerpts_are_included_in_prompt(self, fake_llm, monkeypatch):
        captured = {}

        def fake_complete(system, prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"claims": []})

        monkeypatch.setattr(fake_llm, "complete_json", fake_complete)
        paper = make_paper(method_text="We train a model.")
        rca.verify_claims(paper, tree=["train.py"], readme_content="# X", code_excerpts={"train.py": "def train(): pass"})
        assert "def train(): pass" in captured["prompt"]

    def test_no_code_excerpts_still_works(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"claims": []}))
        paper = make_paper(method_text="We train a model.")
        assert rca.verify_claims(paper, tree=["train.py"], readme_content="# X") == []


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

    def test_repo_profile_also_runs_semantic_review(self, fake_llm, monkeypatch):
        """This is the direct fix for the reported bug: a repo with real code
        but a terse README should no longer be judged only on file
        tree/README -- semantic_review now runs for profile="repo" too,
        using actual code content."""
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [])
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"coverage_score": 85, "findings": []}))
        files = {"README.md": "# X", "train.py": "def train():\n    pass"}
        paper = make_paper(method_text="We train a model.")
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=GOOD_METADATA, profile="repo")
        assert any(c["id"] == "semantic_review" for c in result["checks"])

    def test_semantic_review_weight_dominates_repo_profile_score(self):
        assert rca._REPO_CHECK_WEIGHTS["semantic_review"] == 70
        assert rca._REPO_CHECK_WEIGHTS["has_license"] == 0
        assert rca._REPO_CHECK_WEIGHTS["has_ci"] == 0

    def test_mixup_style_repo_scores_well_when_code_actually_implements_method(self, fake_llm, monkeypatch):
        """Regression test modeling the reported bug: a real repo (terse
        README, no test suite, no CI, archived) whose code genuinely
        implements the paper's method should NOT score near 0 just because
        claim verification only had README/file-tree evidence to work with.
        Semantic review, given the actual code, should recognize the real
        implementation and dominate the score."""
        monkeypatch.setattr(rca, "verify_claims", lambda *a, **k: [
            {"claim": "trains on convex combinations of examples", "verdict": "not_evident", "evidence": "no supporting file/README evidence"},
        ])
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "coverage_score": 90,
            "findings": [{"severity": "low", "issue": "Minor: no explicit alpha validation.", "location": "train.py"}],
        }))
        files = {
            "README.md": "# mixup-cifar10",
            "train.py": (
                "import numpy as np\n\n"
                "def mixup_data(x, y, alpha):\n"
                "    lam = np.random.beta(alpha, alpha)\n"
                "    index = np.random.permutation(x.size(0))\n"
                "    mixed_x = lam * x + (1 - lam) * x[index]\n"
                "    return mixed_x, y, y[index], lam\n"
            ),
        }
        metadata = {**GOOD_METADATA, "archived": True, "license_spdx_id": None, "license_name": None}
        paper = make_paper(method_text="We train a neural network on convex combinations of pairs of examples and their labels.")
        result = rca.evaluate(paper, sorted(files.keys()), files.get, metadata=metadata, profile="repo")
        # Archived status, missing license, no tests/CI no longer crush the
        # score -- they're informational warnings now, not scored deductions.
        assert result["score"] >= 70
        assert result["verdict"] == "Likely reproducible"


class TestPrioritizedCodeExcerpt:
    def test_returns_full_content_if_under_limit(self):
        content = "def foo(): pass"
        assert rca._prioritized_code_excerpt(content, 1000) == content

    def test_prioritizes_function_bodies_over_module_level_boilerplate(self):
        """Regression test for the real bug: mixup-cifar10's train.py has
        ~4000 chars of argparse/setup before def mixup_data() -- a naive
        content[:2000] prefix never reaches it, so the LLM correctly (and
        misleadingly) reported no mixup implementation in what it was shown."""
        boilerplate = "import argparse\n" + ("parser.add_argument('--x')\n" * 200)  # >2000 chars alone
        content = boilerplate + "\ndef mixup_data(x, y, alpha=1.0):\n    lam = 1\n    return x, y, lam\n"
        excerpt = rca._prioritized_code_excerpt(content, 500)
        assert "def mixup_data" in excerpt
        assert len(excerpt) <= 500 + 5  # small slack for the trailing partial-segment slice

    def test_falls_back_to_prefix_when_no_functions_or_classes(self):
        content = "x = 1\n" * 1000
        excerpt = rca._prioritized_code_excerpt(content, 100)
        assert excerpt == content[:100]

    def test_falls_back_to_prefix_on_syntax_error(self):
        content = "def broken(:\n" * 1000
        excerpt = rca._prioritized_code_excerpt(content, 50)
        assert excerpt == content[:50]

    def test_multiple_functions_included_in_file_order(self):
        content = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
        excerpt = rca._prioritized_code_excerpt(content, len(content) - 1)
        assert excerpt.index("def a") < excerpt.index("def b")


class TestSelectClaimCodeFiles:
    def test_prefers_shallow_paths(self):
        tree = ["src/deep/nested/model.py", "model.py"]
        assert rca._select_claim_code_files(tree)[0] == "model.py"

    def test_prefers_method_signal_filenames(self):
        tree = ["utils.py", "train.py"]
        selected = rca._select_claim_code_files(tree)
        assert selected[0] == "train.py"

    def test_excludes_test_files(self):
        tree = ["tests/test_model.py", "test_utils.py", "model.py"]
        selected = rca._select_claim_code_files(tree)
        assert "tests/test_model.py" not in selected
        assert "test_utils.py" not in selected

    def test_respects_max_files_config(self, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "REPRO_CLAIMS_MAX_CODE_FILES", 2)
        tree = ["a.py", "b.py", "c.py", "d.py"]
        assert len(rca._select_claim_code_files(tree)) == 2

    def test_no_py_files_returns_empty(self):
        assert rca._select_claim_code_files(["README.md", "requirements.txt"]) == []

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


class TestSelectClaimCodeFilesPaperSpecific:
    """Regression tests for the confirmed Cutout/SAM bug: paper-specific
    method files (named after the technique itself, not a generic term)
    were never selected."""

    def _cutout_paper(self):
        return IngestedPaper(
            title="Improved Regularization of Convolutional Neural Networks with Cutout",
            abstract="", sections=[], full_text="",
        )

    def test_paper_specific_filename_beats_generic_hints(self):
        """Exact tree from the confirmed bug report against
        uoguelph-mlrg/Cutout: util/cutout.py contains the entire method and
        must be selected; the old version picked train.py/model/__init__.py/
        model/resnet.py instead and never selected it at all."""
        tree = [
            "train.py", "model/__init__.py", "model/resnet.py",
            "model/wide_resnet.py", "util/__init__.py", "util/cutout.py", "util/misc.py",
        ]
        selected = rca._select_claim_code_files(tree, self._cutout_paper())
        assert "util/cutout.py" in selected
        assert selected[0] == "util/cutout.py"  # paper-specific match ranks first

    def test_dunder_files_deprioritized_below_everything_else(self):
        """model/__init__.py must not win purely on alphabetical ordering
        against a real (if unhinted) implementation file."""
        tree = ["model/__init__.py", "model/resnet.py"]
        selected = rca._select_claim_code_files(tree, self._cutout_paper())
        assert selected[0] == "model/resnet.py"

    def test_no_paper_falls_back_to_generic_hints_only(self):
        tree = ["util/cutout.py", "train.py"]
        selected = rca._select_claim_code_files(tree, paper=None)
        assert selected[0] == "train.py"  # generic hint, since no paper to derive from

    def test_depth_does_not_bias_selection(self):
        """Regression for the SAM case: a deeper, actually-relevant file
        must not lose to a shallower irrelevant one just for being shallower.
        Paper title includes the acronym in parens, as real papers virtually
        always do on first mention -- token derivation is title-text-only
        (see _paper_keyword_tokens), so an acronym absent from the title
        entirely is a known, documented limitation, not this bug."""
        paper = IngestedPaper(title="Sharpness-Aware Minimization (SAM) for Efficiently Improving Generalization", abstract="", sections=[], full_text="")
        tree = ["a_shallow_unrelated_file.py", "deep/nested/pkg/sam.py"]
        selected = rca._select_claim_code_files(tree, paper)
        assert selected[0] == "deep/nested/pkg/sam.py"

    def test_generic_title_words_are_not_treated_as_method_names(self):
        """'network', 'deep', 'model' etc. in a title shouldn't cause
        unrelated generically-named files to jump to tier 0."""
        paper = IngestedPaper(title="A Deep Neural Network Model for Improved Learning", abstract="", sections=[], full_text="")
        tokens = rca._paper_keyword_tokens(paper)
        assert tokens == set()  # every word in that title is a stopword


class TestFindManifestPathVendorExclusion:
    def test_excludes_vendored_setup_py(self):
        """Regression for the confirmed Random Erasing bug: a bundled
        third-party library's setup.py must not be picked as the project's
        own dependency manifest."""
        tree = ["vendor/some_lib/setup.py", "requirements.txt"]
        assert rca.find_manifest_path(tree) == "requirements.txt"

    def test_excludes_third_party_dir_variants(self):
        for vendor_dir in ("third_party", "thirdparty", "external", "node_modules", "site-packages"):
            tree = [f"{vendor_dir}/lib/setup.py"]
            assert rca.find_manifest_path(tree) is None

    def test_returns_none_if_only_vendored_manifest_exists(self):
        assert rca.find_manifest_path(["vendor/lib/requirements.txt"]) is None

    def test_still_prefers_root_over_nested_non_vendor(self):
        tree = ["examples/requirements.txt", "requirements.txt"]
        assert rca.find_manifest_path(tree) == "requirements.txt"


class TestPrioritizedCodeExcerptClassMethods:
    """Regression tests for the confirmed Lookahead bug: a class exceeding
    the per-file budget used to fall back to raw prefix-truncation of the
    whole class, cutting off later methods (e.g. step()) entirely."""

    LOOKAHEAD_LIKE = (
        "import torch\n\n"
        "class Lookahead:\n"
        "    def __init__(self, optimizer, k=5, alpha=0.5):\n"
        "        self.optimizer = optimizer\n"
        "        self.k = k\n\n"
        "    def step(self, closure=None):\n"
        "        loss = self.optimizer.step(closure)\n"
        "        return loss\n\n"
        "    def zero_grad(self):\n"
        "        self.optimizer.zero_grad()\n"
    )

    def test_class_broken_into_individual_complete_methods(self):
        import ast as _ast
        segments = rca._flatten_prioritizable_segments(_ast.parse(self.LOOKAHEAD_LIKE), self.LOOKAHEAD_LIKE)
        assert len(segments) == 3  # __init__, step, zero_grad as separate segments
        assert all("def " in s for s in segments)

    def test_later_method_survives_truncation_when_it_fits(self):
        budget = len(self.LOOKAHEAD_LIKE) - 20  # forces truncation, but all 3 methods still fit
        excerpt = rca._prioritized_code_excerpt(self.LOOKAHEAD_LIKE, budget)
        assert "def step" in excerpt and "return loss" in excerpt

    def test_first_method_survives_a_tight_budget_intact(self):
        # first segment (comment prefix + __init__ source) is exactly 130 chars --
        # 150 is tight but sufficient to hold it whole, per the design's own
        # rule of only keeping a segment complete when it actually fits.
        excerpt = rca._prioritized_code_excerpt(self.LOOKAHEAD_LIKE, 150)
        assert "def __init__(self, optimizer, k=5, alpha=0.5):" in excerpt
        assert "self.k = k" in excerpt

    def test_class_with_no_methods_falls_back_to_full_source(self):
        import ast as _ast
        content = "class Config:\n    x = 1\n    y = 2\n"
        segments = rca._flatten_prioritizable_segments(_ast.parse(content), content)
        assert len(segments) == 1
        assert "class Config" in segments[0]