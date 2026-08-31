"""Reproducibility Check Agent: source-agnostic code-quality checks + LLM
review, organized into four scored categories (plus a separate, non-scored
Warnings list) rather than one opaque number.

Deliberately doesn't know or care where the code came from -- it operates on
a plain (paths, get_content, metadata) triple. Two sources feed it:
- repo_fetch.py (a real GitHub repo linked in the paper)
- codegen_agent.py (code PaperPilot generated itself, when no repo exists)

Categories:
- documentation  -- README presence + usage instructions
- hygiene        -- license, CI, dependency manifest, archived status
                     (real repos only -- see the profile note below)
- code_quality   -- ruff static analysis (generated code only)
- correctness    -- tests, pinned dependencies, and (generated code only) an
                     LLM semantic review comparing the code against the paper

Two profiles, not one weighting -- a real repo and code generated fresh in
this session answer different questions:
- profile="repo": full hygiene checklist scored; no code_quality or semantic
  review (we don't fetch every source file's content from GitHub just to
  lint it -- see repo_fetch.py's docstring on why that's a deliberate cost
  tradeoff).
- profile="generated": license/CI/archived-status checks are dropped
  entirely (not run, not shown as N/A clutter) -- code generated fresh in
  one session was never going to have a LICENSE file or CI, and scoring it
  down for that conflates repo-maintenance hygiene with implementation
  quality. In their place: ruff (code_quality) and an LLM semantic review
  (correctness) -- the closest available correctness-adjacent signals
  without executing anything.

Checks that are informational rather than a real pass/fail signal (missing
Dockerfile, missing CITATION file) are collected separately as `warnings`
and excluded from scoring entirely, rather than silently shaping the number.
"""
import ast
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable

from backend import config, llm_client
from backend.ingestion_agent import IngestedPaper

logger = logging.getLogger("paperpilot.repro_check")

CLAIM_SYSTEM_PROMPT = (
    "You are a careful, skeptical code reviewer checking whether a codebase plausibly "
    "implements what a paper describes. You only have a file listing and a README/excerpt, "
    "not the full code, so you flag things as 'unclear' rather than guessing when the "
    "evidence genuinely doesn't say enough. You never claim a match unless the file "
    "listing or README/excerpt gives concrete evidence for it."
)

SEMANTIC_REVIEW_SYSTEM_PROMPT = (
    "You are a rigorous ML code reviewer comparing a generated implementation against "
    "the paper's described method. You look for missing algorithmic steps, calls to "
    "APIs/functions that don't plausibly exist, likely tensor/array shape mismatches, "
    "and other correctness concerns -- not style or formatting. You are specific, cite "
    "the file where an issue occurs, and never invent a problem that isn't evident from "
    "the code actually shown to you."
)

METHOD_HEADINGS = {
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experimental setup", "experiments", "implementation",
}

CATEGORIES = ("documentation", "hygiene", "code_quality", "correctness")

# Category each check belongs to. "warning" is not a scored category --
# checks with this category are informational only (see evaluate()).
# has_license/license_is_recognized/has_ci/repo_not_archived only ever fire
# for profile="repo", so demoting them to "warning" here can't affect
# profile="generated" scoring. has_tests and dependencies_pinned are shared
# across both profiles -- their category is decided per-profile in
# run_checks() instead (see _tests_and_deps_category below), since for
# generated code they're still a meaningful correctness signal but for a
# fetched repo they're SWE-hygiene, not evidence the method is implemented
# correctly (a repo can have full test coverage around a wrong algorithm,
# or zero tests around a correct one -- see semantic_review for the actual
# correctness signal).
_CHECK_CATEGORY = {
    "has_readme": "documentation",
    "readme_has_usage_instructions": "documentation",
    "has_license": "warning",
    "license_is_recognized": "warning",
    "has_dependency_manifest": "hygiene",
    "has_ci": "warning",
    "repo_not_archived": "warning",
    "dependencies_pinned": "correctness",
    "has_tests": "correctness",
    "static_analysis_clean": "code_quality",
    "semantic_review": "correctness",
    "has_dockerfile_or_env_spec": "warning",
    "has_citation_file": "warning",
}


def _tests_and_deps_category(check_id: str, profile: str) -> str:
    """has_tests/dependencies_pinned are structural presence checks -- real
    correctness signal for profile="generated" (there's no other test-
    coverage signal there), but for profile="repo" they measure SWE hygiene,
    not whether the code implements the paper's method. Demoted to warning
    only for repo profile; generated profile keeps the base category."""
    if profile == "repo" and check_id in ("has_tests", "dependencies_pinned"):
        return "warning"
    return _CHECK_CATEGORY[check_id]


# Weights sum to 100 within each profile's *scored* checks (warning-category
# checks aren't weighted -- their weight is irrelevant to scoring).
# Repo-profile weighting: semantic_review (does the code's actual content
# implement the paper's method, per LLM review of real code excerpts) now
# dominates the score. License/CI/archived-status/tests/dependency-pinning
# are demoted to warnings above -- they're software-engineering hygiene,
# not evidence of implementation correctness, and were previously ~50-65%
# of the score despite measuring something orthogonal to reproducibility.
_REPO_CHECK_WEIGHTS = {
    "has_readme": 10,
    "readme_has_usage_instructions": 5,
    "has_license": 0,
    "license_is_recognized": 0,
    "has_dependency_manifest": 15,
    "has_ci": 0,
    "repo_not_archived": 0,
    "dependencies_pinned": 0,
    "has_tests": 0,
    "semantic_review": 70,
    "has_dockerfile_or_env_spec": 0,
    "has_citation_file": 0,
}
_GENERATED_WEIGHTS = {
    "has_readme": 10,
    "readme_has_usage_instructions": 10,
    "has_dependency_manifest": 15,
    "static_analysis_clean": 25,
    "dependencies_pinned": 10,
    "has_tests": 10,
    "semantic_review": 20,
    "has_dockerfile_or_env_spec": 0,
}

DEPENDENCY_MANIFESTS = [
    "requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml",
    "Pipfile", "package.json", "setup.py", "Cargo.toml", "go.mod",
]
RECOGNIZED_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "gpl-3.0", "gpl-2.0",
    "lgpl-3.0", "mpl-2.0", "cc0-1.0", "cc-by-4.0", "unlicense",
}

GetContent = Callable[[str], str | None]


@dataclass
class CheckResult:
    id: str
    label: str
    status: str  # "pass" | "fail" | "warn" | "na"
    detail: str
    weight: int = 0
    category: str = "correctness"


_VENDOR_DIR_MARKERS = {
    "vendor", "vendored", "third_party", "thirdparty", "external", "extern",
    "deps", "node_modules", "site-packages", ".venv", "venv",
}


def find_manifest_path(tree: list[str]) -> str | None:
    """First dependency manifest found, preferring root-level files over
    ones buried in subdirectories (a manifest in examples/ isn't the
    project's real one). Vendored/third-party copies are excluded outright,
    not just deprioritized -- a bundled dependency's own setup.py has
    nothing to do with what THIS project depends on, and depth-sorting
    alone doesn't stop a vendored manifest from being picked when it's
    shallower than (or ties with) the real one."""
    names = {name.lower(): name for name in DEPENDENCY_MANIFESTS}
    candidates = []
    for path in tree:
        filename = path.rsplit("/", 1)[-1]
        if filename not in DEPENDENCY_MANIFESTS and filename.lower() not in names:
            continue
        dir_segments = {seg.lower() for seg in path.split("/")[:-1]}
        if dir_segments & _VENDOR_DIR_MARKERS:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.count("/"))[0]


def find_readme_path(tree: list[str]) -> str | None:
    for path in tree:
        if "/" not in path and path.lower().startswith("readme"):
            return path
    return None


def _dependencies_pinned_ratio(manifest_path: str, content: str) -> float | None:
    """Rough heuristic, format-aware enough to not misjudge pyproject.toml/package.json
    (which pin very differently than requirements.txt) as unpinned."""
    if manifest_path.endswith(".txt"):
        lines = [
            l.strip() for l in content.splitlines()
            if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("-")
        ]
        if not lines:
            return None
        pinned = sum(1 for l in lines if any(op in l for op in ("==", "~=")))
        return pinned / len(lines)
    if manifest_path == "package.json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        if not deps:
            return None
        pinned = sum(1 for v in deps.values() if isinstance(v, str) and not v.startswith(("^", "~", ">", "*")))
        return pinned / len(deps)
    return None


def _run_ruff_on_file(content: str, timeout: float) -> int | None:
    """Lints one file's content with ruff (no execution). Returns the issue
    count, or None if ruff genuinely couldn't be run (not installed, timed
    out, errored) -- distinct from 'ran clean', never silently treated as
    a failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(content)
            tmp_path = tf.name
        # --isolated: ignore any ambient pyproject.toml/ruff.toml. --select=F:
        # pyflakes only (undefined names, unused imports/vars, redefinitions)
        # -- correctness-adjacent, NOT formatting/import-sort/line-length,
        # which would tank the score on trivia unrelated to whether the code
        # makes sense.
        result = subprocess.run(
            ["ruff", "check", "--isolated", "--select=F", "--output-format=json", "--quiet", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("ruff_unavailable_or_failed error=%s", e)
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    try:
        return len(json.loads(result.stdout)) if result.stdout.strip() else 0
    except json.JSONDecodeError:
        logger.warning("ruff_output_unparseable")
        return None


def _static_analysis_check(tree: list[str], get_content: GetContent, weight: int) -> CheckResult:
    py_files = [p for p in tree if p.endswith(".py")]
    if not py_files:
        return CheckResult("static_analysis_clean", "Passes static analysis (ruff)", "na", "No Python files to analyze.", weight, "code_quality")

    total_issues = 0
    files_checked = 0
    for path in py_files:
        content = get_content(path)
        if content is None:
            continue
        count = _run_ruff_on_file(content, config.REPRO_RUFF_TIMEOUT_SECONDS)
        if count is None:
            return CheckResult(
                "static_analysis_clean", "Passes static analysis (ruff)", "na",
                "ruff isn't available in this environment -- static analysis skipped.", weight, "code_quality",
            )
        total_issues += count
        files_checked += 1

    if files_checked == 0:
        return CheckResult("static_analysis_clean", "Passes static analysis (ruff)", "na", "No Python file contents were available to analyze.", weight, "code_quality")
    if total_issues == 0:
        return CheckResult("static_analysis_clean", "Passes static analysis (ruff)", "pass", f"No issues found across {files_checked} Python file(s).", weight, "code_quality")
    status = "warn" if total_issues <= 3 else "fail"
    return CheckResult(
        "static_analysis_clean", "Passes static analysis (ruff)", status,
        f"{total_issues} issue(s) found across {files_checked} Python file(s) (undefined names, unused imports, likely bugs, etc.).",
        weight, "code_quality",
    )


def run_checks(tree: list[str], get_content: GetContent, metadata: dict | None = None, profile: str = "repo") -> list[CheckResult]:
    """Pure, source-agnostic static checks (no LLM). `profile` selects the
    weighting AND which checks even apply -- see the module docstring."""
    metadata = metadata or {}
    weights = _REPO_CHECK_WEIGHTS if profile == "repo" else _GENERATED_WEIGHTS
    cat = _CHECK_CATEGORY
    checks: list[CheckResult] = []

    readme_path = find_readme_path(tree)
    readme_content = get_content(readme_path) if readme_path else None

    checks.append(CheckResult(
        "has_readme", "Has a README",
        "pass" if readme_path else "fail",
        f"Found `{readme_path}`" if readme_path else "No top-level README file found.",
        weights["has_readme"], cat["has_readme"],
    ))

    if profile == "repo":
        has_license_file = any(p.upper().startswith("LICENSE") and "/" not in p for p in tree)
        has_license = bool(metadata.get("license_spdx_id")) or has_license_file
        if not metadata and not has_license_file:
            license_status, license_detail = "na", "No repo metadata available and no LICENSE file to check."
        else:
            license_status = "pass" if has_license else "fail"
            license_detail = metadata.get("license_name") or ("LICENSE file present" if has_license_file else "No license detected.")
        checks.append(CheckResult("has_license", "Has a license", license_status, license_detail, weights["has_license"], cat["has_license"]))

        spdx = (metadata.get("license_spdx_id") or "").lower()
        if not has_license:
            recognized_status, recognized_detail = "na", "No license to evaluate."
        elif spdx in RECOGNIZED_LICENSES:
            recognized_status, recognized_detail = "pass", f"'{spdx}' is a widely recognized OSI-style license."
        elif spdx:
            recognized_status, recognized_detail = "warn", f"License present but not auto-recognized ('{spdx}') -- check terms manually."
        else:
            recognized_status, recognized_detail = "na", "License present but its identifier isn't known (e.g. no repo metadata)."
        checks.append(CheckResult(
            "license_is_recognized", "License is a recognized open license",
            recognized_status, recognized_detail, weights["license_is_recognized"], cat["license_is_recognized"],
        ))

    manifest_path = find_manifest_path(tree)
    checks.append(CheckResult(
        "has_dependency_manifest", "Has a dependency manifest",
        "pass" if manifest_path else "fail",
        f"Found `{manifest_path}`" if manifest_path else "No requirements.txt/pyproject.toml/package.json/etc. found.",
        weights["has_dependency_manifest"], cat["has_dependency_manifest"],
    ))

    if manifest_path:
        manifest_content = get_content(manifest_path)
        ratio = _dependencies_pinned_ratio(manifest_path, manifest_content) if manifest_content else None
        if ratio is None:
            status, detail = "na", "Pinning not evaluated for this manifest format (presence alone counted above)."
        elif ratio >= 0.8:
            status, detail = "pass", f"{ratio:.0%} of dependencies are version-pinned."
        elif ratio >= 0.3:
            status, detail = "warn", f"Only {ratio:.0%} of dependencies are version-pinned."
        else:
            status, detail = "fail", f"{ratio:.0%} of dependencies are version-pinned -- reproducibility risk."
    else:
        status, detail = "na", "No manifest found to check."
    checks.append(CheckResult("dependencies_pinned", "Dependencies are version-pinned", status, detail, weights["dependencies_pinned"], _tests_and_deps_category("dependencies_pinned", profile)))

    has_tests = any(
        "test" in seg.lower()
        for path in tree
        for seg in (path.split("/")[:-1] + [path.rsplit("/", 1)[-1]])
        if "test" in seg.lower()
    )
    checks.append(CheckResult(
        "has_tests", "Has a test suite",
        "pass" if has_tests else "fail",
        "Found test files/directory." if has_tests else "No tests/ directory or test_*/*_test files found.",
        weights["has_tests"], _tests_and_deps_category("has_tests", profile),
    ))

    if profile == "repo":
        has_ci = any(p.startswith((".github/workflows/", ".gitlab-ci")) for p in tree)
        checks.append(CheckResult(
            "has_ci", "Has continuous integration configured",
            "pass" if has_ci else "fail",
            "Found a CI workflow." if has_ci else "No .github/workflows or CI config found.",
            weights["has_ci"], cat["has_ci"],
        ))

    has_docker_or_env = any(p.rsplit("/", 1)[-1] in ("Dockerfile", "environment.yml", "environment.yaml") for p in tree)
    checks.append(CheckResult(
        "has_dockerfile_or_env_spec", "Has a Dockerfile or environment spec",
        "pass" if has_docker_or_env else "warn",
        "Found a Dockerfile/environment spec." if has_docker_or_env else "No containerized/conda environment spec found (informational -- not scored).",
        weights["has_dockerfile_or_env_spec"], cat["has_dockerfile_or_env_spec"],
    ))

    if readme_content:
        lowered = readme_content.lower()
        usage_ok = "```" in readme_content or any(kw in lowered for kw in ("install", "usage", "getting started", "quickstart", "how to run"))
        status = "pass" if usage_ok else "warn"
        detail = "README includes install/usage instructions." if usage_ok else "README doesn't appear to describe how to install or run the code."
    elif readme_path:
        status, detail = "na", "README found but couldn't be fetched to inspect."
    else:
        status, detail = "fail", "No README to check."
    checks.append(CheckResult("readme_has_usage_instructions", "README explains install/usage", status, detail, weights["readme_has_usage_instructions"], cat["readme_has_usage_instructions"]))

    if profile == "repo":
        if "archived" in metadata:
            checks.append(CheckResult(
                "repo_not_archived", "Repository is actively maintained (not archived)",
                "fail" if metadata.get("archived") else "pass",
                "Repository is archived on GitHub." if metadata.get("archived") else "Repository is not archived.",
                weights["repo_not_archived"], cat["repo_not_archived"],
            ))
        else:
            checks.append(CheckResult("repo_not_archived", "Repository is actively maintained (not archived)", "na", "Not applicable (no repo metadata).", weights["repo_not_archived"], cat["repo_not_archived"]))

        has_citation = any(p.upper() in ("CITATION.CFF", "CITATION.BIB", "CITATION") for p in tree)
        checks.append(CheckResult(
            "has_citation_file", "Has a CITATION file",
            "pass" if has_citation else "warn",
            "Found a CITATION file." if has_citation else "No CITATION.cff/CITATION file (informational -- not scored).",
            0, cat["has_citation_file"],
        ))
    else:
        checks.append(_static_analysis_check(tree, get_content, weights["static_analysis_clean"]))

    return checks


def score(checks: list[CheckResult]) -> tuple[int, str]:
    """Overall score across scored (non-warning, non-na) checks."""
    applicable = [c for c in checks if c.category != "warning" and c.status != "na"]
    total_weight = sum(c.weight for c in applicable) or 1
    earned = sum(c.weight for c in applicable if c.status == "pass") + sum(c.weight * 0.5 for c in applicable if c.status == "warn")
    pct = round(100 * earned / total_weight)
    if pct >= 80:
        verdict = "Likely reproducible"
    elif pct >= 50:
        verdict = "Partially reproducible -- some gaps"
    else:
        verdict = "Reproducibility at risk"
    return pct, verdict


def category_scores(checks: list[CheckResult]) -> dict[str, int | None]:
    """Per-category breakdown (e.g. {'documentation': 90, 'hygiene': 60,
    'code_quality': None, 'correctness': 75}) instead of one opaque number.
    None means no scored checks in that category applied (e.g. code_quality
    is always None for profile="repo" -- ruff only runs on generated code)."""
    result: dict[str, int | None] = {}
    for c in CATEGORIES:
        items = [chk for chk in checks if chk.category == c and chk.status != "na"]
        if not items:
            result[c] = None
            continue
        total = sum(chk.weight for chk in items) or 1
        earned = sum(chk.weight for chk in items if chk.status == "pass") + sum(chk.weight * 0.5 for chk in items if chk.status == "warn")
        result[c] = round(100 * earned / total)
    return result


def method_excerpt(paper: IngestedPaper) -> str:
    chunks = [s.text for s in paper.sections if s.heading.lower() in METHOD_HEADINGS]
    text = "\n\n".join(chunks) or paper.abstract
    return text[:5000]


def _flatten_prioritizable_segments(tree: ast.Module, content: str) -> list[str]:
    """Complete, self-contained source units to prioritize: top-level
    functions as-is, and classes broken into one segment PER METHOD (not
    kept as one atomic block) -- so truncation always cuts between complete
    functions/methods, never mid-method. A class with no methods (rare --
    e.g. a plain dataclass-style container) falls back to its full source
    as one segment, since there's nothing smaller to break it into."""
    segments = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = ast.get_source_segment(content, node)
            if src:
                segments.append(src)
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not methods:
                src = ast.get_source_segment(content, node)
                if src:
                    segments.append(src)
                continue
            for method in methods:
                src = ast.get_source_segment(content, method)
                if src:
                    segments.append(f"# (method of class {node.name})\n{src}")
    return segments


def _prioritized_code_excerpt(content: str, max_chars: int) -> str:
    """Real bug this fixes: naive `content[:max_chars]` truncation cut off a
    file's actual implementation entirely when it was defined after
    module-level setup code (imports, argparse, config) -- common in
    research code, and confirmed directly against facebookresearch's
    mixup-cifar10: mixup_data() doesn't start until character ~4100 of
    train.py, well past a 2000-3000 char prefix, so the LLM reviewing that
    prefix correctly reported seeing no mixup implementation -- it never saw
    it. This spends the character budget on complete function/method units
    first (in file order), not a blind prefix, so the algorithm itself is
    what gets shown even when it's not the first thing in the file.

    A class is NOT treated as one atomic unit -- an earlier version did
    that and hit the same failure mode one level down: confirmed against a
    Lookahead-optimizer implementation, where the entire method lived in
    one class, and a class exceeding the per-file budget fell back to
    raw-prefix-truncating the class body, cutting off methods defined later
    (e.g. step()) exactly like the original bug. Classes are now flattened
    into one segment per method (see _flatten_prioritizable_segments), so
    truncation always drops whole trailing methods, never cuts one in half.

    Falls back to a plain prefix if the file has no top-level functions/
    classes to prioritize, or doesn't parse as valid Python."""
    if len(content) <= max_chars:
        return content
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content[:max_chars]

    segments = _flatten_prioritizable_segments(tree, content)
    if not segments:
        return content[:max_chars]

    included = []
    used = 0
    for segment in segments:
        if used + len(segment) + 2 > max_chars:
            remaining = max_chars - used - 2
            if remaining > 100:  # only keep a partial segment if it's still meaningfully sized
                included.append(segment[:remaining])
            break
        included.append(segment)
        used += len(segment) + 2

    return "\n\n".join(included) if included else content[:max_chars]


_CODE_SIGNAL_NAME_HINTS = (
    "train", "main", "model", "method", "algorithm", "core", "loss", "run",
)

# Generic words common in paper titles -- excluded when deriving paper-specific
# filename keywords below, since treating these as "the technique's name"
# would false-positive on almost any file (e.g. "network", "deep", "improved").
_TITLE_STOPWORDS = {
    "the", "and", "for", "with", "using", "via", "based", "improved", "improving",
    "novel", "new", "toward", "towards", "study", "analysis", "approach", "approaches",
    "method", "methods", "model", "models", "network", "networks", "neural", "deep",
    "learning", "training", "regularization", "convolutional", "framework", "system",
    "efficient", "robust", "simple", "generalization", "understanding", "revisiting",
}


def _paper_keyword_tokens(paper: IngestedPaper) -> set[str]:
    """Candidate filename keywords derived from the paper's own title.
    Academic implementations are very often named after the paper's own
    coined term for its technique (cutout.py, mixup.py, lookahead.py,
    sam.py) -- a fixed generic hint list can never anticipate that, since
    it's different for every paper. Returns an empty set (graceful, no
    crash) if the title yields nothing usable after stopword filtering.

    Known limitation: derived from the title only, not the abstract or
    body. A paper whose title doesn't spell out an acronym used as the
    method's actual name (uncommon -- most papers include it in parens on
    first mention, e.g. "Sharpness-Aware Minimization (SAM)") won't surface
    that acronym as a keyword. Not fixed here to keep this deliberately
    narrow in scope and cheap (one field, no extra parsing); worth widening
    to the abstract if this proves to matter in practice."""
    tokens = re.findall(r"[a-zA-Z]{3,}", (paper.title or "").lower())
    return {t for t in tokens if t not in _TITLE_STOPWORDS}


def _select_claim_code_files(tree: list[str], paper: IngestedPaper = None) -> list[str]:
    """Pick a small, bounded set of .py files likely to contain the actual
    method implementation, for claim verification/semantic review to
    actually read. Not exhaustive (see repo_fetch.py's cost tradeoff) --
    just enough that a terse-README repo isn't judged on filenames alone.

    Fixes a real, confirmed failure mode: against uoguelph-mlrg/Cutout's
    tree (train.py, model/__init__.py, model/resnet.py, model/wide_resnet.py,
    util/cutout.py, util/misc.py), the previous version picked
    ['train.py', 'model/__init__.py', 'model/resnet.py'] -- util/cutout.py,
    which contains the ENTIRE method implementation, was never selected,
    and semantic_review then reported the method as missing from a repo
    that actually implements it correctly. Two causes, both fixed here:
    1. The old hint list (train/main/model/...) is generic and will never
       contain a specific paper's own coined term for its technique.
    2. The old tiebreak fell back to alphabetical order with no penalty for
       near-empty files -- `model/__init__.py` beat `util/cutout.py` purely
       because 'm' < 'u'.

    Ranking, most to least preferred:
    0. filename contains a keyword derived from the paper's own title
    1. filename contains a generic implementation-signal keyword (train/main/...)
    2. everything else
    3. __init__.py / __main__.py -- near-empty package boilerplate, actively
       deprioritized rather than left to alphabetical chance

    Directory depth is deliberately NOT part of the ranking -- a shallow
    entry-point script isn't more likely to contain the actual method than
    a file one directory deeper, and repos that deliberately organize core
    algorithm logic into a subpackage (a common, sensible pattern) were
    previously penalized for it. Within a tier, ties break alphabetically."""
    py_files = [p for p in tree if p.endswith(".py") and "/test" not in p.lower() and not p.lower().startswith("test")]
    paper_tokens = _paper_keyword_tokens(paper) if paper is not None else set()

    def sort_key(path: str) -> tuple[int, str]:
        name = path.rsplit("/", 1)[-1].lower()
        stem = name[:-3] if name.endswith(".py") else name

        if stem in ("__init__", "__main__"):
            tier = 3
        elif paper_tokens and any(tok in stem for tok in paper_tokens):
            tier = 0
        elif any(hint in stem for hint in _CODE_SIGNAL_NAME_HINTS):
            tier = 1
        else:
            tier = 2

        return (tier, path)

    return sorted(py_files, key=sort_key)[: config.REPRO_CLAIMS_MAX_CODE_FILES]


def verify_claims(
    paper: IngestedPaper,
    tree: list[str],
    readme_content: str | None,
    code_excerpts: dict[str, str] | None = None,
) -> list[dict]:
    """One bounded LLM call: does the codebase plausibly back up what the
    paper's Methods section claims was implemented? Uses the file tree and
    README always, plus actual content from a small, bounded set of
    high-signal .py files when available (code_excerpts) -- a README rarely
    restates paper-language like "convex combinations of examples", so
    tree/README alone under-confirms real implementations."""
    method_text = method_excerpt(paper)
    if not method_text.strip():
        return []

    tree_summary = "\n".join(tree[: config.REPRO_MAX_TREE_ENTRIES_IN_PROMPT])
    readme_excerpt = (readme_content or "(no README available)")[:3000]
    code_excerpts = code_excerpts or {}
    code_block = (
        "\n\n".join(
            f"--- {name} ---\n{_prioritized_code_excerpt(content, config.REPRO_CLAIMS_CODE_FILE_CHARS)}"
            for name, content in sorted(code_excerpts.items())
        )
        or "(no code excerpts available)"
    )

    prompt = f"""PAPER METHOD/APPROACH EXCERPT:
{method_text}

CODEBASE FILE LISTING:
{tree_summary}

CODEBASE README (excerpt):
{readme_excerpt}

CODE EXCERPTS (a handful of likely-relevant files, not the full codebase):
{code_block}

Identify up to 5 concrete implementation claims from the paper excerpt (e.g. "uses a
transformer encoder", "trained with the Adam optimizer", "releases pretrained
checkpoints", "evaluated with a custom benchmark script"). For each, judge whether the
file listing/README/code excerpts give concrete evidence for it -- prefer the code
excerpts as evidence over the README when both are available, since code is the
ground truth and README prose is often incomplete or stale.

Respond with ONLY a JSON object of this exact shape, no other text:
{{"claims": [
  {{"claim": "<one-sentence claim from the paper>", "verdict": "matches|unclear|not_evident", "evidence": "<short reason citing a file/README/code detail, or 'no supporting evidence'>"}}
]}}"""
    raw = llm_client.complete_json(CLAIM_SYSTEM_PROMPT, prompt, max_tokens=config.REPRO_CLAIMS_MAX_TOKENS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    results = []
    for item in parsed.get("claims", []):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        verdict = str(item.get("verdict", "")).strip().lower()
        evidence = str(item.get("evidence", "")).strip()
        if claim and verdict in ("matches", "unclear", "not_evident") and evidence:
            results.append({"claim": claim, "verdict": verdict, "evidence": evidence})
    return results


def _collect_py_contents(tree: list[str], get_content: GetContent) -> dict[str, str]:
    result = {}
    for p in tree:
        if p.endswith(".py"):
            c = get_content(p)
            if c:
                result[p] = c
    return result


def semantic_review(paper: IngestedPaper, py_files: dict[str, str], profile: str = "generated") -> dict:
    """Reviews ACTUAL code content (not just a file listing + README) against
    the paper's method, looking for missing steps, hallucinated APIs, and
    shape mismatches -- issues ruff and compile() structurally cannot catch.
    Runs for both profiles: for profile="generated", py_files is typically
    every generated .py file (small by construction); for profile="repo",
    callers should pass the same bounded, high-signal excerpt set used for
    verify_claims() (see _select_claim_code_files) rather than the whole
    repo, to respect repo_fetch.py's fetch-cost tradeoff."""
    method_text = method_excerpt(paper)
    if not method_text.strip() or not py_files:
        return {"coverage_score": None, "findings": []}

    code_excerpt = "\n\n".join(
        f"--- {name} ---\n{_prioritized_code_excerpt(content, config.SEMANTIC_REVIEW_PER_FILE_CHARS)}" for name, content in sorted(py_files.items())
    )[: config.SEMANTIC_REVIEW_MAX_CODE_CHARS]
    code_label = "GENERATED CODE" if profile == "generated" else "CODE (excerpt -- a bounded subset of the repo's files, not the whole codebase)"

    prompt = f"""PAPER METHOD/APPROACH EXCERPT:
{method_text}

{code_label}:
{code_excerpt}

Review the code against the paper excerpt. Identify concrete issues: missing
algorithmic steps described in the paper, calls to APIs/functions that don't plausibly
exist, likely tensor/array shape mismatches, and other correctness concerns. Do not
comment on style or formatting -- that's covered elsewhere. If this is only an excerpt
of a larger codebase, judge coverage based on what's shown and say so in a finding
rather than assuming missing pieces are absent from the full repo.

Respond with ONLY a JSON object of exactly this shape, no other text:
{{"coverage_score": <integer 0-100 estimating how completely the paper's method is implemented>,
  "findings": [{{"severity": "high|medium|low", "issue": "<one-sentence description>", "location": "<filename or 'general'>"}}]}}"""
    raw = llm_client.complete_json(SEMANTIC_REVIEW_SYSTEM_PROMPT, prompt, max_tokens=config.SEMANTIC_REVIEW_MAX_TOKENS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"coverage_score": None, "findings": []}

    score_val = parsed.get("coverage_score")
    if not isinstance(score_val, (int, float)) or not (0 <= score_val <= 100):
        score_val = None

    findings = []
    for item in parsed.get("findings", []):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).strip().lower()
        issue = str(item.get("issue", "")).strip()
        location = str(item.get("location", "")).strip() or "general"
        if issue and severity in ("high", "medium", "low"):
            findings.append({"severity": severity, "issue": issue, "location": location})

    return {"coverage_score": int(score_val) if score_val is not None else None, "findings": findings}


def _check_to_dict(c: CheckResult) -> dict:
    return {"id": c.id, "label": c.label, "status": c.status, "detail": c.detail, "weight": c.weight, "category": c.category}


def evaluate(paper: IngestedPaper, tree: list[str], get_content: GetContent, metadata: dict | None = None, profile: str = "repo") -> dict:
    """Run static checks (+ semantic review, generated profile only) and
    claim verification against any codebase, packaged into the shared report
    shape used by both the repo-checker and the generator's self-check.

    Response shape:
    - checks: scored checks only (warning-category checks excluded)
    - warnings: informational, non-scored checks (missing Dockerfile/CITATION)
    - score / verdict: overall, from `checks` only
    - category_scores: per-category breakdown (documentation/hygiene/
      code_quality/correctness), None for categories with no applicable checks
    - claims: paper-vs-code claim verification (both profiles)
    - semantic_findings: LLM semantic review findings (generated profile only)
    """
    all_checks = run_checks(tree, get_content, metadata, profile=profile)

    readme_path = find_readme_path(tree)
    readme_content = get_content(readme_path) if readme_path else None

    # For profile="repo", fetch the bounded, high-signal code excerpt set
    # ONCE and reuse it for both claim verification and semantic review --
    # avoids two separate GitHub fetch passes over the same files.
    code_excerpts: dict[str, str] = {}
    if profile == "repo":
        for path in _select_claim_code_files(tree, paper):
            content = get_content(path)
            if content:
                code_excerpts[path] = content
    else:
        code_excerpts = _collect_py_contents(tree, get_content)

    semantic_findings: list[dict] = []
    weights = _REPO_CHECK_WEIGHTS if profile == "repo" else _GENERATED_WEIGHTS
    if config.SEMANTIC_REVIEW_ENABLED:
        review = semantic_review(paper, code_excerpts, profile=profile)
        weight = weights["semantic_review"]
        if review["coverage_score"] is None:
            all_checks.append(CheckResult(
                "semantic_review", "LLM semantic review vs. paper", "na",
                "Semantic review could not be completed (no method text, no code, or the LLM call failed).",
                weight, "correctness",
            ))
        else:
            cov = review["coverage_score"]
            status = "pass" if cov >= 80 else ("warn" if cov >= 50 else "fail")
            detail = f"Estimated {cov}% coverage of the paper's described method ({len(review['findings'])} finding(s))."
            all_checks.append(CheckResult("semantic_review", "LLM semantic review vs. paper", status, detail, weight, "correctness"))
        semantic_findings = review["findings"]

    scored_checks = [c for c in all_checks if c.category != "warning"]
    warning_checks = [c for c in all_checks if c.category == "warning"]

    pct, verdict = score(scored_checks)
    cat_scores = category_scores(scored_checks)

    claims = []
    if config.REPRO_LLM_CLAIMS_ENABLED:
        claims = verify_claims(paper, tree, readme_content, code_excerpts if profile == "repo" else None)

    return {
        "checks": [_check_to_dict(c) for c in scored_checks],
        "warnings": [_check_to_dict(c) for c in warning_checks],
        "score": pct,
        "category_scores": cat_scores,
        "verdict": verdict,
        "claims": claims,
        "semantic_findings": semantic_findings,
    }