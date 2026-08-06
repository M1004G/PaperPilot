"""Reproducibility Agent: assesses whether a paper's linked code repo is
actually reproducible.

Two layers, deliberately kept separate (same split as gap_agent's
author-acknowledged vs. inferred, or AI4Reproducibility's deterministic
checks vs. bounded LLM judges):

1. STATIC CHECKS (primary, no LLM) -- mechanical facts about repo hygiene:
   README, license, pinned dependencies, tests, CI, Dockerfile, etc. These
   are cheap, deterministic, and never hallucinate.
2. CLAIM VERIFICATION (secondary, one bounded LLM call) -- does the repo's
   structure/README plausibly support what the paper's Methodology section
   *claims* was implemented? This is judgment, not a mechanical fact, so it's
   scoped to a single rubric-style call whose output is treated as one
   input among several, not a verdict on its own.

No code from the target repo is ever executed -- see repo_fetch.py.
"""
import json
import logging
from dataclasses import dataclass, field

from backend import config, llm_client, repo_fetch
from backend.ingestion_agent import IngestedPaper
from backend.repo_fetch import RepoFetchError

logger = logging.getLogger("paperpilot.repro")

CLAIM_SYSTEM_PROMPT = (
    "You are a careful, skeptical code reviewer checking whether a paper's released "
    "code plausibly implements what the paper describes. You only have the repo's file "
    "listing and README, not the full code, so you flag things as 'unclear' rather than "
    "guessing when the file tree/README genuinely doesn't say enough. You never claim a "
    "match unless the file tree or README gives concrete evidence for it."
)

METHOD_HEADINGS = {
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experimental setup", "experiments", "implementation",
}

# (check_id, label, weight) -- weight reflects how heavily code-reproducibility
# hygiene should count vs. "nice to have" documentation extras. Weights sum to
# 100 so the aggregate score is already a percentage.
_CHECK_WEIGHTS = {
    "has_readme": 10,
    "has_license": 10,
    "has_dependency_manifest": 15,
    "dependencies_pinned": 15,
    "has_tests": 15,
    "has_ci": 10,
    "readme_has_usage_instructions": 10,
    "license_is_recognized": 5,
    "repo_not_archived": 10,
    "has_dockerfile_or_env_spec": 5,
    "has_citation_file": 5,
}

DEPENDENCY_MANIFESTS = [
    "requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml",
    "Pipfile", "package.json", "setup.py", "Cargo.toml", "go.mod",
]
RECOGNIZED_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "gpl-3.0", "gpl-2.0",
    "lgpl-3.0", "mpl-2.0", "cc0-1.0", "cc-by-4.0", "unlicense",
}


@dataclass
class CheckResult:
    id: str
    label: str
    status: str  # "pass" | "fail" | "warn" | "na"
    detail: str
    weight: int = 0


@dataclass
class ReproReport:
    repo_url: str | None
    repo_metadata: dict | None
    checks: list[CheckResult] = field(default_factory=list)
    score: int | None = None  # 0-100, weighted static-check score
    verdict: str | None = None
    claims: list[dict] = field(default_factory=list)
    note: str | None = None  # set when there's no repo to analyze / a fetch error occurred


def _find_manifest_path(tree: list[str]) -> str | None:
    """First dependency manifest found, preferring root-level files over
    ones buried in subdirectories (a manifest in examples/ or a vendored
    dependency isn't the project's real one)."""
    by_depth = sorted(tree, key=lambda p: p.count("/"))
    names = {name.lower(): name for name in DEPENDENCY_MANIFESTS}
    for path in by_depth:
        filename = path.rsplit("/", 1)[-1]
        if filename in DEPENDENCY_MANIFESTS or filename.lower() in names:
            return path
    return None


def _find_readme_path(tree: list[str]) -> str | None:
    for path in tree:
        if "/" not in path and path.lower().startswith("readme"):
            return path
    return None


def _dependencies_pinned_ratio(manifest_path: str, content: str) -> float | None:
    """Rough heuristic, format-aware enough to not misjudge pyproject.toml/package.json
    (which pin very differently than requirements.txt) as unpinned."""
    if manifest_path.endswith((".txt",)):  # requirements.txt-style
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
        # npm convention: an exact version (no leading ^ or ~) is a hard pin.
        pinned = sum(1 for v in deps.values() if isinstance(v, str) and not v.startswith(("^", "~", ">", "*")))
        return pinned / len(deps)
    # pyproject.toml / Pipfile / environment.yml / Cargo.toml / go.mod: presence
    # of the file itself already implies a resolvable, checked-in dependency
    # spec (often with a companion lockfile) -- don't penalize format
    # differences we're not parsing in detail.
    return None


def _run_static_checks(metadata: dict, tree: list[str], owner: str, repo: str, branch: str) -> list[CheckResult]:
    checks: list[CheckResult] = []

    readme_path = _find_readme_path(tree)
    readme_content = repo_fetch.fetch_file_content(owner, repo, branch, readme_path) if readme_path else None

    checks.append(CheckResult(
        "has_readme", "Has a README",
        "pass" if readme_path else "fail",
        f"Found `{readme_path}`" if readme_path else "No top-level README file found.",
        _CHECK_WEIGHTS["has_readme"],
    ))

    has_license_file = any(p.upper().startswith("LICENSE") and "/" not in p for p in tree)
    has_license = bool(metadata.get("license_spdx_id")) or has_license_file
    checks.append(CheckResult(
        "has_license", "Has a license",
        "pass" if has_license else "fail",
        metadata.get("license_name") or ("LICENSE file present" if has_license_file else "No license detected."),
        _CHECK_WEIGHTS["has_license"],
    ))

    spdx = (metadata.get("license_spdx_id") or "").lower()
    if not has_license:
        recognized_status = "na"
        recognized_detail = "No license to evaluate."
    elif spdx in RECOGNIZED_LICENSES:
        recognized_status = "pass"
        recognized_detail = f"'{spdx}' is a widely recognized OSI-style license."
    else:
        recognized_status = "warn"
        recognized_detail = f"License present but not auto-recognized ('{spdx or 'unspecified'}') -- check terms manually."
    checks.append(CheckResult(
        "license_is_recognized", "License is a recognized open license",
        recognized_status, recognized_detail, _CHECK_WEIGHTS["license_is_recognized"],
    ))

    manifest_path = _find_manifest_path(tree)
    checks.append(CheckResult(
        "has_dependency_manifest", "Has a dependency manifest",
        "pass" if manifest_path else "fail",
        f"Found `{manifest_path}`" if manifest_path else "No requirements.txt/pyproject.toml/package.json/etc. found.",
        _CHECK_WEIGHTS["has_dependency_manifest"],
    ))

    if manifest_path:
        manifest_content = repo_fetch.fetch_file_content(owner, repo, branch, manifest_path)
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
    checks.append(CheckResult(
        "dependencies_pinned", "Dependencies are version-pinned",
        status, detail, _CHECK_WEIGHTS["dependencies_pinned"],
    ))

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
        _CHECK_WEIGHTS["has_tests"],
    ))

    has_ci = any(p.startswith((".github/workflows/", ".gitlab-ci")) for p in tree)
    checks.append(CheckResult(
        "has_ci", "Has continuous integration configured",
        "pass" if has_ci else "fail",
        "Found a CI workflow." if has_ci else "No .github/workflows or CI config found.",
        _CHECK_WEIGHTS["has_ci"],
    ))

    has_docker_or_env = any(p.rsplit("/", 1)[-1] in ("Dockerfile", "environment.yml", "environment.yaml") for p in tree)
    checks.append(CheckResult(
        "has_dockerfile_or_env_spec", "Has a Dockerfile or environment spec",
        "pass" if has_docker_or_env else "warn",
        "Found a Dockerfile/environment spec." if has_docker_or_env else "No containerized/conda environment spec found (not required, but helps reproducibility).",
        _CHECK_WEIGHTS["has_dockerfile_or_env_spec"],
    ))

    if readme_content:
        lowered = readme_content.lower()
        has_code_block = "```" in readme_content
        has_usage_words = any(kw in lowered for kw in ("install", "usage", "getting started", "quickstart", "how to run"))
        usage_ok = has_code_block or has_usage_words
        detail = "README includes install/usage instructions." if usage_ok else "README doesn't appear to describe how to install or run the code."
        status = "pass" if usage_ok else "warn"
    elif readme_path:
        status, detail = "na", "README found but couldn't be fetched to inspect."
    else:
        status, detail = "fail", "No README to check."
    checks.append(CheckResult(
        "readme_has_usage_instructions", "README explains install/usage",
        status, detail, _CHECK_WEIGHTS["readme_has_usage_instructions"],
    ))

    checks.append(CheckResult(
        "repo_not_archived", "Repository is actively maintained (not archived)",
        "fail" if metadata.get("archived") else "pass",
        "Repository is archived on GitHub." if metadata.get("archived") else "Repository is not archived.",
        _CHECK_WEIGHTS["repo_not_archived"],
    ))

    has_citation = any(p.upper() in ("CITATION.CFF", "CITATION.BIB", "CITATION") for p in tree)
    checks.append(CheckResult(
        "has_citation_file", "Has a CITATION file",
        "pass" if has_citation else "warn",
        "Found a CITATION file." if has_citation else "No CITATION.cff/CITATION file (not required, but good academic practice).",
        _CHECK_WEIGHTS["has_citation_file"],
    ))

    return checks


def _score(checks: list[CheckResult]) -> tuple[int, str]:
    applicable = [c for c in checks if c.status != "na"]
    total_weight = sum(c.weight for c in applicable) or 1
    earned = sum(c.weight for c in applicable if c.status == "pass") + sum(
        c.weight * 0.5 for c in applicable if c.status == "warn"
    )
    score = round(100 * earned / total_weight)
    if score >= 80:
        verdict = "Likely reproducible"
    elif score >= 50:
        verdict = "Partially reproducible -- some gaps"
    else:
        verdict = "Reproducibility at risk"
    return score, verdict


def _method_excerpt(paper: IngestedPaper) -> str:
    chunks = [s.text for s in paper.sections if s.heading.lower() in METHOD_HEADINGS]
    text = "\n\n".join(chunks) or paper.abstract
    return text[:5000]


def verify_claims(paper: IngestedPaper, tree: list[str], readme_content: str | None) -> list[dict]:
    """One bounded LLM call: does the repo's structure/README plausibly back
    up what the paper's Methods section claims to have implemented?"""
    method_text = _method_excerpt(paper)
    if not method_text.strip():
        return []

    tree_summary = "\n".join(tree[: config.REPRO_MAX_TREE_ENTRIES_IN_PROMPT])
    readme_excerpt = (readme_content or "(no README available)")[:3000]

    prompt = f"""PAPER METHOD/APPROACH EXCERPT:
{method_text}

REPO FILE LISTING:
{tree_summary}

REPO README (excerpt):
{readme_excerpt}

Identify up to 5 concrete implementation claims from the paper excerpt (e.g. "uses a
transformer encoder", "trained with the Adam optimizer", "releases pretrained
checkpoints", "evaluated with a custom benchmark script"). For each, judge whether the
file listing/README gives concrete evidence for it.

Respond with ONLY a JSON object of this exact shape, no other text:
{{"claims": [
  {{"claim": "<one-sentence claim from the paper>", "verdict": "matches|unclear|not_evident", "evidence": "<short reason citing a file/README detail, or 'no supporting file/README evidence'>"}}
]}}"""
    raw = llm_client.complete_json(CLAIM_SYSTEM_PROMPT, prompt, max_tokens=600)
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


def analyze(paper: IngestedPaper, repo_url: str | None = None) -> dict:
    """Main entry point. `repo_url` overrides whatever URL (if any) is found
    in the paper's own text -- lets a user point at the right repo when a
    paper doesn't link one, or links the wrong one (e.g. a baseline's repo)."""
    url = repo_url or repo_fetch.extract_code_url(paper.full_text)
    if not url:
        report = ReproReport(
            repo_url=None, repo_metadata=None,
            note="No GitHub repository URL was found in the paper, and none was supplied. "
                 "Pass a repo_url to check a specific repository.",
        )
        return _report_to_dict(report)

    try:
        owner, repo = repo_fetch.parse_github_url(url)
        metadata = repo_fetch.fetch_repo_metadata(owner, repo)
        branch = metadata["default_branch"]
        tree = repo_fetch.fetch_file_tree(owner, repo, branch)
    except RepoFetchError as e:
        logger.warning("repro_fetch_failed url=%s error=%s", url, e)
        report = ReproReport(repo_url=url, repo_metadata=None, note=str(e))
        return _report_to_dict(report)

    checks = _run_static_checks(metadata, tree, owner, repo, branch)
    score, verdict = _score(checks)

    readme_path = _find_readme_path(tree)
    readme_content = repo_fetch.fetch_file_content(owner, repo, branch, readme_path) if readme_path else None
    claims = verify_claims(paper, tree, readme_content) if config.REPRO_LLM_CLAIMS_ENABLED else []

    report = ReproReport(
        repo_url=metadata["html_url"], repo_metadata=metadata,
        checks=checks, score=score, verdict=verdict, claims=claims,
    )
    return _report_to_dict(report)


def _report_to_dict(report: ReproReport) -> dict:
    return {
        "repo_url": report.repo_url,
        "repo_metadata": report.repo_metadata,
        "checks": [
            {"id": c.id, "label": c.label, "status": c.status, "detail": c.detail, "weight": c.weight}
            for c in report.checks
        ],
        "score": report.score,
        "verdict": report.verdict,
        "claims": report.claims,
        "note": report.note,
    }
