"""Reproducibility Check Agent: source-agnostic code-quality checks + claim
verification.

Deliberately doesn't know or care where the code came from -- it operates on
a plain (paths, get_content, metadata) triple. Two sources feed it:
- repo_fetch.py (a real GitHub repo linked in the paper)
- codegen_agent.py (code PaperPilot generated itself, when no repo exists)

That's the point: the same "is this trustworthy/complete" checks apply
whether the code is someone else's or ours, so trust is scored consistently
either way instead of two disconnected notions of "reproducible."

Two layers, kept separate (same split as gap_agent's author-acknowledged vs.
inferred):
1. STATIC CHECKS (primary, no LLM) -- mechanical facts about hygiene: README,
   license, pinned dependencies, tests, CI, Dockerfile, etc.
2. CLAIM VERIFICATION (secondary, one bounded LLM call) -- does the code
   plausibly support what the paper's Methodology section *claims* was
   implemented? Judgment, not a mechanical fact, so it's scoped to a single
   rubric-style call treated as one input among several, not a verdict alone.
"""
import json
import logging
from dataclasses import dataclass, field
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

METHOD_HEADINGS = {
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experimental setup", "experiments", "implementation",
}

# Weights sum to 100 so the aggregate score is already a percentage. Code-hygiene
# items (pinning, tests, CI, manifest) are weighted heavier than doc extras.
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

GetContent = Callable[[str], str | None]


@dataclass
class CheckResult:
    id: str
    label: str
    status: str  # "pass" | "fail" | "warn" | "na"
    detail: str
    weight: int = 0


def find_manifest_path(tree: list[str]) -> str | None:
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


def find_readme_path(tree: list[str]) -> str | None:
    for path in tree:
        if "/" not in path and path.lower().startswith("readme"):
            return path
    return None


def _dependencies_pinned_ratio(manifest_path: str, content: str) -> float | None:
    """Rough heuristic, format-aware enough to not misjudge pyproject.toml/package.json
    (which pin very differently than requirements.txt) as unpinned."""
    if manifest_path.endswith(".txt"):  # requirements.txt-style
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
    # pyproject.toml / Pipfile / environment.yml / Cargo.toml / go.mod: presence
    # of the file itself already implies a resolvable dependency spec -- don't
    # penalize format differences we're not parsing in detail.
    return None


def run_checks(tree: list[str], get_content: GetContent, metadata: dict | None = None) -> list[CheckResult]:
    """Pure, source-agnostic. `tree` is every file path in the codebase;
    `get_content(path)` lazily returns a file's text (or None if unavailable);
    `metadata` is optional repo-level info (license/archived status) --
    generated code has none of that, so those checks come back 'na' rather
    than 'fail' when metadata is absent."""
    metadata = metadata or {}
    checks: list[CheckResult] = []

    readme_path = find_readme_path(tree)
    readme_content = get_content(readme_path) if readme_path else None

    checks.append(CheckResult(
        "has_readme", "Has a README",
        "pass" if readme_path else "fail",
        f"Found `{readme_path}`" if readme_path else "No top-level README file found.",
        _CHECK_WEIGHTS["has_readme"],
    ))

    has_license_file = any(p.upper().startswith("LICENSE") and "/" not in p for p in tree)
    has_license = bool(metadata.get("license_spdx_id")) or has_license_file
    if not metadata and not has_license_file:
        license_status, license_detail = "na", "No repo metadata available and no LICENSE file to check."
    else:
        license_status = "pass" if has_license else "fail"
        license_detail = metadata.get("license_name") or ("LICENSE file present" if has_license_file else "No license detected.")
    checks.append(CheckResult("has_license", "Has a license", license_status, license_detail, _CHECK_WEIGHTS["has_license"]))

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
        recognized_status, recognized_detail, _CHECK_WEIGHTS["license_is_recognized"],
    ))

    manifest_path = find_manifest_path(tree)
    checks.append(CheckResult(
        "has_dependency_manifest", "Has a dependency manifest",
        "pass" if manifest_path else "fail",
        f"Found `{manifest_path}`" if manifest_path else "No requirements.txt/pyproject.toml/package.json/etc. found.",
        _CHECK_WEIGHTS["has_dependency_manifest"],
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
    checks.append(CheckResult("dependencies_pinned", "Dependencies are version-pinned", status, detail, _CHECK_WEIGHTS["dependencies_pinned"]))

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
        usage_ok = "```" in readme_content or any(kw in lowered for kw in ("install", "usage", "getting started", "quickstart", "how to run"))
        status = "pass" if usage_ok else "warn"
        detail = "README includes install/usage instructions." if usage_ok else "README doesn't appear to describe how to install or run the code."
    elif readme_path:
        status, detail = "na", "README found but couldn't be fetched to inspect."
    else:
        status, detail = "fail", "No README to check."
    checks.append(CheckResult("readme_has_usage_instructions", "README explains install/usage", status, detail, _CHECK_WEIGHTS["readme_has_usage_instructions"]))

    if "archived" in metadata:
        checks.append(CheckResult(
            "repo_not_archived", "Repository is actively maintained (not archived)",
            "fail" if metadata.get("archived") else "pass",
            "Repository is archived on GitHub." if metadata.get("archived") else "Repository is not archived.",
            _CHECK_WEIGHTS["repo_not_archived"],
        ))
    else:
        checks.append(CheckResult("repo_not_archived", "Repository is actively maintained (not archived)", "na", "Not applicable (no repo metadata).", _CHECK_WEIGHTS["repo_not_archived"]))

    has_citation = any(p.upper() in ("CITATION.CFF", "CITATION.BIB", "CITATION") for p in tree)
    checks.append(CheckResult(
        "has_citation_file", "Has a CITATION file",
        "pass" if has_citation else "warn",
        "Found a CITATION file." if has_citation else "No CITATION.cff/CITATION file (not required, but good academic practice).",
        _CHECK_WEIGHTS["has_citation_file"],
    ))

    return checks


def score(checks: list[CheckResult]) -> tuple[int, str]:
    applicable = [c for c in checks if c.status != "na"]
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


def method_excerpt(paper: IngestedPaper) -> str:
    chunks = [s.text for s in paper.sections if s.heading.lower() in METHOD_HEADINGS]
    text = "\n\n".join(chunks) or paper.abstract
    return text[:5000]


def verify_claims(paper: IngestedPaper, tree: list[str], readme_content: str | None) -> list[dict]:
    """One bounded LLM call: does the codebase's structure/README plausibly
    back up what the paper's Methods section claims was implemented?"""
    method_text = method_excerpt(paper)
    if not method_text.strip():
        return []

    tree_summary = "\n".join(tree[: config.REPRO_MAX_TREE_ENTRIES_IN_PROMPT])
    readme_excerpt = (readme_content or "(no README available)")[:3000]

    prompt = f"""PAPER METHOD/APPROACH EXCERPT:
{method_text}

CODEBASE FILE LISTING:
{tree_summary}

CODEBASE README (excerpt):
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


def evaluate(paper: IngestedPaper, tree: list[str], get_content: GetContent, metadata: dict | None = None) -> dict:
    """Run static checks + (optionally) claim verification against any
    codebase, and package it into the shared report shape used by both
    the repo-checker and the generator's self-check."""
    checks = run_checks(tree, get_content, metadata)
    pct, verdict = score(checks)
    readme_path = find_readme_path(tree)
    readme_content = get_content(readme_path) if readme_path else None
    claims = verify_claims(paper, tree, readme_content) if config.REPRO_LLM_CLAIMS_ENABLED else []
    return {
        "checks": [{"id": c.id, "label": c.label, "status": c.status, "detail": c.detail, "weight": c.weight} for c in checks],
        "score": pct,
        "verdict": verdict,
        "claims": claims,
    }
