"""Report Agent: compositor that assembles Summary + Gap agent outputs into one Markdown report."""
from backend.ingestion_agent import IngestedPaper


def build_report(
    paper: IngestedPaper,
    tldr: str,
    section_summaries: list[dict],
    key_findings: list[str],
    gaps: dict,
    repro: dict | None = None,
) -> str:
    lines = [f"# {paper.title}", ""]

    lines += ["## Concise Overview", tldr, ""]

    lines += ["## Key Findings"]
    for f in key_findings:
        lines.append(f"- {f}")
    lines.append("")

    lines += ["## Section Summaries"]
    for s in section_summaries:
        lines.append(f"**{s['heading']}** — {s['summary']}")
        lines.append("")

    lines += ["## Research Gaps", "### Author-Acknowledged"]
    if gaps["author_acknowledged"]:
        for g in gaps["author_acknowledged"]:
            lines.append(f"- {g}")
    else:
        lines.append("_None explicitly stated by the authors._")
    lines.append("")

    lines += ["### Inferred (Critical Review)"]
    if gaps["inferred"]:
        for g in gaps["inferred"]:
            lines.append(f"- **{g['gap']}** _(confidence: {g['confidence']})_ → *Suggested direction:* {g['direction']}")
    else:
        lines.append("_None identified._")
    lines.append("")

    lines += ["## Suggested Future Directions"]
    directions = [g["direction"] for g in gaps["inferred"]]
    if directions:
        for d in directions:
            lines.append(f"- {d}")
    else:
        lines.append("_No additional directions identified._")
    lines.append("")

    lines += _reproducibility_section(repro)

    return "\n".join(lines)


_STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "➖"}


def _reproducibility_section(repro: dict | None) -> list[str]:
    lines = ["## Code Reproducibility"]
    if not repro:
        lines.append("_Not evaluated._")
        return lines

    if repro.get("note") and not repro.get("checks"):
        lines.append(f"_{repro['note']}_")
        return lines

    if repro.get("mode") == "repo_check":
        lines.append(f"**Repository:** [{repro['repo_metadata']['full_name']}]({repro['repo_url']})")
    else:
        lines.append(
            "**No repository was linked to this paper.** PaperPilot generated an implementation "
            "attempt from the methodology section and evaluated that instead."
        )
    lines.append(f"**Score:** {repro['score']}/100 — {repro['verdict']}")
    lines.append("")

    lines.append("### Category Scores")
    cat_labels = {"documentation": "Documentation", "hygiene": "Project Hygiene", "code_quality": "Code Quality", "correctness": "Correctness"}
    for cat_id, label in cat_labels.items():
        val = (repro.get("category_scores") or {}).get(cat_id)
        lines.append(f"- **{label}:** {val}/100" if val is not None else f"- **{label}:** N/A (no applicable checks)")
    lines.append("")

    lines.append("### Checks")
    for c in repro["checks"]:
        lines.append(f"- {_STATUS_ICON.get(c['status'], '•')} **{c['label']}** — {c['detail']}")
    lines.append("")

    if repro.get("warnings"):
        lines.append("### Warnings (informational, not scored)")
        for w in repro["warnings"]:
            lines.append(f"- ⚠️ **{w['label']}** — {w['detail']}")
        lines.append("")

    lines.append("### Claim Verification (paper vs. code)")
    if repro.get("claims"):
        verdict_icon = {"matches": "✅", "unclear": "⚠️", "not_evident": "❌"}
        for item in repro["claims"]:
            icon = verdict_icon.get(item["verdict"], "•")
            lines.append(f"- {icon} **{item['claim']}** — {item['evidence']}")
    else:
        lines.append("_No implementation claims were checked (LLM claim verification disabled, or nothing extractable from the Methods section)._")

    if repro.get("semantic_findings"):
        lines.append("")
        lines.append("### Semantic Review Findings (paper vs. generated code)")
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}
        for f in repro["semantic_findings"]:
            icon = severity_icon.get(f["severity"], "•")
            lines.append(f"- {icon} **[{f['location']}]** {f['issue']}")

    if repro.get("mode") == "generated":
        lines.append("")
        lines.append(repro.get("gap_report") or "")
        if repro.get("files"):
            lines.append("")
            lines.append(f"### Generated Files ({len(repro['files'])})")
            for name in sorted(repro["files"]):
                lines.append(f"- `{name}`")
            lines.append("")
            lines.append("_Full file contents are available in the app's Reproducibility tab and via the download endpoint, not inlined into this report._")

    return lines
