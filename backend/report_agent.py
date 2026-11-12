"""Report Agent: compositor that assembles Summary + Gap agent outputs into one Markdown report."""
from backend.ingestion_agent import IngestedPaper


def build_report(
    paper: IngestedPaper,
    tldr: str,
    section_summaries: list[dict],
    key_findings: list[str],
    gaps: dict,
) -> str:
    lines = [f"# {paper.title}", ""] #The # at the front is Markdown syntax for "this is a heading."

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

    return "\n".join(lines)
