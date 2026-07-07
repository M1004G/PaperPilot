"""Ingestion Agent: PDF -> structured text (title, abstract, sections, full text).

Sections now carry page numbers so downstream chunks can be tagged with
"page N" metadata, which the RAG agent surfaces to the LLM for grounding.
"""
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# Common top-level headings in research papers, used to split the document into sections.
KNOWN_HEADINGS = [
    "abstract", "introduction", "related work", "background", "methodology",
    "methods", "materials and methods", "approach", "experiments",
    "experimental setup", "results", "evaluation", "discussion",
    "limitations", "future work", "conclusion", "conclusions",
    "acknowledgments", "acknowledgements", "references",
]

HEADING_LINE_RE = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?({})\s*$".format("|".join(KNOWN_HEADINGS)),
    re.IGNORECASE,
)

# Lines that are almost certainly boilerplate rather than a paper title, seen on
# the first page of conference/preprint PDFs.
TITLE_SKIP_PREFIXES = (
    "arxiv", "http", "doi", "page", "proceedings of", "ieee", "acm",
    "preprint", "accepted at", "to appear", "copyright", "©",
)


@dataclass
class Section:
    heading: str
    text: str
    page_start: int = 1
    page_end: int = 1


@dataclass
class IngestedPaper:
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)
    full_text: str = ""
    num_pages: int = 0


def _extract_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def _guess_title(first_page_text: str) -> str:
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 8 and not line.lower().startswith(TITLE_SKIP_PREFIXES):
            return line
    return "Untitled Paper"


def _split_sections(pages: list[str]) -> list[Section]:
    """Walk every page's lines, splitting into sections while tracking page numbers."""
    sections: list[Section] = []
    current_heading = "Front Matter"
    current_lines: list[str] = []
    current_page_start = 1

    def _flush(end_page: int):
        if current_lines:
            sections.append(
                Section(
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                    page_start=current_page_start,
                    page_end=end_page,
                )
            )

    for page_num, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            match = HEADING_LINE_RE.match(line.strip())
            if match:
                _flush(page_num)
                current_heading = match.group(1).title()
                current_lines = []
                current_page_start = page_num
            else:
                current_lines.append(line)

    _flush(len(pages) or 1)

    # Drop empty/near-empty sections.
    return [s for s in sections if len(s.text) > 20]


def _extract_abstract(sections: list[Section], full_text: str) -> str:
    for s in sections:
        if s.heading.lower() == "abstract":
            return s.text
    # Fallback: first ~1500 chars if no explicit Abstract heading was found.
    return full_text[:1500].strip()


PRIORITY_HEADINGS = {"limitations", "future work", "discussion", "conclusion", "conclusions"}


def prioritized_excerpt(paper: IngestedPaper, limit: int) -> str:
    """Build a truncated excerpt of the paper for prompts with a fixed character budget.

    Naively truncating paper.full_text[:limit] silently drops whatever comes last --
    which for a research paper is often Limitations/Discussion/Conclusion, exactly the
    sections a gap-analysis or summary prompt needs most. This front-loads the abstract,
    then those priority sections, then fills any remaining budget with the rest of the
    paper in original order.
    """
    parts: list[str] = []
    used = 0

    def _add(text: str) -> bool:
        nonlocal used
        text = text.strip()
        if not text or used >= limit:
            return False
        remaining = limit - used
        parts.append(text[:remaining])
        used += min(len(text), remaining)
        return used < limit

    if not _add(paper.abstract):
        return "\n\n".join(parts)

    priority_sections = [s for s in paper.sections if s.heading.lower() in PRIORITY_HEADINGS]
    other_sections = [s for s in paper.sections if s.heading.lower() not in PRIORITY_HEADINGS]

    for section in priority_sections:
        if not _add(f"[{section.heading}]\n{section.text}"):
            return "\n\n".join(parts)

    for section in other_sections:
        if not _add(f"[{section.heading}]\n{section.text}"):
            return "\n\n".join(parts)

    return "\n\n".join(parts)


def ingest(pdf_path: str) -> IngestedPaper:
    """Main entry point: parse a PDF into a structured IngestedPaper."""
    pages = _extract_pages(pdf_path)
    full_text = "\n".join(pages)
    title = _guess_title(pages[0] if pages else "")
    sections = _split_sections(pages)
    abstract = _extract_abstract(sections, full_text)

    return IngestedPaper(
        title=title,
        abstract=abstract,
        sections=sections,
        full_text=full_text,
        num_pages=len(pages),
    )
