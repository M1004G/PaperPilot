"""Ingestion Agent: PDF -> structured text (title, abstract, sections, full text).

Sections carry per-page text segments (not just a page_start/page_end range),
so downstream chunking can tag each chunk with the specific page(s) it actually
came from, instead of the whole section's page span.
"""
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from backend import config

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


class IngestionError(Exception):
    """Raised for problems with the uploaded file itself (bad input, not a bug) --
    corrupt PDFs, password-protected files, oversized uploads, etc. The FastAPI
    layer maps this to a 400, as distinct from unexpected internal errors (500)."""


@dataclass
class Section:
    heading: str
    text: str
    page_start: int = 1
    page_end: int = 1
    # Per-page text within this section: [(page_num, text_on_that_page), ...].
    # Lets chunking assign each chunk its actual page rather than the section's
    # full (possibly multi-page) range.
    page_segments: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class IngestedPaper:
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)
    full_text: str = ""
    num_pages: int = 0


def _extract_pages(pdf_path: str) -> list[str]:
    try:
        doc = fitz.open(pdf_path)
    except fitz.FileDataError as e:
        raise IngestionError(f"This file doesn't look like a valid PDF: {e}") from e
    except Exception as e:  # pragma: no cover - fitz raises assorted error types for bad files
        raise IngestionError(f"Could not open this PDF: {e}") from e

    if doc.is_encrypted:
        needs_pass = doc.needs_pass
        doc.close()
        if needs_pass:
            raise IngestionError(
                "This PDF is password-protected. Please remove the password before uploading."
            )

    if doc.page_count == 0:
        doc.close()
        raise IngestionError("This PDF has no pages.")

    try:
        pages = [page.get_text("text") for page in doc]
    except Exception as e:
        raise IngestionError(f"Could not extract text from this PDF: {e}") from e
    finally:
        doc.close()

    return pages


def _guess_title(first_page_text: str) -> str:
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 8 and not line.lower().startswith(TITLE_SKIP_PREFIXES):
            return line
    return "Untitled Paper"


def _split_sections(pages: list[str]) -> list[Section]:
    """Walk every page's lines, splitting into sections while tracking, per section,
    which lines came from which page (page_segments)."""
    sections: list[Section] = []
    current_heading = "Front Matter"
    # current_page_lines[page_num] = list of lines from that page, in order encountered
    current_page_lines: dict[int, list[str]] = {}
    current_page_start = 1

    def _flush(end_page: int):
        if not current_page_lines:
            return
        page_segments = [
            (page_num, "\n".join(lines).strip())
            for page_num, lines in sorted(current_page_lines.items())
            if "\n".join(lines).strip()
        ]
        if not page_segments:
            return
        full_text = "\n".join(seg_text for _, seg_text in page_segments)
        sections.append(
            Section(
                heading=current_heading,
                text=full_text,
                page_start=current_page_start,
                page_end=end_page,
                page_segments=page_segments,
            )
        )

    for page_num, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            match = HEADING_LINE_RE.match(line.strip())
            if match:
                _flush(page_num)
                current_heading = match.group(1).title()
                current_page_lines = {}
                current_page_start = page_num
            else:
                current_page_lines.setdefault(page_num, []).append(line)

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
    """Main entry point: parse a PDF into a structured IngestedPaper.

    Raises IngestionError for problems with the file itself (corrupt, encrypted,
    empty) so the API layer can return a clear 400 rather than a bare 500.
    """
    pages = _extract_pages(pdf_path)
    full_text = "\n".join(pages)
    if not full_text.strip():
        raise IngestionError(
            "No extractable text found in this PDF -- it may be a scanned image "
            "without OCR text, which isn't supported yet."
        )

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
