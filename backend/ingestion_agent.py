"""Ingestion Agent: PDF -> structured text (title, abstract, sections, full text).

Sections carry per-page text segments (not just a page_start/page_end range),
so downstream chunking can tag each chunk with the specific page(s) it actually
came from, instead of the whole section's page span.

Heading detection combines two signals:
1. A known-heading keyword list (fast, exact, works for "Introduction"/"Results"/etc.)
2. Font-based layout detection (font size relative to the document's body text,
   plus bold) -- so papers with unconventional heading names ("Proposed Framework",
   "Case Study") are still split correctly, not just ones using the standard
   IMRaD vocabulary.
"""
import re
from collections import Counter
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

# A leading section number, e.g. "3.", "3.2", "IV.", stripped off a detected
# heading line before it's used as the section's display name.
NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,4}\.)\s+(?=[A-Z])")

# Lines that *look* like headings by font (short, larger/bolder than body text)
# but are almost never actual section headings -- figure/table/equation captions.
_NON_HEADING_PREFIXES = ("figure", "fig.", "table", "eq.", "equation", "algorithm")

# A layout-detected heading must be noticeably larger than the body text...
_HEADING_SIZE_RATIO = 1.15
# ...or, at normal size, simply bold (headings are often bold-not-bigger in
# single-column conference templates).
_HEADING_BOLD_MIN_RATIO = 0.98
_HEADING_MAX_WORDS = 12
_HEADING_MAX_CHARS = 90
_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


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


def _line_is_bold(spans: list[dict]) -> bool:
    return any((s.get("flags", 0) & _BOLD_FLAG) or "bold" in s.get("font", "").lower() for s in spans)


def _extract_pages_and_layout(pdf_path: str) -> tuple[list[str], list[list[dict]]]:
    """Open the PDF once and return, per page: (1) plain text -- same as before,
    used for full_text/title-guessing -- and (2) a font-layout view: one entry per
    visual line with its text, max font size, and whether it's bold. The layout
    view is what heading detection runs on."""
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
        pages_text: list[str] = []
        pages_layout: list[list[dict]] = []
        for page in doc:
            pages_text.append(page.get_text("text"))
            layout_lines: list[dict] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    layout_lines.append({
                        "text": text,
                        "size": max((s.get("size", 0) for s in spans), default=0),
                        "bold": _line_is_bold(spans),
                    })
            pages_layout.append(layout_lines)
    except Exception as e:
        raise IngestionError(f"Could not extract text from this PDF: {e}") from e
    finally:
        doc.close()

    return pages_text, pages_layout


def _dominant_font_size(pages_layout: list[list[dict]]) -> float:
    """The document's body-text font size, taken as the size accounting for the
    most characters overall (robust to a handful of large-font headings/titles
    skewing a simple average or a per-line mode)."""
    char_counts: Counter = Counter()
    for lines in pages_layout:
        for line in lines:
            char_counts[round(line["size"], 1)] += len(line["text"])
    if not char_counts:
        return 10.0
    return char_counts.most_common(1)[0][0]


def _is_layout_heading(line: dict, body_size: float, require_numbered: bool = False) -> bool:
    """Font-based heading test: short, standalone line that's meaningfully larger
    than -- or bold at roughly -- the document's body text size."""
    text = line["text"]
    if not text or len(text) > _HEADING_MAX_CHARS:
        return False
    words = text.split()
    if not words or len(words) > _HEADING_MAX_WORDS:
        return False
    if text.endswith((".", ",", ";")) and len(words) > 6:
        return False  # reads like a full sentence, not a heading

    size_ratio = (line["size"] / body_size) if body_size else 1.0
    strong_font = size_ratio >= _HEADING_SIZE_RATIO or (line["bold"] and size_ratio >= _HEADING_BOLD_MIN_RATIO)
    # Some templates style headings via case rather than size/weight (no bold,
    # no larger font) -- a short, fully-uppercase line is still a strong signal.
    all_caps = text.isupper() and any(c.isalpha() for c in text) and len(words) <= 8
    if not (strong_font or all_caps):
        return False

    if require_numbered:
        return bool(NUMBERED_HEADING_RE.match(text))

    if text.lower().startswith(_NON_HEADING_PREFIXES):
        return False
    return True


def _clean_heading_text(text: str) -> str:
    """Strip leading numbering ("3.2 ", "IV. ") and normalize an ALL-CAPS
    heading to title case, so display names look consistent regardless of
    how the source PDF styled them."""
    text = NUMBERED_HEADING_RE.sub("", text).strip().rstrip(":").strip()
    if text.isupper():
        text = text.title()
    return text


def _guess_title(first_page_text: str) -> str:
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 8 and not line.lower().startswith(TITLE_SKIP_PREFIXES):
            return line
    return "Untitled Paper"


def _split_sections(pages_layout: list[list[dict]], body_size: float) -> list[Section]:
    """Walk every page's lines, splitting into sections while tracking, per section,
    which lines came from which page (page_segments).

    A line starts a new section if either:
    1. It matches the known-heading keyword list (Abstract, Introduction, ...), or
    2. It's font-detected as a heading -- noticeably larger/bolder than the
       document's body text, short, and not a figure/table caption -- which
       catches unconventional section names a keyword list can't anticipate.

    Font-detected headings only kick in once at least one heading (of either
    kind) has already been found, UNLESS the line also looks like a numbered
    heading ("3.2 Proposed Framework") -- that pattern is unambiguous enough to
    seed section structure even before any recognized keyword appears. This
    keeps a large-font paper title or author line on page 1 from being
    misread as the first "section".
    """
    sections: list[Section] = []
    current_heading = "Front Matter"
    # current_page_lines[page_num] = list of lines from that page, in order encountered
    current_page_lines: dict[int, list[str]] = {}
    current_page_start = 1
    seen_first_heading = False

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

    for page_num, lines in enumerate(pages_layout, start=1):
        for line in lines:
            text = line["text"]
            match = HEADING_LINE_RE.match(text.strip())
            if match:
                _flush(page_num)
                current_heading = match.group(1).title()
                current_page_lines = {}
                current_page_start = page_num
                seen_first_heading = True
            elif seen_first_heading and _is_layout_heading(line, body_size):
                _flush(page_num)
                current_heading = _clean_heading_text(text)
                current_page_lines = {}
                current_page_start = page_num
            elif not seen_first_heading and _is_layout_heading(line, body_size, require_numbered=True):
                _flush(page_num)
                current_heading = _clean_heading_text(text)
                current_page_lines = {}
                current_page_start = page_num
                seen_first_heading = True
            else:
                current_page_lines.setdefault(page_num, []).append(text)

    _flush(len(pages_layout) or 1)

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
    pages, pages_layout = _extract_pages_and_layout(pdf_path)
    full_text = "\n".join(pages)
    if not full_text.strip():
        raise IngestionError(
            "No extractable text found in this PDF -- it may be a scanned image "
            "without OCR text, which isn't supported yet."
        )

    title = _guess_title(pages[0] if pages else "")
    body_size = _dominant_font_size(pages_layout)
    sections = _split_sections(pages_layout, body_size)
    abstract = _extract_abstract(sections, full_text)

    return IngestedPaper(
        title=title,
        abstract=abstract,
        sections=sections,
        full_text=full_text,
        num_pages=len(pages),
    )
