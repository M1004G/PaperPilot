"""Summary Agent: TL;DR, per-section summaries, and key findings."""
from concurrent.futures import ThreadPoolExecutor

from backend import llm_client, config
from backend.ingestion_agent import IngestedPaper, prioritized_excerpt
from backend.logging_utils import submit_with_context

SYSTEM_PROMPT = (
    "You are a meticulous research-paper summarization assistant. "
    "You only state what is supported by the provided text. "
    "You write in clear, plain English, avoiding unexplained jargon."
)


def tldr(paper: IngestedPaper) -> str:
    excerpt = prioritized_excerpt(paper, limit=4000)
    prompt = f"""Here is a research paper's title, plus a prioritized excerpt of its text
(abstract first, then limitations/discussion/conclusion, then the rest).

Title: {paper.title}

Excerpt:
{excerpt}

Write a single-paragraph TL;DR (5-7 sentences) explaining, for a reader unfamiliar with the paper:
1) what problem it addresses, 2) what the core method/approach is, 3) what the headline result is,
and 4) why it matters. Do not use bullet points."""
    return llm_client.complete(SYSTEM_PROMPT, prompt, max_tokens=500)


def _summarize_one_section(section) -> dict:
    prompt = f"""Section heading: {section.heading}

Section text:
{section.text[:3000]}

Summarize this section in 2-3 sentences, capturing only its key claims/content."""
    summary = llm_client.complete(SYSTEM_PROMPT, prompt, max_tokens=200)
    return {"heading": section.heading, "summary": summary}


def section_summaries(paper: IngestedPaper) -> list[dict]:
    # Skip non-content sections and junk "sections" produced by heading detection on
    # math/pseudocode (e.g. "X", "T", "while θt not converged do").
    sections = [
        s for s in paper.sections
        if s.heading.lower() not in ("references", "acknowledgments", "acknowledgements")
        and len(s.text) > 600
        and sum(c.isalpha() for c in s.heading) >= 4
        and not s.heading.lower().startswith(("while", "return", "require"))
    ]
    if not sections:
        return []

    # Sections are independent, so summarize them concurrently -- but capped, since
    # Groq's free tier enforces a per-minute request limit (not just a token budget).
    # If any section fails the error propagates, so nothing partial gets cached and
    # the next request simply retries.
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_CONCURRENT_CALLS) as executor:
        futures = [submit_with_context(executor, _summarize_one_section, s) for s in sections]
        return [f.result() for f in futures]


def key_findings(paper: IngestedPaper) -> list[str]:
    excerpt = prioritized_excerpt(paper, limit=6000)
    prompt = f"""Title: {paper.title}

Excerpt (abstract first, then limitations/discussion/conclusion, then the rest):
{excerpt}

List the paper's main claimed contributions and findings as concise bullet points (max 8 bullets).
Each bullet should be a single sentence. Output only the bullets, one per line, starting with "- "."""
    raw = llm_client.complete(SYSTEM_PROMPT, prompt, max_tokens=400)
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip().startswith("-")]