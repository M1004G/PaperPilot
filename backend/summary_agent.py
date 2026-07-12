"""Summary Agent: TL;DR, per-section summaries, and key findings."""
import logging
from concurrent.futures import ThreadPoolExecutor

from backend import llm_client, config
from backend.ingestion_agent import IngestedPaper, prioritized_excerpt
from backend.logging_utils import copy_context_call

logger = logging.getLogger("paperpilot.summary")

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
    sections = [
        s for s in paper.sections
        if s.heading.lower() not in ("references", "acknowledgments", "acknowledgements")
    ]
    if not sections:
        return []

    # Sections are independent, so summarize them concurrently -- but capped, since
    # Groq's free tier enforces a per-minute request limit (not just a token budget).
    # Each future's failure is isolated: one section erroring (timeout, malformed
    # content, transient provider issue) shouldn't discard every other section's
    # already-successful summary.
    results: list[dict] = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_CONCURRENT_CALLS) as executor:
        futures = {
            executor.submit(copy_context_call, _summarize_one_section, s): i
            for i, s in enumerate(sections)
        }
        for future, idx in futures.items():
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("section_summary_failed heading=%r error=%s", sections[idx].heading, e)
                results[idx] = {
                    "heading": sections[idx].heading,
                    "summary": "(Summary unavailable for this section due to an error.)",
                }
    return results


def key_findings(paper: IngestedPaper) -> list[str]:
    excerpt = prioritized_excerpt(paper, limit=6000)
    prompt = f"""Title: {paper.title}

Excerpt (abstract first, then limitations/discussion/conclusion, then the rest):
{excerpt}

List the paper's main claimed contributions and findings as concise bullet points (max 8 bullets).
Each bullet should be a single sentence. Output only the bullets, one per line, starting with "- "."""
    raw = llm_client.complete(SYSTEM_PROMPT, prompt, max_tokens=400)
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip().startswith("-")]
