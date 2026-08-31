"""Discovery Agent: given a topic, finds relevant papers and downloads their
open-access PDFs, ready to be handed to the existing ingestion pipeline.

Single API (Semantic Scholar), not two: its paper/search endpoint already
covers arXiv coverage and returns an `openAccessPdf.url` field directly when
a free full-text PDF exists, so one API gives both metadata and the PDF link
rather than juggling arXiv's API separately and cross-referencing IDs.

Real constraint, stated plainly rather than papered over: only candidates
with a populated `openAccessPdf.url` can be auto-ingested. Many relevant
papers exist behind a paywall with no free PDF -- those are reported as
"found but not ingestable", not silently dropped or faked.

No new ingestion logic here -- downloaded PDFs are handed to the same
orchestrator.ingest_paper() that manual uploads already use.
"""
import json
import logging
import tempfile
import time

import requests

from backend import config, llm_client

logger = logging.getLogger("paperpilot.discovery")

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEARCH_FIELDS = "title,abstract,year,externalIds,openAccessPdf,citationCount"

FILTER_SYSTEM_PROMPT = (
    "You are a research assistant selecting which papers are actually relevant to a "
    "topic, from a candidate list of titles and abstracts. You exclude papers that are "
    "only tangentially related, and you never select more than requested even if more "
    "look relevant -- pick the strongest matches."
)


class DiscoveryError(Exception):
    """Raised for problems reaching or parsing the Semantic Scholar API --
    distinct from a bug in our own code."""


def _headers() -> dict:
    headers = {"User-Agent": "PaperPilot-DiscoveryAgent"}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    return headers


def search_papers(topic: str, limit: int = None) -> list[dict]:
    """Queries Semantic Scholar for candidate papers on a topic. One retry on
    a 429 (rate limit) with a short backoff -- the public tier is roughly
    1 request/sec, and a single discovery request only makes one search call,
    so this is a light, occasional bump, not a sustained-throughput problem."""
    limit = limit or config.DISCOVERY_SEARCH_POOL_SIZE
    params = {"query": topic, "limit": limit, "fields": SEARCH_FIELDS}

    resp = None
    for attempt in range(2):
        try:
            resp = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=_headers(),
                timeout=config.DISCOVERY_HTTP_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            raise DiscoveryError(f"Network error reaching Semantic Scholar: {e}") from e

        if resp.status_code == 429:
            if attempt == 0:
                logger.warning("semantic_scholar_rate_limited retrying_once")
                time.sleep(2.0)
                continue
            raise DiscoveryError("Semantic Scholar rate limit persisted after retry.")
        if resp.status_code >= 400:
            raise DiscoveryError(f"Semantic Scholar search failed ({resp.status_code}) for topic {topic!r}.")
        break

    data = resp.json().get("data", [])
    candidates = []
    for p in data:
        if not p.get("title") or not p.get("abstract"):
            continue  # can't relevance-filter a paper with no abstract
        external = p.get("externalIds") or {}
        candidates.append({
            "paper_id": p.get("paperId"),
            "title": p["title"],
            "abstract": p["abstract"],
            "year": p.get("year"),
            "citation_count": p.get("citationCount"),
            "arxiv_id": external.get("ArXiv"),
            "doi": external.get("DOI"),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
        })
    return candidates


def filter_relevant(topic: str, candidates: list[dict], max_results: int) -> list[dict]:
    """One LLM call ranking candidates by actual relevance to the topic, not
    just keyword overlap with the search query. Falls back to the raw
    citation-count-sorted candidate list (still real papers, just unranked
    by an LLM) if the call fails or returns something unusable -- discovery
    shouldn't silently return nothing because one LLM call had a hiccup."""
    if not candidates:
        return []

    listing = "\n".join(f"{i}. {c['title']}\n   {c['abstract'][:400]}" for i, c in enumerate(candidates))
    prompt = f"""TOPIC: {topic}

CANDIDATE PAPERS:
{listing}

Select up to {max_results} papers most relevant to the topic above.

Respond with ONLY a JSON object of exactly this shape, no other text:
{{"selected_indices": [<integers referencing the numbered list above, most relevant first>]}}"""

    try:
        raw = llm_client.complete_json(FILTER_SYSTEM_PROMPT, prompt, max_tokens=config.DISCOVERY_FILTER_MAX_TOKENS)
        parsed = json.loads(raw)
        indices = [i for i in parsed.get("selected_indices", []) if isinstance(i, int) and 0 <= i < len(candidates)]
        if indices:
            return [candidates[i] for i in indices[:max_results]]
    except Exception as e:
        logger.warning("discovery_filter_failed error=%s -- falling back to citation-sorted order", e)

    ranked = sorted(candidates, key=lambda c: c.get("citation_count") or 0, reverse=True)
    return ranked[:max_results]


def download_pdf(url: str) -> str | None:
    """Downloads one candidate's PDF to a temp file, bounded by size and
    timeout, with a content-type check -- the same defensive shape as
    repo_fetch.py's file access, since this is also fetching content from an
    untrusted external URL. Returns None (not an exception) on any failure:
    one broken PDF link shouldn't abort an entire discovery batch."""
    try:
        resp = requests.get(url, timeout=config.DISCOVERY_HTTP_TIMEOUT_SECONDS, stream=True)
    except requests.RequestException as e:
        logger.warning("discovery_pdf_download_failed url=%s error=%s", url, e)
        return None

    if resp.status_code != 200:
        logger.warning("discovery_pdf_download_bad_status url=%s status=%d", url, resp.status_code)
        return None

    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
        logger.warning("discovery_pdf_unexpected_content_type url=%s content_type=%s", url, content_type)
        return None

    max_bytes = config.DISCOVERY_PDF_MAX_SIZE_MB * 1024 * 1024
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    written = 0
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            written += len(chunk)
            if written > max_bytes:
                logger.warning("discovery_pdf_too_large url=%s max_mb=%d", url, config.DISCOVERY_PDF_MAX_SIZE_MB)
                tmp.close()
                return None
            tmp.write(chunk)
    finally:
        tmp.close()

    if written == 0:
        return None
    return tmp.name


def discover(topic: str, max_results: int = None, existing_arxiv_ids: set[str] = None, existing_dois: set[str] = None) -> dict:
    """Main entry point: search, relevance-filter, and download PDFs for a
    topic. Does NOT ingest anything itself -- returns local file paths for
    the caller (orchestrator) to hand to the existing ingestion pipeline, and
    metadata for candidates that were found but couldn't be auto-ingested
    (no open-access PDF, or the download failed), so the caller can report
    those honestly rather than silently dropping them."""
    max_results = max_results or config.DISCOVERY_MAX_RESULTS
    existing_arxiv_ids = existing_arxiv_ids or set()
    existing_dois = existing_dois or set()

    candidates = search_papers(topic)
    selected = filter_relevant(topic, candidates, max_results)

    downloaded = []
    skipped_no_pdf = []
    skipped_duplicate = []
    skipped_download_failed = []

    for c in selected:
        if (c["arxiv_id"] and c["arxiv_id"] in existing_arxiv_ids) or (c["doi"] and c["doi"] in existing_dois):
            skipped_duplicate.append(c)
            continue
        if not c["pdf_url"]:
            skipped_no_pdf.append(c)
            continue
        path = download_pdf(c["pdf_url"])
        if path is None:
            skipped_download_failed.append(c)
            continue
        downloaded.append({**c, "local_path": path})

    return {
        "topic": topic,
        "candidates_found": len(candidates),
        "downloaded": downloaded,
        "skipped_no_pdf": skipped_no_pdf,
        "skipped_duplicate": skipped_duplicate,
        "skipped_download_failed": skipped_download_failed,
    }
