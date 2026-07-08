# Architecture

## Components

### 1. Ingestion Agent (`backend/ingestion_agent.py`)
Input: PDF file path.
Steps:
1. Extract raw text per page with PyMuPDF, validating the file first (rejects
   corrupt, empty, and password-protected PDFs with a clear `IngestionError`
   rather than letting a raw PyMuPDF exception bubble up as a bare 500).
2. Heuristically split into sections (Abstract, Introduction, Related Work,
   Methodology, Results, Discussion, Limitations, Conclusion, References, ...)
   using regex over heading-like lines (numbered / all-caps / title-case short lines).
3. Track each section's text *per page* (`page_segments`), not just an overall
   page range — this lets chunks downstream carry their actual page rather than
   the whole section's span.
4. Extract lightweight metadata: title (first prominent line), abstract text,
   page count.
Output: `IngestedPaper` dataclass — `{title, abstract, sections: [{heading, text, page_start, page_end, page_segments}], full_text, num_pages}`.

This is the only agent that touches the raw file. Everything downstream
consumes its structured output, so re-parsing never happens twice.

**Known limitation**: heading detection is still a fixed keyword list matched
via regex (see Phase 1 below) — papers with unconventional heading names won't
split correctly.

### 2. Summary Agent (`backend/summary_agent.py`)
- `tldr(paper)` → 1 paragraph plain-English summary.
- `section_summaries(paper)` → 2-3 sentence summary per detected section.
- `key_findings(paper)` → bullet list of the paper's main claimed contributions/results.

All three are separate, small, cheap LLM calls (not one giant prompt) so each
is independently cacheable and debuggable.

### 3. Gap Analysis Agent (`backend/gap_agent.py`)
Two-pass approach:
1. **Extraction pass** — pulls verbatim-grounded material from the paper's own
   Limitations/Discussion/Future-Work sections (if present).
2. **Inference pass** — given the full paper + extraction pass, the LLM is asked
   to propose *additional* gaps not explicitly stated (e.g. untested edge cases,
   missing baselines, scalability claims without evidence), each tagged with a
   confidence label and a suggested research direction.

Output distinguishes `author_acknowledged_gaps` vs `inferred_gaps` so the report
never conflates "the authors admit this" with "the model thinks this." Both
passes get structured JSON back from the LLM (`llm_client.complete_json`) rather
than delimited-string parsing, so a reformatted line doesn't silently drop an item.

### 4. RAG Agent (`backend/rag_agent.py`)
- Chunks each section's sentences into ~800-char windows (150 overlap) without
  ever splitting mid-sentence; each chunk carries the section heading and the
  *actual* page(s) its own sentences came from (via `page_segments`), not the
  whole section's page range.
- Embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2` (no extra API key).
- Indexes with FAISS (in-memory, per document, rebuilt from persisted chunks on restart).
- `retrieve(query, k=5)` → pulls a wider candidate pool via cosine similarity,
  then reranks with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and
  keeps the top-k (toggle via `RERANK_ENABLED` in `config.py`).
- `answer(query, history)` → retrieves, bounds chat history by character budget
  (not just turn count), then calls the LLM (Groq) with the retrieved chunks as
  context, returning an answer + chunk metadata (heading, page, score) for a
  "sources" display in the UI.

### 5. Report Agent (`backend/report_agent.py`)
Pure compositor: takes the outputs of Summary + Gap agents (+ metadata) and
renders a single Markdown report with consistent structure:

```
# <Title>
## TL;DR
## Section Summaries
## Key Findings
## Research Gaps
  ### Author-Acknowledged
  ### Inferred
## Suggested Future Directions
```

No fresh "write a report" LLM call — this avoids the report drifting from what
the Summary/Gap tabs already showed the user.

### 6. Orchestrator (`backend/orchestrator.py`)
- Holds `sessions: dict[doc_id -> DocSession]`, where `DocSession` bundles the
  `IngestedPaper`, the RAG index, cached summary/gap results, and chat history.
- All access to `self.sessions` goes through a `threading.Lock`, since FastAPI's
  sync endpoints run on a threadpool and can hit the dict concurrently.
- Public methods: `ingest_paper`, `get_summary`, `get_gaps`, `get_report`, `chat`.
- Lazily computes and caches summary/gaps on first request per document, so
  uploading is fast and analysis happens on demand. Every cache update is also
  persisted (see Persistence Agent below).
- On startup, rehydrates every previously-persisted session from SQLite —
  ingested papers, cached summaries/gaps, and chat history survive a restart.

### 7. API layer (`backend/main.py`, FastAPI)
Thin REST wrapper around the orchestrator:
- `POST /upload` → `{doc_id, title, num_pages, sections}`
- `GET /summary/{doc_id}`
- `GET /gaps/{doc_id}`
- `GET /report/{doc_id}` (markdown text) and `GET /report/{doc_id}/download` (file)
- `POST /chat` `{doc_id, query, history}` → `{answer, sources}`

Validates uploads (PDF-only, size limit via `MAX_PDF_SIZE_MB`) before they reach
the orchestrator. Errors are mapped to distinct HTTP statuses rather than a flat
500: `IngestionError` → 400, unknown `doc_id` → 404, `LLMProviderError` (Groq
failed after retries) → 502, anything else → 500 (logged server-side with full
traceback, generic message to the client). A logging middleware records every
request's method, path, status, and timing.

### 8. Frontend (`frontend/app.py`, Streamlit)
- Upload widget → calls `/upload`, stores `doc_id` in `st.session_state`.
- Tabs: **Summary**, **Report** (with download button), **Research Gaps**, **Chat**.
- Chat tab keeps its own message history in session state and displays
  retrieved source snippets under each answer.

### 9. Persistence Agent (`backend/persistence.py`)
SQLite-backed store (`data/sessions.db`) so a server restart doesn't lose
everything. Persists the ingested paper (as JSON), chunk text+metadata,
chat history, and cached summary/gap results. The FAISS index itself is
*not* serialized — on load, chunks are re-embedded locally via the same
embedding model, which is cheap (no Groq calls) and avoids FAISS
version/serialization compatibility issues across restarts or machines.

### 10. Eval Agent (`backend/eval_agent.py`) + `scripts/run_eval.py`
A practical quality-check tool, not a formal benchmark (no labeled dataset):
- **Retrieval hit-rate** — given a question with expected keywords, checks
  whether the retrieved chunks actually contain them.
- **LLM-as-judge faithfulness** — scores (1-5, with reasoning) whether a
  generated summary/gap/chat-answer is actually supported by its source text,
  flagging unsupported claims. Useful for catching hallucination and spotting
  regressions while testing across many papers.
- `scripts/run_eval.py` is a CLI: point it at a PDF (optionally with a JSON
  file of test questions) and it prints a full report.

## Data flow for a single user session

```
upload PDF ─▶ Ingestion Agent ─▶ IngestedPaper
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
       Summary Agent            Gap Agent                RAG Agent
             │                       │                    (index built,
             ▼                       ▼                     idle until chat)
        cached summary        cached gap analysis
             └───────────┬───────────┘
                         ▼
                   Report Agent ──▶ Markdown report (view / download)

chat query ──▶ Orchestrator ──▶ RAG Agent.answer() ──▶ Groq ──▶ answer + sources
```

Every write to a session (new ingestion, computed summary/gaps, appended chat
turn) is mirrored to SQLite via the Persistence Agent, so this whole flow
resumes correctly after a server restart without re-uploading the PDF.

## V2 Roadmap

The V1 implementation optimizes for a small, readable, dependency-light
reference build. The table below captures where the current design makes
simplifying assumptions, and the planned enhancement for each as the system
matures toward handling arbitrary real-world papers. This is the working
scope for the next iteration.

### Phase 1 — Extraction & document understanding

| Area | Current design (V1) | Planned enhancement (V2) | Rationale |
|---|---|---|---|
| PDF text extraction | Single-pass linear text extraction (`page.get_text("text")`) | Layout-aware extraction using PyMuPDF's block/dict mode, with blocks sorted by column then position | Correctly orders text on two-column academic layouts (ACL/IEEE/NeurIPS style), which is the dominant format for research papers |
| Scanned documents | Assumes a text layer is present | Detect low-text-density pages and fall back to OCR (e.g. Tesseract) | Extends coverage to scanned/legacy PDFs |
| Section detection | Regex match against a curated list of common heading names | Add an LLM-guided segmentation pass as a fallback when regex detection yields too few or ambiguous sections | Covers papers using non-standard section naming (e.g. "Threats to Validity," "Broader Impact") without hand-maintaining an ever-growing heading list |
| Header/footer noise | Not filtered | Detect and strip lines repeated near-identically across pages before splitting/chunking | Keeps running headers/page numbers out of section text and chunks |

### Phase 2 — Analysis quality

| Area | Original design (V1) | Status |
|---|---|---|
| Document context for Summary/Gap agents | Fixed prefix of the document (first several thousand characters) | ✅ **Done** — `prioritized_excerpt()` front-loads the abstract + Limitations/Discussion/Conclusion before filling remaining budget in order, so those sections survive truncation regardless of paper length. Not a full map-reduce over every section, but directly fixes the "Limitations gets cut off" failure mode. |
| Gap agent output format | Delimited text format (`GAP: ... \|\| CONFIDENCE: ... \|\| DIRECTION: ...`), parsed with string splitting | ✅ **Done** — structured JSON output via `llm_client.complete_json`, with malformed/invalid entries filtered rather than crashing. |
| Section-summary generation | Sequential LLM calls, one per section | ✅ **Done** — parallelized via a capped thread pool (`LLM_MAX_CONCURRENT_CALLS`, default 3) sized to stay under Groq's free-tier requests-per-minute limit. |

### Phase 3 — Retrieval quality

| Area | Original design (V1) | Status |
|---|---|---|
| Vector store | FAISS, in-memory, rebuilt per session | ✅ **Partially done** — sessions (and their chunks) now persist in SQLite and survive a restart; FAISS itself stays in-memory but is cheaply rebuilt from persisted chunks on load. Not a switch to ChromaDB, but achieves the actual goal (surviving a restart) without adding a new vector-store dependency. |
| Chunking | Fixed-size character windows with overlap | ✅ **Done** — sentence-aware chunking that never splits mid-sentence, packed to a target size with overlap. |
| Retrieval | Top-k cosine similarity | ✅ **Done** (cross-encoder, not MMR) — pulls a wider FAISS candidate pool, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeps top-k. MMR (for result diversity specifically) is still open if redundant chunks become a problem in practice. |
| Source attribution | Chunk index + text preview | ✅ **Done** — each chunk carries section heading and its *actual* page (via per-page-segment tracking, not the whole section's page range), surfaced in chat citations and the sources list. |
| Multi-turn chat | Query embedded as-is | **Still open** — no query-rewrite step yet; follow-up questions like "what about its limitations?" rely on the retrieved-context + conversation-history in the prompt rather than a rewritten standalone query. |

### Phase 4 — Productionization (not required for single-user local use)

| Area | Original plan | Status |
|---|---|---|
| Session storage | Move from in-memory dict to SQLite/Redis with TTL-based eviction | ✅ **Partially done** — SQLite persistence added (`persistence.py`); no TTL-based eviction yet, so old sessions accumulate indefinitely. |
| Access control | Add API-key/session auth if deployed beyond localhost | **Still open** — no auth; fine for local single-user use, required before any shared/public deployment. |
| Chat responses | Stream tokens instead of returning the full answer at once | **Still open**. |
| Observability | Token/cost logging per call; upload size and page-count limits | ✅ **Done** — every Groq call logs timing + prompt/completion/total tokens; upload size is capped (`MAX_PDF_SIZE_MB`); every API request is logged (method, path, status, timing). Dollar-cost tracking specifically isn't computed, just raw token counts. |
| Error handling | Everything surfaces as a raw 500 | ✅ **Done** — distinct status codes for bad input (400), unknown doc (404), upstream LLM failure (502), and unexpected errors (500, logged server-side with full traceback). |
| Testing | Unit tests for section splitting, chunking, and orchestrator routing, with the LLM client mocked | **Still open** — verified manually during development (chunking, persistence round-trip, error mapping, retry/backoff all exercised with mocked Groq/embedding calls), but there's no committed automated test suite yet. |

### Phase 5 — Quality evaluation (added, not in original roadmap)

| Area | Status |
|---|---|
| Retrieval hit-rate checking | ✅ **Done** — `eval_agent.retrieval_hit_rate()` checks whether expected keywords appear in retrieved chunks for a set of test questions. |
| Faithfulness / hallucination checking | ✅ **Done** — `eval_agent.judge_faithfulness()` uses the LLM itself to score (1-5) whether generated summaries/gaps/answers are supported by their source text, flagging unsupported claims. |
| CLI harness | ✅ **Done** — `scripts/run_eval.py` runs the full pipeline against a PDF and prints a report; supports an optional test-question file and a JSON report export. |
| Ground-truth benchmark dataset | **Still open** — the eval harness is a practical regression/spot-check tool, not a rigorous benchmark against labeled correct answers. |

### Recommended next iteration scope
With Phases 2 and most of Phase 3/4 now implemented, the highest-value remaining
work is **Phase 1** (heading detection is still the weakest link — see the
"Known limitation" note under the Ingestion Agent above) and building out a
small labeled test set so the Phase 5 eval harness can report an actual
accuracy number rather than just relative faithfulness scores.

