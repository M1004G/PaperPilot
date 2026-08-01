# Architecture

## Components

### 1. Ingestion Agent (`backend/ingestion_agent.py`)
Input: PDF file path.
Steps:
1. Extract raw text per page with PyMuPDF, validating the file first (rejects
   corrupt, empty, and password-protected PDFs with a clear `IngestionError`
   rather than letting a raw PyMuPDF exception bubble up as a bare 500).
2. Split into sections using two combined signals: (a) regex match against a
   known-heading keyword list (Abstract, Introduction, Related Work,
   Methodology, Results, Discussion, Limitations, Conclusion, References, ...),
   and (b) font-layout detection — a line whose font is noticeably larger
   and/or bolder than the document's dominant body-text size (or, absent that,
   a short ALL-CAPS line) is also treated as a heading. (b) lets unconventional
   section names ("Proposed Framework", "Case Study") split correctly without
   being in the keyword list; a numbered heading line ("3.2 Case Study") can
   even establish section structure before any keyword-list heading has been
   seen, for papers that don't use standard IMRaD naming at all.
3. Track each section's text *per page* (`page_segments`), not just an overall
   page range — this lets chunks downstream carry their actual page rather than
   the whole section's span.
4. Extract lightweight metadata: title (first prominent line), abstract text,
   page count.
Output: `IngestedPaper` dataclass — `{title, abstract, sections: [{heading, text, page_start, page_end, page_segments}], full_text, num_pages}`.

This is the only agent that touches the raw file. Everything downstream
consumes its structured output, so re-parsing never happens twice.

**Known limitation**: heading detection is a font/case/keyword heuristic, not
a true layout model — a paper whose headings are styled identically to its
body text (same size, same weight, mixed case, not in the keyword list) still
won't split correctly on those unconventional names.

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

## Roadmap

The tables below track each area against its original, simpler V1 design:
what's been implemented since, and what's intentionally still open. Phase 1
(document understanding) is the highest-value remaining work; Phases 2-5 are
largely complete.

### Phase 1 — Extraction & document understanding

| Area | Current design | Planned enhancement | Rationale |
|---|---|---|---|
| PDF text extraction | Single-pass linear text extraction (`page.get_text("text")`) | Layout-aware extraction using PyMuPDF's block/dict mode, with blocks sorted by column then position | Correctly orders text on two-column academic layouts (ACL/IEEE/NeurIPS style), which is the dominant format for research papers. **Still open** — dict-mode is now used for font metadata (see Section detection below), but blocks aren't yet reordered for two-column layouts. |
| Scanned documents | Assumes a text layer is present | Detect low-text-density pages and fall back to OCR (e.g. Tesseract) | Extends coverage to scanned/legacy PDFs |
| Section detection | ~~Regex match against a curated list of common heading names~~ | Add an LLM-guided segmentation pass as a fallback when regex/font detection yields too few or ambiguous sections | ✅ **Partially done** — keyword regex is now combined with font-layout detection (`page.get_text("dict")` font size + bold, or short ALL-CAPS lines, relative to the document's dominant body-text size), so non-standard names ("Proposed Framework," "Case Study") split correctly without being hand-added to the keyword list. Not an LLM-guided fallback, so a paper with zero visual distinction between headings and body text (same size/weight/case) still won't split on its unconventional names — that residual case is the honest remaining gap. |
| Header/footer noise | Not filtered | Detect and strip lines repeated near-identically across pages before splitting/chunking | Keeps running headers/page numbers out of section text and chunks |

### Phase 2 — Analysis quality

| Area | Original design | Status |
|---|---|---|
| Document context for Summary/Gap agents | Fixed prefix of the document (first several thousand characters) | ✅ **Done** — `prioritized_excerpt()` front-loads the abstract + Limitations/Discussion/Conclusion before filling remaining budget in order, so those sections survive truncation regardless of paper length. Not a full map-reduce over every section, but directly fixes the "Limitations gets cut off" failure mode. |
| Gap agent output format | Delimited text format (`GAP: ... \|\| CONFIDENCE: ... \|\| DIRECTION: ...`), parsed with string splitting | ✅ **Done** — structured JSON output via `llm_client.complete_json`, with malformed/invalid entries filtered rather than crashing. |
| Section-summary generation | Sequential LLM calls, one per section | ✅ **Done** — parallelized via a capped thread pool (`LLM_MAX_CONCURRENT_CALLS`, default 3) sized to stay under Groq's free-tier requests-per-minute limit. |

### Phase 3 — Retrieval quality

| Area | Original design | Status |
|---|---|---|
| Vector store | FAISS, in-memory, rebuilt per session | ✅ **Partially done** — sessions (and their chunks) now persist in SQLite and survive a restart; FAISS itself stays in-memory but is cheaply rebuilt from persisted chunks on load. Not a switch to ChromaDB, but achieves the actual goal (surviving a restart) without adding a new vector-store dependency. |
| Chunking | Fixed-size character windows with overlap | ✅ **Done** — sentence-aware chunking that never splits mid-sentence, packed to a target size with overlap. |
| Retrieval | Top-k cosine similarity | ✅ **Done** (cross-encoder, not MMR) — pulls a wider FAISS candidate pool, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeps top-k. MMR (for result diversity specifically) is still open if redundant chunks become a problem in practice. |
| Source attribution | Chunk index + text preview | ✅ **Done** — each chunk carries section heading and its *actual* page (via per-page-segment tracking, not the whole section's page range), surfaced in chat citations and the sources list. |
| Multi-turn chat | Query embedded as-is | **Still open** — no query-rewrite step yet; follow-up questions like "what about its limitations?" rely on the retrieved-context + conversation-history in the prompt rather than a rewritten standalone query. |

### Phase 4 — Productionization (not required for single-user local use)

| Area | Original design | Status |
|---|---|---|
| Session storage | Move from in-memory dict to SQLite/Redis with TTL-based eviction | ✅ **Partially done** — SQLite persistence added (`persistence.py`); no TTL-based eviction yet, so old sessions accumulate indefinitely. |
| Access control | Add API-key/session auth if deployed beyond localhost | **Still open** — no auth; fine for local single-user use, required before any shared/public deployment. |
| Chat responses | Stream tokens instead of returning the full answer at once | **Still open**. |
| Observability | Token/cost logging per call; upload size and page-count limits | ✅ **Done** — every Groq call logs timing + prompt/completion/total tokens; upload size is capped (`MAX_PDF_SIZE_MB`); every API request is logged (method, path, status, timing). Dollar-cost tracking specifically isn't computed, just raw token counts. |
| Error handling | Everything surfaces as a raw 500 | ✅ **Done** — distinct status codes for bad input (400), unknown doc (404), upstream LLM failure (502), and unexpected errors (500, logged server-side with full traceback). |
| Testing | Unit tests for section splitting, chunking, and orchestrator routing, with the LLM client mocked | ✅ **Done** — 40+ pytest tests across ingestion, chunking/retrieval, persistence (temp SQLite), `llm_client` retry/backoff (mocked), and end-to-end API flows (`TestClient`). Runs in CI (`.github/workflows/ci.yml`) on every push/PR. No real network calls (Groq or Hugging Face) anywhere in the suite. |

### Phase 5 — Quality evaluation (added, not in original roadmap)

| Area | Status |
|---|---|
| Retrieval hit-rate checking | ✅ **Done** — `eval_agent.retrieval_hit_rate()` checks whether expected keywords appear in retrieved chunks for a set of test questions. |
| Faithfulness / hallucination checking | ✅ **Done** — `eval_agent.judge_faithfulness()` uses the LLM itself to score (1-5) whether generated summaries/gaps/answers are supported by their source text, flagging unsupported claims. |
| CLI harness | ✅ **Done** — `scripts/run_eval.py` runs the full pipeline against a PDF and prints a report; supports an optional test-question file and a JSON report export. |
| Ground-truth benchmark dataset | **Still open** — the eval harness is a practical regression/spot-check tool, not a rigorous benchmark against labeled correct answers. |

### Recommended next iteration scope
With Phases 2 and most of Phase 3/4 now implemented, and section detection in
Phase 1 upgraded from keyword-only to font/layout-based, the highest-value
remaining work is two-column layout ordering and OCR fallback (still open in
Phase 1 — see the "Known limitation" note under the Ingestion Agent above) and
building out a small labeled test set so the Phase 5 eval harness can report an
actual accuracy number rather than just relative faithfulness scores.

## Design decisions (the "why," not just the "what")

**Why FAISS instead of Chroma/Pinecone/a hosted vector DB?**
Each document gets its own small index (typically tens to low hundreds of
chunks) — this isn't a shared corpus being queried across documents. FAISS's
`IndexFlatIP` is an exact (not approximate) nearest-neighbor search, which at
this scale is fast enough that there's no accuracy/speed tradeoff to make.
Adding Chroma would mean a persistent service dependency for a workload that
doesn't need one yet. If this became a multi-document, shared-corpus system,
that calculus changes — that's a real limitation (see Phase 3), not a case
for switching preemptively.

**Why SQLite instead of Postgres?**
Single-process, single-machine, low write-concurrency (one person uploading
papers, not many concurrent users hammering writes). SQLite's file-based
simplicity means zero setup — no separate DB server to run, configure, or
containerize. Postgres would be the right call the moment this needs to serve
multiple concurrent users or run across multiple app instances (it doesn't
handle concurrent writers as gracefully) — that migration path is real but not
worth pre-paying for now.

**Why rebuild embeddings on restart instead of persisting the FAISS index itself?**
FAISS index serialization is tied to the FAISS library version that wrote it,
and re-embedding a document's chunks locally (no Groq/API calls involved) takes
well under a second for anything this app handles. Persisting raw chunk
text+metadata in SQLite and rebuilding the index from that on load sidesteps a
whole category of "works on my machine, breaks after a library upgrade" bugs
for a cost that's not actually noticeable.

**Why rerank after retrieval instead of just taking top-k from FAISS directly?**
Cosine similarity from a small bi-encoder embedding model (`all-MiniLM-L6-v2`)
is fast but coarse — it can rank a superficially-similar-but-irrelevant chunk
above a genuinely relevant one. A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
looks at the query and chunk *together* rather than as separate vectors, which
is slower per-comparison but much more accurate — so the pattern here is
"cast a wide net cheaply (FAISS, top ~20), then rerank precisely on that
smaller set" rather than doing expensive reranking over the whole document.

**Why hand-rolled orchestration instead of LangChain/LangGraph/CrewAI/AutoGen?**
Every step in this pipeline (ingest → chunk → embed → retrieve → prompt →
parse) is a plain function call with a clear input/output contract — there's
no dynamic agent-to-agent negotiation, tool-calling loop, or planning step that
would benefit from a framework's abstractions. A framework here would mean
learning and working around someone else's abstraction for orchestration logic
that's ~150 lines of straightforward Python (`orchestrator.py`). The tradeoff
would flip if the system needed genuine multi-step agentic planning (e.g. an
agent deciding *which* tools to call and in what order based on intermediate
results) — that's a different problem than this pipeline solves.

**Why a thread pool for concurrency instead of async/await throughout?**
The actual bottleneck (Groq API calls) is I/O-bound and would benefit from
async in a fully async system, but every I/O call here already blocks
synchronously (PyMuPDF, SentenceTransformer, FAISS, SQLite are all sync
libraries) — converting `main.py`'s route handlers to `async def` wouldn't
gain anything unless the *whole* chain underneath were also async, which would
mean async wrappers or replacements for several libraries that don't offer
them. A bounded `ThreadPoolExecutor` around the one place that's actually
parallelizable (independent section summaries) gets most of the concurrency
benefit without that larger rewrite. Under heavy concurrent load, this is the
part of the design that would need revisiting first — full async support
throughout the stack, not just at the FastAPI layer.

**Design tradeoff: the Orchestrator centralizes several responsibilities.**
It currently owns session state, caching, routing between agents, and
persistence coordination. That's a reasonable scope for the current feature
set, but every new capability (multi-document comparison, background jobs,
auth) would naturally extend the same class. The clear extension point, if
scope grows, is splitting it into narrower application services (e.g. a
`SessionService`, a `SummaryService`) that the API layer calls directly, with
the Orchestrator either shrinking to pure request routing or being retired.