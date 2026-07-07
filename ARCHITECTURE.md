# Architecture

## Components

### 1. Ingestion Agent (`backend/ingestion_agent.py`)
Input: PDF file path.
Steps:
1. Extract raw text per page with PyMuPDF.
2. Heuristically split into sections (Abstract, Introduction, Related Work,
   Methodology, Results, Discussion, Limitations, Conclusion, References, ...)
   using regex over heading-like lines (numbered / all-caps / title-case short lines).
3. Extract lightweight metadata: title (first prominent line), abstract text,
   page count.
Output: `IngestedPaper` dataclass — `{title, abstract, sections: [{heading, text}], full_text, num_pages}`.

This is the only agent that touches the raw file. Everything downstream
consumes its structured output, so re-parsing never happens twice.

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
never conflates "the authors admit this" with "the model thinks this."

### 4. RAG Agent (`backend/rag_agent.py`)
- Chunks `full_text` (~800 chars, 150 overlap).
- Embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2` (no extra API key).
- Indexes with FAISS (in-memory, per document).
- `retrieve(query, k=5)` → top-k chunks by cosine similarity.
- `answer(query, history)` → retrieves, then calls the LLM (Groq) with the retrieved
  chunks as context and conversation history, returns an answer + the chunk
  indices it used (for a "sources" display in the UI).

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
- Public methods: `ingest`, `get_summary`, `get_gaps`, `get_report`, `chat`.
- Lazily computes and caches summary/gaps on first request per document, so
  uploading is fast and analysis happens on demand.

### 7. API layer (`backend/main.py`, FastAPI)
Thin REST wrapper around the orchestrator:
- `POST /upload` → `{doc_id, title, num_pages, sections}`
- `GET /summary/{doc_id}`
- `GET /gaps/{doc_id}`
- `GET /report/{doc_id}` (markdown text) and `GET /report/{doc_id}/download` (file)
- `POST /chat` `{doc_id, query, history}` → `{answer, sources}`

### 8. Frontend (`frontend/app.py`, Streamlit)
- Upload widget → calls `/upload`, stores `doc_id` in `st.session_state`.
- Tabs: **Summary**, **Report** (with download button), **Research Gaps**, **Chat**.
- Chat tab keeps its own message history in session state and displays
  retrieved source snippets under each answer.

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

| Area | Current design (V1) | Planned enhancement (V2) | Rationale |
|---|---|---|---|
| Document context for Summary/Gap agents | Uses a fixed prefix of the document (first several thousand characters) | Map-reduce over the full document: summarize per-section (or per-chunk), then synthesize | Ensures Results/Evaluation/Discussion — often the most information-dense sections — are represented in TL;DR, key findings, and gap analysis regardless of paper length |
| Gap agent output format | Delimited text format (`GAP: ... \|\| CONFIDENCE: ... \|\| DIRECTION: ...`), parsed with string splitting | Structured JSON output via tool-use/schema enforcement | More robust parsing, easier to extend with new fields later |
| Section-summary generation | Sequential LLM calls, one per section | Parallelized calls (async) with retry/backoff | Faster generation, resilient to transient rate-limit/timeout errors |

### Phase 3 — Retrieval quality

| Area | Current design (V1) | Planned enhancement (V2) | Rationale |
|---|---|---|---|
| Vector store | FAISS, in-memory, rebuilt per session | ChromaDB with on-disk persistence | Index survives backend restarts; supports multiple documents/sessions cleanly |
| Chunking | Fixed-size character windows with overlap | Sentence/paragraph-aware chunking, optionally aligned to section boundaries | Avoids splitting mid-sentence, improving embedding and answer quality |
| Retrieval | Top-k cosine similarity | Add MMR re-ranking (and optionally a cross-encoder re-rank) | Improves diversity of retrieved context, reducing redundant chunks |
| Source attribution | Chunk index + text preview | Carry page number and section heading as chunk metadata through to the UI | Lets a user actually locate a cited claim in the source PDF |
| Multi-turn chat | Query embedded as-is | Add a query-rewrite step that resolves the question against chat history before retrieval | Improves retrieval quality on follow-up questions ("what about its limitations?") |

### Phase 4 — Productionization (not required for single-user local use)

| Area | Planned enhancement |
|---|---|
| Session storage | Move from in-memory dict to SQLite/Redis with TTL-based eviction |
| Access control | Add API-key/session auth if deployed beyond localhost |
| Chat responses | Stream tokens instead of returning the full answer at once |
| Observability | Token/cost logging per call; upload size and page-count limits |
| Testing | Unit tests for section splitting, chunking, and orchestrator routing, with the LLM client mocked |

### Recommended next iteration scope
Phase 1 and Phase 3 give the highest return for a working research assistant:
layout-aware extraction and LLM-guided section fallback (Phase 1) ensure the
rest of the pipeline is working from accurate document structure, while
better chunking, MMR, chunk metadata, and a ChromaDB-backed index (Phase 3)
directly improve retrieval and chat quality. Phase 2's map-reduce summarization
is the natural follow-on once full-document context is needed by the Summary
and Gap agents. Phase 4 is deferred until there's a reason to run this beyond
a local, single-user setup.

