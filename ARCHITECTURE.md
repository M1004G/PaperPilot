# System Architecture

## Overview

PaperPilot is a multi-agent system organized around a single pipeline: a PDF is ingested once into a structured representation, and independent agents consume that representation to produce summaries, gap analysis, retrieval-augmented chat, and a consolidated report. Each agent is a stateless module with a defined input/output contract; session state, caching, and coordination are handled centrally by the orchestrator.

## System Diagram

```
upload PDF --> Ingestion Agent --> IngestedPaper
                                       |
                +----------------------+----------------------+
                v                      v                      v
          Summary Agent            Gap Agent              RAG Agent
                |                      |                (index built,
                v                      v                 idle until chat)
          cached summary      cached gap analysis
                +----------+-----------+
                           v
                     Report Agent
                           |
                           v
                   Markdown report

chat query --> Orchestrator --> RAG Agent --> LLM --> answer + sources
```

Every session write (ingestion, computed summary/gaps, chat turn) is persisted to SQLite, so a server restart does not require re-uploading the document.

## Components

### 1. Ingestion Agent (`backend/ingestion_agent.py`)

**Input:** PDF file.

**Process:**
1. Extracts text per page via PyMuPDF, validating the file (rejects corrupt, empty, and encrypted PDFs).
2. Splits text into sections using a combination of keyword matching (Abstract, Introduction, Methodology, Results, etc.) and font-based layout detection (relative font size, weight, and casing).
3. Tracks each section's text per page (`page_segments`) for precise downstream page attribution.
4. Extracts title, abstract, and page count.

**Output:** `IngestedPaper` — `{title, abstract, sections: [{heading, text, page_start, page_end, page_segments}], full_text, num_pages}`.

**Limitations:** Heading detection is heuristic. Papers with no visual distinction between headings and body text (uniform size, weight, and case) may not segment correctly on non-standard section names.

### 2. Summary Agent (`backend/summary_agent.py`)

Produces a concise overview, per-section summaries, and key findings via three independent LLM calls.

### 3. Gap Analysis Agent (`backend/gap_agent.py`)

Two-pass structured extraction:
1. **Extraction** — gaps explicitly stated in the paper's Limitations/Discussion/Future Work sections.
2. **Inference** — additional gaps proposed by the LLM (untested edge cases, missing baselines, unsupported scalability claims), each with a confidence label and suggested direction.

Output distinguishes `author_acknowledged_gaps` from `inferred_gaps`.

### 4. RAG Agent (`backend/rag_agent.py`)

Retrieval infrastructure built on LangChain; LLM generation uses a dedicated Groq client (`llm_client.py`).

- **Chunking** — `RecursiveCharacterTextSplitter`, applied per page-segment (`chunk_size=800`, `overlap=150`), so every chunk maps to exactly one page.
- **Embedding** — `HuggingFaceEmbeddings` wrapping `sentence-transformers/all-MiniLM-L6-v2`.
- **Indexing** — `langchain_chroma.Chroma`, cosine distance, one persistent collection per document keyed by document ID (`data/chroma/`).
- **Retrieval** — a candidate pool is retrieved via similarity search, then reranked with `HuggingFaceCrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`), keeping the top-k. The paper's Abstract chunk is always included in the result set.
- **Generation** — retrieved chunks and bounded chat history are passed to Groq via `llm_client`, returning an answer with section/page citations.

### 5. Report Agent (`backend/report_agent.py`)

Assembles a single Markdown report from Summary and Gap outputs. No additional LLM call is made; the report reflects exactly what other agents already produced.

```
# <Title>
## Concise Overview
## Section Summaries
## Key Findings
## Research Gaps
  ### Author-Acknowledged
  ### Inferred
## Suggested Future Directions
```

### 6. Orchestrator (`backend/orchestrator.py`)

Coordinates all agents and holds session state (`sessions: dict[doc_id -> DocSession]`), guarded by a lock for thread-safe access under FastAPI's threaded request handling.

**Public interface:** `ingest_paper`, `get_summary`, `get_gaps`, `get_report`, `chat`.

Results are computed lazily on first request and cached; every cache update is persisted. On startup, all sessions are rehydrated from SQLite.


### 7. API Layer (`backend/main.py`, FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Ingest a PDF |
| `GET` | `/summary/{doc_id}` | Summary |
| `GET` | `/gaps/{doc_id}` | Gap analysis |
| `GET` | `/report/{doc_id}` | Markdown report |
| `POST` | `/chat` | Grounded Q&A |

Errors are mapped to distinct HTTP status codes: `IngestionError` → 400, unknown `doc_id` → 404, LLM provider failure → 502, unexpected errors → 500. All requests are logged with method, path, status, and timing.

### 8. Frontend (`frontend/app.py`, Streamlit)

Tabs: Summary, Report, Research Gaps, Chat. The Chat tab displays retrieved source snippets alongside each answer.

### 9. Persistence Agent (`backend/persistence.py`)

SQLite-backed session store (`data/sessions.db`). Persists the ingested paper, chunk text and metadata, chat history, and cached summary and gap results. Vector embeddings are persisted separately in per-document Chroma collections (`data/chroma/`), not re-embedded on restart.

### 10. Eval Agent (`backend/eval_agent.py`, `scripts/run_eval.py`)

- **Retrieval hit-rate** — checks whether retrieved chunks contain expected keywords for a set of test questions.
- **Faithfulness scoring** — LLM-as-judge scoring (1-5) of whether generated summaries/gaps/answers are supported by their source text.
- `scripts/run_eval.py` provides a CLI for running the full pipeline against a PDF.

## Configuration Reference

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Model identifier | `openai/gpt-oss-120b` |
| `MAX_PDF_SIZE_MB` | Upload size limit | `25` |
| `LLM_MAX_CONCURRENT_CALLS` | Thread pool size for section summarization | `3` |
| `RERANK_ENABLED` | Enable cross-encoder reranking | `true` |

## Design Principles

- **Vector storage** — ChromaDB, scoped per document via a dedicated persistent collection keyed by document ID. Embeddings survive a restart without re-computation; a chunk-count mismatch against the stored collection triggers a full reset and re-embed rather than reusing possibly-stale data.
- **Persistence** — SQLite for session/chunk metadata, chosen for single-process local deployment. Vector embeddings are persisted separately by ChromaDB rather than recomputed on load.
- **Retrieval framework** — LangChain provides chunking, vector store, and reranking abstractions. LLM generation calls use a dedicated Groq client with retry/backoff and structured error mapping, independent of the retrieval framework.
- **Concurrency** — a bounded thread pool parallelizes independent section-summary calls. The remainder of the request path is synchronous, consistent with the underlying I/O libraries (PyMuPDF, ChromaDB, SQLite).
- **Orchestration** — agent coordination is implemented directly rather than through an agent framework, reflecting a linear pipeline with no dynamic planning or tool-selection requirement.

## Known Limitations

- Two-column academic layouts are not reordered during text extraction.
- No OCR fallback for scanned documents.
- Heading detection may not segment sections with no visual distinction from body text.
- No query rewriting for multi-turn chat.
- No authentication layer; suitable for local, single-user deployment.
- No TTL-based session eviction.

## Roadmap

| Area | Status |
|---|---|
| Layout-aware text extraction (two-column ordering) | Planned |
| OCR fallback for scanned documents | Planned |
| LLM-guided section segmentation fallback | Planned |
| Query rewriting for multi-turn chat | Planned |
| Authentication | Planned |
| Ground-truth evaluation benchmark | Planned |
