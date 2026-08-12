# System Architecture

## Overview

PaperPilot is a multi-agent system organized around a single pipeline: a PDF is ingested once into a structured representation, and independent agents consume that representation to produce summaries, gap analysis, retrieval-augmented chat, and a code reproducibility evaluation. Each agent is a stateless module with a defined input/output contract; session state, caching, and coordination are handled centrally by the orchestrator.

## System Diagram

```
upload PDF ─▶ Ingestion Agent ─▶ IngestedPaper
                                     │
             ┌───────────────────────┼───────────────────────┬───────────────────────┐
             ▼                       ▼                       ▼                       ▼
       Summary Agent            Gap Agent                RAG Agent          Reproducibility Suite
             │                       │                  (index built,       (repository fetch or
             ▼                       ▼                   idle until chat)    code generation)
        cached summary        cached gap analysis                                    │
             └───────────┬───────────┘                                              ▼
                         ▼                                          Reproducibility Check Agent
                   Report Agent ◀───────────────────────────────────────────────────┘
                         │
                         ▼
                Markdown report

chat query ──▶ Orchestrator ──▶ RAG Agent ──▶ LLM ──▶ answer + sources
```

Every session write (ingestion, computed summary/gaps, chat turn, reproducibility result) is persisted to SQLite, so a server restart does not require re-uploading the document.

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

Assembles a single Markdown report from Summary, Gap, and Reproducibility Suite outputs. No additional LLM call is made; the report reflects exactly what other agents already produced.

```
# <Title>
## Concise Overview
## Section Summaries
## Key Findings
## Research Gaps
  ### Author-Acknowledged
  ### Inferred
## Suggested Future Directions
## Code Reproducibility
  ### Category Scores
  ### Checks
  ### Warnings
  ### Claim Verification
```

### 6. Orchestrator (`backend/orchestrator.py`)

Coordinates all agents and holds session state (`sessions: dict[doc_id -> DocSession]`), guarded by a lock for thread-safe access under FastAPI's threaded request handling.

**Public interface:** `ingest_paper`, `get_summary`, `get_gaps`, `get_reproducibility`, `get_cached_reproducibility`, `get_report`, `chat`.

Results are computed lazily on first request and cached; every cache update is persisted. On startup, all sessions are rehydrated from SQLite.

`get_reproducibility(doc_id, repo_url=None)` selects between the two Reproducibility Suite sources: a linked or supplied repository is fetched and evaluated; otherwise the Codegen Agent generates an implementation, which is evaluated by the same check agent. The response includes a `mode` field (`"repo_check"` or `"generated"`) but uses an identical scoring structure either way.

### 7. API Layer (`backend/main.py`, FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Ingest a PDF |
| `GET` | `/summary/{doc_id}` | Summary |
| `GET` | `/gaps/{doc_id}` | Gap analysis |
| `GET` | `/reproducibility/{doc_id}` | Reproducibility evaluation |
| `GET` | `/reproducibility/{doc_id}/download` | Download generated files |
| `GET` | `/report/{doc_id}` | Markdown report |
| `POST` | `/chat` | Grounded Q&A |

Errors are mapped to distinct HTTP status codes: `IngestionError` → 400, unknown `doc_id` → 404, LLM provider failure → 502, unexpected errors → 500. All requests are logged with method, path, status, and timing.

### 8. Frontend (`frontend/app.py`, Streamlit)

Tabs: Summary, Report, Research Gaps, Reproducibility, Chat. The Reproducibility tab renders repository checks or generated code (with an inline viewer and download option) depending on evaluation mode. The Chat tab displays retrieved source snippets alongside each answer.

### 9. Persistence Agent (`backend/persistence.py`)

SQLite-backed session store (`data/sessions.db`). Persists the ingested paper, chunk text and metadata, chat history, and cached summary/gap/reproducibility results. Vector embeddings are persisted separately in per-document Chroma collections (`data/chroma/`), not re-embedded on restart. Schema changes are applied via additive migrations at startup.

### 10. Eval Agent (`backend/eval_agent.py`, `scripts/run_eval.py`)

- **Retrieval hit-rate** — checks whether retrieved chunks contain expected keywords for a set of test questions.
- **Faithfulness scoring** — LLM-as-judge scoring (1-5) of whether generated summaries/gaps/answers are supported by their source text.
- `scripts/run_eval.py` provides a CLI for running the full pipeline against a PDF.

### 11. Reproducibility Check Agent (`backend/repro_check_agent.py`) and Repo Fetch (`backend/repo_fetch.py`)

Evaluates a codebase for trustworthiness and completeness. The evaluation logic is source-agnostic, operating on `(tree, get_content, metadata)` rather than a GitHub-specific interface, so the same checks apply whether the code was fetched from a repository or generated from the paper.

**Scoring categories:**

| Category | Repository mode | Generation mode |
|---|---|---|
| Documentation | README, usage instructions | README, usage instructions |
| Project Hygiene | License, CI, dependency manifest, archive status | Dependency manifest |
| Code Quality | Not applicable | Static analysis (ruff, pyflakes rules) |
| Correctness | Dependency pinning, tests | Dependency pinning, tests, LLM semantic review |

Informational checks (missing Dockerfile/environment spec, missing CITATION file) are reported as warnings and excluded from scoring.

**Claim verification** — a bounded LLM call comparing the paper's Methods section against the codebase's file tree and README, applicable to both modes.

`repo_fetch.py` accesses repositories exclusively through the GitHub REST API (metadata, file tree, individual file contents); no repository is cloned and no code is executed. Only `github.com` URLs are accepted.

**Limitations:** This evaluation assesses plausibility and structural completeness. It does not execute code and cannot confirm that a codebase produces the paper's reported results.

### 12. Codegen Agent (`backend/codegen_agent.py`)

Generates an implementation attempt when no repository is linked to a paper.

**Pipeline:**
1. **Extraction** — structured extraction of datasets, model architecture, training configuration, and evaluation details from the paper's methodology, with an explicit `gaps` list for unspecified details.
2. **Planning** — determines the file set and inter-file dependencies before generation. Filenames are validated against path traversal and deduplicated.
3. **Generation** — one LLM call per file, in dependency order, with each file's prompt including the content of its dependencies.
4. **Validation** — generated Python files are checked with `compile()`; a syntax error triggers one retry with the error included in the follow-up prompt. Refusal-shaped or empty output is rejected without retry.

Generated output is evaluated by the Reproducibility Check Agent using the same scoring system applied to fetched repositories.

**Limitations:** Validation is syntactic, not semantic. Generated code is not executed at any stage.

## Configuration Reference

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Model identifier | `llama-3.3-70b-versatile` |
| `MAX_PDF_SIZE_MB` | Upload size limit | `25` |
| `LLM_MAX_CONCURRENT_CALLS` | Thread pool size for section summarization | `3` |
| `GITHUB_TOKEN` | GitHub API token | optional |
| `REPRO_LLM_CLAIMS_ENABLED` | Enable claim verification | `true` |
| `SEMANTIC_REVIEW_ENABLED` | Enable LLM semantic review | `true` |
| `CODEGEN_ENABLED` | Enable code generation | `true` |
| `RERANK_ENABLED` | Enable cross-encoder reranking | `true` |
| `CODEGEN_MAX_FILES` | Maximum files per generated implementation | `8` |

## Design Principles

- **Vector storage** — ChromaDB, scoped per document via a dedicated persistent collection keyed by document ID. Embeddings survive a restart without re-computation; a chunk-count mismatch against the stored collection triggers a full reset and re-embed rather than reusing possibly-stale data.
- **Persistence** — SQLite for session/chunk metadata, chosen for single-process local deployment. Vector embeddings are persisted separately by ChromaDB rather than recomputed on load.
- **Retrieval framework** — LangChain provides chunking, vector store, and reranking abstractions. LLM generation calls use a dedicated Groq client with retry/backoff and structured error mapping, independent of the retrieval framework.
- **Reproducibility scoring** — evaluation logic is source-agnostic and category-based, applied identically regardless of whether code originates from a repository or is generated.
- **Code generation** — implementation attempts follow a plan-then-generate pipeline (extraction, planning, per-file generation, validation) rather than single-pass generation, to preserve inter-file consistency and allocate token budget per file.
- **Code execution** — no generated or fetched code is executed at any point in the system. Validation is limited to syntax checking and static analysis.
- **Concurrency** — a bounded thread pool parallelizes independent section-summary calls. The remainder of the request path is synchronous, consistent with the underlying I/O libraries (PyMuPDF, ChromaDB, SQLite).
- **Orchestration** — agent coordination is implemented directly rather than through an agent framework, reflecting a linear pipeline with no dynamic planning or tool-selection requirement.

## Known Limitations

- Two-column academic layouts are not reordered during text extraction.
- No OCR fallback for scanned documents.
- Heading detection may not segment sections with no visual distinction from body text.
- No query rewriting for multi-turn chat.
- No authentication layer; suitable for local, single-user deployment.
- No TTL-based session eviction.
- Reproducibility evaluation does not execute code and cannot confirm numerical reproduction of results.

## Roadmap

| Area | Status |
|---|---|
| Layout-aware text extraction (two-column ordering) | Planned |
| OCR fallback for scanned documents | Planned |
| LLM-guided section segmentation fallback | Planned |
| Query rewriting for multi-turn chat | Planned |
| Authentication | Planned |
| Sandboxed code execution and benchmark comparison | Under evaluation |
| Reference-code retrieval for reproducibility checks | Under evaluation |
| Ground-truth evaluation benchmark | Planned |
