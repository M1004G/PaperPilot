# PaperPilot — Autonomous Research Assistant

Multi-agent system that ingests a research paper (PDF) and produces:
- A TL;DR + section-by-section summary
- A structured report (Markdown)
- A research-gap analysis (author-acknowledged vs. critically inferred)
- A RAG-based chatbot grounded in the paper, with section+page citations
- A code-reproducibility check against the paper's linked GitHub repo (repo hygiene + a lightweight paper-vs-code claim check)

See `ARCHITECTURE.md` for the full design.

## Setup

### Option A: Docker (recommended — one command, no local Python setup)

```bash
cd research-assistant
cp .env.example .env
# edit .env and add your GROQ_API_KEY (free, no card — https://console.groq.com)

docker compose up
```

Backend: http://localhost:8000 · Frontend: http://localhost:8501 · Data persists in `./data` on your host machine across container restarts/rebuilds.

### Option B: Local Python

```bash
cd research-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY
# (free, no credit card required — get one at https://console.groq.com)
```

## Run (Option B — local Python only; Docker's `docker compose up` already starts both services)

Terminal 1 — backend:
```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 — frontend:
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL it prints (usually http://localhost:8501), upload a PDF, and use the tabs.

Sessions persist across restarts in `data/sessions.db` (SQLite) — re-uploading isn't needed after a backend restart; ingested papers, cached summaries/gaps, and chat history are reloaded automatically on startup.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

40+ tests covering ingestion (heading/page detection, validation errors),
chunking/retrieval (sentence-aware splitting, page precision, reranking),
persistence (temp SQLite, corrupt-row handling), `llm_client` retry/backoff
(mocked), and end-to-end API flows (`TestClient`). No test hits a real Groq
or Hugging Face endpoint — everything's mocked or generated locally, so the
suite runs fast and deterministically offline. Runs automatically in CI
(`.github/workflows/ci.yml`) on every push/PR to `main`.

## Evaluating quality

Rather than eyeballing output, `scripts/run_eval.py` runs the full pipeline against a single PDF and scores it:

```bash
# Faithfulness scoring only (no known-answer questions needed)
python -m scripts.run_eval path/to/paper.pdf

# Also check retrieval hit-rate + chat answer faithfulness against known questions
python -m scripts.run_eval path/to/paper.pdf --questions path/to/questions.json --report-out report.json
```

`questions.json` format:
```json
[
  {"question": "What dataset was used?", "expected_keywords": ["dataset", "collected"]},
  {"question": "What is the main limitation?", "expected_keywords": ["limitation"]}
]
```

This uses an LLM-as-judge to score whether generated summaries/gaps/chat answers are actually supported by the source text (1-5, with reasoning) — useful for catching hallucination and spotting regressions as you test across many papers. It's a practical tool, not a rigorous benchmark (no labeled ground-truth dataset).

## Project layout

```
backend/
  config.py              # env/config loading (Groq settings, rate-limit tuning, storage, logging)
  llm_client.py          # Groq API wrapper: retry/backoff on 429s, timeout, token-usage logging
  logging_utils.py       # request-ID propagation across threads/loggers via contextvars
  ingestion_agent.py     # PDF -> text, sections, per-page metadata; validates corrupt/encrypted/empty files
  summary_agent.py       # TL;DR + section summaries (parallelized, capped concurrency, per-section error isolation)
  gap_agent.py           # research gap analysis (structured JSON output, not string parsing)
  repo_fetch.py          # safe, read-only GitHub access (metadata, file tree, file contents) -- no clone, no code execution
  repro_check_agent.py   # source-agnostic code-quality checks + LLM claim verification (works on a real repo OR generated code)
  codegen_agent.py       # when no repo exists: extracts structured methodology info, generates an implementation attempt
  rag_agent.py           # sentence-aware chunking, section+page metadata, cross-encoder reranking, chat
  report_agent.py        # assembles final Markdown report (no LLM call — deterministic)
  persistence.py         # SQLite-backed session storage, survives server restarts
  eval_agent.py          # retrieval hit-rate + LLM-as-judge faithfulness scoring
  orchestrator.py        # coordinates all agents, thread-safe session state, per-stage timing
  main.py                # FastAPI app / routes, request-ID + logging middleware, structured error responses
frontend/
  app.py                 # Streamlit UI
scripts/
  run_eval.py            # CLI: run the pipeline against a PDF and print a quality report
tests/                   # pytest suite (see Testing section below)
data/
  sessions.db            # SQLite session store (created automatically, gitignored)
Dockerfile               # single image, used for both backend and frontend services
docker-compose.yml       # wires backend + frontend + a persistent ./data volume
.github/workflows/ci.yml # runs the test suite on every push/PR
```

## Error handling

The API distinguishes error types instead of collapsing everything into a raw 500:
- **400** — bad input: non-PDF file, oversized upload (`MAX_PDF_SIZE_MB` in `config.py`), corrupt/encrypted/empty PDF
- **404** — unknown `doc_id`
- **502** — the LLM provider (Groq) failed after all retries
- **500** — unexpected internal error (logged server-side with full traceback; client gets a generic message)

Every request is logged (method, path, status, timing) via middleware in `main.py`; every Groq call logs timing and token usage in `llm_client.py`.

## Notes / extension points
- `config.py` centralizes the model name (`llama-3.3-70b-versatile` on Groq by default) — change it in one place.

## Reproducibility Suite
Answers "can I trust/reuse code for this paper?" — via `GET /reproducibility/{doc_id}` (optionally `?repo_url=...`). Two sources feed the **same** evaluation, so trust means the same thing either way (response includes a `mode` field):

- **`mode: "repo_check"`** — a repo is linked in the paper (or supplied via `repo_url`). Fetched read-only through GitHub's REST API only (`api.github.com` / `raw.githubusercontent.com`) — no `git clone`, no code from the target repo is ever executed. Set `GITHUB_TOKEN` in `.env` to raise the rate limit from 60/hr to 5000/hr.
- **`mode: "generated"`** — no repo is linked (the common case). `codegen_agent.py` extracts structured methodology info (datasets/model/training config, with explicit `gaps` for anything unspecified), plans a small file set + dependencies, then generates each file in dependency order (`model.py`, `dataset.py`, `train.py`, `requirements.txt`, `README.md`). Each `.py` file is validated with `compile(..., "exec")` before being accepted — a syntax error triggers one retry with the actual error shown to the model; refusal-shaped or suspiciously short output is rejected outright, no retry. Filenames are sanitized against path traversal/absolute paths (flat single-directory only) both when the plan is built and again, independently, in the download route. None of this executes the generated code — it's a syntax/shape check, not a correctness one. Download as a zip via `GET /reproducibility/{doc_id}/download`. Toggle off with `CODEGEN_ENABLED=false`.

Either way, `repro_check_agent.py` runs the same evaluation, organized into **four scored categories** (Documentation, Project Hygiene, Code Quality, Correctness) plus a separate, **non-scored Warnings list** — not one opaque number, and not informational checks silently shaping the score:
- **`profile="repo"`** (real repos) — Documentation (README, usage instructions), Hygiene (license, recognized-license check, dependency manifest, CI, archived status), Correctness (pinned deps, tests). Code Quality is `N/A` — we don't fetch every source file's content from GitHub just to lint it.
- **`profile="generated"`** (self-checked generated code) — license/CI/archived-status checks are **dropped entirely**, not just marked `na`: code generated fresh in one session was never going to have a LICENSE file or CI, and scoring it down for that conflates repo-maintenance hygiene with implementation quality. In their place: Code Quality is a **ruff-based static analysis check** (pyflakes rules only — undefined names, unused imports/vars; cosmetic style like import ordering is excluded), and Correctness includes an **LLM semantic review** comparing the actual generated code (not just the file tree) against the paper — missing algorithmic steps, hallucinated APIs, tensor-shape concerns.
- **Warnings, not deductions** — missing Dockerfile/env spec and missing CITATION file are informational only, shown separately, and don't affect the score at all (previously they cost partial credit even for otherwise-complete code).
- **Claim verification (one bounded LLM call, optional)** — does the codebase plausibly implement what the paper's Methods section claims? Toggle off with `REPRO_LLM_CLAIMS_ENABLED=false`. Works for either profile (only needs the file tree/README).

**What this score is and isn't**: it measures "does this look trustworthy/complete" — repo hygiene, static code quality, and (for generated code) an LLM's read on paper-vs-code faithfulness — not "is this proven to be a correct implementation." Neither profile executes any code (see the Codegen Agent section above); a high score means "worth reviewing further," not "verified."

- Groq's free tier is rate-limited (not credit-metered) — `llm_client.py` retries on 429s (configurable timeout via `LLM_TIMEOUT_SECONDS`), and section summarization runs on a capped thread pool so it doesn't burst past the requests-per-minute limit. One section failing doesn't discard the others' results.
- Every request gets a request ID (returned as an `X-Request-ID` header and threaded through every log line, including inside the section-summary thread pool via `contextvars`) — grep any log by request ID to see everything that happened for one request. Per-stage timing (ingestion, index build, retrieval/rerank, per-Groq-call) and cache hit/miss are all logged.
- The FAISS index itself isn't persisted — on restart, chunks are reloaded from SQLite and re-embedded locally (no Groq calls involved, so this is cheap and avoids FAISS version/serialization issues).
- Heading detection (`ingestion_agent.py`) combines a known-heading keyword list (Introduction, Results, etc.) with font-based layout detection — a line is also treated as a heading if it's noticeably larger and/or bolder than the document's body text (or, failing that, a short ALL-CAPS line), so unconventional section names ("Proposed Framework", "Case Study") are split correctly too. This is heuristic, not perfect: papers whose headings are styled identically to body text (no size/weight/case distinction at all) still won't be split on those unconventional names — a genuinely robust fix would need a layout-ML model rather than font heuristics.
- **Architectural tradeoffs** (see `ARCHITECTURE.md`'s Design Decisions section for the full reasoning): the `Orchestrator` centralizes session state, caching, routing, and persistence coordination — the natural next step if scope grows is splitting it into narrower services; the request path is synchronous throughout, matching the underlying libraries (PyMuPDF, SentenceTransformer, FAISS, SQLite are all sync); there's no auth layer, appropriate for local single-user use and a prerequisite for any shared deployment.