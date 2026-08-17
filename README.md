# PaperPilot

Multi-agent research assistant that ingests a research paper PDF and produces a structured summary, gap analysis, retrieval-augmented Q&A, and an automated code reproducibility assessment.


## Features

- **Document ingestion** — PDF parsing with section detection and page-level text tracking
- **Summarization** — concise overview, per-section summaries, and key findings
- **Gap analysis** — author-acknowledged and inferred research gaps, each with a suggested direction
- **Retrieval-augmented chat** — question answering grounded in the paper, with section and page citations
- **Reproducibility Suite** — evaluates a paper's linked code repository, or generates an implementation from the described methodology when no repository exists
- **Report generation** — consolidated Markdown report, viewable and downloadable

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| LLM | Groq (Llama 3.3 70B) |
| PDF processing | PyMuPDF |
| Retrieval | LangChain, ChromaDB, sentence-transformers, cross-encoder reranking |
| Code analysis | ruff, GitHub REST API |
| Persistence | SQLite |
| Frontend | Streamlit |
| Testing | pytest |

## Prerequisites

- Python 3.11+
- A Groq API key ([console.groq.com](https://console.groq.com))
- Docker (optional)

## Installation

### Option A — Docker

```bash
cd research-assistant
cp .env.example .env
# add GROQ_API_KEY to .env

docker compose up
```

Backend: `http://localhost:8000` · Frontend: `http://localhost:8501` · Data persists in `./data`.

### Option B — Local Python

```bash
cd research-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add GROQ_API_KEY to .env
```

## Running

Backend:
```bash
uvicorn backend.main:app --reload --port 8000
```

Frontend:
```bash
streamlit run frontend/app.py
```

Sessions persist across restarts in `data/sessions.db`.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `MAX_PDF_SIZE_MB` | Upload size limit | `25` |
| `GITHUB_TOKEN` | GitHub API token (raises rate limit from 60/hr to 5000/hr) | optional |
| `CODEGEN_ENABLED` | Enable code generation when no repo is linked | `true` |
| `REPRO_LLM_CLAIMS_ENABLED` | Enable LLM claim verification | `true` |
| `SEMANTIC_REVIEW_ENABLED` | Enable LLM semantic review of generated code | `true` |
| `RERANK_ENABLED` | Enable cross-encoder reranking | `true` |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and ingest a PDF |
| `GET` | `/summary/{doc_id}` | Overview, section summaries, key findings |
| `GET` | `/gaps/{doc_id}` | Research gap analysis |
| `GET` | `/reproducibility/{doc_id}` | Reproducibility evaluation (`?repo_url=` optional) |
| `GET` | `/reproducibility/{doc_id}/download` | Download generated implementation as a zip |
| `GET` | `/report/{doc_id}` | Full Markdown report |
| `GET` | `/report/{doc_id}/download` | Download report as a file |
| `POST` | `/chat` | Grounded Q&A with citations |

## Project Structure

```
backend/
  config.py               # environment/configuration
  llm_client.py            # Groq API client with retry/backoff
  logging_utils.py         # request-ID propagation
  ingestion_agent.py        # PDF parsing and section extraction
  summary_agent.py         # summarization
  gap_agent.py              # gap analysis
  rag_agent.py              # chunking, retrieval, reranking, chat
  repro_check_agent.py      # reproducibility evaluation
  codegen_agent.py          # code generation from methodology
  repo_fetch.py             # GitHub repository access
  report_agent.py           # Markdown report assembly
  persistence.py            # SQLite session storage
  eval_agent.py              # retrieval/faithfulness evaluation
  orchestrator.py           # agent coordination and session state
  main.py                    # FastAPI application
frontend/
  app.py                    # Streamlit UI
scripts/
  run_eval.py                # evaluation CLI
tests/                       # pytest suite
data/
  sessions.db                # SQLite store (created at runtime)
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite covers ingestion, chunking/retrieval, persistence, the LLM client, the reproducibility suite, and end-to-end API flows. All external services (Groq, Hugging Face, GitHub) are mocked. Runs automatically in CI on every push and pull request to `main`.

## Evaluation

`scripts/run_eval.py` runs the full pipeline against a PDF and produces a quality report:

```bash
python -m scripts.run_eval path/to/paper.pdf
python -m scripts.run_eval path/to/paper.pdf --questions questions.json --report-out report.json
```

`questions.json` format:
```json
[
  {"question": "What dataset was used?", "expected_keywords": ["dataset", "collected"]}
]
```

Scoring uses retrieval hit-rate and an LLM-as-judge faithfulness check.

## Error Handling

| Status | Cause |
|---|---|
| 400 | Invalid upload (non-PDF, oversized, corrupt, encrypted) |
| 404 | Unknown document ID |
| 502 | LLM provider failure after retries |
| 500 | Unexpected internal error |

## Reproducibility Suite

Evaluates whether a paper's code can be trusted or reused, via `GET /reproducibility/{doc_id}`.

**Repository mode** — used when a repository is linked in the paper or supplied via `repo_url`. Accessed read-only through the GitHub REST API; no code is cloned or executed.

**Generation mode** — used when no repository exists. Methodology is extracted into structured form, a file plan is produced, and each file is generated in dependency order. Generated Python files are syntax-validated before acceptance. Output is downloadable as a zip via `GET /reproducibility/{doc_id}/download`.

Both modes are evaluated using the same scoring system, organized into four categories:

| Category | Description |
|---|---|
| Documentation | README presence, usage instructions |
| Project Hygiene | License, CI configuration, dependency manifest, archive status (repository mode only) |
| Code Quality | Static analysis via ruff (generation mode only) |
| Correctness | Test presence, dependency pinning, LLM semantic review against the paper (generation mode only) |

Informational checks (missing Dockerfile, missing CITATION file) are reported separately and do not affect the score. Claim verification compares the codebase against claims extracted from the paper's Methods section.

No code is executed as part of this evaluation. The score reflects structural completeness and plausibility, not confirmed numerical reproduction of the paper's results.
