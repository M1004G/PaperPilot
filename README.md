# PaperPilot

Multi-agent research assistant that ingests a research paper PDF and produces a structured summary, gap analysis, retrieval-augmented Q&A, and a consolidated report.


## Features

- **Document ingestion** — PDF parsing with section detection and page-level text tracking
- **Summarization** — concise overview, per-section summaries, and key findings
- **Gap analysis** — author-acknowledged and inferred research gaps, each with a suggested direction
- **Retrieval-augmented chat** — question answering grounded in the paper, with section and page citations
- **Report generation** — consolidated Markdown report, viewable and downloadable

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| LLM | Groq API (default model: `openai/gpt-oss-120b`) |
| PDF processing | PyMuPDF |
| Retrieval | LangChain, ChromaDB, sentence-transformers, cross-encoder reranking |
| Persistence | SQLite |
| Frontend | Streamlit |

## Prerequisites

- Python 3.11+
- A Groq API key ([console.groq.com](https://console.groq.com))

## Installation

```bash
cd PaperPilot
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
| `GROQ_MODEL` | Groq model name | `openai/gpt-oss-120b` |
| `MAX_PDF_SIZE_MB` | Upload size limit | `25` |
| `RERANK_ENABLED` | Enable cross-encoder reranking | `true` |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and ingest a PDF |
| `GET` | `/summary/{doc_id}` | Overview, section summaries, key findings |
| `GET` | `/gaps/{doc_id}` | Research gap analysis |
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
  discovery_agent.py        # paper search + PDF download (standalone, not wired into the API)
  report_agent.py           # Markdown report assembly
  persistence.py            # SQLite session storage
  eval_agent.py              # retrieval/faithfulness evaluation
  orchestrator.py           # agent coordination and session state
  main.py                    # FastAPI application
frontend/
  app.py                    # Streamlit UI
scripts/
  run_eval.py                # evaluation CLI
data/
  sessions.db                # SQLite store (created at runtime)
```

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

Scoring uses retrieval hit-rate, an LLM-as-judge faithfulness check (summaries and chat answers), and a gap-validity check (inferred gaps).

## Error Handling

| Status | Cause |
|---|---|
| 400 | Invalid upload (non-PDF, oversized, corrupt, encrypted) |
| 404 | Unknown document ID |
| 502 | LLM provider failure after retries |
| 500 | Unexpected internal error |
