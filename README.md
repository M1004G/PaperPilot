# PaperPilot — Autonomous Research Assistant

Multi-agent system that ingests a research paper (PDF) and produces:
- A TL;DR + section-by-section summary
- A structured report (Markdown)
- A research-gap analysis (author-acknowledged vs. critically inferred)
- A RAG-based chatbot grounded in the paper, with section+page citations

See `ARCHITECTURE.md` for the full design.

## Setup

```bash
cd research-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY
# (free, no credit card required — get one at https://console.groq.com)
```

## Run

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
  llm_client.py          # Groq API wrapper: retry/backoff on 429s, token-usage logging
  ingestion_agent.py     # PDF -> text, sections, per-page metadata; validates corrupt/encrypted/empty files
  summary_agent.py       # TL;DR + section summaries (parallelized, capped concurrency)
  gap_agent.py           # research gap analysis (structured JSON output, not string parsing)
  rag_agent.py           # sentence-aware chunking, section+page metadata, cross-encoder reranking, chat
  report_agent.py        # assembles final Markdown report (no LLM call — deterministic)
  persistence.py         # SQLite-backed session storage, survives server restarts
  eval_agent.py          # retrieval hit-rate + LLM-as-judge faithfulness scoring
  orchestrator.py        # coordinates all agents, thread-safe session state
  main.py                # FastAPI app / routes, request logging, structured error responses
frontend/
  app.py                 # Streamlit UI
scripts/
  run_eval.py            # CLI: run the pipeline against a PDF and print a quality report
data/
  sessions.db            # SQLite session store (created automatically, gitignored)
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
- Groq's free tier is rate-limited (not credit-metered) — `llm_client.py` retries on 429s, and section summarization runs on a capped thread pool so it doesn't burst past the requests-per-minute limit.
- The FAISS index itself isn't persisted — on restart, chunks are reloaded from SQLite and re-embedded locally (no Groq calls involved, so this is cheap and avoids FAISS version/serialization issues).
- Heading detection (`ingestion_agent.py`) still relies on a fixed list of known section names (Introduction, Results, etc.) matched via regex. Papers with unconventional headings ("Proposed Framework", "Case Study") won't be split correctly — this is a known, deliberately out-of-scope limitation; a real fix needs font/layout-based heading detection rather than a keyword list.
