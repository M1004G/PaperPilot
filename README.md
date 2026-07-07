# PaperPilot — Autonomous Research Assistant

Multi-agent system that ingests a research paper (PDF) and produces:
- A TL;DR + section-by-section summary
- A structured report (Markdown)
- A research-gap analysis
- A RAG-based chatbot grounded in the paper

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

## Project layout

```
backend/
  config.py            # env/config loading
  llm_client.py         # thin wrapper around Groq API (retry/backoff on rate limits)
  ingestion_agent.py     # PDF -> text, sections, metadata
  summary_agent.py       # TL;DR + section summaries
  gap_agent.py           # research gap analysis
  rag_agent.py           # chunk/embed/index/retrieve/answer
  report_agent.py        # assembles final Markdown report
  orchestrator.py        # coordinates all agents + session state
  main.py                # FastAPI app / routes
frontend/
  app.py                 # Streamlit UI
```

## Notes / extension points
- Swap FAISS for a persistent vector DB (Chroma, Qdrant) if you need sessions to survive a restart.
- The orchestrator's session store is in-memory (a Python dict) — fine for a single-user demo, swap for Redis/SQLite for multi-user or persistence.
- `config.py` centralizes the model name (`llama-3.3-70b-versatile` on Groq by default) — change it in one place.
- Groq's free tier is rate-limited (not credit-metered) — `llm_client.py` retries on 429s, and section summarization runs on a capped thread pool so it doesn't burst past the requests-per-minute limit.
