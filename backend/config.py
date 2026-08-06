"""Central configuration for the whole backend."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Storage ----------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "sessions.db"

# ---------- Upload validation ----------
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "25"))

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------- LLM provider (Groq) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq free tier is rate-limited (~30 RPM / 6000 TPM depending on model), not just
# a credits budget, so calls need retry/backoff and bounded concurrency baked in.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_BASE_SECONDS = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.0"))
LLM_MAX_CONCURRENT_CALLS = int(os.getenv("LLM_MAX_CONCURRENT_CALLS", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

# RAG tuning
CHUNK_SIZE = 800  # target characters per chunk (sentence-aware, so this is a soft target)
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Retrieval reranking: pull a wider candidate pool from FAISS, then rerank with a
# cross-encoder and keep the best TOP_K. Set RERANK_ENABLED=false to skip this step.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_POOL = 20

# Chat history sent to the LLM is capped by character budget (not just turn count),
# since a handful of long turns can still blow past a reasonable prompt size.
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "6000"))

# LLM call defaults
# Groq's free-tier TPM cap (as low as 6000 tokens/min on some models) is the real
# ceiling here, well below what a full paper would take -- truncation stays necessary.
DEFAULT_MAX_TOKENS = 1500

# ---------- Reproducibility Agent ----------
# Optional: raises the GitHub API rate limit from 60/hr (unauthenticated) to
# 5000/hr. A fine-grained PAT with no scopes (public-repo read only) is enough.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

REPRO_HTTP_TIMEOUT_SECONDS = float(os.getenv("REPRO_HTTP_TIMEOUT_SECONDS", "15"))
# Per-file byte cap when fetching README/manifest contents -- these are read
# for static checks, not archived, so there's no need for full files.
REPRO_MAX_FILE_BYTES = int(os.getenv("REPRO_MAX_FILE_BYTES", "200_000".replace("_", "")))
# Cap on how many file paths we keep from the repo tree API call (very large
# monorepos would otherwise bloat memory and any downstream LLM prompt).
REPRO_MAX_TREE_ENTRIES = int(os.getenv("REPRO_MAX_TREE_ENTRIES", "3000"))
# Separate, much smaller cap on how many of those paths actually get pasted
# into the claim-verification LLM prompt.
REPRO_MAX_TREE_ENTRIES_IN_PROMPT = int(os.getenv("REPRO_MAX_TREE_ENTRIES_IN_PROMPT", "150"))
# The claim-verification LLM call is judgment, not a mechanical fact -- toggle
# off to get static-checks-only (deterministic, no LLM cost/latency).
REPRO_LLM_CLAIMS_ENABLED = os.getenv("REPRO_LLM_CLAIMS_ENABLED", "true").lower() == "true"
