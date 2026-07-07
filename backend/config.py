"""Central configuration for the whole backend."""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- LLM provider (Groq) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq free tier is rate-limited (~30 RPM / 6000 TPM depending on model), not just
# a credits budget, so calls need retry/backoff and bounded concurrency baked in.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_BASE_SECONDS = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.0"))
LLM_MAX_CONCURRENT_CALLS = int(os.getenv("LLM_MAX_CONCURRENT_CALLS", "3"))

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

# LLM call defaults
# Groq's free-tier TPM cap (as low as 6000 tokens/min on some models) is the real
# ceiling here, well below what a full paper would take -- truncation stays necessary.
DEFAULT_MAX_TOKENS = 1500
