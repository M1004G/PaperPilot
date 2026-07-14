"""Shared pytest fixtures.

Tests never hit real network services (Groq, Hugging Face) -- the embedder,
reranker, and llm_client functions are all faked/mocked, so the suite runs
fast and deterministically offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic fake embedder: same text always -> same vector, so
    similarity comparisons in tests are reproducible without downloading
    a real sentence-transformers model."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2**32))
            v = rng.rand(16).astype("float32")
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs, dtype="float32")


class FakeReranker:
    """Deterministic fake cross-encoder: scores by word overlap between
    query and chunk text, instead of downloading a real model."""

    def predict(self, pairs):
        return [
            float(len(set(q.lower().split()) & set(t.lower().split())))
            for q, t in pairs
        ]


@pytest.fixture(autouse=True)
def fake_embedding_models(monkeypatch):
    """Applied to every test automatically: no test should need real
    sentence-transformers/cross-encoder downloads."""
    from backend import rag_agent
    monkeypatch.setattr(rag_agent, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(rag_agent, "_get_reranker", lambda: FakeReranker())


@pytest.fixture
def fake_llm(monkeypatch):
    """Patches llm_client's public functions with simple, deterministic fakes.
    Returns the module so tests can further monkeypatch specific functions."""
    from backend import llm_client
    monkeypatch.setattr(llm_client, "complete", lambda system, user_message, max_tokens=None, temperature=0.3: "fake completion")
    monkeypatch.setattr(llm_client, "complete_json", lambda system, user_message, max_tokens=None, temperature=0.2: '{"gaps": []}')
    monkeypatch.setattr(llm_client, "chat_complete", lambda system, messages, max_tokens=None, temperature=0.3: "fake chat answer")
    return llm_client


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Points config.DATA_DIR/DB_PATH at a fresh temp directory for the duration
    of a test, so persistence tests never touch the real data/sessions.db."""
    from backend import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sessions.db")
    # persistence.py caches its connection at module level -- reset it so a
    # fresh test gets a fresh connection pointed at the new DB_PATH.
    from backend import persistence
    monkeypatch.setattr(persistence, "_conn", None)
    return tmp_path


def make_test_pdf(tmp_path, sections: dict[str, str]) -> str:
    """Build a real (small) multi-page PDF via PyMuPDF for ingestion tests.
    `sections` maps heading -> body text; each becomes its own page, with the
    heading as the first line so ingestion_agent's regex can detect it."""
    import fitz
    doc = fitz.open()
    for heading, body in sections.items():
        page = doc.new_page()
        page.insert_text((72, 72), f"{heading}\n{body}")
    path = str(tmp_path / "test_paper.pdf")
    doc.save(path)
    doc.close()
    return path
