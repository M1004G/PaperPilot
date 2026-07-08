"""SQLite-backed persistence for document sessions.

Without this, every DocSession lives only in the orchestrator's in-memory dict --
restart the server and every ingested paper, cached summary, gap analysis, and
chat history is gone. This stores everything needed to reconstruct a session on
startup. The FAISS index itself is NOT serialized; chunks (text + metadata) are
stored instead, and the index is rebuilt locally via the embedding model on load
(no Groq API calls involved, so it's cheap and avoids FAISS version/serialization
compatibility issues across restarts or machines).
"""
import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from backend import config
from backend.ingestion_agent import IngestedPaper, Section

logger = logging.getLogger("paperpilot.persistence")

_lock = threading.Lock()
_conn: sqlite3.Connection = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                doc_id TEXT PRIMARY KEY,
                paper_json TEXT NOT NULL,
                chunks_json TEXT NOT NULL,
                chat_history_json TEXT NOT NULL DEFAULT '[]',
                tldr TEXT,
                section_summaries_json TEXT,
                key_findings_json TEXT,
                gaps_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def init_db():
    """Ensure the DB and table exist. Safe to call multiple times."""
    _get_conn()


def save_new_session(doc_id: str, paper: IngestedPaper, chunks: list[dict]):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO sessions (doc_id, paper_json, chunks_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, json.dumps(asdict(paper)), json.dumps(chunks),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    logger.info("persisted_new_session doc_id=%s", doc_id)


def update_chat_history(doc_id: str, chat_history: list[dict]):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE sessions SET chat_history_json = ? WHERE doc_id = ?",
            (json.dumps(chat_history), doc_id),
        )
        conn.commit()


def update_summary_cache(doc_id: str, tldr: str, section_summaries: list, key_findings: list):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE sessions SET tldr = ?, section_summaries_json = ?, key_findings_json = ? "
            "WHERE doc_id = ?",
            (tldr, json.dumps(section_summaries), json.dumps(key_findings), doc_id),
        )
        conn.commit()


def update_gaps_cache(doc_id: str, gaps: dict):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE sessions SET gaps_json = ? WHERE doc_id = ?",
            (json.dumps(gaps), doc_id),
        )
        conn.commit()


def _row_to_paper(row: sqlite3.Row) -> IngestedPaper:
    data = json.loads(row["paper_json"])
    sections = [Section(**s) for s in data.get("sections", [])]
    data["sections"] = sections
    return IngestedPaper(**data)


def load_all_sessions() -> list[dict]:
    """Return a list of raw row dicts for every persisted session, for the
    orchestrator to reconstruct DocSession + RAGIndex objects on startup."""
    with _lock:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM sessions ORDER BY created_at ASC")
        rows = cursor.fetchall()

    sessions = []
    for row in rows:
        try:
            paper = _row_to_paper(row)
            chunks = json.loads(row["chunks_json"])
            sessions.append({
                "doc_id": row["doc_id"],
                "paper": paper,
                "chunks": chunks,
                "chat_history": json.loads(row["chat_history_json"] or "[]"),
                "tldr": row["tldr"],
                "section_summaries": json.loads(row["section_summaries_json"]) if row["section_summaries_json"] else None,
                "key_findings": json.loads(row["key_findings_json"]) if row["key_findings_json"] else None,
                "gaps": json.loads(row["gaps_json"]) if row["gaps_json"] else None,
            })
        except Exception as e:
            # A corrupt/unreadable row shouldn't take down the whole app on startup --
            # skip it and keep loading the rest.
            logger.error("failed_to_load_session doc_id=%s error=%s", row["doc_id"], e)
    return sessions
