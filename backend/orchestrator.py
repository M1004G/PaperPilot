"""Orchestrator Agent: owns per-document sessions and routes work to specialist agents."""
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from backend import ingestion_agent, summary_agent, gap_agent, rag_agent, report_agent, persistence
from backend.ingestion_agent import IngestedPaper
from backend.rag_agent import RAGIndex

logger = logging.getLogger("paperpilot.orchestrator")


@dataclass
class DocSession:
    doc_id: str
    paper: IngestedPaper
    rag_index: RAGIndex
    chat_history: list[dict] = field(default_factory=list)
    # caches, populated lazily on first request
    _tldr: str = None
    _section_summaries: list = None
    _key_findings: list = None
    _gaps: dict = None


class Orchestrator:
    """Single stateful coordinator for the whole system.

    self.sessions is accessed from multiple FastAPI request-handling threads
    (uvicorn's default sync-endpoint threadpool), so all reads/writes go through
    self._lock. Sessions are persisted to SQLite (backend/persistence.py) and
    reloaded on startup, so a server restart doesn't lose ingested papers.
    """

    def __init__(self):
        self.sessions: dict[str, DocSession] = {}
        self._lock = threading.Lock()
        persistence.init_db()
        self._load_persisted_sessions()

    def _load_persisted_sessions(self):
        for row in persistence.load_all_sessions():
            try:
                # Rebuilding the FAISS index locally re-embeds chunk text via the
                # embedding model -- no Groq API calls, so this is cheap and avoids
                # needing to serialize/deserialize FAISS indexes across restarts.
                rag_index = rag_agent.RAGIndex(row["chunks"])
                session = DocSession(
                    doc_id=row["doc_id"],
                    paper=row["paper"],
                    rag_index=rag_index,
                    chat_history=row["chat_history"],
                    _tldr=row["tldr"],
                    _section_summaries=row["section_summaries"],
                    _key_findings=row["key_findings"],
                    _gaps=row["gaps"],
                )
                self.sessions[row["doc_id"]] = session
            except Exception as e:
                logger.error("failed_to_rehydrate_session doc_id=%s error=%s", row["doc_id"], e)
        if self.sessions:
            logger.info("rehydrated_sessions count=%d", len(self.sessions))

    # ---------- ingestion ----------
    def ingest_paper(self, pdf_path: str) -> dict:
        t0 = time.monotonic()
        paper = ingestion_agent.ingest(pdf_path)
        t1 = time.monotonic()
        rag_index = rag_agent.build_index(paper)
        t2 = time.monotonic()
        logger.info(
            "ingest_timing parse=%.2fs index_build=%.2fs total=%.2fs pages=%d sections=%d chunks=%d",
            t1 - t0, t2 - t1, t2 - t0, paper.num_pages, len(paper.sections), len(rag_index.chunks),
        )
        doc_id = str(uuid.uuid4())[:8]
        session = DocSession(doc_id=doc_id, paper=paper, rag_index=rag_index)
        with self._lock:
            self.sessions[doc_id] = session
        persistence.save_new_session(doc_id, paper, rag_index.chunks)
        return {
            "doc_id": doc_id,
            "title": paper.title,
            "num_pages": paper.num_pages,
            "sections": [s.heading for s in paper.sections],
        }

    def _get_session(self, doc_id: str) -> DocSession:
        with self._lock:
            if doc_id not in self.sessions:
                raise KeyError(f"Unknown doc_id: {doc_id}")
            return self.sessions[doc_id]

    # ---------- summary ----------
    def get_summary(self, doc_id: str) -> dict:
        session = self._get_session(doc_id)
        changed = False
        if session._tldr is None:
            logger.info("cache_miss doc_id=%s field=tldr", doc_id)
            t0 = time.monotonic()
            session._tldr = summary_agent.tldr(session.paper)
            logger.info("stage_timing doc_id=%s stage=tldr elapsed=%.2fs", doc_id, time.monotonic() - t0)
            changed = True
        else:
            logger.info("cache_hit doc_id=%s field=tldr", doc_id)
        if session._section_summaries is None:
            logger.info("cache_miss doc_id=%s field=section_summaries", doc_id)
            t0 = time.monotonic()
            session._section_summaries = summary_agent.section_summaries(session.paper)
            logger.info("stage_timing doc_id=%s stage=section_summaries elapsed=%.2fs", doc_id, time.monotonic() - t0)
            changed = True
        else:
            logger.info("cache_hit doc_id=%s field=section_summaries", doc_id)
        if session._key_findings is None:
            logger.info("cache_miss doc_id=%s field=key_findings", doc_id)
            t0 = time.monotonic()
            session._key_findings = summary_agent.key_findings(session.paper)
            logger.info("stage_timing doc_id=%s stage=key_findings elapsed=%.2fs", doc_id, time.monotonic() - t0)
            changed = True
        else:
            logger.info("cache_hit doc_id=%s field=key_findings", doc_id)
        if changed:
            persistence.update_summary_cache(
                doc_id, session._tldr, session._section_summaries, session._key_findings
            )
        return {
            "tldr": session._tldr,
            "section_summaries": session._section_summaries,
            "key_findings": session._key_findings,
        }

    # ---------- gaps ----------
    def get_gaps(self, doc_id: str) -> dict:
        session = self._get_session(doc_id)
        if session._gaps is None:
            logger.info("cache_miss doc_id=%s field=gaps", doc_id)
            t0 = time.monotonic()
            session._gaps = gap_agent.analyze(session.paper)
            logger.info("stage_timing doc_id=%s stage=gaps elapsed=%.2fs", doc_id, time.monotonic() - t0)
            persistence.update_gaps_cache(doc_id, session._gaps)
        else:
            logger.info("cache_hit doc_id=%s field=gaps", doc_id)
        return session._gaps

    # ---------- report ----------
    def get_report(self, doc_id: str) -> str:
        session = self._get_session(doc_id)
        summary = self.get_summary(doc_id)
        gaps = self.get_gaps(doc_id)
        return report_agent.build_report(
            paper=session.paper,
            tldr=summary["tldr"],
            section_summaries=summary["section_summaries"],
            key_findings=summary["key_findings"],
            gaps=gaps,
        )

    # ---------- chat / RAG ----------
    def chat(self, doc_id: str, query: str) -> dict:
        session = self._get_session(doc_id)
        result = rag_agent.answer(session.rag_index, query, history=session.chat_history)
        with self._lock:
            session.chat_history.append({"role": "user", "content": query})
            session.chat_history.append({"role": "assistant", "content": result["answer"]})
            history_snapshot = list(session.chat_history)
        persistence.update_chat_history(doc_id, history_snapshot)
        return result


# Single shared instance used by the FastAPI app.
orchestrator = Orchestrator()
