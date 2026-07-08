"""Orchestrator Agent: owns per-document sessions and routes work to specialist agents."""
import logging
import threading
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
        paper = ingestion_agent.ingest(pdf_path)
        rag_index = rag_agent.build_index(paper)
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
            session._tldr = summary_agent.tldr(session.paper)
            changed = True
        if session._section_summaries is None:
            session._section_summaries = summary_agent.section_summaries(session.paper)
            changed = True
        if session._key_findings is None:
            session._key_findings = summary_agent.key_findings(session.paper)
            changed = True
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
            session._gaps = gap_agent.analyze(session.paper)
            persistence.update_gaps_cache(doc_id, session._gaps)
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
