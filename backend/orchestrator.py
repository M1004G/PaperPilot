"""Orchestrator Agent: owns per-document sessions and routes work to specialist agents."""
import uuid
from dataclasses import dataclass, field

from backend import ingestion_agent, summary_agent, gap_agent, rag_agent, report_agent
from backend.ingestion_agent import IngestedPaper
from backend.rag_agent import RAGIndex


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
    """Single stateful coordinator for the whole system."""

    def __init__(self):
        self.sessions: dict[str, DocSession] = {}

    # ---------- ingestion ----------
    def ingest_paper(self, pdf_path: str) -> dict:
        paper = ingestion_agent.ingest(pdf_path)
        rag_index = rag_agent.build_index(paper)
        doc_id = str(uuid.uuid4())[:8]
        self.sessions[doc_id] = DocSession(doc_id=doc_id, paper=paper, rag_index=rag_index)
        return {
            "doc_id": doc_id,
            "title": paper.title,
            "num_pages": paper.num_pages,
            "sections": [s.heading for s in paper.sections],
        }

    def _get_session(self, doc_id: str) -> DocSession:
        if doc_id not in self.sessions:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        return self.sessions[doc_id]

    # ---------- summary ----------
    def get_summary(self, doc_id: str) -> dict:
        session = self._get_session(doc_id)
        if session._tldr is None:
            session._tldr = summary_agent.tldr(session.paper)
        if session._section_summaries is None:
            session._section_summaries = summary_agent.section_summaries(session.paper)
        if session._key_findings is None:
            session._key_findings = summary_agent.key_findings(session.paper)
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
        session.chat_history.append({"role": "user", "content": query})
        session.chat_history.append({"role": "assistant", "content": result["answer"]})
        return result


# Single shared instance used by the FastAPI app.
orchestrator = Orchestrator()
