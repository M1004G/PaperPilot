"""Orchestrator Agent: owns per-document sessions and routes work to specialist agents."""
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from backend import ingestion_agent, summary_agent, gap_agent, rag_agent, report_agent, repro_check_agent, codegen_agent, repo_fetch, persistence
from backend.ingestion_agent import IngestedPaper
from backend.rag_agent import RAGIndex
from backend.repo_fetch import RepoFetchError

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
    _repro: dict = None
    _repro_url_used: str = None  # the repo_url the cached _repro result was computed for


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
                # Chroma collections are persisted to disk keyed by doc_id, so
                # this reuses already-embedded vectors when the chunk count
                # matches rather than re-embedding from scratch -- see
                # RAGIndex's docstring in rag_agent.py.
                rag_index = rag_agent.RAGIndex(row["chunks"], row["doc_id"])
                session = DocSession(
                    doc_id=row["doc_id"],
                    paper=row["paper"],
                    rag_index=rag_index,
                    chat_history=row["chat_history"],
                    _tldr=row["tldr"],
                    _section_summaries=row["section_summaries"],
                    _key_findings=row["key_findings"],
                    _gaps=row["gaps"],
                    _repro=row["repro"],
                    _repro_url_used=row["repro_url_used"],
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
        doc_id = str(uuid.uuid4())[:8]
        rag_index = rag_agent.build_index(paper, doc_id)
        t2 = time.monotonic()
        logger.info(
            "ingest_timing parse=%.2fs index_build=%.2fs total=%.2fs pages=%d sections=%d chunks=%d",
            t1 - t0, t2 - t1, t2 - t0, paper.num_pages, len(paper.sections), len(rag_index.chunks),
        )
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

    # ---------- reproducibility ----------
    def get_reproducibility(self, doc_id: str, repo_url: str | None = None) -> dict:
        """Two sources feed the same evaluation, per-doc cached (cache key
        includes repo_url so overriding it forces a fresh run):
        - a real repo is linked/supplied -> fetch it and evaluate it
        - no repo -> generate an implementation attempt, then evaluate THAT
          with the exact same checks, so "trustworthy" means the same thing
          either way instead of two disconnected notions of reproducibility.
        """
        session = self._get_session(doc_id)
        cache_valid = session._repro is not None and session._repro_url_used == repo_url
        if not cache_valid:
            logger.info("cache_miss doc_id=%s field=repro repo_url=%s", doc_id, repo_url)
            t0 = time.monotonic()
            session._repro = self._run_reproducibility(session.paper, repo_url)
            session._repro_url_used = repo_url
            logger.info("stage_timing doc_id=%s stage=repro elapsed=%.2fs", doc_id, time.monotonic() - t0)
            persistence.update_repro_cache(doc_id, session._repro, repo_url)
        else:
            logger.info("cache_hit doc_id=%s field=repro", doc_id)
        return session._repro

    def get_cached_reproducibility(self, doc_id: str) -> dict | None:
        """Whatever reproducibility result (if any) is already cached for this
        doc, regardless of which repo_url produced it. Used by the download
        endpoint so it serves exactly what the person already saw in the UI --
        it must NOT trigger a fresh run, since re-running with a default
        repo_url=None could silently switch modes (e.g. from a specific repo
        check back to code generation) and hand back files that don't match
        what was actually displayed."""
        session = self._get_session(doc_id)
        return session._repro


    def _run_reproducibility(self, paper: IngestedPaper, repo_url: str | None) -> dict:
        url = repo_url or repo_fetch.extract_code_url(paper.full_text)

        if url:
            try:
                owner, repo = repo_fetch.parse_github_url(url)
                metadata = repo_fetch.fetch_repo_metadata(owner, repo)
                branch = metadata["default_branch"]
                tree = repo_fetch.fetch_file_tree(owner, repo, branch)
            except RepoFetchError as e:
                logger.warning("repro_fetch_failed url=%s error=%s", url, e)
                return {
                    "mode": "repo_check", "repo_url": url, "repo_metadata": None,
                    "checks": [], "score": None, "verdict": None, "claims": [],
                    "paper_info": None, "files": {}, "gap_report": None, "note": str(e),
                }
            get_content = lambda path: repo_fetch.fetch_file_content(owner, repo, branch, path)
            evaluation = repro_check_agent.evaluate(paper, tree, get_content, metadata, profile="repo")
            return {
                "mode": "repo_check", "repo_url": metadata["html_url"], "repo_metadata": metadata,
                "paper_info": None, "files": {}, "gap_report": None, "note": None,
                **evaluation,
            }

        # No repo found or supplied -- generate an implementation and self-check it.
        generated = codegen_agent.analyze(paper)
        files = generated["files"]
        if not files:
            return {
                "mode": "generated", "repo_url": None, "repo_metadata": None,
                "paper_info": generated["paper_info"], "files": {}, "gap_report": generated["gap_report"],
                "checks": [], "score": None, "verdict": None, "claims": [],
                "note": "No repository was found or supplied, and code generation did not produce any files "
                        "(check GROQ_API_KEY / CODEGEN_ENABLED / logs for the underlying error).",
            }
        tree = sorted(files.keys())
        evaluation = repro_check_agent.evaluate(paper, tree, files.get, metadata=None, profile="generated")
        return {
            "mode": "generated", "repo_url": None, "repo_metadata": None,
            "paper_info": generated["paper_info"], "files": files, "gap_report": generated["gap_report"],
            "note": None,
            **evaluation,
        }

    # ---------- report ----------
    def get_report(self, doc_id: str) -> str:
        session = self._get_session(doc_id)
        summary = self.get_summary(doc_id)
        gaps = self.get_gaps(doc_id)
        repro = self.get_reproducibility(doc_id)
        return report_agent.build_report(
            paper=session.paper,
            tldr=summary["tldr"],
            section_summaries=summary["section_summaries"],
            key_findings=summary["key_findings"],
            gaps=gaps,
            repro=repro,
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
