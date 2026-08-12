"""RAG Agent: chunk + embed + index + retrieve + grounded chat answer.

Retrieval infrastructure is built on LangChain: RecursiveCharacterTextSplitter
for chunking, langchain_chroma's Chroma vectorstore + HuggingFace embeddings
for indexing, and HuggingFaceCrossEncoder for reranking. The final
answer-generation call deliberately still goes through our own llm_client
(not langchain_groq's chat model) -- llm_client already has tested
retry/backoff and rate-limit handling for Groq's free tier.

Each document gets its own persistent Chroma collection (data/chroma/,
collection name derived from doc_id), so a server restart reuses the
already-embedded vectors instead of re-embedding chunk text from scratch --
unlike the previous FAISS-based version, which held vectors in memory only
and rebuilt them from persisted chunk text on every load. Each chunk carries
section heading + page metadata, so the LLM can ground answers in "where
this came from" rather than an anonymous blob of text.
"""
import logging
import time
import uuid

from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import llm_client, config
from backend.ingestion_agent import IngestedPaper, Section

logger = logging.getLogger("paperpilot.rag")

_embeddings = None
_reranker = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_reranker() -> HuggingFaceCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL_NAME)
    return _reranker


def chunk_section(section: Section, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Split one section into chunks via LangChain's RecursiveCharacterTextSplitter,
    one page-segment at a time so every resulting chunk maps to exactly one
    page (page_start == page_end always)."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    segments = section.page_segments or [(section.page_start, section.text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for page_num, seg_text in segments:
        if not seg_text.strip():
            continue
        doc = Document(page_content=seg_text.strip())
        for split_doc in splitter.split_documents([doc]):
            text = split_doc.page_content.strip()
            if text:
                chunks.append({"text": text, "heading": section.heading, "page_start": page_num, "page_end": page_num})
    return chunks


def chunk_paper(paper: IngestedPaper, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Chunk every section of a paper, falling back to the raw full text if
    section splitting failed to find any structure (e.g. an unusual layout)."""
    if paper.sections:
        chunks: list[dict] = []
        for section in paper.sections:
            if section.heading.lower() in ("references",):
                continue
            chunks.extend(chunk_section(section, chunk_size, overlap))
        if chunks:
            return chunks

    fallback_section = Section(heading="Full Text", text=paper.full_text, page_start=1, page_end=paper.num_pages or 1)
    return chunk_section(fallback_section, chunk_size, overlap)


def _chroma_kwargs(doc_id: str | None) -> dict:
    """Persistent, doc_id-keyed collection for real sessions; an ephemeral,
    unpersisted, uniquely-named collection when doc_id is None (tests, or any
    ad-hoc index that shouldn't leave data behind on disk)."""
    if doc_id is None:
        return {"collection_name": f"ephemeral_{uuid.uuid4().hex[:12]}"}
    return {"collection_name": f"doc_{doc_id}", "persist_directory": str(config.CHROMA_PERSIST_DIR)}


class RAGIndex:
    """Holds the Chroma vectorstore + chunk metadata for a single document.

    `doc_id`, when provided, makes the underlying Chroma collection durable
    across restarts (see module docstring). If a collection already exists
    for that doc_id with a chunk count matching `chunks`, it's reused as-is
    -- no re-embedding. A mismatched count (e.g. chunking logic changed
    between versions, or the collection is otherwise stale) triggers a full
    reset and re-embed, rather than trusting possibly-inconsistent data.
    """

    def __init__(self, chunks: list[dict], doc_id: str = None):
        self.chunks = chunks
        self.doc_id = doc_id
        t0 = time.monotonic()

        self.vectorstore = Chroma(
            embedding_function=_get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
            **_chroma_kwargs(doc_id),
        )
        existing_count = self.vectorstore._collection.count()

        if existing_count == len(chunks) and existing_count > 0:
            reused = True
        else:
            if existing_count:
                self.vectorstore.reset_collection()
            docs = [
                Document(
                    page_content=c["text"],
                    metadata={"chunk_id": i, "heading": c["heading"], "page_start": c["page_start"], "page_end": c["page_end"]},
                )
                for i, c in enumerate(chunks)
            ]
            if docs:
                self.vectorstore.add_documents(docs, ids=[str(i) for i in range(len(docs))])
            reused = False

        embed_elapsed = time.monotonic() - t0
        self._abstract_chunk_idx = next((i for i, c in enumerate(chunks) if c["heading"].lower() == "abstract"), None)
        logger.info(
            "rag_index_built doc_id=%s chunks=%d reused_existing=%s elapsed=%.2fs",
            doc_id, len(chunks), reused, embed_elapsed,
        )

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        k = k or config.TOP_K
        pool_size = min(config.RERANK_CANDIDATE_POOL if config.RERANK_ENABLED else k, len(self.chunks)) or 1

        t0 = time.monotonic()
        scored_docs = self.vectorstore.similarity_search_with_score(query, k=pool_size)
        search_elapsed = time.monotonic() - t0

        candidates = [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "text": doc.page_content,
                "heading": doc.metadata["heading"],
                "page_start": doc.metadata["page_start"],
                "page_end": doc.metadata["page_end"],
                # Chroma returns a distance (lower = more similar); invert to a
                # similarity-style score so "higher = more relevant" holds for
                # every score this module produces, matching what callers
                # (frontend, sources list) already expect.
                "embedding_score": -float(score),
            }
            for doc, score in scored_docs
        ]

        rerank_elapsed = 0.0
        if config.RERANK_ENABLED and len(candidates) > 1:
            reranker = _get_reranker()
            pairs = [(query, c["text"]) for c in candidates]
            t1 = time.monotonic()
            rerank_scores = reranker.score(pairs)
            rerank_elapsed = time.monotonic() - t1
            for c, rs in zip(candidates, rerank_scores):
                c["score"] = float(rs)
            candidates.sort(key=lambda c: c["score"], reverse=True)
        else:
            for c in candidates:
                c["score"] = c["embedding_score"]

        logger.info(
            "retrieve_timing search=%.3fs rerank=%.3fs candidates=%d",
            search_elapsed, rerank_elapsed, len(candidates),
        )
        results = candidates[:k]

        if self._abstract_chunk_idx is not None and not any(c["chunk_id"] == self._abstract_chunk_idx for c in results):
            abstract_chunk = self.chunks[self._abstract_chunk_idx]
            pinned = {
                "chunk_id": self._abstract_chunk_idx,
                "text": abstract_chunk["text"],
                "heading": abstract_chunk["heading"],
                "page_start": abstract_chunk["page_start"],
                "page_end": abstract_chunk["page_end"],
                "embedding_score": 0.0,
                "score": 0.0,
            }
            results = results[:-1] + [pinned] if results else [pinned]

        return results


def build_index(paper: IngestedPaper, doc_id: str = None) -> RAGIndex:
    chunks = chunk_paper(paper)
    return RAGIndex(chunks, doc_id)


SYSTEM_PROMPT = (
    "You are a research-paper Q&A assistant. Answer ONLY using the provided context chunks "
    "from the paper. If the context does not contain the answer, say so plainly instead of "
    "guessing. Each chunk is tagged with its section heading and page number -- cite them "
    "inline like [Results, p.4] when you use a fact from that chunk."
)


def _format_chunk(r: dict) -> str:
    page_label = f"p.{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp.{r['page_start']}-{r['page_end']}"
    return f"[{r['heading']}, {page_label}]\n{r['text']}"


def _bounded_history(history: list[dict]) -> list[dict]:
    kept: list[dict] = []
    used = 0
    for turn in reversed(history):
        length = len(turn.get("content", ""))
        if kept and used + length > config.MAX_HISTORY_CHARS:
            break
        kept.insert(0, turn)
        used += length
    return kept


def answer(index: RAGIndex, query: str, history: list[dict] = None, k: int = None) -> dict:
    history = history or []
    retrieved = index.retrieve(query, k=k)
    context_block = "\n\n".join(_format_chunk(r) for r in retrieved)

    convo = [{"role": t["role"], "content": t["content"]} for t in _bounded_history(history)]

    user_message = f"""Context from the paper:
{context_block}

Question: {query}

Answer the question using only the context above, citing [section, page] where relevant."""
    convo.append({"role": "user", "content": user_message})

    reply = llm_client.chat_complete(SYSTEM_PROMPT, convo, max_tokens=700)
    return {
        "answer": reply,
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "heading": r["heading"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "score": round(r["score"], 3),
                "preview": r["text"][:200],
            }
            for r in retrieved
        ],
    }