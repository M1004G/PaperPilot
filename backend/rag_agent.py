"""RAG Agent: chunk + embed + index + retrieve + grounded chat answer.

Chunking is now sentence-aware (chunks never split mid-sentence) and each chunk
carries section heading + page metadata, so the LLM can ground answers in
"where this came from" rather than an anonymous blob of text. Retrieval also
adds an optional cross-encoder rerank pass over a wider FAISS candidate pool.
"""
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend import llm_client, config
from backend.ingestion_agent import IngestedPaper, Section

_embedder = None
_reranker = None

# Sentence boundary: a period/question/exclamation followed by whitespace and a
# capital letter or digit (good-enough heuristic that also copes with "et al."
# style abbreviations reasonably well since it requires the following token to
# start a new sentence-looking chunk).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL_NAME)
    return _reranker


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Collapse whitespace within paragraphs first so the regex sees clean gaps.
    text = re.sub(r"[ \t]+", " ", text)
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_section(section: Section, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Pack a section's sentences into ~chunk_size chunks without splitting mid-sentence."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    sentences = _split_sentences(section.text)
    if not sentences:
        return []

    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    def _flush():
        if current:
            chunks.append({
                "text": " ".join(current).strip(),
                "heading": section.heading,
                "page_start": section.page_start,
                "page_end": section.page_end,
            })

    for sentence in sentences:
        # A single sentence longer than chunk_size is kept whole rather than cut mid-word;
        # over-long chunks are rare and better than corrupting a sentence.
        if current_len + len(sentence) + 1 > chunk_size and current:
            _flush()
            # Carry the tail of the previous chunk forward for overlap/context continuity.
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence) + 1

    _flush()
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

    # Fallback: treat the whole document as one unlabeled section.
    fallback_section = Section(heading="Full Text", text=paper.full_text, page_start=1, page_end=paper.num_pages or 1)
    return chunk_section(fallback_section, chunk_size, overlap)


class RAGIndex:
    """Holds the FAISS index + chunk metadata for a single document."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        embedder = _get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        k = k or config.TOP_K
        embedder = _get_embedder()
        q_emb = embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        pool_size = min(config.RERANK_CANDIDATE_POOL if config.RERANK_ENABLED else k, len(self.chunks))
        scores, indices = self.index.search(q_emb, pool_size)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            candidates.append({
                "chunk_id": int(idx),
                "text": chunk["text"],
                "heading": chunk["heading"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "embedding_score": float(score),
            })

        if config.RERANK_ENABLED and len(candidates) > 1:
            reranker = _get_reranker()
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = reranker.predict(pairs)
            for c, rs in zip(candidates, rerank_scores):
                c["score"] = float(rs)
            candidates.sort(key=lambda c: c["score"], reverse=True)
        else:
            for c in candidates:
                c["score"] = c["embedding_score"]

        return candidates[:k]


def build_index(paper: IngestedPaper) -> RAGIndex:
    chunks = chunk_paper(paper)
    return RAGIndex(chunks)


SYSTEM_PROMPT = (
    "You are a research-paper Q&A assistant. Answer ONLY using the provided context chunks "
    "from the paper. If the context does not contain the answer, say so plainly instead of "
    "guessing. Each chunk is tagged with its section heading and page number -- cite them "
    "inline like [Results, p.4] when you use a fact from that chunk."
)


def _format_chunk(r: dict) -> str:
    page_label = f"p.{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp.{r['page_start']}-{r['page_end']}"
    return f"[{r['heading']}, {page_label}]\n{r['text']}"


def answer(index: RAGIndex, query: str, history: list[dict] = None, k: int = None) -> dict:
    history = history or []
    retrieved = index.retrieve(query, k=k)
    context_block = "\n\n".join(_format_chunk(r) for r in retrieved)

    convo = []
    for turn in history[-6:]:  # keep last few turns for context
        convo.append({"role": turn["role"], "content": turn["content"]})

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
