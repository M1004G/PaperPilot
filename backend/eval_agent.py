"""Evaluation Agent: practical quality checks for the pipeline's actual output.

This is NOT a rigorous benchmark (no labeled ground-truth dataset) -- it's a tool
for catching regressions and getting a rough read on quality as you test the app
across real papers. Two kinds of checks:

1. Retrieval hit-rate: for a question with known expected keywords, does the
   retrieved context actually contain them? Cheap, deterministic, no LLM call.
2. LLM-as-judge faithfulness: is a generated summary/gap/answer actually
   supported by the source text it's supposed to be based on? Uses the same
   Groq model to score 1-5 with a short reason. Not perfectly reliable (the
   judge can be wrong), but useful as a directional signal, especially for
   spotting hallucination.
"""
import json
import logging

from backend import llm_client
from backend.ingestion_agent import IngestedPaper
from backend.rag_agent import RAGIndex

logger = logging.getLogger("paperpilot.eval")

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You compare a SOURCE text against a GENERATED "
    "text and judge whether every claim in the generated text is actually supported "
    "by the source. You are not judging writing quality, only factual faithfulness. "
    "Penalize any claim not directly supported by the source, even if it sounds plausible."
)


def judge_faithfulness(source_text: str, generated_text: str, label: str = "output") -> dict:
    """Score how well `generated_text` is supported by `source_text`, 1-5.
    5 = fully supported, no invented claims. 1 = largely unsupported/hallucinated."""
    prompt = f"""SOURCE TEXT:
{source_text[:5000]}

GENERATED {label.upper()}:
{generated_text}

Score how faithfully the generated {label} reflects only what's in the source text.
Respond with ONLY a JSON object of this exact shape, no other text:
{{"score": <integer 1-5>, "reasoning": "<one sentence explaining the score>", "unsupported_claims": ["<claim not found in source>", ...]}}"""
    raw = llm_client.complete_json(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=300)
    try:
        parsed = json.loads(raw)
        return {
            "label": label,
            "score": int(parsed.get("score", 0)),
            "reasoning": str(parsed.get("reasoning", "")),
            "unsupported_claims": [c for c in parsed.get("unsupported_claims", []) if isinstance(c, str)],
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("judge_faithfulness_parse_failed raw=%r", raw[:200])
        return {"label": label, "score": None, "reasoning": "Judge response could not be parsed.", "unsupported_claims": []}


def evaluate_summary(paper: IngestedPaper, summary: dict) -> dict:
    """Judge the TL;DR against the abstract+full text, and each section summary
    against its own section text."""
    results = {"tldr": None, "section_summaries": []}

    reference = f"{paper.abstract}\n\n{paper.full_text[:8000]}"
    results["tldr"] = judge_faithfulness(reference, summary.get("tldr", ""), label="tldr")

    section_lookup = {s.heading: s.text for s in paper.sections}
    for item in summary.get("section_summaries", []):
        heading = item.get("heading", "")
        source = section_lookup.get(heading, "")
        if not source:
            continue
        results["section_summaries"].append(
            judge_faithfulness(source, item.get("summary", ""), label=f"section summary ({heading})")
        )
    return results


def evaluate_gaps(paper: IngestedPaper, gaps: dict) -> dict:
    """Judge inferred gaps against the full paper text -- a gap should be something
    the paper's text plausibly supports as *missing*, not an invented critique."""
    reference = paper.full_text[:8000]
    results = []
    for item in gaps.get("inferred", []):
        gap_text = f"Gap: {item.get('gap', '')} Direction: {item.get('direction', '')}"
        results.append(judge_faithfulness(reference, gap_text, label="inferred gap"))
    return {"inferred_gaps": results}


def retrieval_hit_rate(rag_index: RAGIndex, test_cases: list[dict], k: int = 5) -> dict:
    """test_cases: [{"question": str, "expected_keywords": [str, ...]}, ...]
    A question 'hits' if at least one expected keyword appears (case-insensitive
    substring match) in the retrieved chunks' text."""
    per_question = []
    hits = 0
    for case in test_cases:
        question = case["question"]
        expected = [kw.lower() for kw in case.get("expected_keywords", [])]
        retrieved = rag_index.retrieve(question, k=k)
        combined_text = " ".join(r["text"].lower() for r in retrieved)
        matched = [kw for kw in expected if kw in combined_text]
        hit = len(matched) > 0 if expected else None
        if hit:
            hits += 1
        per_question.append({
            "question": question,
            "expected_keywords": expected,
            "matched_keywords": matched,
            "hit": hit,
            "top_sources": [f"{r['heading']} p.{r['page_start']}" for r in retrieved[:3]],
        })
    scored = [q for q in per_question if q["hit"] is not None]
    hit_rate = hits / len(scored) if scored else None
    return {"per_question": per_question, "hit_rate": hit_rate}


def evaluate_chat_answers(rag_index: RAGIndex, qa_results: list[dict]) -> list[dict]:
    """qa_results: [{"question": str, "answer": str, "sources": [...]}]
    Judges whether each chat answer is faithful to the chunks it cited."""
    evaluations = []
    for qa in qa_results:
        source_text = "\n\n".join(s.get("preview", "") for s in qa.get("sources", []))
        evaluations.append({
            "question": qa["question"],
            **judge_faithfulness(source_text, qa["answer"], label="chat answer"),
        })
    return evaluations


def summarize_scores(*score_lists: list[dict]) -> dict:
    """Aggregate average faithfulness score across any number of judged-item lists."""
    all_scores = [item["score"] for lst in score_lists for item in lst if item.get("score") is not None]
    if not all_scores:
        return {"count": 0, "average": None}
    return {"count": len(all_scores), "average": round(sum(all_scores) / len(all_scores), 2)}
