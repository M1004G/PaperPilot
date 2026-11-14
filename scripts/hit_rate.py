"""Retrieval hit rate over one or more papers. Ingestion + retrieval only: no LLM calls, no Groq tokens.

Usage (pairs of PDF and questions file):
    python -m scripts.hit_rate paper1.pdf q1.json paper2.pdf q2.json ...

Each questions file has the same format as for run_eval:
    [{"question": "What dataset was used?", "expected_keywords": ["MNIST", "CIFAR"]}, ...]
A question is a hit if at least one expected keyword appears in the top-5 retrieved chunks.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import eval_agent
from backend.orchestrator import orchestrator


def main():
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit("usage: python -m scripts.hit_rate paper1.pdf questions1.json [paper2.pdf questions2.json ...]")

    total_hits = total_scored = 0
    for pdf_path, questions_path in zip(args[::2], args[1::2]):
        with open(questions_path) as f:
            test_cases = json.load(f)
        info = orchestrator.ingest_paper(pdf_path)
        report = eval_agent.retrieval_hit_rate(orchestrator.sessions[info["doc_id"]].rag_index, test_cases)

        scored = [q for q in report["per_question"] if q["hit"] is not None]
        hits = sum(q["hit"] for q in scored)
        total_hits += hits
        total_scored += len(scored)

        print(f"\n=== {Path(pdf_path).name} ({len(info['sections'])} sections) ===")
        for q in report["per_question"]:
            status = {True: "HIT", False: "MISS", None: "N/A"}[q["hit"]]
            print(f"  [{status}] {q['question']} -- matched: {q['matched_keywords']} -- top: {q['top_sources']}")
        print(f"  hit rate: {hits}/{len(scored)}")

    print(f"\nOVERALL: {total_hits}/{total_scored} = {total_hits / total_scored:.0%}" if total_scored else "\nNo scored questions.")


if __name__ == "__main__":
    main()
