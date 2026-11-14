"""Retrieval metrics over one or more papers. Ingestion + retrieval only: no LLM calls, no Groq tokens.

Usage (pairs of PDF and questions file):
    python -m scripts.hit_rate paper1.pdf q1.json paper2.pdf q2.json ...

Each questions file has the same format as for run_eval:
    [{"question": "What dataset was used?", "expected_keywords": ["MNIST", "CIFAR"]}, ...]

A question is a hit at k if at least one expected keyword appears in the top-k retrieved chunks.
Reported: hit@1/3/5, MRR (mean reciprocal rank of the first chunk containing a keyword),
keyword coverage (share of expected keywords found in the top 5), ingestion time and retrieval latency.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import eval_agent
from backend.orchestrator import orchestrator

KS = (1, 3, 5)


def main():
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit("usage: python -m scripts.hit_rate paper1.pdf questions1.json [paper2.pdf questions2.json ...]")

    all_questions = []   # scored questions across every paper
    ingest_times = []

    for pdf_path, questions_path in zip(args[::2], args[1::2]):
        with open(questions_path) as f:
            test_cases = json.load(f)
        start = time.monotonic()
        info = orchestrator.ingest_paper(pdf_path)
        ingest_times.append(time.monotonic() - start)
        report = eval_agent.retrieval_hit_rate(orchestrator.sessions[info["doc_id"]].rag_index, test_cases)

        scored = [q for q in report["per_question"] if q["hit"] is not None]
        all_questions += scored

        print(f"\n=== {Path(pdf_path).name} ({len(info['sections'])} sections, ingested in {ingest_times[-1]:.1f}s) ===")
        for q in report["per_question"]:
            status = {True: "HIT", False: "MISS", None: "N/A"}[q["hit"]]
            rank = f"rank {q['rank']}" if q["rank"] else "no rank"
            print(f"  [{status}] ({rank}) {q['question']} -- matched: {q['matched_keywords']} -- top: {q['top_sources']}")
        hits = sum(q["hit"] for q in scored)
        print(f"  hit rate: {hits}/{len(scored)}")

    if not all_questions:
        sys.exit("\nNo scored questions.")

    n = len(all_questions)
    print(f"\n=== OVERALL ({n} questions, {len(ingest_times)} papers) ===")
    for k in KS:
        hits = sum(q["rank"] is not None and q["rank"] <= k for q in all_questions)
        print(f"  hit@{k}: {hits}/{n} = {hits / n:.0%}")
    mrr = sum(1 / q["rank"] for q in all_questions if q["rank"]) / n
    coverage = sum(len(q["matched_keywords"]) / len(q["expected_keywords"]) for q in all_questions) / n
    latencies = sorted(q["latency_s"] for q in all_questions)
    print(f"  MRR: {mrr:.2f}")
    print(f"  keyword coverage (top 5): {coverage:.0%}")
    print(f"  retrieval latency: median {statistics.median(latencies) * 1000:.0f} ms, max {latencies[-1] * 1000:.0f} ms")
    print(f"  ingestion (parse + chunk + embed): mean {statistics.mean(ingest_times):.1f}s per paper")


if __name__ == "__main__":
    main()
