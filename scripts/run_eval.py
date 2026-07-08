"""CLI to run the pipeline against a single PDF and print a quality report.

Usage:
    python -m scripts.run_eval path/to/paper.pdf
    python -m scripts.run_eval path/to/paper.pdf --questions path/to/questions.json

questions.json (optional) format:
[
  {"question": "What dataset was used?", "expected_keywords": ["dataset", "collected"]},
  {"question": "What is the main limitation?", "expected_keywords": ["limitation"]}
]

Without --questions, only faithfulness scoring runs (on the summary and gaps
that get generated); retrieval hit-rate is skipped since it needs known-answer
questions to check against.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import Orchestrator
from backend import eval_agent


def _print_header(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run PaperPilot end-to-end and report quality.")
    parser.add_argument("pdf_path", help="Path to a research paper PDF")
    parser.add_argument("--questions", help="Optional JSON file of {question, expected_keywords} test cases")
    parser.add_argument("--report-out", help="Optional path to write the full JSON report")
    args = parser.parse_args()

    test_cases = []
    if args.questions:
        with open(args.questions) as f:
            test_cases = json.load(f)

    orch = Orchestrator()

    _print_header("INGESTING")
    ingest_result = orch.ingest_paper(args.pdf_path)
    doc_id = ingest_result["doc_id"]
    print(f"doc_id={doc_id} title={ingest_result['title']!r} pages={ingest_result['num_pages']}")
    print(f"sections detected: {ingest_result['sections']}")

    _print_header("GENERATING SUMMARY + GAPS")
    summary = orch.get_summary(doc_id)
    gaps = orch.get_gaps(doc_id)
    print(f"TL;DR ({len(summary['tldr'])} chars): {summary['tldr'][:150]}...")
    print(f"Section summaries: {len(summary['section_summaries'])}")
    print(f"Author-acknowledged gaps: {len(gaps['author_acknowledged'])}")
    print(f"Inferred gaps: {len(gaps['inferred'])}")

    session = orch.sessions[doc_id]

    _print_header("FAITHFULNESS EVAL (summary)")
    summary_eval = eval_agent.evaluate_summary(session.paper, summary)
    print(f"TL;DR faithfulness: {summary_eval['tldr']}")
    for s in summary_eval["section_summaries"]:
        flag = " ⚠️" if s["score"] and s["score"] < 4 else ""
        print(f"  [{s['label']}] score={s['score']}{flag} — {s['reasoning']}")

    _print_header("FAITHFULNESS EVAL (gaps)")
    gaps_eval = eval_agent.evaluate_gaps(session.paper, gaps)
    for g in gaps_eval["inferred_gaps"]:
        flag = " ⚠️" if g["score"] and g["score"] < 4 else ""
        print(f"  score={g['score']}{flag} — {g['reasoning']}")

    qa_results = []
    retrieval_report = None
    if test_cases:
        _print_header("RETRIEVAL HIT-RATE")
        retrieval_report = eval_agent.retrieval_hit_rate(session.rag_index, test_cases)
        for q in retrieval_report["per_question"]:
            status = "HIT" if q["hit"] else "MISS"
            print(f"  [{status}] {q['question']} — matched: {q['matched_keywords']} — top: {q['top_sources']}")
        hr = retrieval_report["hit_rate"]
        print(f"\nOverall hit rate: {hr:.0%}" if hr is not None else "\nOverall hit rate: n/a")

        _print_header("CHAT ANSWER FAITHFULNESS")
        for case in test_cases:
            result = orch.chat(doc_id, case["question"])
            qa_results.append({"question": case["question"], **result})
        chat_eval = eval_agent.evaluate_chat_answers(session.rag_index, qa_results)
        for c in chat_eval:
            flag = " ⚠️" if c["score"] and c["score"] < 4 else ""
            print(f"  score={c['score']}{flag} — {c['question']} — {c['reasoning']}")
    else:
        chat_eval = []
        print("\n(no --questions provided, skipping retrieval hit-rate + chat eval)")

    _print_header("SUMMARY")
    overall = eval_agent.summarize_scores(
        [summary_eval["tldr"]], summary_eval["section_summaries"], gaps_eval["inferred_gaps"], chat_eval
    )
    print(f"Average faithfulness score across {overall['count']} judged items: {overall['average']}")
    if retrieval_report and retrieval_report["hit_rate"] is not None:
        print(f"Retrieval hit rate: {retrieval_report['hit_rate']:.0%}")

    if args.report_out:
        full_report = {
            "doc_id": doc_id,
            "title": ingest_result["title"],
            "summary_eval": summary_eval,
            "gaps_eval": gaps_eval,
            "retrieval_report": retrieval_report,
            "chat_eval": chat_eval,
            "overall": overall,
        }
        with open(args.report_out, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nFull report written to {args.report_out}")


if __name__ == "__main__":
    main()
