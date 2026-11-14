"""Gap Analysis Agent: identifies research gaps, split into author-acknowledged vs inferred."""
import json

from backend import llm_client
from backend.ingestion_agent import IngestedPaper, prioritized_excerpt

SYSTEM_PROMPT = (
    "You are a critical, senior research reviewer. You carefully distinguish between "
    "gaps/limitations the authors *explicitly* state, versus gaps you infer yourself from "
    "reading the work critically. You never invent limitations that contradict the text. "
    "You are specific and actionable, not generic."
)

LIMITATION_HEADINGS = {"limitations", "future work", "discussion", "conclusion", "conclusions"}


def _parse_json(raw: str) -> dict:
    """Parse the model's JSON reply. An unusable reply is an error (so nothing is cached and a
    retry works), never an empty result that looks like a genuine "no gaps found"."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise llm_client.LLMProviderError(
            f"The model returned invalid JSON (the reply may have been cut off): {e}"
        ) from e
    if not isinstance(data, dict):
        raise llm_client.LLMProviderError("The model returned JSON in an unexpected shape.")
    return data


def _limitation_text(paper: IngestedPaper) -> str:
    chunks = [s.text for s in paper.sections if s.heading.lower() in LIMITATION_HEADINGS]
    return "\n\n".join(chunks)[:5000]


def author_acknowledged_gaps(paper: IngestedPaper) -> list[str]:
    limitation_text = _limitation_text(paper)
    if not limitation_text.strip():
        return []
    prompt = f"""Below are the Limitations/Future Work/Discussion/Conclusion sections of a paper.

{limitation_text}

Extract the limitations or open problems the AUTHORS THEMSELVES explicitly acknowledge.

Respond with ONLY a JSON object of this exact shape, no other text:
{{"gaps": ["<one specific sentence per limitation>", ...]}}
If nothing explicit is acknowledged, respond with {{"gaps": []}}."""
    raw = llm_client.complete_json(SYSTEM_PROMPT, prompt, max_tokens=400)
    parsed = _parse_json(raw)
    return [g.strip() for g in parsed.get("gaps", []) if isinstance(g, str) and g.strip()]


def inferred_gaps(paper: IngestedPaper, acknowledged: list[str]) -> list[dict]:
    acknowledged_block = "\n".join(f"- {g}" for g in acknowledged) or "(none found)"
    excerpt = prioritized_excerpt(paper, limit=6000)
    prompt = f"""Title: {paper.title}

Excerpt (abstract first, then limitations/discussion/conclusion, then the rest):
{excerpt}

The authors' own acknowledged limitations are:
{acknowledged_block}

Now, as a critical reviewer, identify ADDITIONAL research gaps not already listed above -- e.g.
missing baselines/comparisons, untested edge cases or populations, scalability claims without
evidence, reproducibility concerns, unaddressed ethical/societal considerations, or open
questions the related work suggests but this paper doesn't resolve.

Respond with ONLY a JSON object of this exact shape, no other text, at most 6 items:
{{"gaps": [
  {{"gap": "<one-sentence description>", "confidence": "low|medium|high", "direction": "<one-sentence suggested future research direction>"}}
]}}"""
    raw = llm_client.complete_json(SYSTEM_PROMPT, prompt, max_tokens=1000)
    parsed = _parse_json(raw)

    results = []
    for item in parsed.get("gaps", []):
        if not isinstance(item, dict):
            continue
        gap = str(item.get("gap", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()
        direction = str(item.get("direction", "")).strip()
        if confidence not in ("low", "medium", "high"):
            confidence = "medium"  # keep the gap even if the model words the confidence oddly
        if gap and direction:
            results.append({"gap": gap, "confidence": confidence, "direction": direction})
    return results


def analyze(paper: IngestedPaper) -> dict:
    acknowledged = author_acknowledged_gaps(paper)
    inferred = inferred_gaps(paper, acknowledged)
    return {"author_acknowledged": acknowledged, "inferred": inferred}
