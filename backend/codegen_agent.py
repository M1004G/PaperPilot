"""Codegen Agent: when a paper doesn't link real code (the common case), this
generates an attempted implementation from the paper's own text instead of
just reporting "no code found."

Pipeline: EXTRACT (structured info from the paper) -> PLAN (file set +
dependencies) -> GENERATE (one call per file, dependency-ordered) -> VALIDATE
(each .py file must actually parse; refusal-shaped/empty output is rejected,
not silently stored; one retry attempt on a syntax error before giving up).

The output is then handed to repro_check_agent.evaluate() by the
orchestrator, exactly like a fetched GitHub repo would be -- so a generated
implementation gets the same hygiene/claim scoring as a real one, instead of
being trusted blindly. Validation here answers a narrower question --
"is this at least parseable, non-refusal output?" -- not "does it run
correctly" or "does it reproduce the paper's results"; see the "Known
limitation" note in ARCHITECTURE.md.
"""
import json
import logging
import os
import re

from backend import config, llm_client
from backend.ingestion_agent import IngestedPaper

logger = logging.getLogger("paperpilot.codegen")

EXTRACT_SYSTEM_PROMPT = (
    "You are an expert ML researcher extracting exactly what's needed to reproduce a "
    "paper's experiments. You distinguish clearly between what the paper explicitly "
    "states and what would have to be guessed -- anything not explicitly stated goes in "
    "'gaps', never silently invented as a stated fact."
)

GENERATE_SYSTEM_PROMPT = (
    "You are an expert ML engineer writing a best-effort PyTorch implementation of a "
    "paper's described method. Write real, runnable, idiomatic code -- not pseudocode. "
    "Where the paper leaves a detail unspecified (see the 'gaps' list you're given), pick "
    "a reasonable, clearly-commented default (e.g. '# LR not specified in paper; using "
    "1e-3 as a common default') rather than inventing a fake citation for the choice."
)

PLAN_SYSTEM_PROMPT = (
    "You are a software architect planning a PyTorch implementation of a paper's method, "
    "before any code is written. You design a small, sensible set of files and their "
    "dependencies -- not an exhaustive framework, just what this specific method needs."
)

EXTRACT_PROMPT_TEMPLATE = """Read this research paper excerpt and extract everything needed to reproduce
its experiments.

PAPER TEXT:
{paper_text}

Respond with ONLY a JSON object of exactly this shape, no other text:
{{
  "title": "paper title",
  "task": "the ML task being solved (e.g. image classification, NLP, object detection)",
  "framework": "pytorch",
  "datasets": [
    {{"name": "...", "source": "e.g. a URL, or a library like torchvision/HuggingFace", "splits": "train/val/test if mentioned", "preprocessing": "steps described, or empty string"}}
  ],
  "model": {{
    "architecture": "description of the architecture",
    "base_model": "pretrained base model if any, else empty string",
    "key_components": "notable layers/blocks/modules described",
    "input_shape": "input dimensions if mentioned, else empty string",
    "output_shape": "output dimensions / num classes if mentioned, else empty string"
  }},
  "training": {{
    "optimizer": "optimizer name if mentioned, else empty string",
    "learning_rate": "value if mentioned, else empty string",
    "batch_size": "value if mentioned, else empty string",
    "epochs": "value if mentioned, else empty string",
    "loss_function": "if mentioned, else empty string"
  }},
  "evaluation": {{"metrics": ["metric names"], "reported_results": "key numbers reported, or empty string"}},
  "gaps": ["one entry per hyperparameter/detail the paper did NOT specify, needed for a real run"]
}}"""

# PLANNING (before any code is written) -- adapted from PaperCoder/Paper2Code
# (https://github.com/going-doer/Paper2Code): plan the file set and their
# dependencies FIRST, then generate each file with awareness of the files it
# depends on, instead of asking one call to produce every file blind and
# unordered. A single "write 5 files at once" call is the weaker approach
# this replaces -- it starves each file of output budget and produces files
# that don't actually agree with each other's function signatures.
PLAN_PROMPT_TEMPLATE = """Based on this structured summary of a paper's method, plan the files needed
for a PyTorch implementation attempt. Keep it small and specific to this method -- typically
4-7 files (e.g. dataset loading, model definition, training loop, requirements, README; add
an eval/config file only if the method clearly needs one).

PAPER INFO:
{paper_info_json}

Respond with ONLY a JSON object of exactly this shape, no other text:
{{"files": [
  {{"filename": "model.py", "purpose": "one sentence describing what this file implements", "depends_on": ["list of other filenames from this same plan that this file needs to import/reference, or empty list"]}}
]}}
Filenames must be flat (no directories, e.g. "model.py" not "src/model.py") and unique.
requirements.txt and README.md should have empty depends_on lists."""

GENERATE_FILE_PROMPT_TEMPLATE = """Write the complete contents of `{filename}` for a PyTorch implementation
of this paper's method.

PURPOSE OF THIS FILE: {purpose}

PAPER INFO:
{paper_info_json}
{dependencies_block}
Respond with ONLY the raw file contents -- no markdown code fences, no explanation before or after."""

# Follow-up used when the first attempt at a .py file fails to compile --
# gives the model its own broken output plus the actual error, rather than
# asking it to guess again from scratch.
GENERATE_FILE_RETRY_TEMPLATE = """Your previous attempt to write `{filename}` has a Python syntax error.

PURPOSE OF THIS FILE: {purpose}

PAPER INFO:
{paper_info_json}
{dependencies_block}
PREVIOUS ATTEMPT:
{previous_content}

SYNTAX ERROR: {error}

Write a corrected, complete, syntactically valid version of `{filename}` that fixes this error.
Respond with ONLY the raw file contents -- no markdown code fences, no explanation before or after."""

_DEFAULT_PLAN = [
    {"filename": "dataset.py", "purpose": "Dataset loading and preprocessing", "depends_on": []},
    {"filename": "model.py", "purpose": "Model architecture", "depends_on": []},
    {"filename": "train.py", "purpose": "Training loop", "depends_on": ["dataset.py", "model.py"]},
    {"filename": "requirements.txt", "purpose": "Python dependencies", "depends_on": []},
    {"filename": "README.md", "purpose": "What this is and how to run it", "depends_on": []},
]

# Flat, single-segment filenames only: letters/digits/./_/- , no leading dot
# (blocks ".", "..", and hidden files), no separators.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

_REFUSAL_PREFIXES = (
    "i cannot", "i can't", "i am sorry", "i'm sorry", "as an ai",
    "i am unable", "i'm unable", "sorry, i", "i won't", "i will not",
    "unfortunately, i",
)


def sanitize_filename(name) -> str | None:
    """Guards against path traversal / absolute paths / nested directories in
    a filename coming from an LLM-produced plan, before it's ever used as a
    real filesystem or zip-entry path. Only a flat, single-segment filename
    is accepted. Public (not `_`-prefixed) so main.py's zip download route
    can reuse this exact guard as defense in depth -- the download route
    shouldn't rely solely on plan_files() having already sanitized things,
    since the files it serves come back out of persisted/cached storage,
    not directly from this function's return value."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name or name in (".", ".."):
        return None
    if "/" in name or "\\" in name or "\x00" in name:
        return None
    if os.path.isabs(name):
        return None
    if name.startswith("."):
        return None
    if not _FILENAME_RE.match(name):
        return None
    return name


def _looks_like_refusal_or_empty(content: str) -> bool:
    """Lightweight check for won't-implement/refusal-shaped output: either
    far too short to be a real file, or opens with a refusal phrase. Checking
    only the opening (not substring search across the whole file) avoids
    false positives on legitimate code that happens to contain a word like
    "cannot" inside a string or comment."""
    stripped = content.strip()
    if len(stripped) < config.CODEGEN_MIN_FILE_CHARS:
        return True
    return stripped.lower().startswith(_REFUSAL_PREFIXES)


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return content


def _python_syntax_error(content: str, filename: str) -> SyntaxError | None:
    try:
        compile(content, filename, "exec")
        return None
    except SyntaxError as e:
        return e


def _paper_text_for_extraction(paper: IngestedPaper) -> str:
    """Prioritizes methodology/experiment-bearing sections over the whole
    paper, within a generous but bounded character budget -- unlike the
    2000-char truncation in the reference script this replaces, which cut
    the input down to roughly the abstract alone."""
    from backend.repro_check_agent import METHOD_HEADINGS

    method_chunks = [f"## {s.heading}\n{s.text}" for s in paper.sections if s.heading.lower() in METHOD_HEADINGS]
    if method_chunks:
        text = paper.abstract + "\n\n" + "\n\n".join(method_chunks)
    else:
        text = paper.full_text
    return text[: config.CODEGEN_MAX_INPUT_CHARS]


def extract_paper_info(paper: IngestedPaper) -> dict:
    prompt = EXTRACT_PROMPT_TEMPLATE.format(paper_text=_paper_text_for_extraction(paper))
    raw = llm_client.complete_json(EXTRACT_SYSTEM_PROMPT, prompt, max_tokens=config.CODEGEN_EXTRACT_MAX_TOKENS)
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("codegen_extract_json_parse_failed")
        info = {"title": paper.title, "gaps": ["Could not parse structured extraction from the LLM response."]}
    info.setdefault("gaps", [])
    return info


def plan_files(paper_info: dict) -> list[dict]:
    """Planning stage (before any code is written): decide the file set and
    their dependencies. Falls back to a fixed, reasonable default plan if the
    LLM call fails or returns something unusable, rather than generating
    nothing.

    Each candidate filename is sanitized (path traversal / absolute paths /
    nested directories rejected) and deduplicated (first occurrence wins,
    later duplicates dropped with a warning) before being accepted into the
    plan, so nothing downstream -- generation, the zip download route --
    has to trust the LLM's filenames blindly."""
    prompt = PLAN_PROMPT_TEMPLATE.format(paper_info_json=json.dumps(paper_info, indent=2))
    raw = llm_client.complete_json(PLAN_SYSTEM_PROMPT, prompt, max_tokens=config.CODEGEN_PLAN_MAX_TOKENS)
    try:
        parsed = json.loads(raw)
        raw_files = parsed.get("files", [])

        seen_names: set[str] = set()
        candidates = []
        for f in raw_files:
            if not isinstance(f, dict):
                continue
            name = sanitize_filename(f.get("filename"))
            if name is None:
                logger.warning("codegen_plan_invalid_filename_dropped raw=%r", f.get("filename"))
                continue
            if name in seen_names:
                logger.warning("codegen_plan_duplicate_filename_dropped filename=%s", name)
                continue
            seen_names.add(name)
            candidates.append({
                "filename": name,
                "purpose": str(f.get("purpose", "")).strip() or "Implementation file.",
                "depends_on": f.get("depends_on", []),
            })

        plan = []
        for f in candidates:
            depends_on = [d for d in f["depends_on"] if isinstance(d, str) and d in seen_names and d != f["filename"]]
            plan.append({"filename": f["filename"], "purpose": f["purpose"], "depends_on": depends_on})
        if plan:
            return plan[: config.CODEGEN_MAX_FILES]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        pass
    logger.warning("codegen_plan_failed_using_default")
    return list(_DEFAULT_PLAN)


def _topo_order(plan: list[dict]) -> list[dict]:
    """Kahn's algorithm so each file is generated after the files it depends
    on (dependency content can then be shown to the LLM writing it, the way
    PaperCoder's generation stage follows its planning-determined execution
    order). Falls back to the plan's own order on a cycle -- a malformed
    dependency graph shouldn't block generation entirely."""
    by_name = {f["filename"]: f for f in plan}
    in_degree = {f["filename"]: 0 for f in plan}
    for f in plan:
        for dep in f["depends_on"]:
            if dep in in_degree:
                in_degree[f["filename"]] += 1

    ready = [name for name, deg in in_degree.items() if deg == 0]
    ordered = []
    remaining_deps = {f["filename"]: set(f["depends_on"]) for f in plan}
    while ready:
        ready.sort()  # deterministic order among independent files
        name = ready.pop(0)
        ordered.append(by_name[name])
        for f in plan:
            if name in remaining_deps[f["filename"]]:
                remaining_deps[f["filename"]].discard(name)
                if not remaining_deps[f["filename"]] and f["filename"] not in [o["filename"] for o in ordered] and f["filename"] not in ready:
                    ready.append(f["filename"])

    if len(ordered) != len(plan):
        logger.warning("codegen_plan_has_cycle_using_declared_order")
        return plan
    return ordered


def _build_dependencies_block(f: dict, files: dict[str, str]) -> str:
    dep_contents = {dep: files[dep] for dep in f["depends_on"] if dep in files}
    if not dep_contents:
        return ""
    return "\n\nFILES THIS DEPENDS ON (already written -- match their actual names/signatures):\n" + "\n\n".join(
        f"--- {name} ---\n{content[:2000]}" for name, content in dep_contents.items()
    )


def _generate_and_validate_file(f: dict, dep_block: str, paper_info: dict) -> str | None:
    """Generates one file, then validates it rather than trusting it blindly:
    - refusal-shaped or too-short output is rejected outright (no retry --
      a follow-up call is unlikely to fix a refusal, unlike a syntax error)
    - .py files must actually compile(); on a SyntaxError, one retry is made
      with the previous attempt + the real error message included, since
      that's a concrete, fixable signal (unlike a refusal)
    - a file that's still invalid after the retry (or whose retry call itself
      fails) is dropped, the same way an LLM exception on the first attempt
      is already handled -- silently storing broken output would make the
      Reproducibility Check Agent's checks (and the person reading them)
      trust code that was never actually valid Python
    """
    filename = f["filename"]
    paper_info_json = json.dumps(paper_info, indent=2)
    prompt = GENERATE_FILE_PROMPT_TEMPLATE.format(
        filename=filename, purpose=f["purpose"], paper_info_json=paper_info_json, dependencies_block=dep_block,
    )
    try:
        content = llm_client.complete(GENERATE_SYSTEM_PROMPT, prompt, max_tokens=config.CODEGEN_PER_FILE_MAX_TOKENS)
    except Exception as e:
        logger.warning("codegen_file_generation_failed filename=%s error=%s", filename, e)
        return None

    content = _strip_code_fences(content)
    if _looks_like_refusal_or_empty(content):
        logger.warning("codegen_file_refusal_or_empty filename=%s", filename)
        return None
    if not filename.endswith(".py"):
        return content

    error = _python_syntax_error(content, filename)
    if error is None:
        return content

    logger.warning("codegen_file_syntax_error filename=%s error=%s -- retrying once", filename, error)
    retry_prompt = GENERATE_FILE_RETRY_TEMPLATE.format(
        filename=filename, purpose=f["purpose"], paper_info_json=paper_info_json,
        dependencies_block=dep_block, previous_content=content[:4000], error=str(error),
    )
    try:
        retried = llm_client.complete(GENERATE_SYSTEM_PROMPT, retry_prompt, max_tokens=config.CODEGEN_PER_FILE_MAX_TOKENS)
    except Exception as e:
        logger.warning("codegen_file_retry_failed filename=%s error=%s", filename, e)
        return None

    retried = _strip_code_fences(retried)
    if _looks_like_refusal_or_empty(retried):
        logger.warning("codegen_file_retry_refusal_or_empty filename=%s", filename)
        return None
    retry_error = _python_syntax_error(retried, filename)
    if retry_error is not None:
        logger.warning("codegen_file_still_invalid_after_retry filename=%s error=%s", filename, retry_error)
        return None
    return retried


def generate_code_files(paper_info: dict) -> dict[str, str]:
    """Plan the file set, order it dependency-first, then generate (and
    validate) each file with the actual contents of the files it depends on
    in context -- so e.g. train.py is written having seen model.py's real
    class/function names, instead of every file being guessed independently."""
    plan = plan_files(paper_info)
    ordered = _topo_order(plan)

    files: dict[str, str] = {}
    for f in ordered:
        dep_block = _build_dependencies_block(f, files)
        content = _generate_and_validate_file(f, dep_block, paper_info)
        if content:
            files[f["filename"]] = content
    return files


def build_gap_report(paper_info: dict) -> str:
    lines = ["## Extraction Gaps", "", "Details the paper didn't specify, so the generated code had to assume them:", ""]
    gaps = paper_info.get("gaps") or []
    if gaps:
        for g in gaps:
            lines.append(f"- {g}")
    else:
        lines.append("_No gaps identified -- the paper's methodology section was unusually complete._")
    return "\n".join(lines)


def analyze(paper: IngestedPaper) -> dict:
    """Main entry point: extract structured info, generate an implementation
    attempt from it. Does NOT run the generated code and does NOT itself
    score it -- that's repro_check_agent.evaluate()'s job, called by the
    orchestrator against this function's `files` output."""
    paper_info = extract_paper_info(paper)
    files = generate_code_files(paper_info) if config.CODEGEN_ENABLED else {}
    gap_report = build_gap_report(paper_info)
    return {"paper_info": paper_info, "files": files, "gap_report": gap_report}
