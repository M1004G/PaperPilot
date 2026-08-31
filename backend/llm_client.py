"""Thin wrapper around the Groq API so every agent calls the LLM the same way.

Groq's free tier is rate-limited (requests/min, tokens/min, requests/day) rather
than credit-metered, so 429s are an expected part of normal operation here 
not just an edge case. Every call goes through retry-with-backoff, and every
call is logged with timing + token usage for basic observability.
"""
import logging
import random
import time

import groq
from groq import Groq

from backend import config

logger = logging.getLogger("paperpilot.llm")


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider call fails after all retries are exhausted.
    The FastAPI layer maps this to a 502 (upstream failure), distinct from bugs
    in our own code (500) or bad input from the user (400)."""


_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(get one free, no credit card required, at https://console.groq.com)."
            )
        _client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT_SECONDS)
    return _client


def _call_with_retry(**create_kwargs) -> str:
    """Call the Groq chat completion endpoint, retrying on rate limits / transient errors."""
    client = _get_client()
    last_error = None
    call_start = time.monotonic()

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            attempt_start = time.monotonic()
            resp = client.chat.completions.create(**create_kwargs)
            elapsed = time.monotonic() - attempt_start
            usage = resp.usage
            logger.info(
                "groq_call_ok model=%s attempt=%d elapsed=%.2fs prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s",
                create_kwargs.get("model"), attempt + 1, elapsed,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )
            return resp.choices[0].message.content or ""
        except groq.RateLimitError as e:
            last_error = e
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (AttributeError, ValueError, TypeError):
                pass
            delay = retry_after if retry_after else config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
            delay += random.uniform(0, 0.5)
            logger.warning(
                "groq_rate_limited attempt=%d retrying_in=%.2fs", attempt + 1, delay
            )
            time.sleep(delay)
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_error = e
            delay = config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "groq_transient_error attempt=%d error=%s retrying_in=%.2fs",
                attempt + 1, type(e).__name__, delay
            )
            time.sleep(delay)
        except (groq.NotFoundError, groq.AuthenticationError, groq.PermissionDeniedError, groq.BadRequestError) as e:
            # Not transient -- retrying an invalid model name, a bad/missing API
            # key, or a malformed request will fail identically every time.
            # Fail immediately with a clear message instead of burning through
            # every retry attempt only to raise the exact same error, and
            # instead of letting it propagate unhandled into a raw 500 (this
            # is exactly what LLMProviderError -> 502 mapping in main.py exists
            # for -- it just wasn't being reached for this error class).
            logger.error("groq_call_failed_non_retryable error=%s detail=%s", type(e).__name__, e)
            raise LLMProviderError(
                f"Groq API rejected the request ({type(e).__name__}): {e}. "
                f"Check GROQ_MODEL ('{create_kwargs.get('model')}') is a currently valid model "
                f"and GROQ_API_KEY is correct."
            ) from e

    total_elapsed = time.monotonic() - call_start
    logger.error(
        "groq_call_failed after %d retries in %.2fs: %s",
        config.LLM_MAX_RETRIES, total_elapsed, last_error,
    )
    raise LLMProviderError(
        f"Groq API call failed after {config.LLM_MAX_RETRIES} retries: {last_error}"
    ) from last_error


def complete(system: str, user_message: str, max_tokens: int = None, temperature: float = 0.3) -> str:
    """Single-turn completion: system prompt + one user message -> text response."""
    return _call_with_retry(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )


def chat_complete(system: str, messages: list[dict], max_tokens: int = None, temperature: float = 0.3) -> str:
    """Multi-turn completion: messages is a list of {"role": "user"/"assistant", "content": str}."""
    full_messages = [{"role": "system", "content": system}] + messages
    return _call_with_retry(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
        temperature=temperature,
        messages=full_messages,
    )


def _extract_json_object(text: str) -> str:
    """Extracts a JSON object from raw model output without depending on
    provider-side JSON-mode enforcement (see complete_json's docstring for
    why that enforcement is the actual thing that broke on a model swap).
    Strips a markdown code fence if present, then finds the first '{' and
    its balanced closing '}' via a depth counter -- not just the first-to-
    last brace, since nested objects would confuse that -- discarding any
    reasoning preamble/postamble text around the JSON. Falls back to
    returning the (fence-stripped) text unchanged if no '{' is found at
    all, or if braces never balance (e.g. a truncated response) -- in both
    cases letting the caller's own json.loads raise a clear parse error
    rather than silently returning something misleading."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def complete_json(system: str, user_message: str, max_tokens: int = None, temperature: float = 0.2) -> str:
    """Single-turn completion constrained to JSON output.

    Deliberately does NOT use Groq's response_format={"type": "json_object"}
    mode. That mode's enforcement is provider/model-specific -- confirmed
    directly: swapping GROQ_MODEL to a reasoning model (openai/gpt-oss-120b)
    made every call to this function fail with a 400 "Failed to validate
    JSON" from Groq's own server-side validator, even though the model's
    actual answer was reasonable. That makes model choice a fragile,
    code-breaking decision instead of the config-only swap it's supposed to
    be (GROQ_MODEL is an env var specifically so this doesn't require a
    code change).

    Instead: the system prompt is given an explicit JSON-only instruction
    (on top of whatever instruction the caller's own prompt already
    includes), and the JSON object is extracted from whatever the model
    actually returns via _extract_json_object -- tolerant of markdown
    fences or stray reasoning text around it. This works identically across
    models regardless of how strictly each one honors "JSON mode"."""
    system_with_json_instruction = (
        f"{system}\n\nRespond with ONLY a single valid JSON object -- no markdown code "
        f"fences, no reasoning, no explanation before or after."
    )
    raw = _call_with_retry(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_with_json_instruction},
            {"role": "user", "content": user_message},
        ],
    )
    return _extract_json_object(raw)