"""Thin wrapper around the Groq API so every agent calls the LLM the same way.

Groq's free tier is rate-limited (requests/min, tokens/min, requests/day) rather
than credit-metered, so 429s are an expected part of normal operation here --
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


def complete_json(system: str, user_message: str, max_tokens: int = None, temperature: float = 0.2) -> str:
    """Single-turn completion constrained to JSON output (used for structured parsing)."""
    return _call_with_retry(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
