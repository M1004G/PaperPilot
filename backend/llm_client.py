"""Thin wrapper around the Groq API so every agent calls the LLM the same way.

Groq's free tier is rate-limited (requests/min, tokens/min, requests/day) rather
than credit-metered, so 429s are an expected part of normal operation here --
not just an edge case. Every call goes through retry-with-backoff.
"""
import random
import time

import groq
from groq import Groq

from backend import config

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(get one free, no credit card required, at https://console.groq.com)."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _call_with_retry(**create_kwargs) -> str:
    """Call the Groq chat completion endpoint, retrying on rate limits / transient errors."""
    client = _get_client()
    last_error = None

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            return resp.choices[0].message.content or ""
        except groq.RateLimitError as e:
            last_error = e
            # Respect a server-provided retry-after if present, else exponential backoff + jitter.
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (AttributeError, ValueError, TypeError):
                pass
            delay = retry_after if retry_after else config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
            delay += random.uniform(0, 0.5)
            time.sleep(delay)
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_error = e
            delay = config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)

    raise RuntimeError(
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
