"""Tests for backend/llm_client.py -- retry/backoff logic, fully mocked (no real API calls)."""
from unittest.mock import MagicMock

import groq
import pytest

from backend import llm_client, config


def _fake_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


class TestRetryBackoff:
    def test_succeeds_on_first_try(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_response("hello")
        monkeypatch.setattr(llm_client, "_client", fake_client)

        result = llm_client.complete("system", "user message")
        assert result == "hello"
        assert fake_client.chat.completions.create.call_count == 1

    def test_recovers_after_transient_rate_limit(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 3)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

        call_count = {"n": 0}

        def flaky(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                fake_resp = MagicMock()
                fake_resp.headers = {}
                raise groq.RateLimitError("rate limited", response=fake_resp, body=None)
            return _fake_response("success")

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = flaky
        monkeypatch.setattr(llm_client, "_client", fake_client)

        result = llm_client.complete("system", "user")
        assert result == "success"
        assert call_count["n"] == 3

    def test_raises_llm_provider_error_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

        def always_fails(**kwargs):
            fake_resp = MagicMock()
            fake_resp.headers = {}
            raise groq.RateLimitError("rate limited", response=fake_resp, body=None)

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = always_fails
        monkeypatch.setattr(llm_client, "_client", fake_client)

        with pytest.raises(llm_client.LLMProviderError):
            llm_client.complete("system", "user")

    def test_retries_on_connection_error(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

        call_count = {"n": 0}

        def flaky(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise groq.APIConnectionError(request=MagicMock())
            return _fake_response("recovered")

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = flaky
        monkeypatch.setattr(llm_client, "_client", fake_client)

        result = llm_client.complete("system", "user")
        assert result == "recovered"


class TestNonRetryableErrors:
    """A bad model name / API key / malformed request will fail identically
    on every attempt -- these should fail immediately with a clear
    LLMProviderError, not burn through every retry only to raise the same
    error, and not propagate raw into a generic 500 (this is a real bug that
    happened: Groq deprecated llama-3.3-70b-versatile, and the resulting
    groq.NotFoundError was uncaught here, producing an opaque 500 instead of
    the clean 502 mapping this app already has for provider failures)."""

    @pytest.mark.parametrize("error_cls", [
        groq.NotFoundError, groq.AuthenticationError, groq.PermissionDeniedError, groq.BadRequestError,
    ])
    def test_fails_immediately_without_retrying(self, monkeypatch, error_cls):
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 5)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

        fake_resp = MagicMock()
        fake_resp.status_code = 404

        call_count = {"n": 0}

        def always_fails(**kwargs):
            call_count["n"] += 1
            raise error_cls("model not found", response=fake_resp, body=None)

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = always_fails
        monkeypatch.setattr(llm_client, "_client", fake_client)

        with pytest.raises(llm_client.LLMProviderError):
            llm_client.complete("system", "user")
        assert call_count["n"] == 1  # no retries burned on a non-retryable error

    def test_error_message_includes_model_name_for_debuggability(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
        monkeypatch.setattr(config, "GROQ_MODEL", "some-deprecated-model")

        fake_resp = MagicMock()
        fake_resp.status_code = 404

        def always_fails(**kwargs):
            raise groq.NotFoundError("model not found", response=fake_resp, body=None)

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = always_fails
        monkeypatch.setattr(llm_client, "_client", fake_client)

        with pytest.raises(llm_client.LLMProviderError, match="some-deprecated-model"):
            llm_client.complete("system", "user")


class TestNoApiKey:
    def test_raises_clear_error_without_api_key(self, monkeypatch):
        monkeypatch.setattr(llm_client, "_client", None)
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            llm_client.complete("system", "user")