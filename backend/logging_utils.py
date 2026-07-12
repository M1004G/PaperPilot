"""Request-ID propagation for logs.

A single incoming request touches many modules (orchestrator, ingestion, rag_agent,
llm_client) and, for section summaries, several worker threads. Threading a
request_id parameter through every function signature would be noisy; instead
this uses a contextvar set once per request in main.py's middleware, and a
logging.Filter that injects it into every log record automatically.

contextvars don't propagate into concurrent.futures.ThreadPoolExecutor workers
by default, so `copy_context_call` is used when submitting work to a pool
(see summary_agent.py) to carry the current request_id into those threads too.
"""
import contextvars
import logging
import uuid

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def set_request_id(request_id: str):
    return _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


class RequestIdLogFilter(logging.Filter):
    """Injects the current request_id into every log record as %(request_id)s."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def copy_context_call(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) inside a snapshot of the *current* contextvars context.
    Pass this to ThreadPoolExecutor.submit instead of fn directly to carry the
    current request_id (and any other contextvars) into the worker thread:

        executor.submit(copy_context_call, fn, arg1, arg2)
    """
    ctx = contextvars.copy_context()
    return ctx.run(fn, *args, **kwargs)
