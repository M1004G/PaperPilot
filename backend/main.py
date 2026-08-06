"""FastAPI layer: thin REST wrapper around the Orchestrator."""
import logging
import os
import tempfile
import time
import io
import zipfile

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from backend import config, codegen_agent
from backend.orchestrator import orchestrator
from backend.ingestion_agent import IngestionError
from backend.llm_client import LLMProviderError
from backend.logging_utils import new_request_id, set_request_id, get_request_id, RequestIdLogFilter

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
)
# Filters must go on the handler, not the root logger -- a logger-level filter
# only applies to records logged directly through that logger, not ones
# propagating up to it from child loggers (paperpilot.rag, paperpilot.llm, etc.).
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RequestIdLogFilter())
logger = logging.getLogger("paperpilot.api")

app = FastAPI(title="PaperPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or new_request_id()
    set_request_id(request_id)
    start = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - start
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request method=%s path=%s status=%d elapsed=%.2fs",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


class ChatRequest(BaseModel):
    doc_id: str
    query: str


def _handle_known_errors(e: Exception):
    """Map internal exception types to the right HTTP status, and log the real
    error server-side rather than just echoing raw exception strings to the client."""
    if isinstance(e, KeyError):
        raise HTTPException(404, "Unknown doc_id") from e
    if isinstance(e, IngestionError):
        raise HTTPException(400, str(e)) from e
    if isinstance(e, LLMProviderError):
        logger.error("llm_provider_error: %s", e)
        raise HTTPException(502, "The LLM provider is currently unavailable. Please try again shortly.") from e
    logger.exception("unexpected_error")
    raise HTTPException(500, "An unexpected error occurred. Please try again.") from e


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > config.MAX_PDF_SIZE_MB:
        raise HTTPException(
            400, f"File is {size_mb:.1f}MB, which exceeds the {config.MAX_PDF_SIZE_MB}MB limit."
        )
    if len(content) == 0:
        raise HTTPException(400, "The uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = orchestrator.ingest_paper(tmp_path)
        logger.info("ingested doc_id=%s title=%r pages=%d", result["doc_id"], result["title"], result["num_pages"])
        return result
    except Exception as e:
        _handle_known_errors(e)
    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # Windows can briefly hold a lock on the temp file after PyMuPDF
            # touches it, even on a failed open. Not deleting it immediately
            # isn't harmful -- it's in the OS temp dir and gets cleaned up
            # eventually; we just don't want this to crash the request.
            pass


@app.get("/summary/{doc_id}")
def summary(doc_id: str):
    try:
        return orchestrator.get_summary(doc_id)
    except Exception as e:
        _handle_known_errors(e)


@app.get("/gaps/{doc_id}")
def gaps(doc_id: str):
    try:
        return orchestrator.get_gaps(doc_id)
    except Exception as e:
        _handle_known_errors(e)


@app.get("/reproducibility/{doc_id}")
def reproducibility(doc_id: str, repo_url: str | None = None):
    """Runs (or returns the cached result of) the Reproducibility Suite.
    If a repo is linked/supplied, evaluates it directly. Otherwise, generates
    an implementation attempt from the paper and evaluates that instead --
    see the response's "mode" field ("repo_check" | "generated").
    Pass ?repo_url=... to check a specific repo instead of auto-detection."""
    try:
        return orchestrator.get_reproducibility(doc_id, repo_url=repo_url)
    except Exception as e:
        _handle_known_errors(e)


@app.get("/reproducibility/{doc_id}/download")
def reproducibility_download(doc_id: str):
    """Zips up the generated code files (mode == 'generated' only) for download.
    Serves whatever was already computed by GET /reproducibility/{doc_id} --
    does NOT trigger a fresh run, since that could silently switch modes."""
    try:
        repro = orchestrator.get_cached_reproducibility(doc_id)
    except Exception as e:
        _handle_known_errors(e)
        return
    if not repro or repro.get("mode") != "generated" or not repro.get("files"):
        raise HTTPException(
            status_code=404,
            detail="No generated code files are available. Call GET /reproducibility/{doc_id} first.",
        )

    buf = io.BytesIO()
    written_any = False
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in repro["files"].items():
            # Defense in depth: codegen_agent.plan_files() already sanitizes
            # filenames before generation, but these files come back out of
            # cached/persisted storage here, not directly from that function's
            # return value -- re-checking rather than trusting storage keeps
            # a single missed layer from being enough to write outside the zip.
            safe_name = codegen_agent.sanitize_filename(filename)
            if safe_name is None:
                logger.warning("reproducibility_download_dropped_unsafe_filename doc_id=%s filename=%r", doc_id, filename)
                continue
            zf.writestr(safe_name, content)
            written_any = True
    if not written_any:
        raise HTTPException(status_code=404, detail="No valid generated files were available to download.")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=paperpilot_{doc_id}_generated.zip"},
    )


@app.get("/report/{doc_id}", response_class=PlainTextResponse)
def report(doc_id: str):
    try:
        return orchestrator.get_report(doc_id)
    except Exception as e:
        _handle_known_errors(e)


@app.get("/report/{doc_id}/download")
def report_download(doc_id: str):
    try:
        content = orchestrator.get_report(doc_id)
    except Exception as e:
        _handle_known_errors(e)
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{doc_id}.md"},
    )


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return orchestrator.chat(req.doc_id, req.query)
    except Exception as e:
        _handle_known_errors(e)


@app.get("/health")
def health():
    return {"status": "ok"}
