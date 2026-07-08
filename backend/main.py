"""FastAPI layer: thin REST wrapper around the Orchestrator."""
import logging
import os
import tempfile
import time

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from backend import config
from backend.orchestrator import orchestrator
from backend.ingestion_agent import IngestionError
from backend.llm_client import LLMProviderError

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("paperpilot.api")

app = FastAPI(title="PaperPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - start
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
        os.unlink(tmp_path)


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
