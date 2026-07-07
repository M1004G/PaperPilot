"""FastAPI layer: thin REST wrapper around the Orchestrator."""
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.orchestrator import orchestrator

app = FastAPI(title="PaperPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    doc_id: str
    query: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = orchestrator.ingest_paper(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        os.unlink(tmp_path)

    return result


@app.get("/summary/{doc_id}")
def summary(doc_id: str):
    try:
        return orchestrator.get_summary(doc_id)
    except KeyError:
        raise HTTPException(404, "Unknown doc_id")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/gaps/{doc_id}")
def gaps(doc_id: str):
    try:
        return orchestrator.get_gaps(doc_id)
    except KeyError:
        raise HTTPException(404, "Unknown doc_id")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/report/{doc_id}", response_class=PlainTextResponse)
def report(doc_id: str):
    try:
        return orchestrator.get_report(doc_id)
    except KeyError:
        raise HTTPException(404, "Unknown doc_id")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/report/{doc_id}/download")
def report_download(doc_id: str):
    from fastapi.responses import Response
    try:
        content = orchestrator.get_report(doc_id)
    except KeyError:
        raise HTTPException(404, "Unknown doc_id")
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{doc_id}.md"},
    )


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return orchestrator.chat(req.doc_id, req.query)
    except KeyError:
        raise HTTPException(404, "Unknown doc_id")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
