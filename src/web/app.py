"""FastAPI backend — upload PDFs, run delta, ask questions."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe

from src.canonical.model import CanonicalDocument
from src.config import Config, load_config
from src.delta.align import align_documents
from src.delta.engine import DeltaEntry, compute_delta
from src.delta.report import render_json, render_markdown
from src.ingest.base import auto_adapter
from src.observability.logging import bind_correlation_id, new_correlation_id
from src.observability.tracing import flush, init_tracer

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# App state — holds the last analysis result for chat context
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}
_ALLOWED_UPLOAD_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PID Delta Chat", version="0.1.0")

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    """Attach a correlation ID to every request and bind it to structlog."""
    cid = request.headers.get("X-Request-ID") or new_correlation_id()
    bind_correlation_id(cid)
    start = time.monotonic()
    
    from langfuse import propagate_attributes
    with propagate_attributes(session_id=cid):
        response = await call_next(request)
        
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=elapsed_ms,
        correlation_id=cid,
    )
    response.headers["X-Request-ID"] = cid
    return response


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (_static_dir / "index.html").read_text()


@app.post("/api/delta")
@observe(name="web.delta")
async def api_delta(
    old_file: UploadFile = File(...),  # noqa: B008
    new_file: UploadFile = File(...),  # noqa: B008
) -> JSONResponse:
    """Ingest two PDFs/images and return the delta report."""
    cfg = load_config()

    def _suffix_for_upload(upload: UploadFile) -> str:
        raw_name = upload.filename or ""
        suffix = Path(raw_name).suffix.lower()
        if suffix in _ALLOWED_UPLOAD_EXTS:
            return suffix
        return ".bin"

    old_suffix = _suffix_for_upload(old_file)
    new_suffix = _suffix_for_upload(new_file)
    if old_suffix not in _ALLOWED_UPLOAD_EXTS or new_suffix not in _ALLOWED_UPLOAD_EXTS:
        supported = ", ".join(sorted(_ALLOWED_UPLOAD_EXTS))
        return JSONResponse(
            {"error": f"Unsupported file format. Supported extensions: {supported}"},
            status_code=400,
        )

    # Save uploads to temp files
    with tempfile.NamedTemporaryFile(suffix=old_suffix, delete=False) as tmp_old:
        tmp_old.write(await old_file.read())
        old_path = Path(tmp_old.name)

    with tempfile.NamedTemporaryFile(suffix=new_suffix, delete=False) as tmp_new:
        tmp_new.write(await new_file.read())
        new_path = Path(tmp_new.name)

    try:
        log.info("delta.start", old=old_file.filename, new=new_file.filename)
        start = time.monotonic()

        old_doc = auto_adapter(old_path).ingest(old_path)
        new_doc = auto_adapter(new_path).ingest(new_path)

        alignment = align_documents(old_doc, new_doc, config=cfg.delta)
        entries = compute_delta(alignment)

        # Store for chat
        _state["old_doc"] = old_doc
        _state["new_doc"] = new_doc
        _state["entries"] = entries
        _state["qa"] = None  # reset QA on new analysis

        added = sum(1 for e in entries if e.kind.value == "added")
        removed = sum(1 for e in entries if e.kind.value == "removed")
        modified = sum(1 for e in entries if e.kind.value == "modified")

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        log.info(
            "delta.complete",
            elapsed_ms=elapsed_ms,
            added=added, removed=removed, modified=modified,
        )

        return JSONResponse({
            "report_md": render_markdown(entries, old_doc, new_doc),
            "report_json": render_json(entries, old_doc, new_doc),
            "summary": {
                "old_elements": len(old_doc.elements),
                "old_pages": old_doc.page_count,
                "new_elements": len(new_doc.elements),
                "new_pages": new_doc.page_count,
                "added": added,
                "removed": removed,
                "modified": modified,
                "total": len(entries),
            },
        })
    except Exception as exc:
        log.exception("delta.error", error=str(exc))
        status = 500
        message = str(exc)
        if "No format adapter found" in message:
            status = 400
            message = (
                "No compatible adapter found for one or both files. "
                "Supported: PDF, PNG, JPG, JPEG, TIF, TIFF, BMP, WEBP. "
                "For image/scanned OCR, install the system tesseract binary."
            )
        elif "requires the system 'tesseract' binary" in message:
            status = 400
        return JSONResponse({"error": f"Delta analysis failed: {message}"}, status_code=status)
    finally:
        old_path.unlink(missing_ok=True)
        new_path.unlink(missing_ok=True)
        flush()


@app.post("/api/chat")
@observe(name="web.chat")
async def api_chat(body: dict[str, str]) -> JSONResponse:
    """Answer a question grounded in the uploaded documents + delta."""
    cfg = load_config()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    if "old_doc" not in _state:
        return JSONResponse({"error": "Run a delta analysis first"}, status_code=400)

    if not cfg.chat.llm_api_key:
        return JSONResponse({
            "error": "LLM not configured. Set PID_LLM_API_KEY.",
        }, status_code=400)

    from src.chat.answer import GroundedQA
    from src.chat.index import RetrievalIndex
    from src.chat.llm import LLMClient

    old_doc: CanonicalDocument = _state["old_doc"]
    new_doc: CanonicalDocument = _state["new_doc"]
    entries: list[DeltaEntry] = _state["entries"]

    # Lazily build index + QA
    if _state.get("qa") is None:
        try:
            log.info("chat.index.start")
            start = time.monotonic()
            
            llm = LLMClient(cfg.chat)
            index = RetrievalIndex(cfg.chat.embedding_dim)
            index.add_document(old_doc, "old")
            index.add_document(new_doc, "new")
            index.add_delta_entries(entries)

            texts = [e.text for e in index.entries]
            if texts:
                embeddings = llm.embed(texts)
                emb_map = {e.id: emb for e, emb in zip(index.entries, embeddings)}
                index.set_embeddings(emb_map)

            _state["qa"] = GroundedQA(index, llm, cfg.chat)
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            log.info("chat.index.complete", elapsed_ms=elapsed_ms, entries=len(texts))
        except Exception as exc:
            log.exception("chat.index.error", error=str(exc))
            flush()
            return JSONResponse({"error": f"Failed to build index: {exc}"}, status_code=502)

    try:
        qa: GroundedQA = _state["qa"]
        answer = qa.answer(question)
        log.info("chat.answer.complete", citations=len(answer.citations), citation_rate=answer.citation_rate)
    except Exception as exc:
        log.exception("chat.answer.error", error=str(exc))
        flush()
        return JSONResponse({"error": f"LLM error: {exc}"}, status_code=502)

    return JSONResponse({
        "answer": answer.text,
        "citations": answer.citations,
        "citation_rate": answer.citation_rate,
        "validated_citations": [
            {"raw": c.raw, "valid": c.valid, "reason": c.reason}
            for c in answer.validated_citations
        ],
    })


def launch(config: Config | None = None) -> None:
    """Build and launch the FastAPI app."""
    import uvicorn

    cfg = config or load_config()
    init_tracer(cfg.output_dir / "traces")
    uvicorn.run(
        app,
        host=cfg.web.host,
        port=cfg.web.port,
        log_level=cfg.log_level.lower(),
    )
