"""FastAPI backend — upload PDFs, run delta, ask questions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.canonical.model import CanonicalDocument
from src.config import Config, load_config
from src.delta.align import align_documents
from src.delta.engine import DeltaEntry, compute_delta
from src.delta.report import render_json, render_markdown
from src.ingest.base import auto_adapter

# ---------------------------------------------------------------------------
# App state — holds the last analysis result for chat context
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PID Delta Chat", version="0.1.0")

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (_static_dir / "index.html").read_text()


@app.post("/api/delta")
async def api_delta(
    old_file: UploadFile = File(...),  # noqa: B008
    new_file: UploadFile = File(...),  # noqa: B008
) -> JSONResponse:
    """Ingest two PDFs and return the delta report."""
    cfg = load_config()

    # Save uploads to temp files
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_old:
        tmp_old.write(await old_file.read())
        old_path = Path(tmp_old.name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_new:
        tmp_new.write(await new_file.read())
        new_path = Path(tmp_new.name)

    try:
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
    finally:
        old_path.unlink(missing_ok=True)
        new_path.unlink(missing_ok=True)


@app.post("/api/chat")
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

    qa: GroundedQA = _state["qa"]
    answer = qa.answer(question)

    return JSONResponse({
        "answer": answer.text,
        "citations": answer.citations,
    })


def launch(config: Config | None = None) -> None:
    """Build and launch the FastAPI app."""
    import uvicorn

    cfg = config or load_config()
    uvicorn.run(
        app,
        host=cfg.web.host,
        port=cfg.web.port,
        log_level=cfg.log_level.lower(),
    )
