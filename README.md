# PID Delta Chat

Compare two revisions of a P&ID (Piping & Instrumentation Diagram), compute a structured delta, and ask grounded questions about the changes. Inputs can be PDFs or image files.

## Quick Start

```bash
# Install
uv sync --extra dev

# Copy env and set your LLM key
cp .env.example .env
# Edit .env — set PID_LLM_API_KEY

# Launch web UI
make web
# or: uv run python main.py web

# Optional for OCR on scanned PDFs and image files
# Ubuntu/Debian: sudo apt install tesseract-ocr
# macOS: brew install tesseract
```

Open http://127.0.0.1:7860, upload two PDF revisions, and click **Run Delta Analysis**.

## CLI Usage

```bash
# Full pipeline (ingest → delta → report)
make run OLD=data/samples/old.pdf NEW=data/samples/new.pdf

# Interactive chat
make chat OLD=data/samples/old.pdf NEW=data/samples/new.pdf

# Eval harness
make eval OLD=data/samples/old.pdf NEW=data/samples/new.pdf GT=eval/datasets/ground_truth.json
```

## Configuration

All settings are driven by environment variables (or a `.env` file). See `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `PID_LLM_API_KEY` | (empty) | **Required for chat.** OpenAI-compatible API key |
| `PID_LLM_MODEL` | `gpt-4o-mini` | Model for chat + embeddings |
| `PID_FUZZY_THRESHOLD` | `70` | Alignment sensitivity (0–100) |
| `PID_WEB_PORT` | `7860` | Web UI port |
| `PID_PDF_DPI` | `300` | PDF rendering resolution |

## Architecture

```
PDF A ──┐
        ├── ingest ── canonical ── align ── delta engine ── report
PDF B ──┘                                                    │
                                                     ┌──────┘
                                                     │  retrieval index
                                                     │  + embeddings
                                                     ▼
                                              grounded Q&A (RAG)
```

**Key design decisions:**

- **Canonical model as the seam:** All format adapters (native PDF, scanned OCR, DWG stub) normalise into the same `CanonicalDocument` / `Element` schema. Delta engine and chat are format-agnostic.
- **Stable-ID alignment first:** Elements with tag numbers (`PI-101-01`), line specs, or note numbers match by ID. Unmatched elements fall back to fuzzy text similarity (rapidfuzz).
- **Citation enforcement:** Chat answers must cite `[PID:source:page:element_id]` or `[delta:entry_index]`. The system prompt refuses to answer ungrounded questions.
- **Deterministic delta:** Alignment and classification are deterministic given the same inputs. LLM non-determinism is isolated to the chat layer (temperature=0 by default).

## Project Structure

```
src/
  canonical/model.py      — Format-agnostic document/element schema
  ingest/
    base.py               — FormatAdapter ABC + registry
    pdf_native.py         — PyMuPDF adapter with regex entity extraction
    pdf_scanned.py        — OCR adapter (Tesseract)
    dwg.py                — DWG stub (interface demo)
  delta/
    align.py              — Stable-key + fuzzy alignment
    engine.py             — Change classification (added/removed/modified)
    report.py             — Markdown + JSON report rendering
  chat/
    index.py              — In-memory vector retrieval
    llm.py                — OpenAI-compatible client
    answer.py             — RAG with citation enforcement
  markup/
    overlay.py            — Delta highlights on PDF pages
  web/
    app.py                — Gradio UI (upload + report + chat)
  observability/
    tracing.py            — Span-based JSONL tracing
    logging.py            — structlog setup
  config.py               — Env-var configuration
eval/
  metrics.py              — Delta P/R/F1 + answer scoring
  run_eval.py             — CLI eval harness
main.py                   — CLI entry point (pipeline / chat / web)
```

## Formats In Scope

| Format | Status | Adapter |
|---|---|---|
| Native PDF | **Working** | `pdf_native.py` — extracts text, tags, line specs, notes |
| Scanned PDF | **Working** | `pdf_scanned.py` — OCR via Tesseract (requires `tesseract` binary) |
| Image files (`.png/.jpg/.jpeg/.tif/.tiff/.bmp/.webp`) | **Working** | `image_ocr.py` — OCR via Tesseract (requires `tesseract` binary) |
| DWG | **Stub** | `dwg.py` — demonstrates the adapter seam |

## Running Tests

```bash
make test
# or: uv run pytest tests/ -v
```

## Lint & Type Check

```bash
make lint       # ruff
make typecheck  # mypy --strict
```

## What's Cut (Take-Home Scope)

- **Persistence:** No database. Session state is in-memory per Gradio session.
- **Real vector DB:** Uses in-memory cosine similarity. Swap in Chroma/Pinecone for production.
- **Async pipeline:** All code is synchronous. Gradio runs the pipeline in a thread.
- **Authentication:** No login. The web UI is open to anyone on the network.
- **DWG implementation:** Stub only. The adapter interface is ready for ODA/ezdxf integration.
