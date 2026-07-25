"""Gradio web UI — upload two PIDs, see the delta, ask questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from src.canonical.model import CanonicalDocument
from src.config import Config, load_config
from src.delta.align import align_documents
from src.delta.engine import DeltaEntry, compute_delta
from src.delta.report import render_markdown
from src.ingest.base import auto_adapter


def _run_delta(
    old_path: str | None,
    new_path: str | None,
) -> tuple[str, str]:
    """Ingest both PDFs, compute delta, return (report_md, summary)."""
    if not old_path or not new_path:
        return "*Please upload both PDFs.*", ""

    cfg = load_config()

    old_doc = auto_adapter(Path(old_path)).ingest(Path(old_path))
    new_doc = auto_adapter(Path(new_path)).ingest(Path(new_path))

    alignment = align_documents(old_doc, new_doc, config=cfg.delta)
    entries = compute_delta(alignment)

    report = render_markdown(entries, old_doc, new_doc)

    summary = (
        f"**Old:** {len(old_doc.elements)} elements across {old_doc.page_count} pages  \n"
        f"**New:** {len(new_doc.elements)} elements across {new_doc.page_count} pages  \n"
        f"**Changes:** {len(entries)} total "
        f"({sum(1 for e in entries if e.kind.value == 'added')} added, "
        f"{sum(1 for e in entries if e.kind.value == 'removed')} removed, "
        f"{sum(1 for e in entries if e.kind.value == 'modified')} modified)"
    )

    # Store for chat — we use gr.State, so return via separate mechanism
    _last_run["old_doc"] = old_doc
    _last_run["new_doc"] = new_doc
    _last_run["entries"] = entries

    return report, summary


# Module-level state to share between delta and chat tabs
_last_run: dict[str, Any] = {}


def _chat_respond(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Answer a question grounded in the uploaded documents + delta."""
    if not message.strip():
        return "", history

    cfg = load_config()

    if "old_doc" not in _last_run:
        history.append({"role": "assistant", "content": "Please run a delta analysis first (Analysis tab)."})
        return "", history

    if not cfg.chat.llm_api_key:
        history.append({
            "role": "assistant",
            "content": (
                "LLM not configured. Set `PID_LLM_API_KEY` to enable chat. "
                "You can view the delta report in the Analysis tab without the API key."
            ),
        })
        return "", history

    from src.chat.answer import GroundedQA
    from src.chat.index import RetrievalIndex
    from src.chat.llm import LLMClient

    old_doc: CanonicalDocument = _last_run["old_doc"]
    new_doc: CanonicalDocument = _last_run["new_doc"]
    entries: list[DeltaEntry] = _last_run["entries"]

    # Lazily build index + QA on first question
    if "qa" not in _last_run:
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

        _last_run["qa"] = GroundedQA(index, llm, cfg.chat)

    qa: GroundedQA = _last_run["qa"]
    answer = qa.answer(message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer.text})
    return "", history


def build_app(config: Config | None = None) -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    cfg = config or load_config()

    with gr.Blocks(
        title="PID Delta Chat",
    ) as app:
        gr.Markdown("# PID Revision Delta Analysis")

        with gr.Tabs():
            # ---- Tab 1: Analysis ----
            with gr.Tab("Analysis"):
                gr.Markdown("Upload two revisions of a P&ID to compare them.")
                with gr.Row():
                    old_file = gr.File(
                        label="Original PID (old)",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath",
                    )
                    new_file = gr.File(
                        label="Revised PID (new)",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath",
                    )

                run_btn = gr.Button("Run Delta Analysis", variant="primary")
                summary = gr.Markdown("*Upload two PDFs and click Run Delta Analysis.*")
                report = gr.Markdown(
                    value="",
                    label="Delta Report",
                    height=600,
                )

                run_btn.click(
                    fn=_run_delta,
                    inputs=[old_file, new_file],
                    outputs=[report, summary],
                )

            # ---- Tab 2: Q&A ----
            with gr.Tab("Q&A"):
                if not cfg.chat.llm_api_key:
                    gr.Markdown(
                        "**Chat disabled** — set `PID_LLM_API_KEY` to enable "
                        "grounded Q&A over your documents."
                    )
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="Ask about the delta report...",
                    label="Question",
                )
                _clear = gr.ClearButton([msg, chatbot])
                msg.submit(
                    fn=_chat_respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                )

    return app  # type: ignore[no-any-return]


def launch(config: Config | None = None) -> None:
    """Build and launch the Gradio app."""
    cfg = config or load_config()
    app = build_app(cfg)
    app.launch(
        server_name=cfg.web.host,
        server_port=cfg.web.port,
        share=cfg.web.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
