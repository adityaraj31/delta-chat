"""End-to-end pipeline: ingest → delta report → (optional) chat demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe

from src.config import load_config
from src.observability.logging import setup_logging
from src.observability.tracing import flush, init_tracer


@observe(name="pipeline")
def run_pipeline_logic(old_path: Path, new_path: Path, cfg):
    from src.ingest.base import auto_adapter
    
    old_doc = auto_adapter(old_path).ingest(old_path)
    print(f"Ingested old: {len(old_doc.elements)} elements from {old_path.name}")
    
    new_doc = auto_adapter(new_path).ingest(new_path)
    print(f"Ingested new: {len(new_doc.elements)} elements from {new_path.name}")
    
    from src.delta.align import align_documents
    alignment = align_documents(old_doc, new_doc, config=cfg.delta)
    print(f"Aligned: {len(alignment.matched)} matched, {len(alignment.removed)} removed, {len(alignment.added)} added")
    
    from src.delta.engine import compute_delta
    entries = compute_delta(alignment)
    print(f"Delta: {len(entries)} total changes")
    
    from src.delta.report import write_report
    output_dir = cfg.output_dir
    paths = write_report(entries, old_doc, new_doc, output_dir)
    print("Report written to:")
    for kind, p in paths.items():
        print(f"  {kind}: {p}")
        
    return old_doc, new_doc, entries

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full ingest → delta pipeline."""
    cfg = load_config()
    setup_logging(cfg.log_level)
    init_tracer(cfg.output_dir / "traces")

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        print(f"Error: old document not found: {old_path}", file=sys.stderr)
        sys.exit(1)
    if not new_path.exists():
        print(f"Error: new document not found: {new_path}", file=sys.stderr)
        sys.exit(1)

    old_doc, new_doc, entries = run_pipeline_logic(old_path, new_path, cfg)

    # Optional: run a demo chat question if LLM is configured
    if cfg.chat.llm_api_key:
        _run_chat_demo(cfg, old_doc, new_doc, entries, cfg.output_dir)
    else:
        print("\n(LLM not configured — skipping chat demo. Set PID_LLM_API_KEY to enable.)")

    print("\nDone.")
    flush()

@observe(name="chat_demo")
def _run_chat_demo(cfg, old_doc, new_doc, entries, output_dir):
    """Run a sample chat question to demonstrate grounding."""
    from src.chat.answer import GroundedQA
    from src.chat.index import RetrievalIndex
    from src.chat.llm import LLMClient

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

    qa = GroundedQA(index, llm, cfg.chat)
    question = "What instrument tags were added or removed between the two revisions?"
    answer = qa.answer(question)
    print(f"\nQ: {question}")
    print(f"A: {answer.text}")
    print(f"Citations: {answer.citations}")

@observe(name="chat")
def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat mode over previously generated delta report."""
    cfg = load_config()
    setup_logging(cfg.log_level)
    init_tracer(cfg.output_dir / "traces")

    if not cfg.chat.llm_api_key:
        print("Error: PID_LLM_API_KEY must be set for chat mode.", file=sys.stderr)
        sys.exit(1)

    from src.chat.answer import GroundedQA
    from src.chat.index import RetrievalIndex
    from src.chat.llm import LLMClient
    from src.delta.align import align_documents
    from src.delta.engine import compute_delta
    from src.ingest.base import auto_adapter

    old_path = Path(args.old)
    new_path = Path(args.new)

    old_doc = auto_adapter(old_path).ingest(old_path)
    new_doc = auto_adapter(new_path).ingest(new_path)

    alignment = align_documents(old_doc, new_doc, config=cfg.delta)
    entries = compute_delta(alignment)

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

    qa = GroundedQA(index, llm, cfg.chat)

    print("PID Chat (type 'quit' to exit)\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer = qa.answer(q)
        print(f"A: {answer.text}\n")

    flush()

def cmd_web(_args: argparse.Namespace) -> None:
    """Launch the FastAPI web server."""
    cfg = load_config()
    setup_logging(cfg.log_level)
    from src.web.app import launch
    launch(cfg)

def main() -> None:
    parser = argparse.ArgumentParser(description="PID Delta Chat — revision comparison tool")
    sub = parser.add_subparsers(dest="command")

    p_pipeline = sub.add_parser("pipeline", help="Run ingest → delta → report")
    p_pipeline.add_argument("--old", required=True, help="Path to old revision document/image")
    p_pipeline.add_argument("--new", required=True, help="Path to new revision document/image")

    p_chat = sub.add_parser("chat", help="Interactive Q&A over two revisions")
    p_chat.add_argument("--old", required=True, help="Path to old revision document/image")
    p_chat.add_argument("--new", required=True, help="Path to new revision document/image")

    sub.add_parser("web", help="Launch Gradio web UI")

    args = parser.parse_args()
    if args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
