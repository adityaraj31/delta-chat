"""Run the eval harness over ground truth datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe

from eval.metrics import load_ground_truth, run_eval, score_answer, score_delta
from src.config import load_config
from src.delta.align import align_documents
from src.delta.engine import compute_delta
from src.ingest.base import auto_adapter
from src.observability.tracing import flush, init_tracer


@observe(name="eval")
def run_eval_pipeline(args, cfg, old_path, new_path):
    old_doc = auto_adapter(old_path).ingest(old_path)
    new_doc = auto_adapter(new_path).ingest(new_path)

    alignment = align_documents(old_doc, new_doc, config=cfg.delta)
    entries = compute_delta(alignment)

    gt_delta = load_ground_truth(Path(args.gt_delta))
    delta_metrics = score_delta(entries, gt_delta["entries"])

    answer_metrics: list[dict[str, object]] = []
    if args.gt_answers and cfg.chat.llm_api_key:
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

        gt_qa = load_ground_truth(Path(args.gt_answers))
        for item in gt_qa["questions"]:
            ans = qa.answer(item["question"])
            answer_metrics.append(score_answer(item["question"], ans.text, item["answer"], llm, item.get("citations")))

    run_eval(delta_metrics, answer_metrics)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PID delta eval")
    parser.add_argument("--old", required=True, help="Old revision document/image")
    parser.add_argument("--new", required=True, help="New revision document/image")
    parser.add_argument("--gt-delta", required=True, help="Ground truth delta JSON")
    parser.add_argument("--gt-answers", default=None, help="Ground truth Q&A JSON")
    args = parser.parse_args()

    cfg = load_config()
    from src.observability.logging import setup_logging, new_correlation_id
    setup_logging(cfg.log_level)
    cid = new_correlation_id()
    init_tracer(cfg.output_dir / "traces")
    old_path = Path(args.old)
    new_path = Path(args.new)

    from langfuse import propagate_attributes
    with propagate_attributes(session_id=cid):
        run_eval_pipeline(args, cfg, old_path, new_path)

    flush()

if __name__ == "__main__":
    main()
