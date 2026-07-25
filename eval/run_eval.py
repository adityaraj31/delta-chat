"""Run the eval harness over ground truth datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.metrics import load_ground_truth, run_eval, score_answer, score_delta
from src.config import load_config
from src.delta.align import align_documents
from src.delta.engine import compute_delta
from src.ingest.base import auto_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PID delta eval")
    parser.add_argument("--old", required=True, help="Old revision document/image")
    parser.add_argument("--new", required=True, help="New revision document/image")
    parser.add_argument("--gt-delta", required=True, help="Ground truth delta JSON")
    parser.add_argument("--gt-answers", default=None, help="Ground truth Q&A JSON")
    args = parser.parse_args()

    cfg = load_config()
    old_path = Path(args.old)
    new_path = Path(args.new)

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
            answer_metrics.append(score_answer(ans.text, item["answer"], item.get("citations")))

    run_eval(delta_metrics, answer_metrics)


if __name__ == "__main__":
    main()
