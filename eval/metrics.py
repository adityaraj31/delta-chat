"""Eval harness skeleton — scores delta accuracy and answer quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.delta.engine import DeltaEntry


def load_ground_truth(path: Path) -> dict[str, Any]:
    """Load a hand-labelled ground truth JSON file."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def score_delta(
    predicted: list[DeltaEntry],
    ground_truth: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute precision / recall / F1 for delta classification.

    Ground truth entries are dicts with:
        {"kind": "added"|"removed"|"modified", "tag_or_note": "PI-101-01", ...}

    Matching criterion: same kind AND same tag/note number (if present in GT).
    """
    tp = 0
    fp = 0
    fn = 0

    gt_matched: set[int] = set()

    for pred in predicted:
        found = False
        for i, gt in enumerate(ground_truth):
            if i in gt_matched:
                continue
            if gt["kind"] != pred.kind.value:
                continue
            # Match on structured ID if present
            gt_id = gt.get("tag_or_note")
            if gt_id:
                pred_id = None
                if (
                    (pred.old and pred.old.tag_number == gt_id)
                    or (pred.new and pred.new.tag_number == gt_id)
                    or (pred.old and pred.old.note_number is not None and f"note:{pred.old.note_number}" == gt_id)
                    or (pred.new and pred.new.note_number is not None and f"note:{pred.new.note_number}" == gt_id)
                ):
                    pred_id = gt_id
                if pred_id is None:
                    continue
            found = True
            gt_matched.add(i)
            break
        if found:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - len(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_answer(
    answer_text: str,
    ground_truth_answer: str,
    required_citations: list[str] | None = None,
) -> dict[str, Any]:
    """Score a chat answer against ground truth.

    Returns a dict with exact_match, citation presence, and a simple overlap score.
    """
    import re
    citations_found = re.findall(r"\[(?:PID:[^\]]+|delta:\d+)\]", answer_text)
    has_citations = len(citations_found) > 0
    required_cite_present = True
    if required_citations:
        for req in required_citations:
            if not any(req in c for c in citations_found):
                required_cite_present = False
                break

    return {
        "exact_match": answer_text.strip().lower() == ground_truth_answer.strip().lower(),
        "has_citations": has_citations,
        "citation_count": len(citations_found),
        "required_citations_present": required_cite_present,
    }


def run_eval(
    delta_metrics: dict[str, float],
    answer_metrics: list[dict[str, Any]],
) -> None:
    """Print a summary scorecard."""
    print("\n=== Eval Scorecard ===\n")
    print("Delta Metrics:")
    for k, v in delta_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    if answer_metrics:
        print("\nAnswer Metrics (averaged):")
        avg: dict[str, float] = {}
        for m in answer_metrics:
            for k, v in m.items():
                if isinstance(v, (bool, int, float)):
                    avg.setdefault(k, 0.0)
                    avg[k] += float(v)
        n = len(answer_metrics)
        for k, v in avg.items():
            print(f"  {k}: {v / n:.3f}")
    print()
