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
) -> dict[str, Any]:
    """Compute precision / recall / F1 for delta classification.

    Ground truth entries are dicts with:
        {"kind": "added"|"removed"|"modified", "tag_or_note": "PI-101-01", ...}

    Matching criterion: same kind AND same tag/note number (if present in GT).
    """
    tp = 0
    fp = 0
    fn = 0

    gt_matched: set[int] = set()
    
    fp_items = []
    fn_items = []

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
            desc = pred.description
            if pred.old and pred.old.tag_number: desc = f"({pred.old.tag_number}) " + desc
            elif pred.new and pred.new.tag_number: desc = f"({pred.new.tag_number}) " + desc
            fp_items.append({"kind": pred.kind.value, "desc": desc})

    for i, gt in enumerate(ground_truth):
        if i not in gt_matched:
            fn += 1
            fn_items.append(gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "tp": tp, 
        "fp": fp, 
        "fn": fn,
        "fp_items": fp_items,
        "fn_items": fn_items
    }


def score_answer(
    question: str,
    answer_text: str,
    ground_truth_answer: str,
    llm: Any = None,
    required_citations: list[str] | None = None,
) -> dict[str, Any]:
    """Score a chat answer against ground truth.
    
    Uses exact_match as a baseline, and LLM-as-a-judge if provided.
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
                
    result = {
        "question": question,
        "generated": answer_text,
        "ground_truth": ground_truth_answer,
        "exact_match": answer_text.strip().lower() == ground_truth_answer.strip().lower(),
        "has_citations": has_citations,
        "citation_count": len(citations_found),
        "required_citations_present": required_cite_present,
    }
    
    if llm:
        prompt = f"""You are an expert AI judge. Evaluate the generated answer against the ground truth answer for the following question.
Question: {question}
Ground Truth: {ground_truth_answer}
Generated Answer: {answer_text}

Score the generated answer on two criteria:
1. Correctness: Does it convey the same core information as the ground truth? (1 for yes, 0 for no)
2. Groundedness: Does it avoid hallucinating information not present in the ground truth or the likely source document? (1 for yes, 0 for no)

Output ONLY a JSON object with keys "correctness" and "groundedness", and integer values 0 or 1.
"""
        try:
            resp = llm.chat([
                {"role": "system", "content": "You are a strict, objective evaluation system. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ], temperature=0.0)
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                result["llm_correctness"] = float(parsed.get("correctness", 0))
                result["llm_groundedness"] = float(parsed.get("groundedness", 0))
        except Exception:
            pass

    return result


def run_eval(
    delta_metrics: dict[str, Any],
    answer_metrics: list[dict[str, Any]],
) -> None:
    """Print a summary scorecard and a candid failure table."""
    print("\n" + "="*50)
    print("=== Eval Scorecard ===")
    print("="*50 + "\n")
    
    print("--- DELTA METRICS ---")
    print(f"  Precision: {delta_metrics['precision']:.3f}  (TP: {delta_metrics['tp']}, FP: {delta_metrics['fp']})")
    print(f"  Recall:    {delta_metrics['recall']:.3f}  (TP: {delta_metrics['tp']}, FN: {delta_metrics['fn']})")
    print(f"  F1 Score:  {delta_metrics['f1']:.3f}")
    
    if delta_metrics['fp'] > 0 or delta_metrics['fn'] > 0:
        print("\n  CANDID FAILURE TABLE (Deltas):")
        if delta_metrics['fp'] > 0:
            print("  - False Positives (Invented or Mismatched):")
            for item in delta_metrics['fp_items']:
                print(f"      [{item['kind']}] {item.get('desc', '')}")
        if delta_metrics['fn'] > 0:
            print("  - False Negatives (Missed True Changes):")
            for item in delta_metrics['fn_items']:
                tag = item.get("tag_or_note", "Unknown")
                print(f"      [{item.get('kind')}] {tag}")

    if answer_metrics:
        print("\n--- CHAT METRICS (Averaged) ---")
        avg: dict[str, float] = {}
        for m in answer_metrics:
            for k, v in m.items():
                if isinstance(v, (bool, int, float)) and not isinstance(v, str):
                    avg.setdefault(k, 0.0)
                    avg[k] += float(v)
        n = len(answer_metrics)
        for k, v in avg.items():
            print(f"  {k}: {v / n:.3f}")
            
        # Failures
        chat_failures = []
        for m in answer_metrics:
            # If LLM judge scored 0 on correctness, or exact match is false when no LLM judge
            failed = False
            if "llm_correctness" in m:
                if m["llm_correctness"] < 1.0: failed = True
            elif not m.get("exact_match"):
                failed = True
                
            if not m.get("required_citations_present"):
                failed = True
                
            if failed:
                chat_failures.append(m)
                
        if chat_failures:
            print("\n  CANDID FAILURE TABLE (Chat):")
            for f in chat_failures:
                print(f"  - Q: {f['question']}")
                print(f"    Expected: {f['ground_truth']}")
                print(f"    Got:      {f['generated']}")
                if 'llm_correctness' in f:
                    print(f"    Scores -> Correctness: {f.get('llm_correctness')}, Groundedness: {f.get('llm_groundedness')}, Required Cites: {f.get('required_citations_present')}")
                print()

    print("="*50 + "\n")
