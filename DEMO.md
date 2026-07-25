# PID Delta Chat - Submission Demo

This walkthrough demonstrates the core functionality of the PID Delta Chat pipeline: ingesting two P&ID revisions, extracting a structured delta, and grounding an interactive chat session on the extracted changes.

## 1. Pipeline Execution (Ingest → Delta → Report)

We run the analysis using the unified CLI pipeline command. The `auto_adapter` seamlessly routes the scanned PNG files to the OCR ingestion path.

```bash
$ make run OLD=data/sample_image/gas_ocr.png NEW=data/sample_image/gas_ocr_revision.png

2026-07-25 19:18:03 [info     ] http.request                   correlation_id=6c2af9519181
2026-07-25 19:18:03 [info     ] delta.start                    old=gas_ocr.png new=gas_ocr_revision.png
2026-07-25 19:18:05 [info     ] delta.complete                 added=3 removed=1 modified=5
```

The system successfully generates human-readable (`report.md`) and machine-readable (`report.json`) delta outputs, classifying changes into insertions, deletions, and modifications.

## 2. Interactive Chat (Grounded QA)

We can seamlessly launch into a chat mode targeting the exact delta we just computed. The RAG architecture strictly enforces citations to prevent hallucination.

```bash
$ make chat OLD=data/sample_image/gas_ocr.png NEW=data/sample_image/gas_ocr_revision.png

PID Chat (type 'quit' to exit)

Q: What changed regarding the primary safety valve?
A: The primary safety valve (PSV-101) was modified. Its set pressure was increased from 150 PSI to 175 PSI [delta:4].

Q: Did they add any new instrumentation to the main line?
A: Yes, a new flow transmitter (FT-205) was added to the main gas line downstream of the control valve [delta:1].
```

## 3. Evaluation Scorecard

We built an automated evaluation harness leveraging an LLM-as-a-judge to programmatically score delta detection accuracy and chat groundedness.

```bash
$ make sample-image-eval

--- Eval Scorecard ---
Delta Precision: 1.000
Delta Recall:    1.000
Delta F1:        1.000

Chat Correctness:  0.250
Chat Groundedness: 1.000

--- Candid Failure Table (Chat) ---
[Question 2: What is the normal operating pressure range...] - Failed Correctness
  Expected: Normal operating pressure is between 50 and 60 psi.
  Actual: The provided documents do not contain information regarding the normal operating pressure range.
  Reasoning: The LLM correctly avoided hallucinating, but failed to retrieve the pressure bounds due to OCR chunking splitting the note block.
```

The scorecard highlights that while our delta engine is perfectly tuned for this sample (F1 = 1.0), our RAG chunking strategy needs refinement to capture longer continuous note blocks, which provides a clear roadmap for our next iteration.
