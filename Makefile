.PHONY: install lint typecheck test run chat web serve eval sample-image-eval clean help

help:
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make lint             Run ruff linter"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make test             Run pytest"
	@echo "  make web              Launch Gradio web UI"
	@echo "  make serve            Alias for web"
	@echo "  make run OLD=x NEW=y  Run pipeline (ingest → delta → report)"
	@echo "  make chat OLD=x NEW=y Interactive CLI chat"
	@echo "  make eval OLD=x NEW=y GT=z  Run eval harness"
	@echo "  make sample-image-eval  Run bundled OCR image eval"
	@echo "  make clean            Remove build artifacts"

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ main.py eval/ tests/

typecheck:
	uv run mypy src/ main.py eval/

test:
	uv run pytest tests/ -v

web:
	uv run python main.py web

serve: web

run:
	@if [ -z "$(OLD)" ] || [ -z "$(NEW)" ]; then \
		echo "Usage: make run OLD=path/to/old.pdf NEW=path/to/new.pdf"; exit 1; \
	fi
	uv run python main.py pipeline --old $(OLD) --new $(NEW)

chat:
	@if [ -z "$(OLD)" ] || [ -z "$(NEW)" ]; then \
		echo "Usage: make chat OLD=path/to/old.pdf NEW=path/to/new.pdf"; exit 1; \
	fi
	uv run python main.py chat --old $(OLD) --new $(NEW)

eval:
	@if [ -z "$(OLD)" ] || [ -z "$(NEW)" ] || [ -z "$(GT)" ]; then \
		echo "Usage: make eval OLD=path/to/old.pdf NEW=path/to/new.pdf GT=path/to/ground_truth.json"; exit 1; \
	fi
	uv run python eval/run_eval.py --old $(OLD) --new $(NEW) --gt-delta $(GT)

sample-image-eval:
	uv run python eval/run_eval.py \
		--old data/sample_image/gas_ocr.png \
		--new data/sample_image/gas_ocr_revision.png \
		--gt-delta eval/datasets/sample_image_delta_gt.json \
		--gt-answers eval/datasets/sample_image_qa_gt.json

clean:
	rm -rf output/ .mypy_cache/ .ruff_cache/ __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
