.PHONY: all sync test translate check benchmark clean

all: sync

sync:
	uv sync --dev

test:
	uv run pytest tests/ -v --tb=short

SOURCE_DIR  ?= initial_code
TARGET_FILE ?= ssw.c
OUTPUT_DIR  ?= output

translate:
	uv run python -m src.repair $(SOURCE_DIR) $(TARGET_FILE) $(OUTPUT_DIR)

check:
	uv run python -m src.check $(OUTPUT_DIR)

BENCHMARK_DATASET ?= 1M.fa
benchmark:
	uv run python -m src.benchmark --dataset $(BENCHMARK_DATASET)

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
