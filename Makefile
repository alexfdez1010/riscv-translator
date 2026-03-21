.PHONY: all sync test test-python repair clean

all: sync

sync:
	uv sync --dev

test: test-python

test-python:
	uv run pytest tests/ -v --tb=short

repair:
	@echo "Usage: make repair INPUT=initial_code/ssw.c OUTPUT=output/ssw.repaired.c"
	@echo "Example: uv run python -m src.repair initial_code/ssw.c output/ssw.repaired.c"
	@test -n "$(INPUT)" && test -n "$(OUTPUT)" && uv run python -m src.repair $(INPUT) $(OUTPUT) || true

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	cd initial_code && make clean 2>/dev/null || true
	cd highway_port && make clean 2>/dev/null || true
