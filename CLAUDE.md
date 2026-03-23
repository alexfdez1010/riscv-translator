# CLAUDE.md

## Purpose

This repository provides an LLM-driven pipeline for translating C/C++ libraries
that use x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions via
Google Highway.  The pipeline is **generic** — it is not tied to any specific
library.  The LLM iteratively fixes compiler errors based on feedback until the
translated code compiles and runs correctly.

The current test case is the Striped Smith-Waterman (SSW) library, but the
pipeline works with any SSE-based codebase.

## How it works

1. **Input**: A directory of C/C++ source files using SSE intrinsics.
2. **Translation**: An LLM translates the target file to Google Highway C++.
3. **Compile-fix loop**: The translated code is compiled in Docker with the
   RISC-V toolchain + QEMU emulator.  Compiler errors are fed back to the LLM
   which produces incremental diffs until compilation succeeds.
4. **Simulator validation**: The binary runs under QEMU to verify correctness.
5. **Hardware validation**: If SSH to real RISC-V hardware is available, the
   code is also compiled and run there.
6. **Output**: The final translated source file.

## Repository layout

```
src/               — Python package (translation agent, validators, LLM client)
initial_code/      — Original SSW source files (SSE2) — example input
highway/           — Vendored Google Highway library
tests/             — Python test suite
dataset/           — FASTA test data (for SSW example)
docs/              — RVV reference material for LLM context
```

## Python environment

This project uses **`uv`** exclusively.

```bash
uv sync          # install deps
uv sync --dev    # install dev deps (pytest)
```

## Key commands

| Command | What it does |
|---------|-------------|
| `make test` | Run Python tests (`uv run pytest tests/ -v --tb=short`) |
| `make sync` | Install/update Python dependencies |
| `make translate SOURCE_DIR=... TARGET_FILE=... OUTPUT_DIR=...` | Run translation pipeline |
| `make clean` | Remove build artifacts |

### Direct invocation

```bash
uv run python -m src.repair <source_dir> <target_file> <output_dir> \
    [--build-command "..."] \
    [--ssh-compile "..."] \
    [--ssh-run "..."] \
    [--max-steps N]
```

## Architecture notes

- **Translation agent**: `src/repair.py` — `TranslationAgent` orchestrates the LLM compile-fix loop.
- **LLM client**: `src/llm_utils.py` — `create_llm()` returns an `LLM` protocol object (local → SSH tunnel → OpenRouter fallback).
- **LLM types**: `src/llm_types.py` — `Message`, `LLM`, `llm_fn` (no external dependency).
- **Config**: `src/config.py` — all tunables, overridable via environment variables (see `.env.example`).
- **Validators**: `src/validators.py` — `DockerValidator` (QEMU emulation) and `SSHValidator` (real hardware).
- **Prompts**: `src/prompts.py` — generic SSE→Highway translation prompts (no library-specific hardcoding).
- **Diff utils**: `src/diff_utils.py` — robust unified-diff parsing tolerant of LLM formatting mistakes.
- **Reference material**: `docs/riscv-reference/reference.md` is the authoritative RVV reference for LLM prompts.

## Testing

```bash
uv run pytest tests/ -v --tb=short        # full suite
uv run pytest tests/test_diff_utils.py -q  # diff parser tests
uv run pytest tests/test_repair.py -q      # translation pipeline tests
```

## Conventions

- Keep changes minimal and focused.
- Prefer editing existing files over creating new ones.
- The pipeline must remain generic — no library-specific hardcoded fixes.
- The LLM solves errors using compiler feedback, not pre-programmed patterns.
- Do not invent APIs or build steps not present in the existing code.
