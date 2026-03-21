# CLAUDE.md

## Purpose

This repository provides an LLM-driven pipeline for translating the Striped Smith-Waterman (SSW) library from x86 SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions. It has two approaches:

1. **Automated repair agent** (`src/repair.py`) — uses an LLM to incrementally patch `ssw.c` via `sse2rvv.h`, validated through Docker/QEMU emulation and SSH to real hardware.
2. **Manual Highway port** (`highway_port/`) — a reference implementation using Google Highway, already complete and verified.

## Repository layout

```
src/               — Python package (repair agent, validators, LLM client)
initial_code/      — Original SSW source files (SSE2) + sse2rvv.h translation layer
highway_port/      — Manual Highway SIMD port (reference implementation)
highway/           — Vendored Google Highway library
tests/             — Python test suite
dataset/           — FASTA test data
docs/              — RVV reference material for LLM context
```

## Python environment

This project uses **`uv`** exclusively. All Python commands go through `uv run`.

```bash
uv sync          # install deps
uv sync --dev    # install dev deps (pytest)
```

## Key commands

| Command | What it does |
|---------|-------------|
| `make test` | Run Python tests (`uv run pytest tests/ -v --tb=short`) |
| `make sync` | Install/update Python dependencies |
| `make repair INPUT=... OUTPUT=...` | Run repair agent |
| `make clean` | Remove build artifacts |

## Architecture notes

- **LLM client**: `src/llm_utils.py` provides `create_llm()` returning an `LLM` protocol object. All modules use this.
- **LLM types**: `src/llm_types.py` defines `Message`, `LLM`, `llm_fn` locally (no external dependency).
- **Config**: All tunables in `src/config.py`, overridable via environment variables (see `.env.example`).
- **Validation**: Docker + QEMU for emulated RISC-V, SSH for real hardware. Both paths used by repair agent.
- **Reference material**: `docs/riscv-reference/reference.md` is the authoritative RVV reference for LLM prompts.

## Testing

```bash
uv run pytest tests/ -v --tb=short        # full suite
uv run pytest tests/test_diff_utils.py -q  # diff parser tests
uv run pytest tests/test_repair.py -q      # repair agent tests
```

## Conventions

- Keep changes minimal and focused.
- Prefer editing existing files over creating new ones.
- The `initial_code/` files use Docker with the RISC-V toolchain image for validation.
- Do not invent APIs or build steps not present in the existing code.
