# CLAUDE.md

## Purpose

This repository provides an LLM-driven pipeline for translating C/C++ libraries
that use x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions using
the **sse2rvv.h** drop-in compatibility header.  The pipeline is **generic** —
it is not tied to any specific library.  The LLM iteratively fixes compiler
errors based on feedback until the translated code compiles and runs correctly.

The current test case is the Striped Smith-Waterman (SSW) library, but the
pipeline works with any SSE-based codebase.

## How it works

1. **Input**: A directory of C/C++ source files using SSE intrinsics.
2. **Pre-processing**: SSE `#include` directives are replaced with `#include "sse2rvv.h"`.
3. **Compile-fix loop**: The code is cross-compiled in Docker with the RISC-V
   toolchain.  The binary runs under the Spike RISC-V ISA simulator.  Compiler
   or runtime errors are fed back to the LLM, which produces minimal
   search/replace diffs until the code compiles and runs successfully.
4. **Simulator validation**: The binary runs under Spike to verify execution.
5. **Hardware validation** *(optional)*: If SSH to real RISC-V hardware is
   available, the code is also compiled and run there.
6. **Correctness check** *(optional)*: Output is compared against an Intel x86
   reference to verify functional equivalence.
7. **Output**: The final translated source files.

## SSW source modifications

The original SSW library depends on `zlib` for reading FASTA files (`gzopen`,
`gzread`, etc.).  In this repository, `initial_code/main.c` has been modified
to use standard C file I/O (`fopen`, `fread`, `fgets`) instead.  This avoids
the `zlib` dependency, which is not available in the bare-metal RISC-V
toolchain (`riscv64-unknown-elf-gcc`).  The core algorithm (`ssw.c`) is
unchanged.

## Vector width

The translated code uses a fixed 128-bit vector width to match SSE semantics,
even if the target hardware has wider registers.  The priority in this phase
was correctness: translating SSE intrinsics while simultaneously widening to
an arbitrary VLEN introduced too many degrees of freedom for the LLM to
converge reliably.  Constraining to 128 bits keeps the task tractable.
Future work: an evolutionary algorithm will optimize the initial translation
to use the hardware's full register width.

## Repository layout

```
src/               — Python package (translation agent, validators, LLM client)
initial_code/      — Original SSW source files (SSE2, zlib removed) + sse2rvv.h subset
translations/      — Successfully translated outputs (see below)
datasets/          — FASTA test data for SSW benchmarking
tests/             — Python test suite (pytest)
docs/              — RVV reference material loaded as LLM context
```

### Translations directory

The `translations/` directory contains three variants of the SSW library
translation, each representing a different stage of the pipeline:

| Directory | Approach | Vector intrinsics | VLEN |
|-----------|----------|-------------------|------|
| `sequence-alignment/` | Phase 1 — automated via `src/repair.py` | `sse2rvv.h` (drop-in compatibility) | Fixed 128-bit |
| `sequence-alignment-widened/` | Phase 2 — manually widened | Native `<riscv_vector.h>` | VLEN-agnostic |
| `sequence-alignment-widened-auto/` | Phase 2 — automatically widened via `src/widen.py` | Native `<riscv_vector.h>` | VLEN-agnostic |

- **`sequence-alignment/`**: The direct output of the Phase 1 repair pipeline.
  Uses `sse2rvv.h` to map SSE intrinsics to RVV equivalents at a fixed 128-bit
  width.  Includes the `sse2rvv.h` header in the output.
- **`sequence-alignment-widened/`**: A manually crafted version that replaces
  `sse2rvv.h` calls with native RVV intrinsics (`<riscv_vector.h>`), making the
  code VLEN-agnostic so it can exploit the full width of the hardware's vector
  registers.
- **`sequence-alignment-widened-auto/`**: The output of the automated widening
  pipeline (`src/widen.py`), which uses an LLM to transform the Phase 1
  sse2rvv.h-based code into native RVV intrinsics.  Functionally equivalent to
  the manual widening but produced automatically.

## Python environment

This project uses **`uv`** exclusively.  Only dependency: `python-dotenv`.

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
| `make check OUTPUT_DIR=...` | Validate a translation output directory |
| `make benchmark BENCHMARK_DATASET=...` | Compare Intel vs RISC-V execution (Intel runs locally unless `SSH_JUMP_HOST` is set) |
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
- **LLM client**: `src/llm_utils.py` — `create_llm()` returns an `LLM` protocol object using OpenRouter (with retry/backoff on 429).
- **LLM types**: `src/llm_types.py` — `Message`, `LLM`, `llm_fn` (no external dependency).
- **Config**: `src/config.py` — all tunables, overridable via environment variables (see `.env.example`).
- **Validators**: `src/validators.py` — `DockerValidator` (Spike simulator) and `SSHValidator` (real hardware).
- **Prompts**: `src/prompts.py` — generic SSE→sse2rvv.h translation prompts (no library-specific hardcoding).  Includes critical rules for `sizeof(__m128i)` → 16-byte SSE semantic width.
- **Search/replace**: `src/search_replace.py` — robust search/replace block parsing and application, tolerant of LLM formatting mistakes (trailing whitespace, markdown fences, fuzzy whitespace matching).
- **Benchmark**: `src/benchmark.py` — runs original SSE code on Intel and translated RVV code on RISC-V, compares outputs.
- **Check**: `src/check.py` — standalone validation tool for output directories.
- **Widening pipeline**: `src/widen.py` — Phase 2 pipeline that transforms sse2rvv.h-based code to native RVV intrinsics (`<riscv_vector.h>`), making it VLEN-agnostic.  Uses a multi-pass LLM compile-fix loop with SSH benchmarking and correctness checking.
- **Reference material**: `docs/riscv-reference/reference.md` is the authoritative RVV reference for LLM prompts.

## Testing

```bash
uv run pytest tests/ -v --tb=short        # full suite
uv run pytest tests/test_search_replace.py -q  # search/replace tests
uv run pytest tests/test_repair.py -q      # translation pipeline tests
```

## Conventions

- Keep changes minimal and focused.
- Prefer editing existing files over creating new ones.
- The pipeline must remain generic — no library-specific hardcoded fixes.
- The LLM solves errors using compiler feedback, not pre-programmed patterns.
- Do not invent APIs or build steps not present in the existing code.
