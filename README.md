# RISC-V Translator

LLM-driven pipeline for translating C/C++ libraries that use x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions using the [sse2rvv](https://github.com/pattonkan/sse2rvv) drop-in compatibility header.

## What This Does

This tool takes C/C++ code written with x86 SSE/SSE2 intrinsics and automatically translates it to run on RISC-V hardware. An LLM reads compiler errors, produces minimal source patches, and repeats until the code compiles and runs correctly — no manual porting required.

The translation relies on **[sse2rvv](https://github.com/pattonkan/sse2rvv)**, a header-only library that re-implements SSE/SSE2 intrinsics using RISC-V Vector (RVV) instructions. We include a **subset** of `sse2rvv.h` in this repository — only the intrinsics needed for our current use case — rather than the full upstream header.

### Current Test Case

The library we translated is the **[Complete-Striped-Smith-Waterman-Library (SSW)](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)**, a SIMD-accelerated implementation of the Smith-Waterman algorithm for sequence alignment. The pipeline is generic though — it works with any SSE-based codebase, not just SSW.

## How It Works

```
SSE source code (input directory)
        |
        v
+---------------------+
|  Pre-processing      |  Replace SSE #includes with sse2rvv.h
+---------------------+
        |
        v
+---------------------+
|  LLM Translation    |  SSE intrinsics → sse2rvv.h equivalents
|  & Repair Loop      |  Incremental diffs guided by compiler feedback
|  (src/repair.py)    |
+---------------------+
        |          ^
        v          |
+---------------------+
|  Docker + QEMU      |  Cross-compile with riscv64 toolchain
|  (simulator)        |  Run under qemu-riscv64
+---------------------+
        |
        v
+---------------------+
|  SSH Hardware        |  Compile & run on real RISC-V hardware
|  (optional)         |
+---------------------+
        |
        v
  Translated output file
```

1. **Input**: A directory of C/C++ source files using SSE intrinsics.
2. **Pre-processing**: SSE `#include` directives are replaced with `#include "sse2rvv.h"`.
3. **Compile-fix loop**: The code is compiled in Docker with the RISC-V toolchain + QEMU emulator. Compiler errors are fed back to the LLM, which produces minimal diffs until compilation succeeds.
4. **Simulator validation**: The binary runs under QEMU to verify correctness.
5. **Hardware validation** *(optional)*: If SSH to real RISC-V hardware is available, the code is also compiled and run there.
6. **Output**: The final translated source files.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) (for RISC-V cross-compilation and QEMU)
- Docker image: `luispimo/riscv-toolchain:arm64-2025-10-20`

### Setup

```bash
uv sync --dev
cp .env.example .env   # configure LLM endpoint, SSH host, etc.
```

### Run the Translation Pipeline

```bash
uv run python -m src.repair <source_dir> <target_file> <output_dir> \
    [--build-command "..."] \
    [--ssh-compile "..."] \
    [--ssh-run "..."] \
    [--max-steps N]
```

**Example** with the included SSW library:

```bash
uv run python -m src.repair initial_code ssw.c output/
```

The pipeline will:
1. Use an LLM to translate SSE intrinsics to sse2rvv.h equivalents
2. Validate each patch via Docker + QEMU emulation
3. Feed compiler errors back to the LLM for incremental fixes
4. Optionally validate on real RISC-V hardware via SSH
5. Iterate until the code compiles and runs correctly
6. Write all translated files to the output directory

### Run Tests

```bash
uv run pytest tests/ -v --tb=short
```

## Configuration

All settings are in `src/config.py` and overridable via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | API key for OpenRouter |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | Model to use via OpenRouter |
| `DOCKER_IMAGE` | `luispimo/riscv-toolchain:arm64-2025-10-20` | RISC-V cross-compilation image |
| `RISCVCXX` | `riscv64-unknown-elf-g++` | RISC-V C++ compiler |
| `SSH_HOST` | `final` | SSH host for real hardware validation |
| `VLEN` | `128` | RISC-V vector register width |
| `REACT_MAX_STEPS` | `15` | Max LLM repair iterations |

## Project Structure

```
riscv-translator/
├── src/                    Python package (translation pipeline)
│   ├── repair.py           TranslationAgent — LLM compile-fix loop
│   ├── prompts.py          Generic SSE→sse2rvv.h translation prompts
│   ├── validators.py       DockerValidator (QEMU) + SSHValidator
│   ├── search_replace.py   Robust search/replace parser
│   ├── llm_utils.py        LLM client (OpenRouter)
│   ├── llm_types.py        Message/LLM protocol types
│   ├── config.py           Configuration (env-var overridable)
│   └── logger.py           Terminal logger
├── initial_code/           SSW library source (SSE2) + sse2rvv.h subset
├── tests/                  Test suite
├── dataset/                FASTA test data (for SSW example)
├── docs/                   RVV reference material for LLM context
├── Makefile                Top-level build targets
├── pyproject.toml          Python packaging (uv)
└── .env.example            Environment variables template
```

## Key Dependencies

- **[sse2rvv](https://github.com/pattonkan/sse2rvv)** — Header-only SSE-to-RVV translation layer. We include only the subset of intrinsics required by the current target library.
- **[Complete-Striped-Smith-Waterman-Library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)** — The C library we translated as the first test case.

## License

This project is licensed under the [MIT License](LICENSE).

The bundled third-party components are also MIT-compatible:
- **SSW library** — MIT License (see `initial_code/ssw.c` header)
- **sse2rvv** — MIT License (see [upstream](https://github.com/pattonkan/sse2rvv))
