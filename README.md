# RISC-V Translator

LLM-driven pipeline for translating the [Striped Smith-Waterman Library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library) (SSW) from x86 SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions.

## Overview

The SSW library implements a fast SIMD-accelerated Smith-Waterman sequence alignment algorithm using SSE2 intrinsics (`__m128i`, `_mm_*`). This project provides two approaches to port it to RISC-V:

| Approach | Directory | Method | Status |
|----------|-----------|--------|--------|
| **Automated repair** | `src/` + `initial_code/` | LLM incrementally patches `ssw.c` via `sse2rvv.h` | Production pipeline |
| **Manual Highway port** | `highway_port/` | Complete rewrite using Google Highway | Reference implementation |

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) (for RISC-V cross-compilation and QEMU)
- Docker image: `luispimo/riscv-toolchain:arm64-2025-10-20`

### Setup

```bash
# Install Python dependencies
uv sync --dev

# Copy .env and configure (optional — defaults work for local use)
cp .env.example .env
```

### Run the Repair Agent

The repair agent takes an SSE2-based `ssw.c` and iteratively patches it to compile and run on RISC-V:

```bash
# Repair ssw.c → produce a working RISC-V version
uv run python -m src.repair initial_code/ssw.c output/ssw.repaired.c
```

The agent will:
1. Preprocess the code (fix `sizeof(__m128i)`, inject RVV helpers)
2. Use an LLM to generate targeted diffs fixing compilation errors
3. Validate each patch via Docker + QEMU emulation
4. Optionally validate on real RISC-V hardware via SSH
5. Iterate until the code compiles and runs correctly

### Run Tests

```bash
uv run pytest tests/ -v --tb=short
```

## Architecture

### Automated Repair Pipeline

```
initial_code/ssw.c (SSE2)
        │
        ▼
┌─────────────────────┐
│  preprocess_rvv_compat │  Mechanical fixes (sizeof, pointer arith)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    LLM Repair Loop   │  ReAct agent: observe error → generate diff → validate
│  (src/repair.py)     │
└─────────────────────┘
        │          ▲
        ▼          │
┌─────────────────────┐
│  Docker + QEMU       │  Compile with riscv64-unknown-elf-gcc
│  Validation          │  Run under qemu-riscv64 (vlen=128)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SSH Hardware        │  Compile with clang on real RISC-V
│  Validation          │  Run benchmark on actual hardware
└─────────────────────┘
        │
        ▼
    output/ssw.repaired.c (RVV)
```

### Key Components

| Module | Purpose |
|--------|---------|
| `src/repair.py` | ReAct-style repair agent with multi-step LLM interaction |
| `src/prompts.py` | LLM prompt construction + `preprocess_rvv_compat()` preprocessor |
| `src/diff_utils.py` | Robust unified-diff parser tolerant of LLM formatting mistakes |
| `src/validators.py` | Docker/QEMU emulation + SSH hardware validators |
| `src/fitness.py` | SSW-specific validation (creates workspace, runs Docker) |
| `src/llm_utils.py` | LLM client with OpenRouter fallback |
| `src/config.py` | All configuration constants (env-var overridable) |

### Translation Layer: sse2rvv.h

The `initial_code/sse2rvv.h` header translates SSE2 intrinsics to RVV equivalents. Key challenges it addresses:

- **Sizeless types**: On RVV, `__m128i` maps to `vint32m1_t` (a sizeless type)
- **No `sizeof(__m128i)`**: Must use `_SSW_VEC_BYTES` runtime macro
- **No pointer arithmetic**: `ptr + j` on `__m128i*` is illegal; use byte offsets
- **No struct fields**: `__m128i` cannot be a struct member; use `uint8_t*` buffers

The preprocessor (`preprocess_rvv_compat`) handles these mechanical fixes before the LLM tackles semantic issues.

### Manual Highway Port (Reference)

The `highway_port/` directory contains a complete manual port using [Google Highway](https://github.com/google/highway). This port:

- Uses `FixedTag<uint8_t, 16>` for exact 128-bit vectors
- Changes profile struct from `__m128i*` to typed pointers (`uint8_t*`, `int16_t*`)
- Produces output identical to the original x86 SSW library

To build and test the Highway port:

```bash
docker run --rm \
  --mount type=bind,source=$(pwd),target=/workspace \
  -w /workspace/highway_port \
  luispimo/riscv-toolchain:arm64-2025-10-20 \
  bash -lc 'make clean && make rvv_example && \
    qemu-riscv64 -cpu rv64,v=on,vext_spec=v1.0,vlen=128,rvv_ta_all_1s=on ./rvv_example'
```

## Configuration

All settings are in `src/config.py` and overridable via environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:30000/v1` | OpenAI-compatible LLM endpoint |
| `OPENROUTER_API_KEY` | *(empty)* | Fallback API key for OpenRouter |
| `DOCKER_IMAGE` | `luispimo/riscv-toolchain:arm64-2025-10-20` | RISC-V cross-compilation image |
| `SSH_HOST` | `final` | SSH host for real hardware validation |
| `VLEN` | `128` | RISC-V vector register width |
| `REACT_MAX_STEPS` | `15` | Max LLM repair iterations |

## Project Structure

```
riscv-translator/
├── src/                    Python package (repair agent pipeline)
├── tests/                  Test suite
├── initial_code/           SSW source files (SSE2 starting point)
├── highway_port/           Manual Highway port (reference)
├── highway/                Vendored Google Highway library
├── dataset/                FASTA test data
├── docs/                   RVV reference material
├── Makefile                Top-level build targets
├── pyproject.toml          Python packaging (uv)
├── .env.example            Environment variables template
└── CLAUDE.md               AI assistant instructions
```

## License

The SSW library is MIT/BSD licensed (see source file headers).
The Highway library is Apache-2.0 licensed.
