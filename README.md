# RISC-V Translator

LLM-driven pipeline for translating C/C++ libraries that use x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions via [Google Highway](https://github.com/google/highway).

## Overview

The pipeline is **generic** — it works with any SSE-based codebase, not just a specific library. The LLM iteratively fixes compiler errors based on feedback until the translated code compiles and runs correctly.

```
SSE source code (input directory)
        |
        v
+---------------------+
|  LLM Translation    |  SSE intrinsics -> Google Highway C++
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
  Translated output file (Highway C++)
```

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

On success, the `output/` directory will contain all files needed to compile
and run the translated program (source files + vendored Highway library).

The pipeline will:
1. Use an LLM to translate SSE intrinsics to Google Highway equivalents
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
│   ├── diff_utils.py       Robust search/replace parser
│   ├── llm_utils.py        LLM client (OpenRouter)
│   ├── llm_types.py        Message/LLM protocol types
│   ├── config.py           Configuration (env-var overridable)
│   └── logger.py           Terminal logger
├── tests/                  Test suite
├── initial_code/           Example: SSW library (SSE2 source) + sse2rvv.h
├── docs/                   RVV reference material for LLM context
├── Makefile                Top-level build targets
├── pyproject.toml          Python packaging (uv)
└── .env.example            Environment variables template
```

## License

The SSW library is MIT/BSD licensed (see source file headers).
The Highway library is Apache-2.0 licensed.
