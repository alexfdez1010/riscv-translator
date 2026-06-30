# RISC-V Translator

> 📄 This repository contains the source code accompanying the paper
> **"Porting the Striped Smith-Waterman Library to RISC-V via LLM-Driven
> Translation"** (Fernández Camello, Prieto-Matias & Garcia Sanchez),
> published at the **XXXVI Jornadas SARTECO 2026**.
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21064372.svg)](https://doi.org/10.5281/zenodo.21064372)
> See [Citation](#citation) for how to cite this work.

Automated pipeline that translates C/C++ code using x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions. An LLM reads compiler errors and produces source patches in a loop until the code compiles and runs on RISC-V — no manual porting needed.

Uses **[sse2rvv](https://github.com/pattonkan/sse2rvv)** (a header-only SSE-to-RVV compatibility layer) and the **[SSW library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)** as its test case.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **Python >= 3.12**
- **[Docker](https://www.docker.com/)** — for RISC-V cross-compilation and Spike simulator
- **Docker image**: `luispimo/riscv-toolchain:arm64-2025-10-20`
- **[OpenRouter](https://openrouter.ai/) API key** — for LLM access

## Setup

```bash
git clone <repository-url>
cd riscv-translator
uv sync --dev
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

Pull the Docker toolchain image:

```bash
docker pull luispimo/riscv-toolchain:arm64-2025-10-20
```

## Programs

### 1. `translate` — Phase 1: SSE to sse2rvv.h

Translates SSE intrinsics to their RVV equivalents via the `sse2rvv.h` compatibility header. Output uses a fixed 128-bit vector width matching SSE semantics.

```bash
make translate
# Reads from initial_code/, translates ssw.c, writes to output/
```

Custom paths:

```bash
make translate SOURCE_DIR=my_code TARGET_FILE=my_file.c OUTPUT_DIR=my_output
```

Direct invocation:

```bash
uv run python -m src.repair <source_dir> <target_file> <output_dir> \
    [--build-command "..."] \
    [--ssh-compile "..."] \
    [--ssh-run "..."] \
    [--max-steps N]
```

**Requires**: Docker running, `OPENROUTER_API_KEY` set.
**Optional**: `SSH_HOST` for real hardware validation.

### 2. `widen` — Phase 2: sse2rvv.h to native RVV

Takes Phase 1 output and replaces `sse2rvv.h` calls with native `<riscv_vector.h>` intrinsics, making the code VLEN-agnostic (adapts to the hardware's vector register width at runtime).

```bash
make widen
# Reads from translations/sequence-alignment/, writes to widened/
```

Custom paths:

```bash
make widen WIDEN_SOURCE_DIR=output WIDEN_OUTPUT_DIR=my_widened
```

**Requires**: Docker running, `OPENROUTER_API_KEY` set, `SSH_HOST` configured (for benchmarking during widening).

### 3. `check` — Validate a translation

Validates a translated output directory by:
1. Running the original SSE code on Intel to get reference output
2. Running the translation under Spike at every power-of-2 VLEN from 128 up to `MAX_VLEN`
3. Comparing each VLEN run against the Intel reference
4. Running on real RISC-V hardware via SSH and comparing

```bash
make check OUTPUT_DIR=translations/sequence-alignment
make check OUTPUT_DIR=translations/sequence-alignment-widened
make check OUTPUT_DIR=translations/sequence-alignment-widened-auto
```

Options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | *(required)* | Directory with translated source files |
| `MAX_VLEN` | `4096` | Highest VLEN to test (powers of 2, Spike max is 4096) |
| `CHECK_DATASET` | `10k.fa` | FASTA dataset for the check run |

```bash
make check OUTPUT_DIR=output MAX_VLEN=2048 CHECK_DATASET=1M.fa
```

**Requires**: Docker running, `SSH_HOST` configured, `SSH_JUMP_HOST` set (for Intel reference).

### 4. `benchmark` — Run performance benchmarks

Runs all code variants on RISC-V hardware (10 runs each), validates correctness against Intel SSE reference, and writes results to `benchmarks.csv`.

Comparisons performed:
- naive (scalar) vs `sequence-alignment` — 1k, 10k, 100k datasets
- `sequence-alignment` vs `sequence-alignment-widened-auto` — 10k, 100k, 1M
- `sequence-alignment-widened` vs `sequence-alignment-widened-auto` — 10k, 100k, 1M

Already-completed experiments are skipped (incremental mode).

```bash
make benchmark
```

**Requires**: `SSH_HOST` configured, `SSH_JUMP_HOST` set (for Intel reference).

### 5. `graph` — Generate performance graphs

Reads `benchmarks.csv` and generates comparison plots in the `graphs/` directory:

- `boxplot_combined.png` — execution time distributions
- `bar_translated_variants.png` — variant comparison
- `speedup_vs_naive.png` — speedup over scalar baseline
- `speedup_widened_vs_sse.png` — widened vs sse2rvv speedup
- `scaling_line_chart.png` — scaling across dataset sizes

```bash
make graph
```

**Requires**: `benchmarks.csv` populated (run `make benchmark` first). Uses `matplotlib`.

### 6. `test` — Run the test suite

```bash
make test
# or: uv run pytest tests/ -v --tb=short
```

## Configuration

All settings live in `src/config.py` and are overridable via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | [OpenRouter](https://openrouter.ai/) API key |
| `OPENROUTER_MODEL` | `minimax/minimax-m2.7` | LLM model (recommended) |
| `LLM_TEMPERATURE` | `0` | Sampling temperature (0 = deterministic) |
| `LLM_MAX_COMPLETION_TOKENS` | `0` | Max tokens per LLM response (0 = no limit) |
| `DOCKER_IMAGE` | `luispimo/riscv-toolchain:arm64-2025-10-20` | Docker image with RISC-V toolchain + Spike |
| `RISCVCC` | `riscv64-unknown-elf-gcc` | RISC-V C compiler (used in Docker build commands) |
| `SIMULATOR` | `spike --isa=rv64gcv pk64` | Spike simulator command |
| `SSH_HOST` | `final` | SSH host for RISC-V hardware |
| `SSH_CC` | `clang` | C compiler on the SSH RISC-V host |
| `SSH_JUMP_HOST` | *(unset)* | SSH jump host for Intel reference runs (when unset, Intel runs locally) |
| `REMOTE_DIR` | `/tmp/sse2rvv` | Working directory on remote hosts |
| `DATASETS_DIR` | `datasets/` | Directory containing FASTA test data |
| `REACT_MAX_STEPS` | `15` | Max LLM repair iterations |
| `LLM_VALIDATION_RETRIES` | `2` | Retries per step on edit/parse failure |
| `VALIDATION_TIMEOUT_SECONDS` | `120` | Timeout for Docker/SSH commands |

## Cost

Phase 1 costs ~**$0.05 USD** and Phase 2 (widening) another ~**$0.05 USD** using the recommended model (`minimax/minimax-m2.7` via OpenRouter). Each LLM iteration costs about $0.01 USD.

## Translations

The `translations/` directory contains three pre-built variants:

| Directory | Phase | Intrinsics | VLEN | How |
|-----------|-------|------------|------|-----|
| `sequence-alignment/` | 1 | `sse2rvv.h` | Fixed 128-bit | `src/repair.py` (automated) |
| `sequence-alignment-widened/` | 2 | `<riscv_vector.h>` | Agnostic | Manual porting |
| `sequence-alignment-widened-auto/` | 2 | `<riscv_vector.h>` | Agnostic | `src/widen.py` (automated) |

## Project Structure

```
src/                        Python package
  repair.py                 Phase 1 — LLM compile-fix loop
  widen.py                  Phase 2 — sse2rvv.h to native RVV
  check.py                  Validation across VLEN sizes
  benchmark.py              Performance benchmarking (10-run, incremental)
  graph.py                  Plot generation from benchmarks.csv
  prompts.py                LLM prompts (generic, not library-specific)
  validators.py             DockerValidator (Spike) + SSHValidator (hardware)
  search_replace.py         Search/replace block parser
  llm_utils.py              OpenRouter client with retry/backoff
  llm_types.py              Message/LLM protocol types
  config.py                 Configuration (env-var overridable)
  logger.py                 Terminal logger
initial_code/               SSW source (SSE2, zlib removed) + sse2rvv.h subset
translations/               Pre-built translation variants
datasets/                   FASTA test data (1k, 10k, 100k, 1M, 10M)
tests/                      pytest test suite
docs/riscv-reference/       RVV reference material (loaded as LLM context)
graphs/                     Generated benchmark plots
benchmarks.csv              Benchmark results
```

## Citation

This work was published at the **XXXVI Jornadas SARTECO 2026**. If you use this
software or its results, please cite it.

> Fernández Camello, A., Prieto-Matias, M., & Garcia Sanchez, C. (2026).
> *Porting the Striped Smith-Waterman Library to RISC-V via LLM-Driven
> Translation*. XXXVI Jornadas SARTECO 2026.
> https://doi.org/10.5281/zenodo.21064372

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21064372.svg)](https://doi.org/10.5281/zenodo.21064372)

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided. BibTeX:

```bibtex
@inproceedings{fernandezcamello2026ssw,
  title     = {Porting the Striped Smith-Waterman Library to RISC-V via LLM-Driven Translation},
  author    = {Fern\'andez Camello, Alejandro and Prieto-Matias, Manuel and Garcia Sanchez, Carlos},
  booktitle = {XXXVI Jornadas SARTECO 2026},
  year      = {2026},
  doi       = {10.5281/zenodo.21064372},
  url       = {https://doi.org/10.5281/zenodo.21064372}
}
```

## License

MIT. See [LICENSE](LICENSE).

Bundled third-party components (SSW library, sse2rvv) are also MIT-licensed.
