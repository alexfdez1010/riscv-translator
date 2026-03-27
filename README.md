# RISC-V Translator

LLM-driven pipeline for automatically translating C/C++ libraries that use x86 SSE/SSE2 SIMD intrinsics to RISC-V Vector (RVV) extensions using the [sse2rvv](https://github.com/pattonkan/sse2rvv) drop-in compatibility header.

## What This Does

This tool takes C/C++ code written with x86 SSE/SSE2 intrinsics and automatically translates it to run on RISC-V hardware. An LLM reads compiler errors, produces minimal source patches (as search/replace blocks), and repeats until the code compiles and runs correctly on both an emulator and real hardware — no manual porting required.

The translation relies on **[sse2rvv](https://github.com/pattonkan/sse2rvv)**, a header-only library that re-implements SSE/SSE2 intrinsics using RISC-V Vector (RVV) instructions. We include a **subset** of `sse2rvv.h` in this repository — only the intrinsics needed for the current use case — rather than the full upstream header.

### Current Test Case

The library translated as proof of concept is the **[Complete-Striped-Smith-Waterman-Library (SSW)](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)**, a SIMD-accelerated implementation of the Smith-Waterman algorithm for DNA/protein sequence alignment. The pipeline is generic — it works with any SSE-based codebase, not just SSW.

> **Note on the SSW source code:** The original SSW library uses `zlib` (`gzopen`, `gzread`, etc.) for reading FASTA input files. In this repository, the SSW source (`initial_code/main.c`) has been modified to use standard C file I/O (`fopen`, `fread`, `fgets`) instead. This change was made for compatibility — `zlib` introduces an external dependency that complicates cross-compilation for the bare-metal RISC-V toolchain (`riscv64-unknown-elf-gcc`), which does not include `zlib` by default. The algorithmic core (`ssw.c`) is unchanged.

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
|  LLM Translation    |  SSE intrinsics -> sse2rvv.h equivalents
|  & Repair Loop      |  Incremental diffs guided by compiler feedback
|  (src/repair.py)    |
+---------------------+
        |          ^
        v          |
+---------------------+
|  Docker + Spike     |  Cross-compile with riscv64 toolchain
|  (simulator)        |  Run under Spike RISC-V ISA simulator
+---------------------+
        |
        v
+---------------------+
|  SSH Hardware        |  Compile & run on real RISC-V hardware
|  (optional)         |
+---------------------+
        |
        v
+---------------------+
|  Correctness Check  |  Compare output vs Intel x86 reference
|  (optional)         |
+---------------------+
        |
        v
  Translated output files
```

1. **Input**: A directory of C/C++ source files using SSE intrinsics.
2. **Pre-processing**: SSE `#include` directives are automatically replaced with `#include "sse2rvv.h"`.
3. **Compile-fix loop**: The code is cross-compiled inside Docker with the RISC-V toolchain. The resulting binary is executed under the Spike RISC-V ISA simulator. Compiler or runtime errors are fed back to the LLM, which produces minimal search/replace diffs until the code compiles and runs successfully.
4. **Simulator validation**: The binary runs under Spike to verify it executes without errors.
5. **Hardware validation** *(optional)*: If SSH access to real RISC-V hardware is configured, the code is also compiled and run there.
6. **Correctness check** *(optional)*: If an Intel reference host is available (via SSH jump host), the original SSE code is compiled and run on x86, and its output is compared against the RISC-V translation to verify functional equivalence.
7. **Output**: The final translated source files, written to the output directory.

## Cost

With the recommended model (`minimax/minimax-m2.7` via [OpenRouter](https://openrouter.ai/)), a full translation run of the SSW library costs approximately **$0.05 USD** (5 cents). The typical run completes in 3-8 LLM iterations.

## Vector Width and Future Optimization

The translated code operates with a fixed vector width of **128 bits** (`VLEN=128`), matching the 128-bit width of SSE registers. This is intentional: `sse2rvv.h` preserves exact SSE semantics, so operations like `sizeof(__m128i)`, memory alignment, and lane counts all assume 16-byte (128-bit) vectors. This guarantees functional equivalence with the original x86 code, even when the target RISC-V hardware has wider vector registers (e.g., 256-bit on the SiFive P550 or wider on other implementations).

While this ensures correctness, it means the translation does not exploit the full width of the hardware's vector registers — leaving performance on the table. The priority in this phase was to produce a **provably correct** translation: asking the LLM to simultaneously translate SSE intrinsics *and* widen the vector semantics to an arbitrary `VLEN` introduced too many degrees of freedom, making it difficult for the model to converge on working code. By constraining the translation to 128-bit semantics, the LLM only needs to map SSE operations to their RVV equivalents — a much more tractable task that reliably produces correct results.

In future work, we plan to develop an **evolutionary algorithm** that takes the initial working translation as a starting point and progressively optimizes it to utilize the full `VLEN` of the target hardware, widening vector operations beyond the 128-bit SSE constraint while preserving correctness.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python >= 3.12
- [Docker](https://www.docker.com/) (for RISC-V cross-compilation and Spike simulator)
- Docker image: `luispimo/riscv-toolchain:arm64-2025-10-20`

### Setup

```bash
git clone <repository-url>
cd riscv-translator
uv sync --dev
cp .env.example .env
# Edit .env and add your OpenRouter API key
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

Or using `make`:

```bash
make translate
# equivalent to: make translate SOURCE_DIR=initial_code TARGET_FILE=ssw.c OUTPUT_DIR=output
```

The pipeline will:
1. Load source files and replace SSE headers with `sse2rvv.h`
2. Use an LLM to translate SSE intrinsics to sse2rvv.h equivalents
3. Validate each patch via Docker + Spike emulation
4. Feed compiler/runtime errors back to the LLM for incremental fixes
5. Optionally validate on real RISC-V hardware via SSH
6. Optionally compare output against an Intel x86 reference
7. Write all translated files to the output directory

### Validate a Previous Translation

```bash
make check OUTPUT_DIR=translations/sequence-alignment
```

### Run Benchmarks

```bash
make benchmark BENCHMARK_DATASET=1M.fa
```

This compiles and runs the original SSE code on an Intel host and the translated RVV code on a RISC-V host, comparing outputs and execution times.

### Run Tests

```bash
make test
# or: uv run pytest tests/ -v --tb=short
```

## Configuration

All settings are in `src/config.py` and overridable via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | API key for [OpenRouter](https://openrouter.ai/) |
| `OPENROUTER_MODEL` | `minimax/minimax-m2.7` | LLM model to use via OpenRouter (recommended) |
| `LLM_TEMPERATURE` | `0` | Sampling temperature (0 = deterministic) |
| `LLM_MAX_COMPLETION_TOKENS` | `5000` | Max tokens per LLM response |
| `DOCKER_IMAGE` | `luispimo/riscv-toolchain:arm64-2025-10-20` | Docker image with RISC-V toolchain + Spike |
| `RISCVCC` | `riscv64-unknown-elf-gcc` | RISC-V C compiler |
| `RISCVCXX` | `riscv64-unknown-elf-g++` | RISC-V C++ compiler |
| `VLEN` | `128` | RISC-V vector register width (bits) |
| `SIMULATOR` | `spike --isa=rv64gcv pk64` | RISC-V ISA simulator command |
| `SSH_HOST` | `final` | SSH host for real RISC-V hardware validation |
| `SSH_JUMP_HOST` | *(unset)* | SSH jump host for Intel reference runs. When unset, benchmark runs Intel locally |
| `REMOTE_DIR` | `/tmp/sse2rvv` | Working directory on remote hosts |
| `REACT_MAX_STEPS` | `15` | Max LLM repair iterations |
| `LLM_VALIDATION_RETRIES` | `2` | Retries per LLM step on edit/parse failure |
| `VALIDATION_TIMEOUT_SECONDS` | `120` | Timeout for Docker/SSH validation commands |

## Project Structure

```
riscv-translator/
├── src/                        Python package (translation pipeline)
│   ├── repair.py               TranslationAgent — LLM compile-fix loop orchestrator
│   ├── prompts.py              Generic SSE -> sse2rvv.h translation prompts
│   ├── validators.py           DockerValidator (Spike) + SSHValidator (real hardware)
│   ├── search_replace.py       Robust search/replace block parser (tolerates LLM formatting errors)
│   ├── llm_utils.py            LLM client (OpenRouter API, with retry/backoff)
│   ├── llm_types.py            Message/LLM protocol types (no external dependency)
│   ├── config.py               Configuration constants (all env-var overridable)
│   ├── benchmark.py            Intel vs RISC-V execution comparison
│   ├── check.py                Standalone validation of output directories
│   └── logger.py               Terminal logger with context awareness
├── initial_code/               SSW library source (SSE2) + sse2rvv.h subset
│   ├── ssw.c                   Smith-Waterman algorithm implementation (SSE2 intrinsics)
│   ├── ssw.h                   SSW library header
│   ├── main.c                  Test harness (modified: uses standard C I/O instead of zlib)
│   ├── kseq.h                  FASTA sequence parser
│   ├── sse2rvv.h               SSE -> RVV compatibility header (subset)
│   └── Makefile                Build for x86 (original SSE code)
├── translations/               Successfully translated outputs
│   └── sequence-alignment/     SSW library translated to RVV (ready to compile)
├── datasets/                   FASTA test data for SSW benchmarking
│   ├── 1k.fa                   1,000 base pairs
│   ├── 10k.fa                  10,000 base pairs
│   ├── 1M.fa                   ~1 million base pairs
│   ├── 10M.fa                  ~10 million base pairs
│   └── 54mer_hap1_1.100.fa     Reference sequence
├── tests/                      Python test suite (pytest)
├── docs/                       RVV reference material (loaded as LLM context)
│   └── riscv-reference/
│       └── reference.md        RISC-V Vector programming reference
├── Makefile                    Top-level targets (sync, test, translate, check, benchmark)
├── pyproject.toml              Python project metadata (uv, minimal dependencies)
├── .env.example                Environment variable template
├── CLAUDE.md                   Development guide and architecture notes
└── LICENSE                     MIT License
```

## Reproducing the Translation

To reproduce the SSW library translation from scratch:

1. **Install prerequisites**: `uv`, Docker, and pull the toolchain image:
   ```bash
   docker pull luispimo/riscv-toolchain:arm64-2025-10-20
   ```

2. **Configure the environment**:
   ```bash
   uv sync --dev
   cp .env.example .env
   # Add your OpenRouter API key to .env
   ```

3. **Run the translation**:
   ```bash
   make translate
   ```
   This reads from `initial_code/`, translates `ssw.c`, and writes the result to `output/`.

4. **Validate the output**:
   ```bash
   make check OUTPUT_DIR=output
   ```

5. **Compare against the included reference translation**:
   The `translations/sequence-alignment/` directory contains a previously successful translation that can be used as a reference.

> **Reproducibility note**: Because the pipeline uses an LLM, exact outputs may vary between runs. However, the functional behavior (correct compilation and matching output on RISC-V) should be consistent. Setting `LLM_TEMPERATURE=0` improves determinism.

## Key Dependencies

- **[sse2rvv](https://github.com/pattonkan/sse2rvv)** — Header-only SSE-to-RVV translation layer. We include only the subset of intrinsics required by the current target library.
- **[Complete-Striped-Smith-Waterman-Library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)** — The C library translated as the first test case. Source in `initial_code/` has been modified to remove the `zlib` dependency (standard C I/O is used instead).
- **[Spike](https://github.com/riscv-software-src/riscv-isa-sim)** — RISC-V ISA simulator used for validation (included in the Docker image).

## License

This project is licensed under the [MIT License](LICENSE).

The bundled third-party components are also MIT-compatible:
- **SSW library** — MIT License (see `initial_code/ssw.c` header)
- **sse2rvv** — MIT License (see [upstream](https://github.com/pattonkan/sse2rvv))
