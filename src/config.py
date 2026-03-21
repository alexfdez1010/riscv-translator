import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
KERNEL_FILE = PROJECT_DIR / "csrc" / "sequence_alignment_kernel_rvv.c"
BENCH_FILE = PROJECT_DIR / "csrc" / "bench_remote.c"
VALIDATION_BENCH_FILE = PROJECT_DIR / "csrc" / "bench_validation_small.c"
BLOSUM62_FILE = PROJECT_DIR / "csrc" / "blosum62.h"
FASTA_DATASET = PROJECT_DIR / "dataset" / "uniprot_sprot_varsplic.fasta"
REFERENCE_FILE = PROJECT_DIR / "docs" / "riscv-reference" / "reference.md"

# ---------------------------------------------------------------------------
# SSW (Striped Smith-Waterman) evolutionary target
# ---------------------------------------------------------------------------

SSW_FILE = PROJECT_DIR / "initial_code" / "ssw.repaired.c"
SSW_DIR = PROJECT_DIR / "initial_code"
SSW_BENCH_FILE = PROJECT_DIR / "csrc" / "bench_ssw.c"
BENCHMARK_FASTA = PROJECT_DIR / "dataset" / "10k.fa"

# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

SSH_HOST = "final"
REMOTE_DIR = "/tmp/ea_kernels"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:30000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key-required")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "5000"))

# ---------------------------------------------------------------------------
# Evolutionary algorithm
# ---------------------------------------------------------------------------

MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "20"))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "10"))
NUM_RUNS_FITNESS = int(os.getenv("NUM_RUNS_FITNESS", "5"))
LLM_VALIDATION_RETRIES = int(os.getenv("LLM_VALIDATION_RETRIES", "2"))
VALIDATION_RANDOM_TESTS = int(os.getenv("VALIDATION_RANDOM_TESTS", "6"))
VALIDATION_MIN_LEN = int(os.getenv("VALIDATION_MIN_LEN", "5"))
VALIDATION_MAX_LEN = int(os.getenv("VALIDATION_MAX_LEN", "6"))
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "120"))
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "luispimo/riscv-toolchain:arm64-2025-10-20")
RISCVCC = os.getenv("RISCVCC", "riscv64-unknown-elf-gcc")
VLEN = int(os.getenv("VLEN", "128"))
PK_PATH = os.getenv("PK_PATH", "/opt/riscv/riscv64-unknown-elf/bin/pk64")
SIMULATOR = os.getenv(
    "SIMULATOR",
    f"qemu-riscv64 -cpu rv64,v=on,vext_spec=v1.0,vlen={VLEN},rvv_ta_all_1s=on",
)
VALIDATION_CFLAGS = os.getenv("VALIDATION_CFLAGS", "")
VALIDATION_LDFLAGS = os.getenv("VALIDATION_LDFLAGS", "")

# ---------------------------------------------------------------------------
# RVV Reference context (loaded once for LLM prompts)
# ---------------------------------------------------------------------------

RVV_REFERENCE = REFERENCE_FILE.read_text() if REFERENCE_FILE.exists() else ""

# ---------------------------------------------------------------------------
# HPC Optimization Techniques (randomly selected for mutation prompts)
# ---------------------------------------------------------------------------

HPC_TECHNIQUES = [
    {
        "name": "Loop Unrolling",
        "description": (
            "Manually unroll the inner loop to reduce loop overhead and increase "
            "instruction-level parallelism. Process multiple anti-diagonal elements "
            "per iteration by duplicating the loop body with adjusted indices."
        ),
    },
    {
        "name": "Sequence Pre-encoding",
        "description": (
            "Preprocess seq1 and seq2 into compact amino-acid index arrays before the DP "
            "loop so the hot path no longer calls blosum62_aa_to_idx per lane. Remove "
            "invalid-residue checks from the inner loop whenever the input can be "
            "validated ahead of time."
        ),
    },
    {
        "name": "LMUL Tuning",
        "description": (
            "Experiment with different RVV LMUL settings such as m1, m2, m4, or m8 for "
            "the anti-diagonal kernel. Higher LMUL may improve throughput but can also "
            "increase register pressure and reduce flexibility, so choose the best value "
            "for the actual hardware and sequence lengths."
        ),
    },
    {
        "name": "VLEN-specific Specialization",
        "description": (
            "Create vector-length-specific variants for the target machine instead of using "
            "only fully generic vector-length-agnostic code. Dispatch on vlenb or a known "
            "hardware configuration so the kernel can use fixed unroll factors, fixed "
            "scratch sizes, and shuffle strategies tuned to the measured VLEN."
        ),
    },
    {
        "name": "Substitution Profile Precomputation",
        "description": (
            "Precompute BLOSUM62 access patterns outside the hot loop. Replace per-lane "
            "scalar score construction and temporary substitution buffers with cached "
            "profiles, direct indexed loads, or reusable scratch storage sized for vlmax."
        ),
    },
    {
        "name": "Traversal and Layout Co-design",
        "description": (
            "Align the DP data layout with anti-diagonal traversal so vector memory "
            "operations become contiguous or nearly contiguous. Consider storing active "
            "anti-diagonals contiguously, using a skewed layout, or padding structures to "
            "avoid expensive strided loads and stores."
        ),
    },
    {
        "name": "Alignment and Padding",
        "description": (
            "Align DP buffers and temporary score storage to favorable memory boundaries, and "
            "pad rows, diagonals, or scratch arrays so vector memory operations avoid awkward "
            "tails and misaligned accesses. This can improve unit-stride behavior and reduce "
            "the cost of boundary handling."
        ),
    },
    {
        "name": "Rolling Diagonal Buffers",
        "description": (
            "Replace the full H, E, and F matrices with rolling buffers that keep only the "
            "active and dependent diagonals. This reduces memory footprint, improves cache "
            "locality, and lowers bandwidth pressure while preserving affine-gap "
            "dependencies."
        ),
    },
    {
        "name": "Anti-diagonal Tiling",
        "description": (
            "Tile the wavefront computation into cache-friendly blocks instead of processing "
            "entire anti-diagonals at once. Reuse boundary data inside a tile and reduce "
            "the working set of H, E, and F to improve locality on long sequences."
        ),
    },
    {
        "name": "Register Pressure Reduction",
        "description": (
            "Reduce the number of simultaneously live vector values by reusing registers, "
            "shortening live ranges, and avoiding unnecessary temporaries. This is "
            "especially important when testing larger LMUL settings in the affine-gap "
            "kernel."
        ),
    },
    {
        "name": "Pointer Arithmetic Simplification",
        "description": (
            "Replace repeated index recomputation such as i * cols + j with pointer bumping, "
            "precomputed diagonal bases, and fixed offset increments. Hoist invariant address "
            "calculations out of the hot loop to cut scalar overhead."
        ),
    },
    {
        "name": "VL and VTYPE Control",
        "description": (
            "Minimize vsetvl and vtype reconfiguration overhead by keeping the kernel in a "
            "stable SEW and LMUL mode when possible. Compare fully dynamic vsetvl on every "
            "chunk against a main-path configuration that uses vlmax-sized chunks plus a "
            "smaller cleanup path."
        ),
    },
    {
        "name": "Branch Elimination",
        "description": (
            "Remove conditional logic from the hot loop by validating residues in a "
            "preprocessing pass, restructuring boundary handling, and using predicated "
            "vector operations where needed. This keeps the wavefront kernel more regular "
            "and easier for the compiler to optimize."
        ),
    },
    {
        "name": "Masking Minimization",
        "description": (
            "Prefer controlling the active work with VL and loop bounds instead of forming "
            "extra masks when only the leading lanes matter. Use masking only when required "
            "for correctness, faults, or side effects, and avoid tail-undisturbed behavior "
            "unless preserved inactive lanes are actually needed later."
        ),
    },
    {
        "name": "Software Pipelining",
        "description": (
            "Overlap the current anti-diagonal computation with loads or address setup for "
            "the next chunk. Schedule memory operations early and interleave independent "
            "work to hide load latency and keep the vector unit occupied."
        ),
    },
    {
        "name": "Tail Loop Optimization",
        "description": (
            "Optimize the handling of non-multiple-of-vl anti-diagonal tails. Compare a "
            "single vlmax main loop plus a cleanup step against fully dynamic vsetvl usage, "
            "and test tail-agnostic versus tail-undisturbed policies where supported."
        ),
    },
    {
        "name": "Instruction Scheduling",
        "description": (
            "Reorder dependent and independent vector instructions to reduce pipeline stalls. "
            "Issue loads earlier, separate dependent max/sub/add chains when possible, and "
            "arrange computation to better cover the latency of strided memory operations."
        ),
    },
]
