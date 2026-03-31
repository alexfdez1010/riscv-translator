"""Vector-width optimisation pipeline for RISC-V translated code.

Takes code already translated from SSE to RISC-V (via sse2rvv.h with fixed
128-bit vectors) and rewrites the SIMD operations to use **native** RISC-V
Vector (RVV) intrinsics with runtime VLEN — so the hardware can use its full
vector register width.

    Phase 1 (repair.py):  SSE source  ->  sse2rvv.h  (128-bit, functional)
    Phase 2 (widen.py):   sse2rvv.h   ->  native RVV (VLEN-agnostic, fast)

Pipeline:
  1. Validate that the input (sse2rvv.h) code compiles and runs.
  2. LLM incrementally replaces sse2rvv.h intrinsics with native RVV.
  3. Each pass: LLM proposes diffs -> compile -> validate -> benchmark.
  4. LLM signals ALL_WIDENED when no SSE intrinsics remain.
  5. Correctness is verified against Intel x86 reference output.
"""

import argparse
import re
import shutil
import time
from pathlib import Path

from src.llm_types import LLM, Message
from src.config import (
    DATASETS_DIR,
    PROJECT_DIR,
    REACT_MAX_STEPS,
    REFERENCE_FILE,
    REMOTE_DIR,
    RISCVCC,
    RVV_REFERENCE,
    SIMULATOR,
    SSH_CC,
    SSH_HOST,
    SSH_JUMP_HOST,
)
from src.benchmark import (
    BenchmarkResult,
    REFERENCE_FILE as BENCH_REFERENCE_FILE,
    check_ssh,
    compare_outputs,
    run_on_host,
    upload_datasets,
    upload_to_host,
)
from src.search_replace import (
    apply_search_replace,
    extract_search_replace,
    search_replace_error_feedback,
    search_replace_format_example,
)
from src.llm_utils import create_llm
from src.logger import configure_logging, get_logger
from src.validators import DockerValidator, SSHValidator, ValidationResult

# Re-export shared infrastructure from the Phase-1 repair module so that
# callers (and tests) can reach them via ``widen.<name>``.
from src.repair import (
    MAX_OUTPUT_CHARS,
    SourceSnapshot,
    WorkspaceSet,
    apply_content_to_snapshot,
    create_workspace,
    materialize_snapshot,
    truncate_for_log,
    write_output,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Original SSE source directory — needed for Intel x86 reference compilation.
# The translated RVV code cannot compile on x86, so correctness comparison
# must use the original pre-translation source.
ORIGINAL_SOURCE_DIR = PROJECT_DIR / "initial_code"

CORRECTNESS_DATASET = "10k.fa"


# ---------------------------------------------------------------------------
# Build commands
# ---------------------------------------------------------------------------


def default_docker_build_command() -> str:
    """Build + run command for Docker/QEMU validation of widened code.

    Execution output is suppressed (>/dev/null) to keep LLM feedback
    focused on compilation errors rather than alignment text.
    """
    cflags = "-O2 -I. -march=rv64gcv -mabi=lp64d"
    ldflags = "-lm"
    return (
        f'echo "=== Compiling ===" && '
        f"{RISCVCC} {cflags} main.c ssw.c -o ssw_test {ldflags} 2>&1 && "
        f'echo "=== Compilation succeeded, running ===" && '
        f"ls demo/ 2>&1 && "
        f"{SIMULATOR} ./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa >/dev/null 2>&1 && "
        f'echo "=== Execution succeeded ==="'
    )


def default_ssh_compile_command() -> str:
    """Compile command for real RISC-V hardware via SSH."""
    return (
        f"{SSH_CC} -o ssw_test main.c ssw.c "
        f"--target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"
    )


def default_ssh_run_command() -> str:
    """Run command for real RISC-V hardware via SSH."""
    return f"./ssw_test demo/10k.fa demo/{BENCH_REFERENCE_FILE} 2>&1"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_widen_system_prompt(target_file: str) -> str:
    """System prompt for the sse2rvv -> native RVV widening task."""
    return f"""\
You are an expert systems programmer specialising in RISC-V Vector (RVV)
extensions and x86 SSE SIMD portability.

Your task is to rewrite C code that currently uses SSE intrinsics via the
**sse2rvv.h** emulation layer so that it uses **native RISC-V Vector
intrinsics** and becomes VLEN-agnostic — automatically using the full
hardware vector register width at any VLEN >= 128.

## Background

The input code was translated from x86 SSE to RISC-V using sse2rvv.h, which
emulates SSE intrinsics with RVV but locks all operations to 128 bits (the
SSE semantic width).  Your job: replace sse2rvv.h calls with native RVV
intrinsics that use the full hardware VLEN.

## Transformation overview

### 1. Replace the header and add VLEN-query helpers

Replace ``#include "sse2rvv.h"`` with ``#include <riscv_vector.h>`` and add:

```c
static inline size_t vlmax_e8(void)  {{ return __riscv_vsetvlmax_e8m1(); }}
static inline size_t vlmax_e16(void) {{ return __riscv_vsetvlmax_e16m1(); }}
static inline size_t vregbytes(void) {{ return __riscv_vsetvlmax_e8m1(); }}
```

### 2. Change data types

| Before (sse2rvv.h)            | After (native RVV)                    |
|-------------------------------|---------------------------------------|
| ``__m128i*`` profile members  | ``uint8_t*`` (raw byte arrays)        |
| ``__m128i*`` local arrays     | ``uint8_t*``                          |
| ``__m128i`` vector variables  | ``vuint8m1_t`` / ``vint16m1_t`` etc.  |
| ``__m128i`` zero vector       | ``vuint8m1_t`` / ``vint16m1_t``       |

### 3. Replace hardcoded lane counts with runtime queries

| Before                          | After                                     |
|---------------------------------|-------------------------------------------|
| ``16`` (byte lanes)             | ``vlmax_e8()`` or ``vl``                  |
| ``8`` (word lanes)              | ``vlmax_e16()`` or ``vl``                 |
| ``16`` (memory stride bytes)    | ``vregbytes()`` or ``vb``                 |
| ``(readLen + 15) / 16``         | ``(readLen + lanes - 1) / lanes``         |
| ``j * 16`` (byte offset)       | ``j * vb``                                |
| ``segLen * 16`` (column len)   | ``segLen * vl`` (for tracing)             |
| ``i / 16 + i % 16 * segLen``  | ``i / vl + i % vl * segLen``              |

### 4. Replace SSE intrinsics with native RVV

#### 8-bit (byte) operations
| SSE (sse2rvv.h)                               | RVV (native)                                               |
|------------------------------------------------|------------------------------------------------------------|
| ``_mm_set1_epi32(0)`` (zero)                  | ``__riscv_vmv_v_x_u8m1(0, vl)``                           |
| ``_mm_set1_epi8(x)``                          | ``__riscv_vmv_v_x_u8m1(x, vl)``                           |
| ``_mm_adds_epu8(a, b)``                       | ``__riscv_vsaddu_vv_u8m1(a, b, vl)``                      |
| ``_mm_subs_epu8(a, b)``                       | ``__riscv_vssubu_vv_u8m1(a, b, vl)``                      |
| ``_mm_max_epu8(a, b)``                        | ``__riscv_vmaxu_vv_u8m1(a, b, vl)``                       |
| ``_mm_load_si128(ptr)``                        | ``__riscv_vle8_v_u8m1(ptr, vl)``                          |
| ``_mm_store_si128(ptr, v)``                    | ``__riscv_vse8_v_u8m1(ptr, v, vl)``                       |
| ``_mm_slli_si128(v, 1)``                      | ``__riscv_vslideup_vx_u8m1_tu(vZero, v, 1, vl)``          |
| ``_mm_cmpeq_epi8`` + ``movemask == 0xffff``   | ``vmseq`` + ``vcpop == vl`` (see patterns below)           |
| ``max16`` tree-reduction macro                  | ``__riscv_vredmaxu_vs_u8m1_u8m1(v, vZero, vl)``           |
| ``_mm_extract_epi16(v, 0)``                   | ``__riscv_vmv_x_s_u8m1_u8(v)``                            |

#### 16-bit (word) operations
| SSE (sse2rvv.h)                               | RVV (native)                                               |
|------------------------------------------------|------------------------------------------------------------|
| ``_mm_set1_epi32(0)`` (zero)                  | ``__riscv_vmv_v_x_i16m1(0, vl)``                          |
| ``_mm_set1_epi16(x)`` (signed)                | ``__riscv_vmv_v_x_i16m1(x, vl)``                          |
| ``_mm_set1_epi16(x)`` (unsigned, e.g. gaps)   | ``__riscv_vmv_v_x_u16m1(x, vl)``                          |
| ``_mm_adds_epi16(a, b)``                      | ``__riscv_vsadd_vv_i16m1(a, b, vl)``                      |
| ``_mm_subs_epu16(a, b)``                      | reinterpret to u16, ``__riscv_vssubu_vv_u16m1``, back     |
| ``_mm_max_epi16(a, b)``                       | ``__riscv_vmax_vv_i16m1(a, b, vl)``                       |
| ``_mm_load_si128(ptr)``                        | ``__riscv_vle16_v_i16m1((int16_t*)ptr, vl)``              |
| ``_mm_store_si128(ptr, v)``                    | ``__riscv_vse16_v_i16m1((int16_t*)ptr, v, vl)``           |
| ``_mm_slli_si128(v, 2)``                      | ``__riscv_vslideup_vx_i16m1_tu(vZero, v, 1, vl)``         |
| ``_mm_cmpgt_epi16`` + ``!movemask``           | ``vmsgt`` + ``vcpop == 0``                                 |
| ``max8`` tree-reduction macro                   | ``__riscv_vredmax_vs_i16m1_i16m1(v, vZero, vl)``          |

### 5. Key patterns

**Early exit in Lazy_F loop (8-bit):**
```c
vuint8m1_t vDiff = __riscv_vssubu_vv_u8m1(vF, vH, vl);
vbool8_t all_zero = __riscv_vmseq_vx_u8m1_b8(vDiff, 0, vl);
if (UNLIKELY(__riscv_vcpop_m_b8(all_zero, vl) == vl)) goto lazy_end;
```

**Early exit in Lazy_F loop (16-bit):**
```c
vbool16_t gt_mask = __riscv_vmsgt_vv_i16m1_b16(vF, vH_gap, vl);
if (UNLIKELY(__riscv_vcpop_m_b16(gt_mask, vl) == 0)) goto word_lazy_end;
```

**Horizontal max reduction (8-bit):**
```c
vuint8m1_t vReduce = __riscv_vredmaxu_vs_u8m1_u8m1(vMaxScore, vZero, vl);
uint8_t temp = (uint8_t)__riscv_vmv_x_s_u8m1_u8(vReduce);
```

**Horizontal max reduction (16-bit):**
```c
vint16m1_t vReduce = __riscv_vredmax_vs_i16m1_i16m1(vMaxScore, vZero, vl);
uint16_t temp = (uint16_t)__riscv_vmv_x_s_i16m1_i16(vReduce);
```

**Max-score-changed check (8-bit):**
```c
vbool8_t eq_mask = __riscv_vmseq_vv_u8m1_b8(vMaxMark, vMaxScore, vl);
if (__riscv_vcpop_m_b8(eq_mask, vl) != vl) {{ ... }}
```

**16-bit unsigned saturating subtraction (for gap penalties):**
```c
vuint16m1_t vH_u = __riscv_vreinterpret_v_i16m1_u16m1(vH);
vuint16m1_t vH_gap = __riscv_vssubu_vv_u16m1(vH_u, vGapO_u, vl);
// ... then reinterpret back to signed for max comparison
```

### 6. Update loop bounds

- Lazy_F 8-bit:  ``for (k = 0; k < 16; ++k)`` -> ``for (k = 0; k < (int32_t)vl; ++k)``
- Lazy_F 16-bit: ``for (k = 0; k < 8; ++k)``  -> ``for (k = 0; k < (int32_t)vl; ++k)``

### 7. Rename functions

- ``sw_sse2_byte`` -> ``sw_rvv_byte``
- ``sw_sse2_word`` -> ``sw_rvv_word``
- Update calls in ``ssw_align()``

## CRITICAL: 16-bit path unsigned saturation

The 16-bit word path needs careful handling of unsigned saturating subtraction
for gap penalties.  SSE's ``_mm_subs_epu16`` operates on unsigned values but
scores are ``int16``.  Use ``__riscv_vreinterpret_v_i16m1_u16m1`` /
``__riscv_vreinterpret_v_u16m1_i16m1`` casts around the unsigned operations.

## Rules

- Modify only ``{target_file}``.
- Do NOT touch scalar functions (``banded_sw``, ``cigar_alignment_score``,
  ``mark_mismatch``, ``seq_reverse``, ``add_cigar``, ``store_previous_m``,
  etc.).
- Keep the same algorithmic structure — only change the SIMD implementation.
- The code must remain correct at VLEN=128 (same results as the sse2rvv
  version) and also work at wider VLEN.

## Output format

Return only:
1) A short summary sentence.
2) One or more search/replace blocks.

Each block has this exact format:

<<<<<<< SEARCH
exact lines from the current file to find
=======
replacement lines
>>>>>>> REPLACE

{search_replace_format_example()}

## Edit requirements

- The SEARCH section must be copied EXACTLY from the current file
  (same indentation, same whitespace, character for character).
- The REPLACE section must be DIFFERENT from SEARCH — every block must
  actually change something.
- If the same text appears in multiple places, a single block replaces
  all occurrences.
- You may use multiple search/replace blocks for changes in different
  parts of the file.

## RISC-V Vector (RVV) reference

Reference material from: {REFERENCE_FILE}

{RVV_REFERENCE}
""".strip()


def build_widen_initial_prompt(
    target_file: str,
    source_code: str,
    build_command: str,
    validation_feedback: str | None = None,
) -> str:
    """First user prompt — asks the LLM to begin widening."""
    validation_section = ""
    if validation_feedback:
        validation_section = f"""

Current validation failure:
{validation_feedback}
""".rstrip()

    return f"""\
Task: Widen this RISC-V code from sse2rvv.h (fixed 128-bit) to native RVV
intrinsics (VLEN-agnostic, full hardware vector width).

Goal:
Replace all sse2rvv.h SSE emulation calls with native RISC-V Vector
intrinsics that work at any VLEN >= 128.  The code should automatically
use the full hardware vector width.

Context:
- Target file: {target_file}
- Build & validation command: {build_command}
- The code currently uses sse2rvv.h which locks operations to 128 bits.
- After widening, the code should use ``<riscv_vector.h>`` directly.

What to change in this pass:
1. Replace ``#include "sse2rvv.h"`` with ``#include <riscv_vector.h>``
2. Add the ``vlmax_e8`` / ``vlmax_e16`` / ``vregbytes`` helper functions
3. Change ``struct _profile`` member types from ``__m128i*`` to ``uint8_t*``
4. Rewrite ``qP_byte()`` to use runtime ``vlmax_e8()`` instead of 16
5. Rewrite ``sw_sse2_byte()`` -> ``sw_rvv_byte()`` using native RVV byte intrinsics
   (change signature, local variables, all intrinsic calls, loop bounds, reduction)

Focus on the **8-bit (byte) path** first.  The 16-bit path will be done
in a later pass.

Current code:
```c
{source_code}
```
{validation_section}

Output:
- First, one short summary sentence.
- Then, one or more search/replace blocks with the changes.
{search_replace_format_example()}
""".strip()


def build_widen_repair_prompt(
    target_file: str,
    code: str,
    validation_feedback: str,
) -> str:
    """Follow-up prompt when a widening attempt has compile/runtime errors."""
    return f"""\
Task: Fix the validation failure in {target_file} (widening pipeline).

The code is being widened from sse2rvv.h to native RVV intrinsics.
Fix the error shown below with the smallest correct change.

Context:
- Target file: {target_file}

Current code:
```c
{code}
```

Validation failure details:
{validation_feedback}

Output:
- First, one short summary sentence.
- Then, one or more search/replace blocks with the changes.
{search_replace_format_example()}
""".strip()


def build_widen_continue_prompt(
    target_file: str,
    code: str,
    build_command: str,
    pass_number: int,
    validation_feedback: str | None = None,
) -> str:
    """Prompt for subsequent widening passes after one succeeded."""
    error_section = ""
    if validation_feedback:
        error_section = f"""

The previous pass produced errors.  Fix them AND continue widening
remaining intrinsics in this pass:

{validation_feedback}
"""

    return f"""\
Task: Continue widening {target_file} — this is pass {pass_number}.

Examine the current code and identify remaining sse2rvv.h intrinsics or
patterns that still use fixed 128-bit semantics.
{error_section}
If there are **NO** more SSE / sse2rvv.h intrinsics to widen (i.e. all
SIMD code already uses native RVV) and there are no compilation errors,
respond with ONLY:

    ALL_WIDENED

If there ARE remaining intrinsics to widen or errors to fix, produce
search/replace blocks for the next batch.  Typical remaining work:

- ``qP_word()`` — rewrite with ``vlmax_e16()`` instead of hardcoded 8
- ``sw_sse2_word()`` -> ``sw_rvv_word()`` — 16-bit native RVV path
- ``ssw_align()`` — update ``__m128i* vP`` -> ``uint8_t* vP``, call
  ``sw_rvv_byte`` / ``sw_rvv_word`` instead of ``sw_sse2_byte`` /
  ``sw_sse2_word``
- Any leftover ``_mm_*`` calls, ``__m128i`` types, or hardcoded 16/8
  lane counts

Context:
- Target file: {target_file}
- Build command: {build_command}

Current code:
```c
{code}
```

Output:
- If all widened: respond with ``ALL_WIDENED``
- Otherwise: one short summary sentence + search/replace blocks.
{search_replace_format_example()}
""".strip()


def build_widen_edit_format_feedback(
    file_name: str,
    code: str,
    error_message: str,
) -> str:
    """Feedback when search/replace blocks could not be applied."""
    return search_replace_error_feedback(file_name, code, error_message)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _to_messages(raw: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Widening agent
# ---------------------------------------------------------------------------


class WidenAgent:
    """LLM-driven sse2rvv.h -> native RVV widening pipeline."""

    def __init__(self):
        self.docker_validator = DockerValidator()
        self.ssh_validator = SSHValidator()
        self.llm: LLM = create_llm()
        self._intel_reference: BenchmarkResult | None = None
        self._intel_reference_computed = False

    # -- Intel reference for correctness checking ----------------------------

    def _get_intel_reference(
        self, original_dir: Path, test_data_dir: Path | None,
    ) -> BenchmarkResult | None:
        """Compute reference output by running original SSE code on Intel.

        Uses the *original* pre-translation source (which can compile on
        x86) rather than the RVV-translated source.  Cached after first call.
        """
        if self._intel_reference_computed:
            return self._intel_reference

        self._intel_reference_computed = True
        jump_host = SSH_JUMP_HOST
        if not check_ssh(jump_host):
            logger.warning(
                "Jump host %s not reachable; correctness check disabled",
                jump_host,
            )
            return None

        jump_remote = f"{REMOTE_DIR}-widen-ref"
        dataset_dir = (
            test_data_dir
            if test_data_dir and test_data_dir.is_dir()
            else DATASETS_DIR
        )
        dataset_file = dataset_dir / CORRECTNESS_DATASET
        if not dataset_file.exists():
            logger.warning(
                "Correctness dataset %s not found; correctness check disabled",
                dataset_file,
            )
            return None

        original_files = [p for p in original_dir.iterdir() if p.is_file()]
        logger.info(
            "Uploading original code to %s for correctness reference ...",
            jump_host,
        )
        if not upload_to_host(jump_host, jump_remote, original_files):
            logger.warning("Failed to upload original code; correctness disabled")
            return None
        if not upload_datasets(
            jump_host, jump_remote, dataset_dir, CORRECTNESS_DATASET,
        ):
            logger.warning("Failed to upload datasets; correctness disabled")
            return None

        compile_cmd = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"
        run_cmd = (
            f"./ssw_test demo/{CORRECTNESS_DATASET} "
            f"demo/{BENCH_REFERENCE_FILE} 2>&1"
        )
        result = run_on_host(
            jump_host, jump_remote, compile_cmd, run_cmd, "Intel reference",
        )
        if not result.ok:
            logger.warning(
                "Intel reference run failed; correctness disabled\n%s",
                result.stderr,
            )
            return None

        logger.info(
            "Intel reference output computed (%d chars)", len(result.stdout),
        )
        self._intel_reference = result
        return result

    # -- Correctness validation against Intel reference ----------------------

    def _validate_correctness(
        self, workspace_dir: Path,
    ) -> ValidationResult | None:
        """Compare translated code output against Intel reference."""
        if self._intel_reference is None:
            return None

        final_host = SSH_HOST
        final_remote = f"{REMOTE_DIR}-widen-correctness"

        all_paths = list(workspace_dir.iterdir())
        if not upload_to_host(final_host, final_remote, all_paths):
            logger.warning("Failed to upload for correctness check")
            return None

        if not upload_datasets(
            final_host, final_remote, DATASETS_DIR, CORRECTNESS_DATASET,
        ):
            logger.warning("Failed to upload datasets for correctness check")
            return None

        compile_cmd = (
            f"{SSH_CC} -o ssw_test main.c ssw.c "
            f"--target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"
        )
        run_cmd = (
            f"./ssw_test demo/{CORRECTNESS_DATASET} "
            f"demo/{BENCH_REFERENCE_FILE} 2>&1"
        )
        riscv_result = run_on_host(
            final_host, final_remote, compile_cmd, run_cmd,
            "RISC-V correctness",
        )
        if not riscv_result.ok:
            return ValidationResult(
                ok=False,
                stage="correctness",
                returncode=1,
                stdout=riscv_result.stdout,
                stderr=f"RISC-V correctness run failed:\n{riscv_result.stderr}",
            )

        match, details = compare_outputs(self._intel_reference, riscv_result)
        if match:
            logger.info("Correctness check PASSED")
            return ValidationResult(
                ok=True, stage="correctness", returncode=0,
                stdout=details, stderr="",
            )

        logger.warning("Correctness check FAILED")
        return ValidationResult(
            ok=False,
            stage="correctness",
            returncode=1,
            stdout="",
            stderr=(
                "CORRECTNESS FAILURE: The widened RISC-V code produces "
                "different results than the original Intel SSE code.\n\n"
                f"{details}\n\n"
                "The SIMD widening has a bug.  Check data types, memory "
                "layout strides, lane counts, and reduction operations."
            ),
        )

    # -- Benchmarking on SSH hardware ----------------------------------------

    def _benchmark_on_ssh(
        self,
        workspace_dir: Path,
        ssh_compile_cmd: str,
        ssh_run_cmd: str,
        label: str,
    ) -> float | None:
        """Upload, compile, run on real hardware and return elapsed seconds."""
        if not self.ssh_validator._available:
            return None

        final_host = SSH_HOST
        final_remote = f"{REMOTE_DIR}-widen-bench"

        all_paths = list(workspace_dir.iterdir())
        if not upload_to_host(final_host, final_remote, all_paths):
            logger.warning("Benchmark upload failed for %s", label)
            return None

        if not upload_datasets(
            final_host, final_remote, DATASETS_DIR, CORRECTNESS_DATASET,
        ):
            logger.warning("Benchmark dataset upload failed for %s", label)
            return None

        result = run_on_host(
            final_host, final_remote, ssh_compile_cmd, ssh_run_cmd, label,
        )
        if not result.ok:
            logger.warning("Benchmark run failed for %s", label)
            return None

        logger.info("Benchmark %s: %.2fs", label, result.elapsed_seconds)
        return result.elapsed_seconds

    # -- Single LLM request cycle with retries -------------------------------

    def _generate_single_pass(
        self,
        messages: list[dict[str, str]],
        snapshot: SourceSnapshot,
        file_name: str,
        workspaces: WorkspaceSet,
        build_command: str,
    ) -> tuple[SourceSnapshot | None, ValidationResult]:
        """Run a single LLM call, apply edits, and validate (no retries).

        Returns ``(snapshot, validation)``.
        Detects the ``ALL_WIDENED`` signal and returns a special result.
        """
        # --- LLM request ---
        try:
            logger.info(
                "LLM request with %d message(s)", len(messages),
            )
            response = self.llm(_to_messages(messages))
        except Exception as exc:
            return None, ValidationResult(
                ok=False,
                stage="internal-error",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )

        logger.info(
            "LLM response (%d chars):\n%s",
            len(response),
            truncate_for_log(response, 3000),
        )

        # --- Detect ALL_WIDENED signal ---
        if "ALL_WIDENED" in response:
            logger.info("LLM signalled ALL_WIDENED")
            return None, ValidationResult(
                ok=True,
                stage="all-widened",
                returncode=0,
                stdout="All SSE intrinsics have been widened to native RVV.",
                stderr="",
            )

        # --- Extract and apply search/replace blocks ---
        sr_blocks = extract_search_replace(response)
        if sr_blocks is None:
            logger.warning("No search/replace blocks found in response")
            return None, ValidationResult(
                ok=False,
                stage="edit-failure",
                returncode=None,
                stdout="",
                stderr="Could not extract edits from the response.",
            )

        logger.info("Extracted %d search/replace block(s)", len(sr_blocks))
        try:
            new_content = apply_search_replace(
                snapshot.files[file_name], sr_blocks,
            )
            candidate_snapshot = apply_content_to_snapshot(
                snapshot, file_name, new_content,
            )
        except ValueError as exc:
            logger.warning("Search/replace failed: %s", exc)
            return None, ValidationResult(
                ok=False,
                stage="edit-failure",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )

        # --- Validate in Docker/QEMU ---
        materialize_snapshot(workspaces.workspace_dir, candidate_snapshot)
        validation = self.docker_validator.validate(
            workspaces.workspace_dir, build_command,
        )
        logger.info(
            "Validation: ok=%s stage=%s rc=%s\n%s",
            validation.ok,
            validation.stage,
            validation.returncode,
            truncate_for_log(validation.combined_output, 2000),
        )

        return candidate_snapshot, validation

    # -- Main pipeline -------------------------------------------------------

    def run(
        self,
        source_dir: Path,
        target_file: str,
        output_dir: Path,
        build_command: str | None = None,
        ssh_compile_command: str | None = None,
        ssh_run_command: str | None = None,
        max_steps: int = REACT_MAX_STEPS,
        test_data_dir: Path | None = None,
    ) -> int:
        """Run the full widening pipeline.

        Args:
            source_dir: Directory with sse2rvv.h-translated source files.
            target_file: Name of the file to widen (e.g. ``ssw.c``).
            output_dir: Directory to write all widened files into.
            build_command: Shell command to build+test in Docker.
            ssh_compile_command: Shell command to compile on SSH hardware.
            ssh_run_command: Shell command to run on SSH hardware.
            max_steps: Maximum widening passes.
            test_data_dir: Directory with test data files.
        """
        if build_command is None:
            build_command = default_docker_build_command()
        if ssh_compile_command is None:
            ssh_compile_command = default_ssh_compile_command()
        if ssh_run_command is None:
            ssh_run_command = default_ssh_run_command()

        logger.info(
            "Starting widening for %s in %s", target_file, source_dir,
        )

        # Load all files from the source directory
        file_names = [f.name for f in source_dir.iterdir() if f.is_file()]
        if target_file not in file_names:
            raise ValueError(
                f"Target file {target_file} not found in {source_dir}"
            )

        snapshot = SourceSnapshot(
            files={name: (source_dir / name).read_text() for name in file_names}
        )

        workspaces = create_workspace(source_dir, snapshot, test_data_dir)

        # Compute Intel reference using ORIGINAL source (not translated RVV)
        self._get_intel_reference(ORIGINAL_SOURCE_DIR, test_data_dir)

        try:
            # --- Baseline validation (input must already compile) ---
            baseline = self.docker_validator.validate(
                workspaces.workspace_dir, build_command,
            )
            logger.info(
                "Baseline validation: ok=%s stage=%s rc=%s\n%s",
                baseline.ok,
                baseline.stage,
                baseline.returncode,
                truncate_for_log(baseline.combined_output, 3000),
            )

            if not baseline.ok:
                logger.warning(
                    "Baseline validation FAILED — the input code does not "
                    "compile/run.  Cannot proceed with widening."
                )
                return 1

            # --- Multi-pass widening loop (no per-pass retries) ---
            current_snapshot = snapshot
            last_error: str | None = None

            for step in range(1, max_steps + 1):
                logger.info(
                    "Widening pass %d/%d for %s",
                    step, max_steps, target_file,
                )
                current_code = current_snapshot.files[target_file]

                # Build prompt — include errors from previous pass
                if step == 1:
                    user_content = build_widen_initial_prompt(
                        target_file,
                        current_code,
                        build_command,
                        validation_feedback=last_error,
                    )
                else:
                    user_content = build_widen_continue_prompt(
                        target_file,
                        current_code,
                        build_command,
                        pass_number=step,
                        validation_feedback=last_error,
                    )

                messages = [
                    {
                        "role": "system",
                        "content": build_widen_system_prompt(target_file),
                    },
                    {"role": "user", "content": user_content},
                ]

                result, validation = self._generate_single_pass(
                    messages,
                    current_snapshot,
                    target_file,
                    workspaces,
                    build_command,
                )

                # ALL_WIDENED signal — done
                if validation.ok and validation.stage == "all-widened":
                    logger.info(
                        "All SSE intrinsics widened after %d pass(es)",
                        step - 1,
                    )
                    break

                # Internal error — abort
                if result is None and validation.stage == "internal-error":
                    logger.warning(
                        "Stopping early due to unrecoverable error: %s",
                        validation.stderr,
                    )
                    return 1

                # No valid edits produced — carry error forward
                if result is None:
                    logger.info(
                        "Pass %d did not yield a valid candidate (%s), "
                        "carrying error forward",
                        step, validation.stage,
                    )
                    last_error = validation.stderr or validation.stdout
                    continue

                # Accept the snapshot regardless of validation outcome
                current_snapshot = result

                if validation.ok:
                    logger.info("Pass %d compiled and ran OK", step)
                    last_error = None
                else:
                    logger.info(
                        "Pass %d has errors — carrying forward to next pass",
                        step,
                    )
                    last_error = validation.as_feedback()

            # --- Write output ---
            write_output(output_dir, current_snapshot)
            logger.info("Wrote widened output to %s", output_dir)

            # --- Final benchmark: baseline (original) vs widened ---
            materialize_snapshot(workspaces.workspace_dir, snapshot)
            baseline_time = self._benchmark_on_ssh(
                workspaces.workspace_dir,
                ssh_compile_command,
                ssh_run_command,
                "baseline (sse2rvv 128-bit)",
            )

            materialize_snapshot(workspaces.workspace_dir, current_snapshot)
            final_time = self._benchmark_on_ssh(
                workspaces.workspace_dir,
                ssh_compile_command,
                ssh_run_command,
                "final (widened RVV)",
            )
            if baseline_time and final_time:
                speedup = baseline_time / final_time
                logger.info(
                    "Final speedup: %.2fx (%.2fs -> %.2fs)",
                    speedup, baseline_time, final_time,
                )

            # --- Final correctness check against Intel reference ---
            materialize_snapshot(workspaces.workspace_dir, current_snapshot)
            correctness = self._validate_correctness(
                workspaces.workspace_dir,
            )
            if correctness is None:
                logger.warning(
                    "Correctness check skipped (reference unavailable)"
                )
            elif correctness.ok:
                logger.info(
                    "FINAL CORRECTNESS CHECK PASSED — widened code matches "
                    "Intel SSE reference"
                )
            else:
                logger.warning(
                    "FINAL CORRECTNESS CHECK FAILED — widened code differs "
                    "from Intel SSE reference\n%s",
                    correctness.combined_output,
                )
                return 1

            return 0

        finally:
            logger.debug("Cleaning up workspace at %s", workspaces.root)
            shutil.rmtree(workspaces.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Widen sse2rvv.h-translated RISC-V code to native RVV intrinsics"
        ),
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory with sse2rvv.h-translated source files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write widened files into (created if needed)",
    )
    parser.add_argument(
        "--target-file",
        default="ssw.c",
        help="Name of the file to widen (default: ssw.c)",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Shell command to compile and test (run inside Docker container)",
    )
    parser.add_argument(
        "--ssh-compile",
        default=None,
        help="Shell command to compile on SSH hardware",
    )
    parser.add_argument(
        "--ssh-run",
        default=None,
        help="Shell command to run on SSH hardware",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=REACT_MAX_STEPS,
        help=f"Maximum widening passes (default: {REACT_MAX_STEPS})",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DATASETS_DIR,
        help="Directory with test data files; copied into workspace as demo/",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return WidenAgent().run(
        source_dir=args.source_dir,
        target_file=args.target_file,
        output_dir=args.output_dir,
        build_command=args.build_command,
        ssh_compile_command=args.ssh_compile,
        ssh_run_command=args.ssh_run,
        max_steps=args.max_steps,
        test_data_dir=args.test_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
