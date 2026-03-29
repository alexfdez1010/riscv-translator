"""Vector-width optimization pipeline for RISC-V Vector (RVV) code.

Takes translated RVV code that is constrained to 128-bit (SSE-compatible)
vector operations and iteratively widens it to exploit the full hardware
VLEN.  Uses the same LLM compile-fix loop as ``repair.py``:

  1. Start with 128-bit-constrained translated code.
  2. LLM proposes diffs to widen a portion of the code.
  3. Compile + run on simulator (VLEN=128) to verify no regression.
  4. Compile + run on SSH hardware (VLEN=256) to verify wider execution.
  5. Compare both outputs against Intel reference for correctness.
  6. Feed errors back to LLM and iterate.

On success the widened code is written to the output directory.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from src.llm_types import LLM, Message
from src.config import (
    DATASETS_DIR,
    LLM_VALIDATION_RETRIES,
    PROJECT_DIR,
    REACT_MAX_STEPS,
    REMOTE_DIR,
    RISCVCC,
    RVV_REFERENCE,
    SIMULATOR,
    SSH_CC,
    SSH_HOST,
    SSH_JUMP_HOST,
    VLEN,
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
from src.validators import (
    DockerValidator,
    SSHValidator,
    ValidationResult,
)

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 16000
CORRECTNESS_DATASET = "10k.fa"

DEFAULT_SOURCE_DIR = PROJECT_DIR / "translations" / "sequence-alignment"
ORIGINAL_SOURCE_DIR = PROJECT_DIR / "initial_code"

# Separator used to concatenate .h and .c files for the LLM
_FILE_SEPARATOR = "\n/* ===== END ssw.h / BEGIN ssw.c ===== */\n"
HEADER_FILE = "ssw.h"


def truncate_for_log(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# Snapshot and workspace (reused from repair.py pattern)
# ---------------------------------------------------------------------------


class SourceSnapshot:
    __slots__ = ("files",)

    def __init__(self, files: dict[str, str]):
        self.files = files


class WorkspaceSet:
    __slots__ = ("root", "workspace_dir")

    def __init__(self, root: Path, workspace_dir: Path):
        self.root = root
        self.workspace_dir = workspace_dir


def materialize_snapshot(workspace_dir: Path, snapshot: SourceSnapshot) -> None:
    for name, content in snapshot.files.items():
        (workspace_dir / name).write_text(content)


def create_workspace(
    source_dir: Path,
    snapshot: SourceSnapshot,
    test_data_dir: Path | None = None,
) -> WorkspaceSet:
    root = Path(tempfile.mkdtemp(prefix="widen-"))
    workspace_dir = root / "workspace"
    shutil.copytree(source_dir, workspace_dir)
    materialize_snapshot(workspace_dir, snapshot)
    if test_data_dir is not None and test_data_dir.is_dir():
        demo_dir = workspace_dir / "demo"
        shutil.copytree(test_data_dir, demo_dir)
    return WorkspaceSet(root=root, workspace_dir=workspace_dir)


def apply_content_to_snapshot(
    snapshot: SourceSnapshot, file_name: str, content: str
) -> SourceSnapshot:
    if file_name not in snapshot.files:
        raise ValueError(f"Unknown target file: {file_name}")
    updated = dict(snapshot.files)
    updated[file_name] = content
    return SourceSnapshot(files=updated)


def write_output(output_dir: Path, snapshot: SourceSnapshot) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in snapshot.files.items():
        (output_dir / name).write_text(content)
    logger.info("Wrote %d file(s) to %s", len(snapshot.files), output_dir)


def concat_header_and_source(snapshot: SourceSnapshot, target_file: str) -> str:
    """Concatenate ssw.h + ssw.c into a single string for the LLM."""
    header = snapshot.files.get(HEADER_FILE, "")
    source = snapshot.files[target_file]
    return header + _FILE_SEPARATOR + source


def split_header_and_source(combined: str) -> tuple[str, str]:
    """Split the concatenated string back into (ssw.h, ssw.c)."""
    parts = combined.split(_FILE_SEPARATOR, 1)
    if len(parts) != 2:
        raise ValueError("Could not find file separator in LLM output")
    return parts[0], parts[1]


def _to_messages(raw: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Prompts for widening
# ---------------------------------------------------------------------------


def _rvv_intrinsic_cheatsheet() -> str:
    """Compact RVV intrinsic reference to prevent the LLM from inventing names."""
    return """\
## RVV Intrinsic Quick Reference (MUST use these exact names)

### Naming convention
`__riscv_<op>_<form>_<type>(args..., vl)`
- form: `v` (unary/load), `vv` (vector-vector), `vx` (vector-scalar),
  `vf` (vector-float-scalar)
- ALL vector intrinsics require a `vl` (vector length) parameter as last arg.

### Vector length
- `size_t vl = __riscv_vsetvlmax_e8m1();`   — max VL for 8-bit, LMUL=1
- `size_t vl = __riscv_vsetvlmax_e16m1();`  — max VL for 16-bit, LMUL=1
- `size_t vl = __riscv_vsetvl_e8m1(avl);`   — set VL with AVL cap

### Splat (broadcast scalar to vector) — note the `_x_` infix
- `vuint8m1_t  v = __riscv_vmv_v_x_u8m1(val, vl);`
- `vint8m1_t   v = __riscv_vmv_v_x_i8m1(val, vl);`
- `vuint16m1_t v = __riscv_vmv_v_x_u16m1(val, vl);`
- `vint16m1_t  v = __riscv_vmv_v_x_i16m1(val, vl);`

### Loads — note the `_v_` infix
- `vuint8m1_t  v = __riscv_vle8_v_u8m1(ptr, vl);`
- `vint8m1_t   v = __riscv_vle8_v_i8m1(ptr, vl);`
- `vuint16m1_t v = __riscv_vle16_v_u16m1(ptr, vl);`
- `vint16m1_t  v = __riscv_vle16_v_i16m1(ptr, vl);`

### Stores
- `__riscv_vse8_v_u8m1(ptr, val, vl);`
- `__riscv_vse8_v_i8m1(ptr, val, vl);`
- `__riscv_vse16_v_u16m1(ptr, val, vl);`
- `__riscv_vse16_v_i16m1(ptr, val, vl);`

### Arithmetic (unsigned 8-bit examples — same pattern for i8/u16/i16)
- `__riscv_vadd_vv_u8m1(a, b, vl)`         — add
- `__riscv_vsub_vv_u8m1(a, b, vl)`         — subtract
- `__riscv_vmaxu_vv_u8m1(a, b, vl)`        — unsigned max
- `__riscv_vmax_vv_i16m1(a, b, vl)`        — signed max (for int16)
- `__riscv_vsaddu_vv_u8m1(a, b, vl)`       — saturating add (unsigned)
- `__riscv_vssubu_vv_u8m1(a, b, vl)`       — saturating sub (unsigned)
- `__riscv_vsadd_vv_i16m1(a, b, vl)`       — saturating add (signed)
- `__riscv_vssub_vv_i16m1(a, b, vl)`       — saturating sub (signed)

### Slide (shift lanes)
- `__riscv_vslideup_vx_u8m1(dst, src, offset, vl)`
- `__riscv_vslidedown_vx_u8m1(src, offset, vl)`

### Compare → mask
- `vbool8_t m = __riscv_vmseq_vv_u8m1_b8(a, b, vl);`   — equal
- `vbool8_t m = __riscv_vmseq_vx_u8m1_b8(a, 0, vl);`   — equal to scalar
- `vbool16_t m = __riscv_vmseq_vv_i16m1_b16(a, b, vl);` — 16-bit eq

### Mask → scalar
- `long bits = __riscv_vcpop_m_b8(mask, vl);`  — popcount of mask
- `long first = __riscv_vfirst_m_b8(mask, vl);` — index of first set bit

### Reductions
- `vuint8m1_t r = __riscv_vredmaxu_vs_u8m1_u8m1(vec, scalar_vec, vl);`
- `vint16m1_t r = __riscv_vredmax_vs_i16m1_i16m1(vec, scalar_vec, vl);`
- Extract scalar: `uint8_t s = __riscv_vmv_x_s_u8m1_u8(r);`
- Extract scalar: `int16_t s = __riscv_vmv_x_s_i16m1_i16(r);`

### CRITICAL: Common mistakes to avoid
- WRONG: `__riscv_vmv_v_u8m1(val)` → CORRECT: `__riscv_vmv_v_x_u8m1(val, vl)`
- WRONG: `__riscv_vle8_u8m1(ptr, vl)` → CORRECT: `__riscv_vle8_v_u8m1(ptr, vl)`
- WRONG: any intrinsic without `vl` param → ALL intrinsics need `vl`
- WRONG: `_mm_extract_epi16(v, 0)` for max → use `vredmaxu`/`vredmax` + `vmv_x_s`
- WRONG: `_mm_movemask_epi8(v)` → use `vmseq` + `vcpop` or `vfirst`
"""


def build_widen_system_prompt(target_file: str) -> str:
    # Include the full RVV reference if available, plus the compact cheatsheet
    rvv_ref_section = ""
    if RVV_REFERENCE:
        rvv_ref_section = f"""

## RVV Reference Material

The following is the authoritative reference for RVV C intrinsics.
Use it to verify intrinsic names and signatures before writing code.

{RVV_REFERENCE}
"""

    return f"""\
You are an expert systems programmer specialising in RISC-V Vector (RVV)
optimization.  Your task is to incrementally widen C/C++ code that currently
uses fixed 128-bit vector operations (via sse2rvv.h SSE intrinsics) so
that it exploits the full hardware vector length (VLEN).

## Background

The code was translated from x86 SSE to RISC-V using sse2rvv.h — a
drop-in compatibility header.  Currently every vector operation processes
exactly 16 bytes (128 bits) regardless of the hardware VLEN, because the
code still uses SSE intrinsics like `_mm_load_si128`, `_mm_add_epi8`, etc.

Your job is to replace SSE intrinsics with native RVV intrinsics that
operate on the full hardware VLEN, so that wider hardware (e.g. VLEN=256,
512, 1024) processes more data per instruction.

{_rvv_intrinsic_cheatsheet()}
{rvv_ref_section}
## Widening strategy

Work in SMALL, ATOMIC increments.  Each pass should widen ONE tightly
coupled unit — for example, one function, one loop, or one data structure
— while keeping everything else unchanged and compiling correctly.

**CRITICAL rules for incremental widening:**

- Change at most ONE tightly coupled unit per pass.  If a function
  (e.g. a profile/lookup-table builder) produces data consumed by
  another function (e.g. a DP kernel or inner loop), you MUST widen
  both together in the same pass so the data layout stays consistent.
  Do NOT leave stride or type mismatches between producer and consumer.
- Limit yourself to at most 5 search/replace blocks per response.
  If you need more, stop and do the rest in the next pass.
- Keep your search/replace blocks SHORT — prefer several small blocks
  over one huge block.  Long blocks are more likely to fail matching.
- After widening, ALL remaining code must still compile and work
  correctly.  Do not leave type mismatches between widened and
  non-widened parts.

### Key transformations

1. **Header**: Replace `#include "sse2rvv.h"` with `#include <riscv_vector.h>`.
   Only do this when ALL `_mm_*` intrinsics in the file have been replaced.

2. **Vector types**: Replace `__m128i` with native RVV types (e.g.
   `vint8m1_t`, `vuint8m1_t`, `vint16m1_t`, etc.) chosen to match the
   element type used in that context.  For pointers to vector arrays
   (e.g. `__m128i*` used for profile/DP storage), change to `uint8_t*`
   (raw byte arrays with vectors stored contiguously).

3. **Vector length helpers**: Add small inline helper functions that
   query the runtime vector length.  For example:
   ```c
   static inline size_t vlmax_e8(void)  {{ return __riscv_vsetvlmax_e8m1(); }}
   static inline size_t vlmax_e16(void) {{ return __riscv_vsetvlmax_e16m1(); }}
   static inline size_t vregbytes(void) {{ return __riscv_vsetvlmax_e8m1(); }}
   ```
   Use these throughout instead of hardcoded constants (16 for byte
   lanes, 8 for word lanes).

4. **Memory access**: Replace `_mm_load_si128` / `_mm_store_si128` with
   `__riscv_vle8_v_u8m1(ptr, vl)` / `__riscv_vse8_v_u8m1(ptr, val, vl)`
   (or the appropriate element-width variant).

5. **Arithmetic**: Replace `_mm_add_epi8` with `__riscv_vadd_vv_i8m1`,
   `_mm_max_epu8` with `__riscv_vmaxu_vv_u8m1`, etc.  Use the RVV
   intrinsic that matches the element type and operation.

6. **Allocation**: Replace `calloc(n, 16)` or `malloc(n * 16)` with
   allocation sized to `n * vl` where `vl` is the runtime vector length
   in bytes.

7. **Segment length / loop bounds**: Where the code uses
   `segLen = (len + 15) / 16` or `(len + 7) / 8`, update to use the
   runtime vector length: `segLen = (len + vl - 1) / vl` where `vl`
   is in element units.

8. **Pointer arithmetic**: Replace `(uint8_t*)ptr + j * 16` with
   `(uint8_t*)ptr + j * vl_bytes` where `vl_bytes` is the vector
   register size in bytes.

9. **Shuffles and byte manipulation**: SSE shuffles (`_mm_shuffle_epi8`,
   `_mm_slli_si128`, `_mm_srli_si128`) need careful conversion.  For
   shift-by-N-bytes, use `__riscv_vslideup` / `__riscv_vslidedown`.
   For table lookups, use `__riscv_vrgather`.

10. **Horizontal reductions**: Replace manual tree reductions (macros
    using repeated `srli` + `max`) with RVV reduction intrinsics:
    `__riscv_vredmaxu_vs_u8m1_u8m1` (unsigned 8-bit),
    `__riscv_vredmax_vs_i16m1_i16m1` (signed 16-bit), followed by
    `__riscv_vmv_x_s_*` to extract the scalar result.

11. **Mask operations**: Replace `_mm_movemask_epi8` patterns with RVV
    mask intrinsics: `__riscv_vmseq`/`__riscv_vmsgt` to produce masks,
    `__riscv_vcpop` to count set bits, `__riscv_vfirst` for first set
    bit.

12. **Mixed signed/unsigned**: For operations that mix signed comparison
    with unsigned saturating subtract (common in 16-bit paths), use
    `__riscv_vreinterpret_v_i16m1_u16m1` / `_u16m1_i16m1` to cast
    between signed and unsigned views of the same vector.

## Important constraints

- **Correctness first**: The widened code MUST produce identical output
  to the original 128-bit version.  Algorithms that interleave data
  across vector lanes (striped/segmented layouts) are especially
  sensitive — changing VLEN changes the number of lanes, so data layout,
  profile construction, and any DP recurrence must all be updated
  consistently.

- **Data layout consistency**: If the code uses a segmented/striped
  memory layout (e.g. interleaving positions across vector lanes),
  the builder function and the kernel that consumes the layout must
  use the same vector length.  Widening one without the other will
  silently corrupt results.

- **Keep sse2rvv.h until fully done**: The header is still needed for
  any SSE intrinsics that haven't been widened yet.  Only remove it
  when ALL `_mm_*` intrinsics in the file have been replaced.

- **Compile for rv64gcv**: The code must compile with
  `-march=rv64gcv -mabi=lp64d` or `--target=riscv64-linux-gnu -march=rv64imafdcv`.

## Progress markers

After you finish widening a function or section, add a comment at the
top of that function to mark it as completed:

    /* RVV-WIDENED: this function uses native RVV intrinsics with runtime VLEN */

In subsequent passes, DO NOT modify any function or section that already
has a `RVV-WIDENED` marker.  Focus only on sections that still use
`_mm_*` SSE intrinsics and do NOT have this marker.

## Rules

- You may modify both the header and `{target_file}` (they are shown
  concatenated).
- Make SMALL incremental changes — do not try to widen everything at once.
- Each search/replace block should be small and focused.
- Preserve the algorithm — only change the SIMD layer.
- The code must continue to work correctly at VLEN=128 after widening
  (VLEN=128 is the minimum; the code must be VLEN-agnostic).
- Double-check every intrinsic name against the cheatsheet above before
  writing it.  Wrong names cause compile failures that waste retries.
- Do NOT touch functions marked with `RVV-WIDENED`.
- When renaming functions (e.g. `_sse2_` → `_rvv_`), update ALL call
  sites in the same pass.

## Output format

Return only:
1) A short summary of what you are widening in this pass.
2) One or more search/replace blocks with the changes (max 5 blocks).

{search_replace_format_example()}
""".strip()


def build_widen_initial_prompt(
    target_file: str,
    source_code: str,
    build_command: str,
    validation_feedback: str | None = None,
) -> str:
    validation_section = ""
    if validation_feedback:
        validation_section = f"""

Current validation result:
{validation_feedback}
""".rstrip()

    return f"""\
Task: Widen vector operations in this file to use the full hardware VLEN.

The code currently uses sse2rvv.h SSE intrinsics that process exactly
16 bytes per operation.  Replace a portion of these with native RVV
intrinsics that process `vl` elements per operation, where `vl` is
determined at runtime via `__riscv_vsetvlmax_*()`.

**IMPORTANT**: Make a SMALL change — widen ONE tightly coupled unit
(e.g. one function and its callers' type signatures).  If a function
builds a data structure consumed by another function, you MUST widen
both together so the data layout stays consistent.  Do NOT leave type
or stride mismatches.

Use at most 5 search/replace blocks.  Keep each block short.
Double-check every RVV intrinsic name against the cheatsheet in the
system prompt before writing it.

Context:
- Target file: {target_file}
- Build command: {build_command}

Current code:
```c
{source_code}
```
{validation_section}

Output:
- First, a short summary of what you are widening.
- Then, one or more search/replace blocks (max 5).
{search_replace_format_example()}
""".strip()


def build_widen_repair_prompt(
    target_file: str,
    code: str,
    validation_feedback: str,
) -> str:
    return f"""\
Task: Fix the validation failure from the previous widening attempt.

The last change introduced a compilation or correctness error.
Fix it with the smallest correct change.

IMPORTANT: If the error is about an unknown intrinsic name, check the
RVV intrinsic cheatsheet in the system prompt carefully.  Common mistakes:
- Missing `_v_` infix in loads: `__riscv_vle8_v_u8m1` (not `__riscv_vle8_u8m1`)
- Missing `_x_` infix in splats: `__riscv_vmv_v_x_u8m1` (not `__riscv_vmv_v_u8m1`)
- Missing `vl` parameter: ALL RVV intrinsics require `vl` as last argument
- Wrong reinterpret direction: `__riscv_vreinterpret_v_i16m1_u16m1` converts
  i16→u16, `_u16m1_i16m1` converts u16→i16

Context:
- Target file: {target_file}
- The code is being widened from 128-bit SSE intrinsics to native RVV.

Current code:
```c
{code}
```

Validation failure:
{validation_feedback}

Output:
- First, one short summary sentence.
- Then, one or more search/replace blocks.
{search_replace_format_example()}
""".strip()


def build_widen_continue_prompt(
    target_file: str,
    source_code: str,
    build_command: str,
    pass_number: int,
    validation_feedback: str | None = None,
) -> str:
    if validation_feedback:
        preamble = f"""\
Task: Continue widening vector operations (pass {pass_number}).

The previous pass had validation errors (shown below).  You MUST fix
these errors first before attempting any new widening.  Make the
smallest change that resolves the failure."""
    else:
        preamble = f"""\
Task: Continue widening vector operations (pass {pass_number}).

The previous pass succeeded.  Now widen the next section of the code
that still uses 128-bit SSE intrinsics via sse2rvv.h."""

    validation_section = ""
    if validation_feedback:
        validation_section = f"""

Current validation errors (FIX THESE FIRST):
{validation_feedback}
"""

    return f"""\
{preamble}

Skip any function already marked with `/* RVV-WIDENED */` — those are
done.  Look for remaining `_mm_*` intrinsic calls in unmarked functions
and replace them with native RVV intrinsics.  After widening a function,
add the `/* RVV-WIDENED: ... */` marker comment at the top.

If all functions are already marked `RVV-WIDENED` and no `_mm_*` calls
remain, also:
- Remove `#include "sse2rvv.h"` from any file that no longer uses SSE
  intrinsics (replace with `#include <riscv_vector.h>` if not already
  included, or a comment like `/* RVV-WIDENED: sse2rvv.h no longer needed */`).
- Update any call sites that reference old function names.

If ALL widening is complete and no `_mm_*` calls remain in any file,
respond with exactly:
"ALL_WIDENED: No more SSE intrinsics to widen."

**IMPORTANT**: Make a SMALL change — at most ONE tightly coupled unit
per pass.  Use at most 5 search/replace blocks.  Keep each block short.
Double-check every RVV intrinsic name against the cheatsheet in the
system prompt.

Context:
- Target file: {target_file}
- Build command: {build_command}

Current code:
```c
{source_code}
```
{validation_section}
Output:
- First, a short summary of what you are widening in this pass.
- Then, one or more search/replace blocks (max 5).
{search_replace_format_example()}
""".strip()


def build_widen_edit_format_feedback(
    file_name: str, code: str, error_message: str
) -> str:
    return search_replace_error_feedback(file_name, code, error_message)


# ---------------------------------------------------------------------------
# Build commands
# ---------------------------------------------------------------------------


def default_docker_build_command() -> str:
    cflags = "-O2 -I. -march=rv64gcv -mabi=lp64d"
    ldflags = "-lm"
    return (
        f'echo "=== Compiling ===" && '
        f'{RISCVCC} {cflags} main.c ssw.c -o ssw_test {ldflags} 2>&1 && '
        f'echo "=== Compilation succeeded, running under QEMU ===" && '
        f'{SIMULATOR} ./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa >/dev/null 2>&1 && '
        f'echo "=== Execution succeeded ==="'
    )


def default_ssh_compile_command() -> str:
    return f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"


def default_ssh_run_command() -> str:
    return "./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa 2>&1"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def _detect_target_file(source_dir: Path, file_names: list[str]) -> str:
    """Find the .c file that contains SSE intrinsics (_mm_*)."""
    candidates = []
    for name in file_names:
        if not name.endswith(".c"):
            continue
        content = (source_dir / name).read_text()
        if "_mm_" in content:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # If multiple .c files have SSE intrinsics, pick the one that is NOT main.c
        non_main = [c for c in candidates if c != "main.c"]
        if len(non_main) == 1:
            return non_main[0]
        raise ValueError(
            f"Multiple .c files contain SSE intrinsics: {candidates}. "
            "Specify target_file explicitly."
        )
    raise ValueError(
        f"No .c file with SSE intrinsics (_mm_*) found in {source_dir}. "
        "Specify target_file explicitly."
    )


# ---------------------------------------------------------------------------
# Widening agent
# ---------------------------------------------------------------------------


class WidenAgent:
    """LLM-driven vector-width optimization with compile-fix loop."""

    def __init__(self):
        self.docker_validator = DockerValidator()
        self.ssh_validator = SSHValidator()
        self.llm: LLM = create_llm()
        self._intel_reference: BenchmarkResult | None = None
        self._intel_reference_computed = False

    def _get_intel_reference(self, original_dir: Path, test_data_dir: Path | None) -> BenchmarkResult | None:
        """Compute Intel reference by compiling the *original* SSE code on the jump host.

        The original_dir must contain the unmodified SSE source (e.g. initial_code/)
        — NOT the translated RVV code, which uses sse2rvv.h and won't compile on x86.
        """
        if self._intel_reference_computed:
            return self._intel_reference

        self._intel_reference_computed = True
        jump_host = SSH_JUMP_HOST
        if not check_ssh(jump_host):
            logger.warning("Jump host %s not reachable; correctness check disabled", jump_host)
            return None

        jump_remote = f"{REMOTE_DIR}-widen-ref"
        dataset_dir = test_data_dir if test_data_dir and test_data_dir.is_dir() else DATASETS_DIR

        dataset_file = dataset_dir / CORRECTNESS_DATASET
        if not dataset_file.exists():
            logger.warning("Correctness dataset %s not found; correctness check disabled", dataset_file)
            return None

        # Upload the ORIGINAL SSE source (not the translated RVV code)
        original_files = [p for p in original_dir.iterdir() if p.is_file()]
        logger.info("Uploading original SSE code to %s for Intel reference ...", jump_host)
        if not upload_to_host(jump_host, jump_remote, original_files):
            logger.warning("Failed to upload reference code; correctness check disabled")
            return None
        if not upload_datasets(jump_host, jump_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets; correctness check disabled")
            return None

        run_cmd = f"./ssw_test demo/{CORRECTNESS_DATASET} demo/{BENCH_REFERENCE_FILE} 2>&1"
        compile_cmd = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"

        result = run_on_host(jump_host, jump_remote, compile_cmd, run_cmd, "Intel reference")
        if not result.ok:
            logger.warning("Intel reference run failed; correctness check disabled\n%s", result.stderr)
            return None

        logger.info("Intel reference output computed (%d chars)", len(result.stdout))
        self._intel_reference = result
        return result

    def _validate_correctness(self, workspace_dir: Path) -> ValidationResult | None:
        if self._intel_reference is None:
            return None

        final_host = SSH_HOST
        final_remote = f"{REMOTE_DIR}-widen-correctness"

        all_paths = [p for p in workspace_dir.iterdir()]
        if not upload_to_host(final_host, final_remote, all_paths):
            logger.warning("Failed to upload widened code for correctness check")
            return None

        dataset_dir = DATASETS_DIR
        if not upload_datasets(final_host, final_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets for correctness check")
            return None

        run_cmd = f"./ssw_test demo/{CORRECTNESS_DATASET} demo/{BENCH_REFERENCE_FILE} 2>&1"
        compile_cmd = default_ssh_compile_command()

        riscv_result = run_on_host(final_host, final_remote, compile_cmd, run_cmd, "RISC-V widened")
        if not riscv_result.ok:
            return ValidationResult(
                ok=False,
                stage="correctness",
                returncode=riscv_result.ok,
                stdout=riscv_result.stdout,
                stderr=f"RISC-V widened run failed:\n{riscv_result.stderr}",
            )

        match, details = compare_outputs(self._intel_reference, riscv_result)
        if match:
            logger.info("Correctness check PASSED: widened output matches Intel reference")
            return ValidationResult(ok=True, stage="correctness", returncode=0, stdout=details, stderr="")

        logger.warning("Correctness check FAILED: widened output differs from Intel reference")
        return ValidationResult(
            ok=False,
            stage="correctness",
            returncode=1,
            stdout="",
            stderr=(
                "CORRECTNESS FAILURE: The widened RISC-V code produces different "
                "alignment results than the Intel SSE reference.\n\n"
                f"{details}\n\n"
                "The widening introduced a bug. Common causes:\n"
                "1. Segment length (segLen) not updated consistently — profile "
                "construction and DP loop must use the same vector length.\n"
                "2. Memory allocation still uses hardcoded 16 instead of runtime "
                "vector byte size.\n"
                "3. Pointer arithmetic stride mismatch between write and read paths.\n"
                "4. Shuffle/slide operations not adjusted for wider vectors.\n"
                "5. Boundary conditions (e.g. segLen calculation, loop termination) "
                "not updated for the new vector length.\n\n"
                "Fix the bug while keeping the code VLEN-agnostic."
            ),
        )

    def _benchmark_on_ssh(
        self,
        workspace_dir: Path,
        ssh_compile_cmd: str,
        ssh_run_cmd: str,
        label: str,
    ) -> float | None:
        """Run a timed execution on SSH hardware. Returns elapsed seconds or None on failure."""
        final_host = SSH_HOST
        if not check_ssh(final_host):
            return None

        bench_remote = f"{REMOTE_DIR}-widen-bench"
        all_paths = [p for p in workspace_dir.iterdir()]
        if not upload_to_host(final_host, bench_remote, all_paths):
            return None

        dataset_dir = DATASETS_DIR
        if not upload_datasets(final_host, bench_remote, dataset_dir, CORRECTNESS_DATASET):
            return None

        result = run_on_host(final_host, bench_remote, ssh_compile_cmd, ssh_run_cmd, label)
        if not result.ok:
            logger.warning("Benchmark run failed for %s: %s", label, result.stderr)
            return None

        return result.elapsed_seconds

    def _generate_valid_file(
        self,
        messages: list[dict[str, str]],
        snapshot: SourceSnapshot,
        file_name: str,
        workspaces: WorkspaceSet,
        build_command: str,
    ) -> tuple[SourceSnapshot | None, ValidationResult]:
        """Run one LLM request cycle with retries on diff/validation failure."""
        active_messages = list(messages)
        current_snapshot = snapshot
        latest_validation = ValidationResult(
            ok=False,
            stage="edit-failure",
            returncode=None,
            stdout="",
            stderr="No validation attempted.",
        )

        for attempt in range(LLM_VALIDATION_RETRIES + 1):
            try:
                logger.info(
                    "LLM request attempt %d with %d message(s)",
                    attempt + 1,
                    len(active_messages),
                )
                response = self.llm(_to_messages(active_messages))
            except Exception as exc:
                latest_validation = ValidationResult(
                    ok=False,
                    stage="internal-error",
                    returncode=None,
                    stdout="",
                    stderr=str(exc),
                )
                logger.warning("LLM generation failed on attempt %d: %s", attempt + 1, exc)
                return None, latest_validation

            logger.info(
                "LLM response (attempt %d, %d chars):\n%s",
                attempt + 1,
                len(response),
                truncate_for_log(response, 3000),
            )

            # Check if LLM says all widening is done
            if "ALL_WIDENED" in response and "No more SSE intrinsics" in response:
                latest_validation = ValidationResult(
                    ok=True,
                    stage="all-widened",
                    returncode=0,
                    stdout="LLM reports all SSE intrinsics have been widened.",
                    stderr="",
                )
                return None, latest_validation

            # Extract and apply search/replace blocks
            candidate_snapshot = None
            edit_error = None

            sr_blocks = extract_search_replace(response)
            if sr_blocks is not None:
                logger.info("Extracted %d search/replace block(s)", len(sr_blocks))
                try:
                    combined = concat_header_and_source(current_snapshot, file_name)
                    new_combined = apply_search_replace(combined, sr_blocks)
                    new_header, new_source = split_header_and_source(new_combined)
                    updated_files = dict(current_snapshot.files)
                    updated_files[HEADER_FILE] = new_header
                    updated_files[file_name] = new_source
                    candidate_snapshot = SourceSnapshot(files=updated_files)
                except ValueError as exc:
                    logger.warning("Search/replace failed on attempt %d: %s", attempt + 1, exc)
                    edit_error = str(exc)

            if candidate_snapshot is None:
                error_msg = edit_error or (
                    "Could not extract edits from the response. "
                    "Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
                )
                logger.warning("No valid edits on attempt %d: %s", attempt + 1, error_msg)
                if attempt >= LLM_VALIDATION_RETRIES:
                    return None, latest_validation
                active_messages = active_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_widen_edit_format_feedback(
                            file_name,
                            concat_header_and_source(current_snapshot, file_name),
                            error_msg,
                        ),
                    },
                ]
                continue

            # Validate in Docker/QEMU (VLEN=128 regression check)
            materialize_snapshot(workspaces.workspace_dir, candidate_snapshot)
            latest_validation = self.docker_validator.validate(
                workspaces.workspace_dir, build_command,
            )
            logger.info(
                "Validation result (attempt %d): ok=%s stage=%s rc=%s\n%s",
                attempt + 1,
                latest_validation.ok,
                latest_validation.stage,
                latest_validation.returncode,
                truncate_for_log(latest_validation.combined_output, 2000),
            )

            if latest_validation.ok:
                return candidate_snapshot, latest_validation

            if attempt >= LLM_VALIDATION_RETRIES:
                logger.warning(
                    "Validation failed after %d attempt(s); returning latest snapshot",
                    attempt + 1,
                )
                return candidate_snapshot, latest_validation

            # Feed errors back
            active_messages = active_messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": build_widen_repair_prompt(
                        file_name,
                        concat_header_and_source(candidate_snapshot, file_name),
                        latest_validation.as_feedback(),
                    ),
                },
            ]
            current_snapshot = candidate_snapshot

        return None, latest_validation

    def run(
        self,
        source_dir: Path,
        output_dir: Path,
        target_file: str | None = None,
        build_command: str | None = None,
        ssh_compile_command: str | None = None,
        ssh_run_command: str | None = None,
        max_steps: int = REACT_MAX_STEPS,
        test_data_dir: Path | None = None,
    ) -> int:
        """Run the widening pipeline.

        Args:
            source_dir: Directory with the 128-bit translated code.
            output_dir: Where to write the widened output.
            target_file: File to widen (e.g. "ssw.c").  If None, auto-detected
                as the .c file containing ``_mm_`` SSE intrinsics.
            build_command: Docker build+test command. Auto-generated if None.
            ssh_compile_command: SSH compile command. Auto-generated if None.
            ssh_run_command: SSH run command. Auto-generated if None.
            max_steps: Maximum widening passes.
            test_data_dir: Directory with test data; copied as demo/.
        """
        if build_command is None:
            build_command = default_docker_build_command()
        if ssh_compile_command is None:
            ssh_compile_command = default_ssh_compile_command()
        if ssh_run_command is None:
            ssh_run_command = default_ssh_run_command()

        file_names = [f.name for f in source_dir.iterdir() if f.is_file()]

        # Auto-detect target file if not specified
        if target_file is None:
            target_file = _detect_target_file(source_dir, file_names)

        if target_file not in file_names:
            raise ValueError(f"Target file {target_file} not found in {source_dir}")

        logger.info("Starting widening for %s in %s", target_file, source_dir)

        snapshot = SourceSnapshot(
            files={name: (source_dir / name).read_text() for name in file_names}
        )

        workspaces = create_workspace(source_dir, snapshot, test_data_dir)

        # Compute Intel reference using the ORIGINAL SSE source (initial_code/),
        # not the translated RVV code which won't compile on x86.
        self._get_intel_reference(ORIGINAL_SOURCE_DIR, test_data_dir)

        try:
            # Baseline: verify the input code compiles and runs
            baseline = self.docker_validator.validate(
                workspaces.workspace_dir, build_command,
            )
            logger.info(
                "Baseline validation: ok=%s stage=%s\n%s",
                baseline.ok,
                baseline.stage,
                truncate_for_log(baseline.combined_output, 3000),
            )

            if not baseline.ok:
                logger.error(
                    "Input code does not pass baseline validation; "
                    "fix compilation/runtime errors before widening."
                )
                return 1

            # Benchmark the original translated code to establish baseline timing
            baseline_elapsed = self._benchmark_on_ssh(
                workspaces.workspace_dir, ssh_compile_command, ssh_run_command,
                "original translated (baseline)",
            )
            if baseline_elapsed is not None:
                logger.info("Baseline timing: %.2fs", baseline_elapsed)
            else:
                logger.warning("Could not establish baseline timing; speedup tracking disabled")

            # Main widening loop — each step is one widening pass
            current_snapshot = snapshot
            last_known_good = snapshot
            successful_passes = 0
            pending_feedback: str | None = None  # validation errors to feed forward

            for step in range(1, max_steps + 1):
                logger.info("Widening pass %d/%d for %s", step, max_steps, target_file)
                current_code = concat_header_and_source(current_snapshot, target_file)

                if step == 1:
                    user_content = build_widen_initial_prompt(
                        target_file,
                        current_code,
                        build_command,
                        validation_feedback=pending_feedback,
                    )
                else:
                    user_content = build_widen_continue_prompt(
                        target_file,
                        current_code,
                        build_command,
                        pass_number=step,
                        validation_feedback=pending_feedback,
                    )

                messages = [
                    {"role": "system", "content": build_widen_system_prompt(target_file)},
                    {"role": "user", "content": user_content},
                ]

                repaired_snapshot, latest_validation = self._generate_valid_file(
                    messages,
                    current_snapshot,
                    target_file,
                    workspaces,
                    build_command,
                )

                # Check if LLM says everything is widened
                if latest_validation.stage == "all-widened":
                    logger.info("LLM reports all widening complete after %d pass(es)", successful_passes)
                    write_output(output_dir, last_known_good)
                    return 0

                if repaired_snapshot is None:
                    if latest_validation.stage == "internal-error":
                        logger.warning("Stopping due to unrecoverable error: %s", latest_validation.stderr)
                        write_output(output_dir, last_known_good)
                        return 1
                    logger.info("Pass %d did not yield a valid candidate, continuing with pending feedback", step)
                    pending_feedback = latest_validation.as_feedback() if not latest_validation.ok else None
                    continue

                if not latest_validation.ok:
                    logger.warning(
                        "Pass %d failed Docker/QEMU validation; keeping snapshot and feeding errors forward",
                        step,
                    )
                    current_snapshot = repaired_snapshot
                    pending_feedback = latest_validation.as_feedback()
                    continue

                # Docker/QEMU passed — try SSH hardware
                pending_feedback = None
                if ssh_compile_command and ssh_run_command:
                    ssh_files = [p for p in workspaces.workspace_dir.iterdir()]
                    ssh_result = self.ssh_validator.validate(
                        ssh_files, ssh_compile_command, ssh_run_command,
                    )
                    if not ssh_result.ok:
                        logger.warning(
                            "Pass %d passed QEMU but failed SSH at stage %s; "
                            "keeping snapshot and feeding errors forward\n%s",
                            step, ssh_result.stage, ssh_result.combined_output,
                        )
                        current_snapshot = repaired_snapshot
                        pending_feedback = ssh_result.as_feedback()
                        continue

                    # SSH passed — correctness check
                    correctness = self._validate_correctness(workspaces.workspace_dir)
                    if correctness is not None and not correctness.ok:
                        logger.warning(
                            "Pass %d passed SSH but failed correctness; "
                            "keeping snapshot and feeding errors forward\n%s",
                            step, correctness.combined_output,
                        )
                        current_snapshot = repaired_snapshot
                        pending_feedback = correctness.as_feedback()
                        continue

                # This pass fully validated
                current_snapshot = repaired_snapshot
                last_known_good = repaired_snapshot
                successful_passes += 1

                # Measure speedup vs original translated code
                step_elapsed = self._benchmark_on_ssh(
                    workspaces.workspace_dir, ssh_compile_command, ssh_run_command,
                    f"widened (pass {step})",
                )
                if step_elapsed is not None and baseline_elapsed is not None and baseline_elapsed > 0:
                    speedup = baseline_elapsed / step_elapsed
                    logger.info(
                        "Pass %d succeeded — %.2fs (%.2fx vs baseline %.2fs)",
                        step, step_elapsed, speedup, baseline_elapsed,
                    )
                elif step_elapsed is not None:
                    logger.info(
                        "Pass %d succeeded — %.2fs",
                        step, step_elapsed,
                    )
                else:
                    logger.info(
                        "Widening pass %d succeeded (%d total successful passes)",
                        step, successful_passes,
                    )

            logger.info(
                "Widening completed after %d step(s) (%d successful passes)",
                max_steps, successful_passes,
            )
            write_output(output_dir, last_known_good)
            return 0 if successful_passes > 0 else 1

        finally:
            shutil.rmtree(workspaces.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Widen RISC-V Vector code from 128-bit to full VLEN"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_SOURCE_DIR,
        help="Directory with translated 128-bit code (default: translations/sequence-alignment/)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=PROJECT_DIR / "widened",
        help="Directory for widened output (default: widened/)",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help="File to widen (default: auto-detect .c file with SSE intrinsics)",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Shell command to compile and test (Docker container)",
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
        help="Maximum widening passes",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DATASETS_DIR,
        help="Directory with test data files; copied as demo/",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return WidenAgent().run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_file=args.target_file,
        build_command=args.build_command,
        ssh_compile_command=args.ssh_compile,
        ssh_run_command=args.ssh_run,
        max_steps=args.max_steps,
        test_data_dir=args.test_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
