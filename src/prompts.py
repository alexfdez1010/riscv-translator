"""Prompt construction and RVV preprocessing for the SSW repair agent."""

import re

from src.config import REFERENCE_FILE
from src.diff_utils import diff_error_feedback, diff_format_example
from src.logger import get_logger
from src.validators import QEMU_RVV, RISCVCC

logger = get_logger(__name__)


def load_riscv_reference() -> str:
    return REFERENCE_FILE.read_text()


# ---------------------------------------------------------------------------
# Mechanical pre-processing: fix patterns that are always wrong on RVV
# ---------------------------------------------------------------------------

_VEC_HELPERS = """\
/* --- RVV compatibility helpers (sizeless __m128i) --- */
#if defined(__riscv) || defined(__riscv__)
/* Query actual vector register width at runtime (works for any VLEN). */
static inline int _ssw_vec_bytes(void) {
    return (int)__riscv_vsetvlmax_e8m1();
}
#define _SSW_VEC_BYTES _ssw_vec_bytes()
/* Number of 8-bit lanes per vector register (= VLEN / 8). */
#define _SSW_ELEMS_8  ((int)__riscv_vsetvlmax_e8m1())
static inline __m128i _load_vec(const void *base, int idx) {
    int vb = _ssw_vec_bytes();
    return _mm_load_si128((const __m128i *)((const uint8_t *)base + idx * vb));
}
static inline void _store_vec(void *base, int idx, __m128i v) {
    int vb = _ssw_vec_bytes();
    _mm_store_si128((__m128i *)((uint8_t *)base + idx * vb), v);
}
#define _VEC_PTR(p, i)  ((__m128i *)((uint8_t *)(p) + (i) * _ssw_vec_bytes()))
#else
#define _SSW_VEC_BYTES ((int)sizeof(__m128i))
#define _SSW_ELEMS_8   ((int)(sizeof(__m128i)))
#define _load_vec(base, idx) (*((const __m128i *)(base) + (idx)))
#define _store_vec(base, idx, v) (*(((__m128i *)(base)) + (idx)) = (v))
#define _VEC_PTR(p, i) ((p) + (i))
#endif
"""


def preprocess_rvv_compat(code: str) -> str:
    """Apply mechanical text-level fixes for RVV sizeless-type issues."""
    if "__m128i" not in code:
        return code

    original = code

    # 1. Replace sizeof(__m128i) with _SSW_VEC_BYTES BEFORE injecting helpers
    #    (so the helpers' own sizeof(__m128i) stays untouched).
    code = code.replace("sizeof(__m128i)", "_SSW_VEC_BYTES")

    # 2. Inject helper macros after kroundup32
    if "_SSW_VEC_BYTES" not in original:
        insert_marker = "#define kroundup32"
        idx = code.find(insert_marker)
        if idx != -1:
            eol = code.find("\n", idx)
            if eol != -1:
                code = code[: eol + 1] + "\n" + _VEC_HELPERS + code[eol + 1 :]

    # 3. Replace direct array indexing and pointer arithmetic on known
    #    __m128i* variable names ONLY (to avoid breaking int8_t* etc.)
    _vec_names = {"pvHStore", "pvHLoad", "pvE", "pvHmax", "vProfile", "vP"}

    # ptr + expr (pointer arithmetic on __m128i*) -> _VEC_PTR(ptr, expr)
    for name in _vec_names:
        code = re.sub(
            rf'\b{re.escape(name)}\s*\+\s*([^,;)]+?)\s*(?=[,;)])',
            rf'_VEC_PTR({name}, \1)',
            code,
        )

    # a[x] = b[y]; -> _store_vec(a, x, _load_vec(b, y)); for known names
    for lname in _vec_names:
        for rname in _vec_names:
            code = re.sub(
                rf'\b{re.escape(lname)}\[([^\]]+)\]\s*=\s*{re.escape(rname)}\[([^\]]+)\]\s*;',
                rf'_store_vec({lname}, \1, _load_vec({rname}, \2));',
                code,
            )

    # name[segLen - 1] as rvalue -> _load_vec(name, segLen - 1)
    for name in _vec_names:
        code = re.sub(
            rf'\b{re.escape(name)}\[segLen\s*-\s*1\](?!\s*=)',
            rf'_load_vec({name}, segLen - 1)',
            code,
        )

    # Remaining simple rvalue indexing: name[expr] -> _load_vec(name, expr)
    for name in _vec_names:
        code = re.sub(
            rf'\b{re.escape(name)}\[([^\]]+)\](?!\s*=)',
            rf'_load_vec({name}, \1)',
            code,
        )

    if code != original:
        logger.info("Pre-processing applied RVV compatibility fixes")
    return code


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def build_system_prompt(target_file_name: str) -> str:
    return f"""
You are a senior software engineer acting as a ReAct-style repair agent for a partially broken Intel-to-RISC-V port of the Striped Smith-Waterman library.
Make the smallest correct change needed.

Context:
- Work from the current candidate kept in memory.
- Repair the provided file incrementally until the full project compiles and runs successfully under the provided validation flow.
- Keep the project structure and behavior intact.
- The code uses sse2rvv.h to translate SSE2 intrinsics to RISC-V Vector (RVV).
- On RVV, __m128i is mapped to vint32m1_t which is a sizeless type:
  - sizeof(__m128i) is ILLEGAL — replace with _SSW_VEC_BYTES (a runtime macro that queries the actual vector register width, works for any VLEN).
  - NEVER hardcode 16 for the vector byte width. The hardware VLEN may be 128, 256, or larger. Always use _SSW_VEC_BYTES or _SSW_ELEMS_8 instead.
  - Pointer arithmetic on __m128i* (e.g. ptr + j) is ILLEGAL — use byte-level arithmetic instead: ((__m128i *)((uint8_t *)(ptr) + (j) * _SSW_VEC_BYTES)).
  - Array indexing like pvH[j] on __m128i* is ILLEGAL — use _mm_load_si128 / _mm_store_si128 with byte-offset pointers.
  - Declaring __m128i inside structs is ILLEGAL — use a uint8_t buffer and cast as needed.
  - For segLen calculations with 8-bit elements use _SSW_ELEMS_8, for 16-bit elements use _SSW_ELEMS_8 / 2.

Strategy:
- Fix ONE function (or one category of error) per response.
- Do NOT attempt to fix the entire file at once.
- Address the FIRST error shown in the compiler output.
- If the same pattern (e.g. sizeof(__m128i)) appears in multiple functions, fix it in only ONE function per step.
- The repair loop will call you again with updated code and any remaining errors.

Rules:
- Modify only {target_file_name}.
- Preserve existing style unless the fix requires otherwise.
- Do not change unrelated code.
- Do not invent APIs, functions, files, or build steps.
- Keep each response small and focused on the next high-confidence repair step.
- Prefer the smallest possible diff that fixes the current failure.
- Prefer a single small hunk over broad rewrites whenever possible.
- Do not rewrite unchanged lines just to restyle them.
- If something is ambiguous, make the safest assumption and state it briefly in the summary.

Return only:
1) A short summary sentence.
2) A single-file unified git diff patch in a fenced ```diff block for {target_file_name}.

Patch requirements:
- The patch must modify exactly one file: {target_file_name}.
- Use standard diff headers that match this exact path:
  - `--- a/{target_file_name}`
  - `+++ b/{target_file_name}`
- Each hunk MUST have a proper header: `@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@`
  where OLD_START is the 1-based line number where the context begins in the original file,
  OLD_COUNT is the number of lines from the original (context + removed),
  NEW_START is the corresponding line in the new file,
  NEW_COUNT is the number of lines in the new version (context + added).
- Context lines (unchanged) MUST start with a SPACE character and MUST match the actual file exactly.
- Removed lines start with `-`.
- Added lines start with `+`.
- Include 3 lines of unchanged context before and after each change.
- Do not return full-file rewrites.
- Keep the total patch under 3000 characters.

{diff_format_example(target_file_name)}

Reference material loaded from:
{REFERENCE_FILE}

{load_riscv_reference()}
""".strip()


def build_initial_user_prompt(
    file_name: str,
    code: str,
    target_fasta: str,
    query_fasta: str,
    validation_feedback: str | None = None,
) -> str:
    validation_section = ""
    if validation_feedback:
        validation_section = f"""

Current validation failure:
{validation_feedback}
""".rstrip()
    return f"""
Task: Repair this file for the RISC-V validation flow.

Goal:
Make the smallest correct change so the project compiles and runs successfully with the provided validation command.

Repository context:
- Target file: {file_name}
- Validation command: make clean && make rvv_cli CC="{RISCVCC}" && {QEMU_RVV} ./rvv_ssw_test "{target_fasta}" "{query_fasta}"

What to change:
- Fix the current implementation in {file_name}.
- Keep the repair incremental and localized.
- Prefer the smallest possible patch hunk that addresses the reported failure.

What not to change:
- Do not modify any file other than {file_name}.
- Do not refactor unrelated logic.
- Do not introduce new dependencies or new files.

Acceptance criteria:
- The resulting patch targets only {file_name}.
- The change is minimal and directly addresses the current failure mode.
- The output follows the required format exactly.

Current code:
```c
{code}
```
{validation_section}

Output:
- First, one short summary sentence.
- Then, a single fenced `diff` block containing only the unified diff for `{file_name}`.
- The diff must target only `{file_name}`.
- Use `--- a/{file_name}` and `+++ b/{file_name}`.
- Each hunk header MUST have correct line numbers: `@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@`
- Context lines MUST start with a space and MUST exactly match the current file.
- Include 3 lines of unchanged context before and after each change.
{diff_format_example(file_name)}
""".strip()


def build_repair_prompt(file_name: str, code: str, validation_feedback: str) -> str:
    return f"""
Task: Fix the current validation failure in this file.

Goal:
Make the smallest correct change needed so the project moves closer to passing the RISC-V build/runtime validation flow.

Repository context:
- Target file: {file_name}
- Validation flow: compile with the configured RISC-V toolchain and run the provided CLI validation.

What to change:
- Update only {file_name}.
- Address the failure indicated below.
- Prefer the smallest possible patch hunk that fixes the reported failure.

What not to change:
- Do not modify any file other than {file_name}.
- Do not refactor unrelated code.
- Do not introduce new dependencies or new files.

Acceptance criteria:
- The patch is minimal and localized.
- The patch targets only {file_name}.
- The output follows the required format exactly.

Current code:
```c
{code}
```

Validation failure details:
{validation_feedback}

Output:
- First, one short summary sentence.
- Then, a single fenced `diff` block containing only the unified diff for `{file_name}`.
- Use `--- a/{file_name}` and `+++ b/{file_name}`.
- Each hunk header MUST have correct line numbers: `@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@`
- Context lines MUST start with a space and MUST exactly match the current file.
- Include 3 lines of unchanged context before and after each change.
{diff_format_example(file_name)}
""".strip()


def build_diff_format_feedback(file_name: str, code: str, error_message: str) -> str:
    return diff_error_feedback(file_name, code, error_message)
