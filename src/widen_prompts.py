"""Prompt builders for the vector-width optimization pipeline.

Contains the RVV intrinsic cheatsheet, system prompt, and all
user-facing prompt templates used by the WidenAgent.
"""

from src.config import RVV_REFERENCE
from src.search_replace import (
    search_replace_error_feedback,
    search_replace_format_example,
)


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

### Slide (shift lanes) — MUST use `_tu` (tail-undisturbed) for shift-left
- `__riscv_vslideup_vx_u8m1_tu(zero, src, 1, vl)`  — shift left by 1, insert 0
- `__riscv_vslideup_vx_i16m1_tu(zero, src, 1, vl)` — shift left by 1 element (16-bit)
- `__riscv_vslidedown_vx_u8m1(src, offset, vl)`
NOTE: The `_tu` suffix means tail-undisturbed.  For shift-left-by-N, the first
arg (`dst`) should be a zero vector so the vacated low lanes get zeroed.

### Vector index (iota)
- `vuint8m1_t  idx = __riscv_vid_v_u8m1(vl);`   — {0, 1, 2, ..., vl-1}
- `vuint16m1_t idx = __riscv_vid_v_u16m1(vl);`  — {0, 1, 2, ..., vl-1}

### Merge (conditional select: mask ? b : a)
- `vuint8m1_t  r = __riscv_vmerge_vvm_u8m1(a, b, mask, vl);`  — mask=vbool8_t
- `vuint8m1_t  r = __riscv_vmerge_vxm_u8m1(a, scalar, mask, vl);` — merge with scalar
- `vuint16m1_t r = __riscv_vmerge_vvm_u16m1(a, b, mask, vl);` — mask=vbool16_t
- `vint16m1_t  r = __riscv_vmerge_vvm_i16m1(a, b, mask, vl);` — mask=vbool16_t

### Compare → mask
- `vbool8_t m = __riscv_vmseq_vv_u8m1_b8(a, b, vl);`   — equal
- `vbool8_t m = __riscv_vmseq_vx_u8m1_b8(a, 0, vl);`   — equal to scalar
- `vbool8_t m = __riscv_vmsgtu_vv_u8m1_b8(a, b, vl);`  — unsigned greater than
- `vbool8_t m = __riscv_vmsltu_vx_u8m1_b8(a, val, vl);` — unsigned less than scalar
- `vbool16_t m = __riscv_vmseq_vv_i16m1_b16(a, b, vl);` — 16-bit eq
- `vbool16_t m = __riscv_vmsgtu_vv_u16m1_b16(a, b, vl);` — 16-bit unsigned gt
- `vbool16_t m = __riscv_vmsltu_vx_u16m1_b16(a, val, vl);` — 16-bit unsigned lt scalar

### Mask → scalar
- `long bits = __riscv_vcpop_m_b8(mask, vl);`  — popcount of mask (8-bit)
- `long bits = __riscv_vcpop_m_b16(mask, vl);` — popcount of mask (16-bit)
- `long first = __riscv_vfirst_m_b8(mask, vl);` — index of first set bit

### Reinterpret casts (zero-cost, same register)
- `vuint8m1_t  u = __riscv_vreinterpret_v_i8m1_u8m1(s);`   — signed→unsigned 8-bit
- `vint8m1_t   s = __riscv_vreinterpret_v_u8m1_i8m1(u);`   — unsigned→signed 8-bit
- `vuint16m1_t u = __riscv_vreinterpret_v_i16m1_u16m1(s);`  — signed→unsigned 16-bit
- `vint16m1_t  s = __riscv_vreinterpret_v_u16m1_i16m1(u);`  — unsigned→signed 16-bit

### Reductions
- `vuint8m1_t r = __riscv_vredmaxu_vs_u8m1_u8m1(vec, scalar_vec, vl);`
- `vint16m1_t r = __riscv_vredmax_vs_i16m1_i16m1(vec, scalar_vec, vl);`
- Extract scalar: `uint8_t s = __riscv_vmv_x_s_u8m1_u8(r);`
- Extract scalar: `int16_t s = __riscv_vmv_x_s_i16m1_i16(r);`

### LMUL and mask type correspondence (CRITICAL)
The mask type depends on the element width and LMUL:
- `vbool8_t`  ↔ `*8m1` types (e.g. `vuint8m1_t`, `vint8m1_t`)
- `vbool16_t` ↔ `*16m1` types (e.g. `vuint16m1_t`, `vint16m1_t`)
- `vbool4_t`  ↔ `*8m2` types (e.g. `vuint8m2_t`)
You CANNOT use a `vbool16_t` mask with `u8m1` operations or vice versa.
There is NO intrinsic to convert between mask types (e.g. `vbool16_t` → `vbool8_t`).
If you need to apply a 16-bit comparison result to 8-bit data, compute the
comparison separately in 8-bit space, or use a scalar loop.

### Profile construction pattern (for striped/segmented data layouts)
When building a lookup profile where lane `l` of segment `i` holds
`table[read[i + l * segLen]]`, use a simple scalar loop:
```c
size_t vl = __riscv_vsetvlmax_e8m1();
for (i = 0; i < segLen; i++) {
    for (size_t l = 0; l < vl; l++) {
        int32_t pos = i + (int32_t)l * segLen;
        *t++ = (pos < readLen) ? table[read[pos]] + bias : bias;
    }
}
```
This is simple, correct, and portable.  Do NOT use vrgatherei16 for this —
the LMUL requirements (index must be 2x wider) make it error-prone.

### Lazy_F early exit pattern (MUST use this exact pattern)
For the 8-bit byte path:
```c
vuint8m1_t vDiff = __riscv_vssubu_vv_u8m1(vF, vH, vl);
vbool8_t all_zero = __riscv_vmseq_vx_u8m1_b8(vDiff, 0, vl);
if (UNLIKELY(__riscv_vcpop_m_b8(all_zero, vl) == vl)) goto lazy_end;
```
For the 16-bit word path (signed comparison):
```c
vbool16_t gt_mask = __riscv_vmsgt_vv_i16m1_b16(vF, vH_gap, vl);
if (UNLIKELY(__riscv_vcpop_m_b16(gt_mask, vl) == 0)) goto word_lazy_end;
```
WRONG: checking only lane 0 (`vmv_x_s`).  MUST check ALL lanes via `vcpop`.

### 16-bit signed/unsigned mixing for gap penalties
Gap penalties use unsigned saturating subtract, but scores use signed max:
```c
vuint16m1_t vH_u = __riscv_vreinterpret_v_i16m1_u16m1(vH);
vuint16m1_t vH_gap = __riscv_vssubu_vv_u16m1(vH_u, vGapO_u, vl);
vF = __riscv_vmax_vv_i16m1(
    __riscv_vreinterpret_v_u16m1_i16m1(vF_u),
    __riscv_vreinterpret_v_u16m1_i16m1(vH_gap), vl);
```

### Byte path vs word path vector byte stride
- Byte path (e8): `vl = vlmax_e8()`, `vb = vl` (bytes == lanes)
- Word path (e16): `vl = vlmax_e16()`, `vb = vregbytes()` (bytes = 2 * lanes)
Memory indexed as `base + j * vb` in BOTH paths.

### CRITICAL: Common mistakes to avoid
- WRONG: `__riscv_vmv_v_u8m1(val)` → CORRECT: `__riscv_vmv_v_x_u8m1(val, vl)`
- WRONG: `__riscv_vle8_u8m1(ptr, vl)` → CORRECT: `__riscv_vle8_v_u8m1(ptr, vl)`
- WRONG: any intrinsic without `vl` param → ALL intrinsics need `vl`
- WRONG: `_mm_extract_epi16(v, 0)` for max → use `vredmaxu`/`vredmax` + `vmv_x_s`
- WRONG: `_mm_movemask_epi8(v)` → use `vmseq` + `vcpop` or `vfirst`
- WRONG: using `vbool16_t` mask with `u8m1` operations → mask type MUST match element LMUL
- WRONG: inventing intrinsics like `__riscv_vzext_vf2_b8_u8m1` → there is NO mask type conversion
- WRONG: `__riscv_vrgatherei16_vv_i8m1(ptr, idx, vl)` with a pointer → first arg must be a vector, not a pointer
- WRONG: checking only lane 0 for "all lanes satisfy" → use `vcpop` on a mask to check ALL lanes
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
## Proven transformation patterns (from validated manual widening)

These patterns have been verified to produce correct output at VLEN=128 and
VLEN=256.  Use them as-is — do NOT deviate from these patterns.

### Helper functions (add once at top of file)
```c
static inline size_t vlmax_e8(void)  {{ return __riscv_vsetvlmax_e8m1(); }}
static inline size_t vlmax_e16(void) {{ return __riscv_vsetvlmax_e16m1(); }}
static inline size_t vregbytes(void) {{ return __riscv_vsetvlmax_e8m1(); }}
```

### Struct changes
- `__m128i* profile_byte` → `uint8_t* profile_byte`
- `__m128i* profile_word` → `uint8_t* profile_word`

### qP_byte profile builder (scalar loop — simple and correct)
```c
size_t lanes = vlmax_e8();
int32_t segLen = (readLen + (int32_t)lanes - 1) / (int32_t)lanes;
uint8_t* vProfile = (uint8_t*)malloc(n * segLen * lanes);
int8_t* t = (int8_t*)vProfile;
for (nt = 0; nt < n; nt++) {{
    for (i = 0; i < segLen; i++) {{
        j = i;
        for (segNum = 0; segNum < lanes; segNum++) {{
            *t++ = j >= readLen ? bias : mat[nt * n + read_num[j]] + bias;
            j += segLen;
        }}
    }}
}}
```

### qP_word profile builder (same pattern, 16-bit)
```c
size_t lanes = vlmax_e16();
size_t vb = vregbytes();
int32_t segLen = (readLen + (int32_t)lanes - 1) / (int32_t)lanes;
uint8_t* vProfile = (uint8_t*)malloc(n * segLen * vb);
int16_t* t = (int16_t*)vProfile;
for (nt = 0; nt < n; nt++) {{
    for (i = 0; i < segLen; i++) {{
        j = i;
        for (segNum = 0; segNum < lanes; segNum++) {{
            *t++ = j >= readLen ? 0 : mat[nt * n + read_num[j]];
            j += segLen;
        }}
    }}
}}
```

### Alignment position traceback (8-bit path)
```c
int32_t column_len = segLen * (int32_t)vl;
for (i = 0; i < column_len; ++i, ++t) {{
    if (*t == max) {{
        temp = i / (int32_t)vl + i % (int32_t)vl * segLen;
        if (temp < end_read) end_read = temp;
    }}
}}
```

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
- Mask type mismatch: `vbool8_t` goes with `*8m1` types, `vbool16_t` with
  `*16m1` types.  There is NO mask conversion intrinsic.
- Do NOT invent intrinsic names.  If the exact name is not in the cheatsheet,
  use a scalar fallback instead of guessing.

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
        is_correctness = "CORRECTNESS FAILURE" in validation_feedback
        extra = ""
        if is_correctness:
            extra = """

IMPORTANT: This is a CORRECTNESS bug — the code compiles and runs but
produces WRONG results.  Carefully compare your widened code against the
original 128-bit version (shown below the error details).  The most
common root cause is a mismatch between:
  (a) How the profile/lookup table is BUILT (qP_byte or equivalent), and
  (b) How the DP kernel READS that profile (pointer stride, segment length).
Both must use the same runtime vector length (vl).  Also check:
  - Lazy_F loop exit must test ALL lanes (vcpop on mask == 0), not lane 0.
  - vslideup/vslidedown shift amounts must be correct for the wider vector.
  - Reduction operations (max score tracking) must cover all lanes."""

        preamble = f"""\
Task: Continue widening vector operations (pass {pass_number}).

The previous pass had validation errors (shown below).  You MUST fix
these errors first before attempting any new widening.  Make the
smallest change that resolves the failure.{extra}"""
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
