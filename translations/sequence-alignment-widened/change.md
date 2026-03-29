# RVV-WIDENED: Manual Translation Changes

## Summary

Manually rewrote `ssw.c` from SSE-via-sse2rvv.h (fixed 128-bit) to native RISC-V
Vector (RVV) intrinsics with runtime VLEN. The code is now VLEN-agnostic and will
automatically use the full hardware vector width (128, 256, 512, etc.).

## Files Modified

### ssw.c — Complete rewrite of SIMD sections

#### 1. Replaced `#include "sse2rvv.h"` with `#include <riscv_vector.h>`
- Direct use of native RVV intrinsics instead of SSE emulation layer.
- Eliminates the sse2rvv.h abstraction that locked operations to 128 bits.

#### 2. Added RVV helper functions for runtime VLEN queries
```c
static inline size_t vlmax_e8(void)  { return __riscv_vsetvlmax_e8m1(); }
static inline size_t vlmax_e16(void) { return __riscv_vsetvlmax_e16m1(); }
static inline size_t vregbytes(void) { return __riscv_vsetvlmax_e8m1(); }
```
These replace all hardcoded `16` (byte lanes) and `8` (word lanes) constants.

#### 3. Changed `struct _profile` member types
- **Before:** `__m128i* profile_byte` and `__m128i* profile_word`
- **After:** `uint8_t* profile_byte` and `uint8_t* profile_word`
- Raw byte arrays that store VLEN-width vectors contiguously.
- Profile arrays are now allocated as `n * segLen * vregbytes()` bytes.

#### 4. Rewrote `qP_byte()` — byte query profile builder
- **Before:** `segLen = (readLen + 15) / 16`, loops `segNum < 16`
- **After:** `segLen = (readLen + lanes - 1) / lanes` where `lanes = vlmax_e8()`
- Profile elements are packed into VLEN-width vectors instead of 128-bit.

#### 5. Rewrote `sw_sse2_byte()` → `sw_rvv_byte()` — 8-bit striped SW
All SSE intrinsics replaced with native RVV:

| SSE (sse2rvv.h)              | RVV (native)                                  | Notes |
|------------------------------|-----------------------------------------------|-------|
| `_mm_set1_epi8(x)`          | `__riscv_vmv_v_x_u8m1(x, vl)`               | Broadcast scalar |
| `_mm_adds_epu8(a, b)`       | `__riscv_vsaddu_vv_u8m1(a, b, vl)`          | Unsigned sat add |
| `_mm_subs_epu8(a, b)`       | `__riscv_vssubu_vv_u8m1(a, b, vl)`          | Unsigned sat sub |
| `_mm_max_epu8(a, b)`        | `__riscv_vmaxu_vv_u8m1(a, b, vl)`           | Unsigned max |
| `_mm_load_si128(ptr)`        | `__riscv_vle8_v_u8m1(ptr, vl)`              | Vector load |
| `_mm_store_si128(ptr, v)`    | `__riscv_vse8_v_u8m1(ptr, v, vl)`           | Vector store |
| `_mm_slli_si128(v, 1)`      | `__riscv_vslideup_vx_u8m1_tu(zero, v, 1, vl)` | Byte shift left |
| `_mm_cmpeq_epi8 + movemask` | `__riscv_vmseq + __riscv_vcpop`              | All-lanes check |
| `max16` tree reduction       | `__riscv_vredmaxu_vs_u8m1_u8m1`             | Horizontal max |

Key algorithmic changes:
- **Lazy_F loop:** Iterates `vl` times (was hardcoded 16).
- **Early exit:** Uses `vcpop` on comparison mask instead of `movemask == 0xffff`.
- **Reduction:** Uses `vredmaxu` instead of manual tree fold (srli + max x 4 stages).
- **Memory layout:** Vector arrays indexed as `base + j * vb` where `vb = vregbytes()`.
- **Alignment tracing:** Uses `i / vl + i % vl * segLen` (was `i / 16 + i % 16 * segLen`).

#### 6. Rewrote `qP_word()` — 16-bit query profile builder
- **Before:** `segLen = (readLen + 7) / 8`, loops `segNum < 8`
- **After:** `segLen = (readLen + lanes - 1) / lanes` where `lanes = vlmax_e16()`

#### 7. Rewrote `sw_sse2_word()` → `sw_rvv_word()` — 16-bit striped SW
Similar to byte path but with 16-bit types:

| SSE                          | RVV                                           |
|------------------------------|-----------------------------------------------|
| `_mm_set1_epi16(x)`         | `__riscv_vmv_v_x_i16m1(x, vl)` / `_u16m1`  |
| `_mm_adds_epi16(a, b)`      | `__riscv_vsadd_vv_i16m1(a, b, vl)`          |
| `_mm_max_epi16(a, b)`       | `__riscv_vmax_vv_i16m1(a, b, vl)`           |
| `_mm_subs_epu16(a, b)`      | `__riscv_vssubu_vv_u16m1(a, b, vl)` (via reinterpret) |
| `_mm_cmpgt_epi16(a, b)`     | `__riscv_vmsgt_vv_i16m1_b16(a, b, vl)`      |
| `_mm_slli_si128(v, 2)`      | `__riscv_vslideup_vx_i16m1_tu(zero, v, 1, vl)` |
| `max8` tree reduction        | `__riscv_vredmax_vs_i16m1_i16m1`             |

Special handling for mixed signed/unsigned in word path:
- Gap penalties use `__riscv_vssubu_vv_u16m1` (unsigned saturating sub) via
  `__riscv_vreinterpret_v_i16m1_u16m1` / `_u16m1_i16m1` casts.
- Score comparisons use `__riscv_vmax_vv_i16m1` (signed max) to match SSE semantics.
- Lazy_F early exit uses `vmsgt + vcpop == 0` instead of `movemask(cmpgt) == 0`.

#### 8. Updated `ssw_align()` function
- Changed `__m128i* vP` to `uint8_t* vP` (profile pointer type).
- Changed calls from `sw_sse2_byte`/`sw_sse2_word` to `sw_rvv_byte`/`sw_rvv_word`.
- Changed `qP_byte`/`qP_word` return type handling.

#### 9. `banded_sw()`, `cigar_alignment_score()`, `mark_mismatch()` — UNCHANGED
These functions are scalar (no SIMD) and required no modifications.

### ssw.h — Minimal change
- Removed `#include "sse2rvv.h"` (no longer needed; the public API uses no SIMD types).
- The `_profile` struct is only forward-declared in the header; the actual definition
  with `uint8_t*` members is in ssw.c.

### main.c — Minimal change
- Removed `#include "sse2rvv.h"` / `#include "sse2neon.h"` conditional block.
- main.c only uses the SSW library API and has no SIMD code.

### sse2rvv.h — NOT MODIFIED
- Still present in the directory for reference, but no longer included by any file.
- Could be removed entirely; kept for compatibility reference.

## Validation
- Docker/QEMU (Spike, VLEN=128): **PASSED**
- SSH hardware (real RISC-V): **PASSED**

## How Widening Works

At VLEN=128, the code behaves identically to the original SSE version:
- `vlmax_e8()` = 16 bytes, `vlmax_e16()` = 8 words -- same as SSE
- Segment lengths, memory layout, and algorithm all match

At VLEN=256 (or wider):
- `vlmax_e8()` = 32 bytes -- process 32 read positions per vector
- `vlmax_e16()` = 16 words -- process 16 read positions per vector
- Fewer segments per read -- fewer loop iterations -- faster
- Lazy_F loop runs more iterations (32 vs 16 for byte path) but each
  iteration processes proportionally more data
