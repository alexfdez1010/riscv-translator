# SSE to RISC-V Translation Changes — `ssw.c`

This document describes the modifications made to the original `initial_code/ssw.c` to produce the translated `translations/sequence-alignment/ssw.c`, enabling the Striped Smith-Waterman library to run on RISC-V using the `sse2rvv.h` compatibility layer.

## 1. Header Replacement

The x86 SSE2 intrinsics header was replaced with the RISC-V compatibility header:

```diff
-#include <emmintrin.h>
+#include "sse2rvv.h"
```

This is the core enabler: `sse2rvv.h` provides drop-in replacements for SSE2 intrinsic functions implemented using RISC-V Vector (RVV) instructions.

## 2. `sizeof(__m128i)` Replaced with Literal `16`

Every occurrence of `sizeof(__m128i)` in memory allocation was replaced with the hardcoded constant `16`.

**Why:** On RISC-V with the Vector extension, `__m128i` is typedef'd to a scalable vector type (`vint32m1_t` or similar) whose `sizeof` returns the *hardware* vector register width, not the SSE semantic width of 128 bits (16 bytes). Since the algorithm was designed for 128-bit SIMD lanes, all allocations and pointer arithmetic must use 16 bytes to preserve correctness.

### Affected lines — `malloc` / `calloc` calls

```diff
 // ssw_gen_query_profile (8-bit version)
-__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
+__m128i* vProfile = (__m128i*)malloc(n * segLen * 16);

 // ssw_gen_query_profile (16-bit version)
-__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
+__m128i* vProfile = (__m128i*)malloc(n * segLen * 16);

 // ssw_align (8-bit inner function)
-__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvHLoad  = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvE      = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvHmax   = (__m128i*) calloc(segLen, sizeof(__m128i));
+__m128i* pvHStore = (__m128i*) calloc(segLen, 16);
+__m128i* pvHLoad  = (__m128i*) calloc(segLen, 16);
+__m128i* pvE      = (__m128i*) calloc(segLen, 16);
+__m128i* pvHmax   = (__m128i*) calloc(segLen, 16);

 // ssw_align (16-bit inner function) — same pattern
-__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvHLoad  = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvE      = (__m128i*) calloc(segLen, sizeof(__m128i));
-__m128i* pvHmax   = (__m128i*) calloc(segLen, sizeof(__m128i));
+__m128i* pvHStore = (__m128i*) calloc(segLen, 16);
+__m128i* pvHLoad  = (__m128i*) calloc(segLen, 16);
+__m128i* pvE      = (__m128i*) calloc(segLen, 16);
+__m128i* pvHmax   = (__m128i*) calloc(segLen, 16);
```

## 3. Pointer Arithmetic Changed from Array Indexing to Byte-Offset Casting

On x86, `__m128i` is a plain 16-byte struct, so `ptr + j` advances by `j * sizeof(__m128i)` = `j * 16` bytes — exactly one SSE register width. On RISC-V, `sizeof(__m128i)` can be larger than 16 bytes, making `ptr + j` overshoot. All pointer arithmetic involving `__m128i*` arrays was rewritten to use explicit byte-level offsets with casts.

**Pattern:**
```diff
 // Loads
-_mm_load_si128(pvArray + j)
+_mm_load_si128((const __m128i*)((uint8_t*)pvArray + j * 16))

 // Stores
-_mm_store_si128(pvArray + j, value)
+_mm_store_si128((__m128i*)((uint8_t*)pvArray + j * 16), value)

 // Direct reads (implicit load)
-__m128i vH = pvHStore[segLen - 1];
+__m128i vH = _mm_load_si128((const __m128i*)((uint8_t*)pvHStore + (segLen - 1) * 16));

 // Profile pointer offset
-const __m128i* vP = vProfile + ref[i] * segLen;
+const __m128i* vP = (const __m128i*)((uint8_t*)vProfile + ref[i] * segLen * 16);

 // Bulk copy (pvHmax = pvHStore per-element)
-pvHmax[j] = pvHStore[j];
+_mm_store_si128((__m128i*)((uint8_t*)pvHmax + j * 16),
+                _mm_load_si128((const __m128i*)((uint8_t*)pvHStore + j * 16)));
```

### All affected operations (both 8-bit and 16-bit code paths)

| Operation | Count | Locations |
|-----------|-------|-----------|
| `_mm_load_si128` with byte-offset cast | 14 | Inner loops, Lazy_F loops, initial vH load |
| `_mm_store_si128` with byte-offset cast | 10 | Inner loops, Lazy_F loops, pvHmax copy |
| `vProfile` pointer offset | 2 | One in each bit-width variant |
| Initial `vH` load from `pvHStore[segLen-1]` | 2 | One in each bit-width variant |
| `pvHmax[j] = pvHStore[j]` bulk copy | 2 | One in each bit-width variant |

## Summary

| Change category | Reason |
|----------------|--------|
| Header swap (`emmintrin.h` -> `sse2rvv.h`) | Use RVV-backed intrinsic implementations |
| `sizeof(__m128i)` -> `16` | Enforce 128-bit SSE semantic width regardless of hardware VLEN |
| Array indexing -> byte-offset pointer arithmetic | Prevent `sizeof`-dependent pointer stride from corrupting memory layout |

All three categories stem from the same root cause: on RISC-V, `__m128i` maps to a scalable vector type whose `sizeof` reflects the physical register width, not the 16-byte SSE logical width. The translation enforces 16-byte semantics throughout to maintain functional equivalence with the original x86 code.
