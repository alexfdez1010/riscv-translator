# RISC-V Vector (RVV) C Programming Concepts

## Scope
- RVV C intrinsics and vector-length-agnostic execution model
- Instruction forms, policies, and data types
- Memory access patterns (contiguous/strided/segmented/gather/slide)
- Reduction, softmax-style normalization, NTT-style modular arithmetic, and CRC folding patterns
- Perf counter usage with `rdinstret` / `rdcycle`
- Build/toolchain details intentionally omitted

## Examples (Host/Kernel Interaction, Self-Contained)

The examples below are self-contained summaries of the main kernels and how a typical host harness drives them. The host is the scalar driver that allocates buffers, initializes inputs, calls the RVV kernel, and reads perf counters. The kernel is the RVV intrinsics code that runs on the same core but uses vector instructions (device-style computation inside a host-driven loop).

Common host pattern:

```c
// allocate + initialize
float *src = ...; float *dst = ...; size_t n = ...;
// perf counter
unsigned long start = rdinstret();
kernel(dst, src, n);
unsigned long stop = rdinstret();
printf("%lu instructions\n", stop - start);
```

Common kernel pattern:

```c
size_t avl = n;
while (avl > 0) {
  size_t vl = __riscv_vsetvl_e32m1(avl);
  vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
  // ... compute ...
  __riscv_vse32_v_f32m1(dst, v, vl);
  src += vl; dst += vl; avl -= vl;
}
```

### 1. Vector Add (baseline kernel + bench harness)
**Kernel behavior:** load two vectors, add, store.

```c
vfloat32m1_t a = __riscv_vle32_v_f32m1(lhs, vl);
vfloat32m1_t b = __riscv_vle32_v_f32m1(rhs, vl);
vfloat32m1_t c = __riscv_vfadd_vv_f32m1(a, b, vl);
__riscv_vse32_v_f32m1(dst, c, vl);
```

**Host interaction:** allocate `lhs/rhs/dst`, run once, print instruction count. This is the simplest “host drives kernel” example.

### 2. Matrix Transpose (multiple kernels + benchmark menu)
**Kernel behaviors:**
- Strided-store transpose (write columns using `vsse32`).
- Strided-load transpose (read columns using `vlse32`).
- Segmented load/store for 4x4.
- In-register permutation with `vrgather` or `vslide`.

```c
vfloat32m1_t row = __riscv_vle32_v_f32m1(row_src, vl);
__riscv_vsse32(row_dst, sizeof(float) * n, row, vl);
```

**Host interaction:** build a list of implementations, run each one on the same input, verify output and compare perf counters.

### 3. Softmax (algorithmic variants + accuracy/perf bench)
**Kernel behaviors:**
- Vectorized exponent approximation (polynomial + exponent reconstruction).
- Accumulate sum with tail-undisturbed vector accumulator.
- Normalize with a vector multiply by reciprocal.

```c
vsum = __riscv_vfadd_vv_f32m1_tu(vsum, vsum, vexp, vl);
vfloat32m1_t row = __riscv_vle32_v_f32m1(dst, vl);
row = __riscv_vfmul_vf_f32m1(row, inv_sum, vl);
__riscv_vse32(dst, row, vl);
```

**Host interaction:** generate random inputs, compute a golden reference, run multiple kernels, and print perf + error metrics.

### 4. Polynomial Multiplication / NTT (vectorized butterfly stages)
**Kernel behaviors:**
- Split even/odd coefficients with strided loads or compressed loads.
- Vectorized butterfly: multiply by twiddle factors, add/sub, then reduce modulo.
- Barrett reduction and masked corrections to keep coefficients in range.

```c
vec_odd = __riscv_vmul_vv_i32m8(vec_odd, vec_twiddle, vl);
vec_even = __riscv_vadd_vv_i32m8(vec_even, vec_odd, vl);
vec_even = __riscv_vrem_vx_i32m8(vec_even, modulo, vl);
```

**Host interaction:** allocate polynomial/ring buffers, call forward NTT / multiply / inverse NTT routines, then display results or perf counters.

### 5. Reduction (vector reductions to scalar)
**Kernel behaviors:**
- Use `vred*`/`vfred*` to reduce a vector into a scalar lane.
- Extract the scalar with `vfmv.f.s` or `vmv.x.s`.

```c
vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
float s = __riscv_vfmv_f_s_f32m1_f32(sum);
```

**Host interaction:** set up inputs, run the reduction kernel, and print perf counters or the reduced scalar.

### 6. CRC (vector folding using carry-less multiply)
**Kernel behaviors:**
- Load expanded data, use carry-less multiply (`vclmul`/`vclmulh`).
- Byte-reversal and endianness handling via `vrev8`/`vbrev`.
- Reduce folded vectors with `vredxor`.

```c
vuint64m1_t lo = __riscv_vclmul_vv_u64m1(a, b, vl);
vuint64m1_t hi = __riscv_vclmulh_vv_u64m1(a, b, vl);
```

**Host interaction:** prepare buffers for CRC input, call the vector CRC kernel(s), and compare against a scalar CRC reference.

### 7. Microbenchmarks (instruction-level kernels)
**Kernel behaviors:**
- Short inline loops to measure latency/throughput for a specific instruction pattern.
- Variations by SEW/LMUL to isolate vector configuration costs.

**Host interaction:** run each microbenchmark in a loop and print perf counters to interpret instruction costs.

## Core RVV Concepts

### 1. Vector Length (VL) and Application Vector Length (AVL)

RVV is vector-length agnostic. You do not assume a fixed VL. The common pattern is:

```c
size_t avl = n;
while (avl > 0) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
    // ... compute ...
    __riscv_vse32_v_f32m1(ptr, v, vl);
    avl -= vl;
    ptr += vl;
}
```

Key points:
- `avl` is the number of elements left to process.
- `__riscv_vsetvl_*` returns the VL actually used (<= avl).
- The loop updates pointers by `vl` elements.

### 2. VLMAX and Full-Width Accumulators

Some reductions use a full-length accumulator initialized with `vsetvlmax`:

```c
size_t vlmax = __riscv_vsetvlmax_e32m1();
vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
```

This is critical when you need to keep partial sums across loop iterations and must preserve tail lanes.

### 3. Tail and Mask Policies

When operating on partial vectors (vl < VLMAX), the code sometimes uses tail-undisturbed variants (suffix `_tu`, `_tumu`, `_mu`) so that inactive lanes are preserved (important for accumulators):

- Tail agnostic: `ta`
- Tail undisturbed: `tu`
- Mask agnostic: `ma`
- Mask undisturbed: `mu`

You will see intrinsics such as `__riscv_vfadd_vv_f32m1_tu(...)` to preserve tail elements.

### 4. Intrinsics Naming and Types

- Types: `vfloat32m1_t`, `vint32m1_t`, `vuint32mf2_t`, `vfloat32m4_t`, etc.
  - `m1`, `m2`, `m4`, `m8` represent LMUL.
  - `mf2`, `mf4` are fractional LMUL.
- Intrinsics encode operation, data type, and LMUL, for example:
  - `__riscv_vle32_v_f32m1` (load 32-bit floats, LMUL=1)
  - `__riscv_vfadd_vv_f32m1` (vector-vector add)
  - `__riscv_vfadd_vf_f32m1` (vector-scalar add)

General format:
`__riscv_<op>_<suffix>` where suffix encodes element type and LMUL.

### 5. Memory Access Patterns

- **Contiguous**: `vle32`, `vse32`.
- **Strided**: `vlse32`, `vsse32` (stride in bytes).
- **Segmented**: `vlseg4e32`, `vsseg4e32` for interleaved data.
- **Gather/Permute**: `vrgather`, `vrgatherei16` for reordering.
- **Slide**: `vslideup`, `vslidedown` for shifting lanes.

### 6. Reductions

Reductions appear in two styles:

- **Vector reduction instructions** (e.g., `vredmin.vs`, `vredsum.vs`, `vfredusum.vs`):
  - Works on a vector input and reduces to a scalar element in a vector register.
  - Often paired with `vfmv.f.s` or `vmv.x.s` to extract scalar.

- **Two-step reduction** for LMUL>1:
  - First reduce within each vector register.
  - Then reduce the final vector to a scalar (common for min/sum).

### 7. Conversions and Rounding

- Float to int conversion: `__riscv_vfcvt_x_f_v_i32m1`.
- For deterministic float-to-int conversions, set rounding explicitly:
  - `fesetround(FE_TONEAREST);`

### 8. Widening/Narrowing and Reinterpretation

- Widening: `vwmul`, `vwsll`, `vzext`.
- Narrowing: `vnsra`, `vnsrl`.
- Reinterpretation: `__riscv_vreinterpret_v_i32m1_f32m1` (bitwise reinterpret without conversion).

### 9. Masks

- Mask types are `vboolX_t` and depend on SEW/LMUL.
- Masked intrinsics are used for conditional updates (example: Barrett reduction in polynomial mult).

### 10. Bit Manipulation and Carry-less Multiply

CRC-style workloads often use:
- `vclmul` / `vclmulh` (Zvbc extension)
- `vbrev`, `vbrev8`, `vrev8` (Zvbb/Zvkb style operations)
- Endianness is handled through byte-reversal steps.

## Common Configuration Macros

These macros affect how the code is specialized:

- `LMUL`, `WLMUL`, `NLMUL`: vector grouping for polynomial mult.
- `E32_MASK`: mask type width derived from LMUL.
- `USE_PRECOMPUTED_ROOT_POWERS`: select precomputed tables in NTT.
- `USE_VREM_MODULO`: choose between remainder instruction vs Barrett reduction.
- `COUNT_INSTRET`, `COUNT_CYCLE`: select perf counter.
- `HAS_ZVBB_SUPPORT`: toggle for bit-manip instructions.
- `LINUX_PERF_COUNT`: uses perf_event_open for host perf counters.

## Core Patterns and Algorithmic Templates

### 1. Vector Add Pattern

Minimal vector loop template:

```c
void vec_add(float *dst, const float *lhs, const float *rhs, size_t n) {
    size_t avl = n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t a = __riscv_vle32_v_f32m1(lhs, vl);
        vfloat32m1_t b = __riscv_vle32_v_f32m1(rhs, vl);
        vfloat32m1_t c = __riscv_vfadd_vv_f32m1(a, b, vl);
        __riscv_vse32_v_f32m1(dst, c, vl);
        lhs += vl;
        rhs += vl;
        dst += vl;
        avl -= vl;
    }
}
```

### 2. Matrix Transpose Patterns

Common strategies:
- Strided stores: `__riscv_vsse32` (transpose by storing into columns).
- Strided loads: `__riscv_vlse32`.
- Segmented load/store: `vlseg4e32`, `vsseg4e32`.
- Permutation-based transpose: `vrgatherei16`, `vslideup`, `vslidedown`.

Important patterns:
- For nxn transpose, a nested loop iterates over rows, and each row uses a vector loop with strided stores/loads.
- For 4x4 transpose, in-register permutations reduce memory traffic.

### 3. Reduction

Two main approaches appear:

1. Intrinsic-based micro-benchmarks of reduction instructions:
   - `vredmin.vs`, `vredsum.vs`, `vfredosum.vs`, `vfredusum.vs`.
2. Inline-assembly loop for `rvv_min` and dot-product:
   - Explicit `vsetvli` / `vle32.v` / `vredmin.vs` and `vfredosum.vs` sequences.

Concepts to carry forward:
- Pre-initialize accumulator vector registers.
- When LMUL>1, it is common to run a second reduction pass to combine partial accumulators.

### 4. Softmax-Style Normalization

Pattern summary:
- Compute maximum using `vfredmax` with a `-INFINITY` accumulator.
- Compute element-wise exponentials (often using a polynomial approximation).
- Accumulate a sum with `vfredusum` or vector accumulators.
- Normalize each element by multiplying by the reciprocal of the sum.

Notable RVV usage:
- `vfredmax` for max reduction.
- `vfmadd` / `vfnmsac` for polynomial evaluation.
- `vfredusum` for sum reduction.
- Tail-undisturbed updates for accumulators.

### 5. Polynomial Multiplication / NTT

Core data types:

```c
typedef struct {
  int n;
  int modulo;
  int rootOfUnity;
  int invRootOfUnity;
  int invDegree;
} ring_t;

typedef struct {
  int degree;
  int modulo;
  int* coeffs;
  size_t coeffSize;
} polynomial_t;
```

Key facts:
- The primary ring uses `modulo = 3329` and degree 127 (Kyber-style).
- Root parameters: `rootOfUnity = 33`, `invRootOfUnity = 2522`, `invDegree = 3303`.
- Modulo polynomial: `x^128 - 1` (represented with coefficients at 0 and 128).

RVV implementation details:
- Use precomputed coefficient index arrays for NTT butterfly schedules.
- Use macro metaprogramming to generate intrinsic names for configurable LMUL.
- Use Barrett reduction to avoid expensive modulo operations.
- Use `vmsge` + masked add/sub for conditional modular corrections.
- Variants often include strided load, indexed load, compressed, Barrett reduction, and assembly implementations.

If you author new RVV polynomial code, follow the same patterns:
- Use LMUL-parameterized macros.
- Ensure modulo reductions are consistent (especially for negative results).
- Keep NTT data layout compatible with precomputed index arrays.

### 6. CRC

CRC workloads often use RVV with carry-less multiply (Zvbc):

- Vector folding uses `vclmul` / `vclmulh` on expanded data.
- Endianness handling uses `vrev8`, `vbrev`, or `vbrev8`.
- Reduction is done via `vredxor` into accumulator vectors.

CRC code commonly distinguishes BE vs LE handling, and may use Zvbc32e if available.

### 7. Microbenchmarks and Measurement Loops

Microbenchmarks use inline assembly loops to measure latency/throughput. Common goals:
- How to construct dependency chains for latency measurement.
- How to create unrolled, parallel instruction streams for throughput measurement.
- How to vary LMUL and SEW for vector instruction benchmarks.

### 8. Perf Counters

Perf counters are read using:
- Inline asm `rdinstret` (retired instructions) or `rdcycle` (cycles).
- Optional Linux perf_event support when running on Linux hosts.

## Intrinsic Patterns

Common intrinsics used in RVV C code:

### Vector Length
- `__riscv_vsetvl_e32m1(avl)`
- `__riscv_vsetvlmax_e32m1()`

### Loads / Stores
- `__riscv_vle32_v_f32m1(ptr, vl)`
- `__riscv_vse32_v_f32m1(ptr, v, vl)`
- `__riscv_vlse32_v_f32m1(ptr, stride_bytes, vl)`
- `__riscv_vsse32(ptr, stride_bytes, v, vl)`
- `__riscv_vlseg4e32_v_f32m1x4(ptr, vl)`
- `__riscv_vsseg4e32_v_f32m1x4(ptr, v, vl)`

### Arithmetic
- `__riscv_vfadd_vv_f32m1(a, b, vl)`
- `__riscv_vfadd_vf_f32m1(a, scalar, vl)`
- `__riscv_vfmul_vv_f32m1(a, b, vl)`
- `__riscv_vfmul_vf_f32m1(a, scalar, vl)`
- `__riscv_vfmadd(...)`, `__riscv_vfnmsac(...)` (FMA and fused neg multiply-add)

### Reductions
- `__riscv_vfredusum_vs_f32m1_f32m1(vec, acc, vl)`
- `__riscv_vfredosum_vs_f32m1_f32m1(vec, acc, vl)`
- `__riscv_vredmin_vs_i32m1_i32m1(vec, acc, vl)`
- `__riscv_vredxor_vs_u64m1_u64m1(vec, acc, vl)`

### Conversions / Reinterpret
- `__riscv_vfcvt_x_f_v_i32m1(vec, vl)`
- `__riscv_vfcvt_f_x_v_f32m1(vec, vl)`
- `__riscv_vreinterpret_v_i32m1_f32m1(vec)`

### Widen/Narrow
- `__riscv_vwmul_vx_i64(...)`
- `__riscv_vnsra_wx_i32(...)`
- `__riscv_vzext_vf2_u64m1(...)`

### Permute / Slide / Gather
- `__riscv_vslideup_vx_*` / `__riscv_vslidedown_vx_*`
- `__riscv_vrgatherei16_vv_*`
- `__riscv_vget_v_f32m4_f32m1` / `__riscv_vcreate_v_f32m1_f32m4`

### Bit Ops and Carry-less Multiply
- `__riscv_vclmul_vv_u64m1(...)`
- `__riscv_vclmulh_vv_u64m1(...)`
- `__riscv_vrev8_v_u32m1(...)`
- `__riscv_vbrev_v_u64m1(...)`

## Template: Authoring New RVV C Code

1. Include headers:

```c
#include <riscv_vector.h>
#include <stddef.h>
```

2. Use AVL/VL loop structure:

```c
void my_kernel(float *dst, const float *src, size_t n) {
    size_t avl = n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
        // ... compute ...
        __riscv_vse32_v_f32m1(dst, v, vl);
        src += vl;
        dst += vl;
        avl -= vl;
    }
}
```

3. For reductions, initialize a full-length accumulator and use tail-undisturbed ops.

4. If using float-to-int conversion, explicitly set the rounding mode (as in softmax).

5. If your algorithm needs permutation, evaluate whether strided loads/stores, slides, or gather are most efficient.

## Glossary

- **AVL**: Application Vector Length (elements left to process).
- **VL**: Actual vector length set by vsetvl; used in intrinsics.
- **VLMAX**: Maximum vector length for the selected SEW/LMUL.
- **SEW**: Selected Element Width (8, 16, 32, 64).
- **LMUL**: Vector register grouping multiplier (mf2, m1, m2, m4, m8).
- **Tail Policy**: How inactive lanes are handled (agnostic vs undisturbed).
- **Mask Policy**: Whether masked-off elements are preserved.
- **Zvbc/Zvbc32e**: RISC-V vector carry-less multiply extensions.
