#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ssw.h"

/*
 * Smith-Waterman local alignment with affine gap penalties.
 * Uses RVV intrinsics with wavefront (anti-diagonal) parallelism.
 *
 * DP recurrence (1-indexed, i = read position, j = ref position):
 *   E[i][j] = max(0, max(H[i][j-1] - gapO, E[i][j-1] - gapE))
 *   F[i][j] = max(0, max(H[i-1][j] - gapO, F[i-1][j] - gapE))
 *   H[i][j] = max(0, H[i-1][j-1] + score(i,j), E[i][j], F[i][j])
 *
 * Anti-diagonal d = i + j: all cells on the same anti-diagonal are
 * independent (dependencies are on d-1 and d-2), so we vectorize
 * across the anti-diagonal using RVV strided loads/stores.
 */

#define IDX(i, j, cols) ((size_t)(i) * (size_t)(cols) + (size_t)(j))

s_profile *ssw_init(const int8_t *read, int32_t readLen,
                    const int8_t *mat, int32_t n)
{
    s_profile *p = (s_profile *)calloc(1, sizeof(s_profile));
    p->read = read;
    p->mat = mat;
    p->readLen = readLen;
    p->n = n;
    return p;
}

void init_destroy(s_profile *p) { free(p); }
void align_destroy(s_align *a) { free(a); }

typedef struct { int32_t score; int32_t ref; int32_t read; } align_end;

static align_end sw_core(const int8_t *ref, int8_t ref_dir,
                         int32_t refLen, const int8_t *read_num,
                         int32_t readLen, uint8_t gapO, uint8_t gapE,
                         const int8_t *mat, int32_t n,
                         int32_t terminate)
{
    size_t rows = (size_t)readLen + 1;
    size_t cols = (size_t)refLen + 1;

    int32_t *H = (int32_t *)calloc(rows * cols, sizeof(int32_t));
    int32_t *E = (int32_t *)calloc(rows * cols, sizeof(int32_t));
    int32_t *F = (int32_t *)calloc(rows * cols, sizeof(int32_t));

    align_end best = { 0, -1, readLen - 1 };

    ptrdiff_t stride = (ptrdiff_t)(cols - 1) * (ptrdiff_t)sizeof(int32_t);

    int8_t *ref_mapped = NULL;
    if (ref_dir == 1) {
        ref_mapped = (int8_t *)malloc(refLen * sizeof(int8_t));
        for (int32_t k = 0; k < refLen; k++)
            ref_mapped[k] = ref[refLen - 1 - k];
    }
    const int8_t *ref_seq = ref_dir ? ref_mapped : ref;

    int32_t total_diags = readLen + refLen;
    int32_t stopped = 0;

    for (int32_t d = 2; d <= total_diags && !stopped; d++) {
        int32_t i_min = (d > refLen) ? (d - refLen) : 1;
        int32_t i_max = (d - 1 < readLen) ? (d - 1) : readLen;
        if (i_min > i_max) continue;

        int32_t diag_len = i_max - i_min + 1;

        /* Build substitution score buffer. */
        int32_t *sub_buf = (int32_t *)malloc(diag_len * sizeof(int32_t));
        for (int32_t k = 0; k < diag_len; k++) {
            int32_t ii = i_min + k;
            int32_t jj = d - ii;
            sub_buf[k] = mat[read_num[ii - 1] * n + ref_seq[jj - 1]];
        }

        /* Vectorized wavefront pass. */
        int32_t done = 0;
        int32_t i = i_min;
        while (done < diag_len) {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(diag_len - done));
            int32_t j = d - i;

            vint32m4_t v_sub = __riscv_vle32_v_i32m4(&sub_buf[done], vl);
            vint32m4_t vZero = __riscv_vmv_v_x_i32m4(0, vl);

            /* Diagonal: H[i-1][j-1] + score */
            vint32m4_t vH_diag = __riscv_vlse32_v_i32m4(
                &H[IDX(i - 1, j - 1, cols)], stride, vl);
            vint32m4_t diag_sc = __riscv_vadd_vv_i32m4(vH_diag, v_sub, vl);

            /* E[i][j] = max(0, H[i][j-1]-gapO, E[i][j-1]-gapE) */
            vint32m4_t vH_left = __riscv_vlse32_v_i32m4(
                &H[IDX(i, j - 1, cols)], stride, vl);
            vint32m4_t vE_left = __riscv_vlse32_v_i32m4(
                &E[IDX(i, j - 1, cols)], stride, vl);
            vint32m4_t vE = __riscv_vmax_vv_i32m4(
                __riscv_vsub_vx_i32m4(vH_left, (int32_t)gapO, vl),
                __riscv_vsub_vx_i32m4(vE_left, (int32_t)gapE, vl), vl);
            vE = __riscv_vmax_vv_i32m4(vE, vZero, vl);
            __riscv_vsse32_v_i32m4(&E[IDX(i, j, cols)], stride, vE, vl);

            /* F[i][j] = max(0, H[i-1][j]-gapO, F[i-1][j]-gapE) */
            vint32m4_t vH_up = __riscv_vlse32_v_i32m4(
                &H[IDX(i - 1, j, cols)], stride, vl);
            vint32m4_t vF_up = __riscv_vlse32_v_i32m4(
                &F[IDX(i - 1, j, cols)], stride, vl);
            vint32m4_t vF = __riscv_vmax_vv_i32m4(
                __riscv_vsub_vx_i32m4(vH_up, (int32_t)gapO, vl),
                __riscv_vsub_vx_i32m4(vF_up, (int32_t)gapE, vl), vl);
            vF = __riscv_vmax_vv_i32m4(vF, vZero, vl);
            __riscv_vsse32_v_i32m4(&F[IDX(i, j, cols)], stride, vF, vl);

            /* H[i][j] = max(0, diag+score, E, F) */
            vint32m4_t vH = __riscv_vmax_vv_i32m4(diag_sc, vE, vl);
            vH = __riscv_vmax_vv_i32m4(vH, vF, vl);
            vH = __riscv_vmax_vv_i32m4(vH, vZero, vl);
            __riscv_vsse32_v_i32m4(&H[IDX(i, j, cols)], stride, vH, vl);

            i    += (int32_t)vl;
            done += (int32_t)vl;
        }

        /* Track global best and early termination (scalar). */
        for (int32_t k = 0; k < diag_len; k++) {
            int32_t ci = i_min + k;
            int32_t cj = d - ci;
            int32_t h = H[IDX(ci, cj, cols)];
            int32_t ref_idx = ref_dir ? (refLen - cj) : (cj - 1);

            if (h > best.score) {
                best.score = h;
                best.ref = ref_idx;
            }
            if (terminate > 0 && h >= terminate) {
                stopped = 1;
                break;
            }
        }

        free(sub_buf);
    }

    /* Find read end: smallest read position achieving best_score in best column. */
    if (best.ref >= 0) {
        int32_t col = ref_dir ? (refLen - best.ref) : (best.ref + 1);
        best.read = readLen - 1;
        for (int32_t ri = 1; ri <= readLen; ri++) {
            if (H[IDX(ri, col, cols)] == best.score) {
                best.read = ri - 1;
                break;
            }
        }
    }

    free(H);
    free(E);
    free(F);
    if (ref_mapped) free(ref_mapped);
    return best;
}

static int8_t *seq_reverse(const int8_t *seq, int32_t end)
{
    int8_t *rev = (int8_t *)calloc(end + 1, sizeof(int8_t));
    int32_t s = 0, e = end;
    while (s <= e) {
        rev[s] = seq[e];
        rev[e] = seq[s];
        s++; e--;
    }
    return rev;
}

s_align *ssw_align(const s_profile *prof, const int8_t *ref, int32_t refLen,
                   uint8_t weight_gapO, uint8_t weight_gapE,
                   uint8_t flag, uint16_t filters,
                   int32_t filterd, int32_t maskLen)
{
    s_align *r = (s_align *)calloc(1, sizeof(s_align));
    r->ref_begin1 = -1;
    r->read_begin1 = -1;

    /* Forward pass: optimal score and end positions. */
    align_end fwd = sw_core(ref, 0, refLen, prof->read, prof->readLen,
                            weight_gapO, weight_gapE, prof->mat, prof->n, 0);

    if (fwd.score <= 0)
        return r;

    r->score1    = (uint16_t)fwd.score;
    r->ref_end1  = fwd.ref;
    r->read_end1 = fwd.read;

    if (flag == 0 || (flag == 2 && r->score1 < filters))
        return r;

    /* Reverse pass: begin positions. */
    int8_t *read_rev = seq_reverse(prof->read, r->read_end1);
    align_end rev = sw_core(ref, 1, r->ref_end1 + 1,
                            read_rev, r->read_end1 + 1,
                            weight_gapO, weight_gapE,
                            prof->mat, prof->n, r->score1);
    free(read_rev);

    r->ref_begin1  = rev.ref;
    r->read_begin1 = r->read_end1 - rev.read;

    return r;
}
