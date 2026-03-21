/* The MIT License

   Copyright (c) 2012-2015 Boston College.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* The 2-clause BSD License

   Copyright 2006 Michael Farrar.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  ssw_hwy.cpp
 *
 *  Port of ssw.c to Google Highway SIMD library.
 *  Original created by Mengyao Zhao on 6/22/10.
 *  Copyright 2010 Boston College. All rights reserved.
 *  Version 1.2.6
 *
 *  The lazy-F loop implementation was derived from SWPS3, which is
 *  MIT licensed under ETH Zürich, Institute of Computational Science.
 *
 *  The core SW loop referenced the swsse2 implementation, which is
 *  BSD licensed under Michael Farrar.
 */

// ============================================================
// Section 1: Standard includes and type definitions.
// Guarded so that foreach_target re-inclusions don't redefine them.
// ============================================================
#ifndef SSW_HWY_DEFS_
#define SSW_HWY_DEFS_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "ssw.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Convert the coordinate in the scoring matrix into the coordinate in one line of the band. */
#define set_u(u, w, i, j) { int _x=(i)-(w); _x=_x>0?_x:0; (u)=(j)-_x+1; }

/* Convert the coordinate in the direction matrix into the coordinate in one line of the band. */
#define set_d(u, w, i, j, p) { int _x=(i)-(w); _x=_x>0?_x:0; _x=(j)-_x; (u)=_x*3+p; }

#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

typedef struct {
	uint16_t score;
	int32_t ref;	 //0-based position
	int32_t read;    //alignment ending position on read, 0-based
} alignment_end;

typedef struct {
	uint32_t* seq;
	int32_t length;
} cigar;

struct _profile {
	uint8_t* profile_byte;	// was __m128i*
	int16_t* profile_word;	// was __m128i*
	const int8_t* read;
	const int8_t* mat;
	int32_t readLen;
	int32_t n;
	uint8_t bias;
};

#endif // SSW_HWY_DEFS_

// ============================================================
// Section 2: Highway target dispatch setup.
// ============================================================
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ssw_hwy.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

// ============================================================
// Section 3: Per-target SIMD code (compiled once per enabled target).
// ============================================================
HWY_BEFORE_NAMESPACE();
namespace ssw_hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Fixed 128-bit tags matching the original SSE2 register width.
using D8  = hn::FixedTag<uint8_t, 16>;   // 16 x uint8
using D16 = hn::FixedTag<int16_t, 8>;    // 8 x int16
using DU16 = hn::FixedTag<uint16_t, 8>;  // 8 x uint16 (for unsigned sat sub)

// ----------------------------------------------------------------
// sw_byte: 8-bit striped Smith-Waterman kernel (port of sw_sse2_byte)
// ----------------------------------------------------------------
alignment_end* SswByte(const int8_t* ref,
                       int8_t ref_dir,
                       int32_t refLen,
                       int32_t readLen,
                       const uint8_t weight_gapO,
                       const uint8_t weight_gapE,
                       const uint8_t* vProfile,
                       uint8_t terminate,
                       uint8_t bias,
                       int32_t maskLen) {

	const D8 d8;

	uint8_t max = 0;
	int32_t end_read = readLen - 1;
	int32_t end_ref = -1;
	int32_t segLen = (readLen + 15) / 16;

	/* array to record the largest score of each reference position */
	uint8_t* maxColumn = (uint8_t*) calloc(refLen, 1);

	/* array to record the alignment read ending position of the largest score of each reference position */
	int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

	auto vZero = hn::Zero(d8);

	/* Aligned buffers - each segment is 16 bytes (one 128-bit vector) */
	uint8_t* pvHStore = (uint8_t*) calloc(segLen, 16);
	uint8_t* pvHLoad  = (uint8_t*) calloc(segLen, 16);
	uint8_t* pvE      = (uint8_t*) calloc(segLen, 16);
	uint8_t* pvHmax   = (uint8_t*) calloc(segLen, 16);

	int32_t i, j, k;
	auto vGapO = hn::Set(d8, weight_gapO);
	auto vGapE = hn::Set(d8, weight_gapE);
	auto vBias = hn::Set(d8, bias);

	auto vMaxScore = vZero;
	auto vMaxMark = vZero;
	decltype(vZero) vTemp;
	int32_t edge, begin = 0, end = refLen, step = 1;

	/* outer loop to process the reference sequence */
	if (ref_dir == 1) {
		begin = refLen - 1;
		end = -1;
		step = -1;
	}
	for (i = begin; LIKELY(i != end); i += step) {
		auto e = vZero;
		auto vF = vZero;
		auto vMaxColumn = vZero;

		auto vH = hn::Load(d8, pvHStore + (segLen - 1) * 16);
		vH = hn::ShiftLeftBytes<1>(d8, vH);
		const uint8_t* vP = vProfile + ref[i] * segLen * 16;

		/* Swap the 2 H buffers. */
		uint8_t* pv = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pv;

		/* inner loop to process the query sequence */
		for (j = 0; LIKELY(j < segLen); ++j) {
			vH = hn::SaturatedAdd(vH, hn::Load(d8, vP + j * 16));
			vH = hn::SaturatedSub(vH, vBias); /* vH will be always > 0 */

			/* Get max from vH, vE and vF. */
			e = hn::Load(d8, pvE + j * 16);
			vH = hn::Max(vH, e);
			vH = hn::Max(vH, vF);
			vMaxColumn = hn::Max(vMaxColumn, vH);

			/* Save vH values. */
			hn::Store(vH, d8, pvHStore + j * 16);

			/* Update vE value. */
			vH = hn::SaturatedSub(vH, vGapO);
			e = hn::SaturatedSub(e, vGapE);
			e = hn::Max(e, vH);
			hn::Store(e, d8, pvE + j * 16);

			/* Update vF value. */
			vF = hn::SaturatedSub(vF, vGapE);
			vF = hn::Max(vF, vH);

			/* Load the next vH. */
			vH = hn::Load(d8, pvHLoad + j * 16);
		}

		/* Lazy_F loop */
		for (k = 0; LIKELY(k < 16); ++k) {
			vF = hn::ShiftLeftBytes<1>(d8, vF);
			for (j = 0; LIKELY(j < segLen); ++j) {
				vH = hn::Load(d8, pvHStore + j * 16);
				vH = hn::Max(vH, vF);
				vMaxColumn = hn::Max(vMaxColumn, vH);
				hn::Store(vH, d8, pvHStore + j * 16);
				vH = hn::SaturatedSub(vH, vGapO);
				vF = hn::SaturatedSub(vF, vGapE);
				vTemp = hn::SaturatedSub(vF, vH);
				if (UNLIKELY(hn::AllTrue(d8, hn::Eq(vTemp, vZero)))) goto lazy_f_done;
			}
		}

lazy_f_done:
		vMaxScore = hn::Max(vMaxScore, vMaxColumn);
		if (!hn::AllTrue(d8, hn::Eq(vMaxMark, vMaxScore))) {
			uint8_t temp;
			vMaxMark = vMaxScore;
			temp = hn::ReduceMax(d8, vMaxScore);

			if (LIKELY(temp > max)) {
				max = temp;
				if (max + bias >= 255) break;	//overflow
				end_ref = i;

				/* Store the column with the highest alignment score */
				for (j = 0; LIKELY(j < segLen); ++j) {
					hn::Store(hn::Load(d8, pvHStore + j * 16), d8, pvHmax + j * 16);
				}
			}
		}

		/* Record the max score of current column. */
		maxColumn[i] = hn::ReduceMax(d8, vMaxColumn);
		if (maxColumn[i] == terminate) break;
	}

	/* Trace the alignment ending position on read. */
	uint8_t *t = pvHmax;
	int32_t column_len = segLen * 16;
	for (i = 0; LIKELY(i < column_len); ++i, ++t) {
		int32_t temp;
		if (*t == max) {
			temp = i / 16 + i % 16 * segLen;
			if (temp < end_read) end_read = temp;
		}
	}

	free(pvHmax);
	free(pvE);
	free(pvHLoad);
	free(pvHStore);

	/* Find the most possible 2nd best alignment. */
	alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
	bests[0].score = max + bias >= 255 ? 255 : max;
	bests[0].ref = end_ref;
	bests[0].read = end_read;

	bests[1].score = 0;
	bests[1].ref = 0;
	bests[1].read = 0;

	edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
	for (i = 0; i < edge; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}
	edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
	for (i = edge + 1; i < refLen; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}

	free(maxColumn);
	free(end_read_column);
	return bests;
}

// ----------------------------------------------------------------
// sw_word: 16-bit striped Smith-Waterman kernel (port of sw_sse2_word)
// ----------------------------------------------------------------
alignment_end* SswWord(const int8_t* ref,
                       int8_t ref_dir,
                       int32_t refLen,
                       int32_t readLen,
                       const uint8_t weight_gapO,
                       const uint8_t weight_gapE,
                       const int16_t* vProfile,
                       uint16_t terminate,
                       int32_t maskLen) {

	const D16 d16;
	const DU16 du16;

	uint16_t max = 0;
	int32_t end_read = readLen - 1;
	int32_t end_ref = 0;
	int32_t segLen = (readLen + 7) / 8;

	uint16_t* maxColumn = (uint16_t*) calloc(refLen, 2);
	int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

	auto vZero = hn::Zero(d16);

	/* Each segment is 8 x int16_t = 16 bytes */
	int16_t* pvHStore = (int16_t*) calloc(segLen * 8, sizeof(int16_t));
	int16_t* pvHLoad  = (int16_t*) calloc(segLen * 8, sizeof(int16_t));
	int16_t* pvE      = (int16_t*) calloc(segLen * 8, sizeof(int16_t));
	int16_t* pvHmax   = (int16_t*) calloc(segLen * 8, sizeof(int16_t));

	int32_t i, j, k;
	auto vGapO = hn::Set(d16, (int16_t)weight_gapO);
	auto vGapE = hn::Set(d16, (int16_t)weight_gapE);

	/* BitCast-friendly unsigned versions for saturating subtract */
	auto vGapO_u = hn::Set(du16, (uint16_t)weight_gapO);
	auto vGapE_u = hn::Set(du16, (uint16_t)weight_gapE);

	auto vMaxScore = vZero;
	auto vMaxMark = vZero;
	decltype(vZero) vTemp;
	int32_t edge, begin = 0, end = refLen, step = 1;

	if (ref_dir == 1) {
		begin = refLen - 1;
		end = -1;
		step = -1;
	}
	for (i = begin; LIKELY(i != end); i += step) {
		auto e = vZero;
		auto vF = vZero;
		auto vH = hn::Load(d16, pvHStore + (segLen - 1) * 8);
		vH = hn::ShiftLeftBytes<2>(d16, vH);

		/* Swap the 2 H buffers. */
		int16_t* pv = pvHLoad;

		auto vMaxColumn = vZero;

		const int16_t* vP = vProfile + ref[i] * segLen * 8;
		pvHLoad = pvHStore;
		pvHStore = pv;

		/* inner loop to process the query sequence */
		for (j = 0; LIKELY(j < segLen); j ++) {
			vH = hn::SaturatedAdd(vH, hn::Load(d16, vP + j * 8));

			/* Get max from vH, vE and vF. */
			e = hn::Load(d16, pvE + j * 8);
			vH = hn::Max(vH, e);
			vH = hn::Max(vH, vF);
			vMaxColumn = hn::Max(vMaxColumn, vH);

			/* Save vH values. */
			hn::Store(vH, d16, pvHStore + j * 8);

			/* Update vE value: unsigned saturating subtract via BitCast */
			vH = hn::BitCast(d16, hn::SaturatedSub(hn::BitCast(du16, vH), vGapO_u));
			e  = hn::BitCast(d16, hn::SaturatedSub(hn::BitCast(du16, e),  vGapE_u));
			e = hn::Max(e, vH);
			hn::Store(e, d16, pvE + j * 8);

			/* Update vF value. */
			vF = hn::BitCast(d16, hn::SaturatedSub(hn::BitCast(du16, vF), vGapE_u));
			vF = hn::Max(vF, vH);

			/* Load the next vH. */
			vH = hn::Load(d16, pvHLoad + j * 8);
		}

		/* Lazy_F loop */
		for (k = 0; LIKELY(k < 8); ++k) {
			vF = hn::ShiftLeftBytes<2>(d16, vF);
			for (j = 0; LIKELY(j < segLen); ++j) {
				vH = hn::Load(d16, pvHStore + j * 8);
				vH = hn::Max(vH, vF);
				vMaxColumn = hn::Max(vMaxColumn, vH);
				hn::Store(vH, d16, pvHStore + j * 8);
				vH = hn::BitCast(d16, hn::SaturatedSub(hn::BitCast(du16, vH), vGapO_u));
				vF = hn::BitCast(d16, hn::SaturatedSub(hn::BitCast(du16, vF), vGapE_u));
				if (UNLIKELY(hn::AllFalse(d16, hn::Gt(vF, vH)))) goto lazy_f_done_w;
			}
		}

lazy_f_done_w:
		vMaxScore = hn::Max(vMaxScore, vMaxColumn);
		if (!hn::AllTrue(d16, hn::Eq(vMaxMark, vMaxScore))) {
			uint16_t temp;
			vMaxMark = vMaxScore;
			temp = static_cast<uint16_t>(hn::ReduceMax(d16, vMaxScore));

			if (LIKELY(temp > max)) {
				max = temp;
				end_ref = i;
				for (j = 0; LIKELY(j < segLen); ++j) {
					hn::Store(hn::Load(d16, pvHStore + j * 8), d16, pvHmax + j * 8);
				}
			}
		}

		/* Record the max score of current column. */
		maxColumn[i] = static_cast<uint16_t>(hn::ReduceMax(d16, vMaxColumn));
		if (maxColumn[i] == terminate) break;
	}

	/* Trace the alignment ending position on read. */
	uint16_t *t = (uint16_t*)pvHmax;
	int32_t column_len = segLen * 8;
	for (i = 0; LIKELY(i < column_len); ++i, ++t) {
		int32_t temp;
		if (*t == max) {
			temp = i / 8 + i % 8 * segLen;
			if (temp < end_read) end_read = temp;
		}
	}

	free(pvHmax);
	free(pvE);
	free(pvHLoad);
	free(pvHStore);

	/* Find the most possible 2nd best alignment. */
	alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
	bests[0].score = max;
	bests[0].ref = end_ref;
	bests[0].read = end_read;

	bests[1].score = 0;
	bests[1].ref = 0;
	bests[1].read = 0;

	edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
	for (i = 0; i < edge; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}
	edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
	for (i = edge; i < refLen; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}

	free(maxColumn);
	free(end_read_column);
	return bests;
}

}  // namespace HWY_NAMESPACE
}  // namespace ssw_hwy
HWY_AFTER_NAMESPACE();

// ============================================================
// Section 4: Code compiled once (scalar functions + public API).
// ============================================================
#if HWY_ONCE

// Export SIMD dispatch tables
namespace ssw_hwy {

HWY_EXPORT(SswByte);
HWY_EXPORT(SswWord);

static alignment_end* CallSswByte(const int8_t* ref, int8_t ref_dir,
                                   int32_t refLen, int32_t readLen,
                                   uint8_t weight_gapO, uint8_t weight_gapE,
                                   const uint8_t* vProfile,
                                   uint8_t terminate, uint8_t bias,
                                   int32_t maskLen) {
	return HWY_DYNAMIC_DISPATCH(SswByte)(ref, ref_dir, refLen, readLen,
	    weight_gapO, weight_gapE, vProfile, terminate, bias, maskLen);
}

static alignment_end* CallSswWord(const int8_t* ref, int8_t ref_dir,
                                   int32_t refLen, int32_t readLen,
                                   uint8_t weight_gapO, uint8_t weight_gapE,
                                   const int16_t* vProfile,
                                   uint16_t terminate, int32_t maskLen) {
	return HWY_DYNAMIC_DISPATCH(SswWord)(ref, ref_dir, refLen, readLen,
	    weight_gapO, weight_gapE, vProfile, terminate, maskLen);
}

}  // namespace ssw_hwy

// ----------------------------------------------------------------
// encoded_ops table
// ----------------------------------------------------------------
extern "C" {

const uint8_t encoded_ops[] = {
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0 /*   */, 0 /* ! */, 0 /* " */, 0 /* # */,
	0 /* $ */, 0 /* % */, 0 /* & */, 0 /* ' */,
	0 /* ( */, 0 /* ) */, 0 /* * */, 0 /* + */,
	0 /* , */, 0 /* - */, 0 /* . */, 0 /* / */,
	0 /* 0 */, 0 /* 1 */, 0 /* 2 */, 0 /* 3 */,
	0 /* 4 */, 0 /* 5 */, 0 /* 6 */, 0 /* 7 */,
	0 /* 8 */, 0 /* 9 */, 0 /* : */, 0 /* ; */,
	0 /* < */, 7 /* = */, 0 /* > */, 0 /* ? */,
	0 /* @ */, 0 /* A */, 0 /* B */, 0 /* C */,
	2 /* D */, 0 /* E */, 0 /* F */, 0 /* G */,
	5 /* H */, 1 /* I */, 0 /* J */, 0 /* K */,
	0 /* L */, 0 /* M */, 3 /* N */, 0 /* O */,
	6 /* P */, 0 /* Q */, 0 /* R */, 4 /* S */,
	0 /* T */, 0 /* U */, 0 /* V */, 0 /* W */,
	8 /* X */, 0 /* Y */, 0 /* Z */, 0 /* [ */,
	0 /* \ */, 0 /* ] */, 0 /* ^ */, 0 /* _ */,
	0 /* ` */, 0 /* a */, 0 /* b */, 0 /* c */,
	0 /* d */, 0 /* e */, 0 /* f */, 0 /* g */,
	0 /* h */, 0 /* i */, 0 /* j */, 0 /* k */,
	0 /* l */, 0 /* m */, 0 /* n */, 0 /* o */,
	0 /* p */, 0 /* q */, 0 /* r */, 0 /* s */,
	0 /* t */, 0 /* u */, 0 /* v */, 0 /* w */,
	0 /* x */, 0 /* y */, 0 /* z */, 0 /* { */,
	0 /* | */, 0 /* } */, 0 /* ~ */, 0 /*  */
};

}  // extern "C"

// ----------------------------------------------------------------
// Profile building functions (no SIMD needed - just fill byte arrays)
// ----------------------------------------------------------------

/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
static uint8_t* qP_byte (const int8_t* read_num,
                          const int8_t* mat,
                          const int32_t readLen,
                          const int32_t n,
                          uint8_t bias) {

	int32_t segLen = (readLen + 15) / 16;
	uint8_t* vProfile = (uint8_t*)malloc(n * segLen * 16);
	int8_t* t = (int8_t*)vProfile;
	int32_t nt, i, j, segNum;

	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
				*t++ = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias;
				j += segLen;
			}
		}
	}
	return vProfile;
}

static int16_t* qP_word (const int8_t* read_num,
                          const int8_t* mat,
                          const int32_t readLen,
                          const int32_t n) {

	int32_t segLen = (readLen + 7) / 8;
	int16_t* vProfile = (int16_t*)malloc(n * segLen * 8 * sizeof(int16_t));
	int16_t* t = vProfile;
	int32_t nt, i, j;
	int32_t segNum;

	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 8) ; segNum ++) {
				*t++ = j>= readLen ? 0 : mat[nt * n + read_num[j]];
				j += segLen;
			}
		}
	}
	return vProfile;
}

// ----------------------------------------------------------------
// Scalar helper functions (unchanged from original ssw.c)
// ----------------------------------------------------------------

static cigar* banded_sw (const int8_t* ref,
                 const int8_t* read,
                 int32_t refLen,
                 int32_t readLen,
                 int32_t score,
                 const uint32_t weight_gapO,
                 const uint32_t weight_gapE,
                 int32_t band_width,
                 const int8_t* mat,
                 int32_t n) {

	uint32_t *c = (uint32_t*)malloc(16 * sizeof(uint32_t)), *c1;
	int32_t i, j, e, f, temp1, temp2, s = 16, s1 = 8, l, max = 0, len;
	int32_t max_i = 0, max_j = 0;
	int64_t s2 = 1024;
	char op, prev_op;
	int32_t width, width_d, *h_b, *e_b, *h_c;
	int8_t *direction, *direction_line;
	const int32_t neg_inf = INT32_MIN / 2;
    len = refLen > readLen ? refLen : readLen;
	cigar* result = (cigar*)malloc(sizeof(cigar));
	h_b = (int32_t*)malloc(s1 * sizeof(int32_t));
	e_b = (int32_t*)malloc(s1 * sizeof(int32_t));
	h_c = (int32_t*)malloc(s1 * sizeof(int32_t));
	direction = (int8_t*)malloc(s2 * sizeof(int8_t));

	do {
		width = band_width * 2 + 3, width_d = band_width * 2 + 1;
		while (width >= s1) {
			++s1;
			kroundup32(s1);
			h_b = (int32_t*)realloc(h_b, s1 * sizeof(int32_t));
			e_b = (int32_t*)realloc(e_b, s1 * sizeof(int32_t));
			h_c = (int32_t*)realloc(h_c, s1 * sizeof(int32_t));
		}
		while (width_d * readLen * 3 >= s2) {
			++s2;
			kroundup32(s2);
			direction = (int8_t*)realloc(direction, s2 * sizeof(int8_t));
		}
		direction_line = direction;
		for (j = 1; LIKELY(j < width - 1); j ++) h_b[j] = 0;
		for (i = 0; LIKELY(i < readLen); i ++) {
			int32_t beg = 0, end = refLen - 1, u = 0, edge;
			j = i - band_width;	beg = beg > j ? beg : j;
			j = i + band_width; end = end < j ? end : j;
			edge = end + 1 < width - 1 ? end + 1 : width - 1;
			f = neg_inf;
			h_b[0] = h_b[edge] = h_c[0] = 0;
			e_b[0] = e_b[edge] = neg_inf;
			direction_line = direction + width_d * i * 3;

			for (j = beg; LIKELY(j <= end); j ++) {
				int32_t b, e1, f1, d, de, df, dh;
				set_u(u, band_width, i, j);	set_u(e, band_width, i - 1, j);
				set_u(b, band_width, i, j - 1); set_u(d, band_width, i - 1, j - 1);
				set_d(de, band_width, i, j, 0);
				set_d(df, band_width, i, j, 1);
				set_d(dh, band_width, i, j, 2);

				temp1 = i == 0 ? -(int32_t)weight_gapO : h_b[e] - (int32_t)weight_gapO;
				temp2 = i == 0 ? neg_inf : e_b[e] - (int32_t)weight_gapE;
				e_b[u] = temp1 > temp2 ? temp1 : temp2;
				direction_line[de] = temp1 > temp2 ? 3 : 2;

				temp1 = h_c[b] - (int32_t)weight_gapO;
				temp2 = f - (int32_t)weight_gapE;
				f = temp1 > temp2 ? temp1 : temp2;
				direction_line[df] = temp1 > temp2 ? 5 : 4;

				e1 = e_b[u] > 0 ? e_b[u] : 0;
				f1 = f > 0 ? f : 0;
				temp1 = e1 > f1 ? e1 : f1;
				temp2 = h_b[d] + mat[ref[j] * n + read[i]];
				h_c[u] = temp1 > temp2 ? temp1 : temp2;

				if (h_c[u] > max) {
					max = h_c[u];
					max_i = i;
					max_j = j;
				}

				if (temp1 <= temp2) direction_line[dh] = 1;
				else direction_line[dh] = e1 > f1 ? direction_line[de] : direction_line[df];
			}
			for (j = 1; j <= u; j ++) h_b[j] = h_c[j];
		}
		band_width *= 2;
	} while (max < score && band_width <= len);
	band_width /= 2;

	// trace back
	i = max_i;
	j = max_j;
	e = 0;
	l = 0;
	op = prev_op = 'M';
	temp2 = 2;
	direction_line = direction + width_d * i * 3;
    while (LIKELY(i >= 0 && j > 0)) {
		set_d(temp1, band_width, i, j, temp2);
		switch (direction_line[temp1]) {
			case 1:
				--i;
				--j;
				temp2 = 2;
				direction_line -= width_d * 3;
				op = 'M';
				break;
			case 2:
			 	--i;
				temp2 = 0;
				direction_line -= width_d * 3;
				op = 'I';
				break;
			case 3:
				--i;
				temp2 = 2;
				direction_line -= width_d * 3;
				op = 'I';
				break;
			case 4:
				--j;
				temp2 = 1;
				op = 'D';
				break;
			case 5:
				--j;
				temp2 = 2;
				op = 'D';
				break;
			default:
				fprintf(stderr, "Trace back error: %d.\n", direction_line[temp1 - 1]);
				free(direction);
				free(h_c);
				free(e_b);
				free(h_b);
				free(c);
				free(result);
				return 0;
		}
		if (op == prev_op) ++e;
		else {
			++l;
			while (l >= s) {
				++s;
				kroundup32(s);
				c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
			}
			c[l - 1] = to_cigar_int(e, prev_op);
			prev_op = op;
			e = 1;
		}
	}
	if (op == 'M') {
		++l;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		c[l - 1] = to_cigar_int(e + 1, op);
	}else {
		l += 2;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		c[l - 2] = to_cigar_int(e, op);
		c[l - 1] = to_cigar_int(1, 'M');
	}

	// reverse cigar
	c1 = (uint32_t*)malloc(l * sizeof(uint32_t));
	s = 0;
	e = l - 1;
	while (LIKELY(s <= e)) {
		c1[s] = c[e];
		c1[e] = c[s];
		++ s;
		-- e;
	}
	result->seq = c1;
	result->length = l;

	free(direction);
	free(h_c);
	free(e_b);
	free(h_b);
	free(c);
	return result;
}

static int32_t cigar_alignment_score(const cigar* path,
                                     const int8_t* ref,
                                     const int8_t* read,
                                     const int8_t* mat,
                                     int32_t n,
                                     const uint32_t weight_gapO,
                                     const uint32_t weight_gapE) {
	int32_t score = 0;
	int32_t ref_pos = 0, read_pos = 0;
	for (int32_t i = 0; i < path->length; ++i) {
		uint32_t len = cigar_int_to_len(path->seq[i]);
		char op = cigar_int_to_op(path->seq[i]);
		if (op == 'M') {
			for (uint32_t j = 0; j < len; ++j) {
				score += mat[ref[ref_pos] * n + read[read_pos]];
				++ref_pos;
				++read_pos;
			}
		} else {
			int32_t penalty = weight_gapO + (len > 1 ? (len - 1) * weight_gapE : 0);
			score -= penalty;
			if (op == 'I') read_pos += len;
			else if (op == 'D') ref_pos += len;
		}
	}
	return score;
}

static int8_t* seq_reverse(const int8_t* seq, int32_t end) {
	int8_t* reverse = (int8_t*)calloc(end + 1, sizeof(int8_t));
	int32_t start = 0;
	while (LIKELY(start <= end)) {
		reverse[start] = seq[end];
		reverse[end] = seq[start];
		++ start;
		-- end;
	}
	return reverse;
}

// ----------------------------------------------------------------
// Public API - wrapped in extern "C" for C linkage
// ----------------------------------------------------------------
extern "C" {

s_profile* ssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size) {
	s_profile* p = (s_profile*)calloc(1, sizeof(struct _profile));
	p->profile_byte = 0;
	p->profile_word = 0;
	p->bias = 0;

	if (score_size == 0 || score_size == 2) {
		int32_t bias = 0, i;
		for (i = 0; i < n*n; i++) if (mat[i] < bias) bias = mat[i];
		bias = abs(bias);

		p->bias = bias;
		p->profile_byte = qP_byte (read, mat, readLen, n, bias);
	}
	if (score_size == 1 || score_size == 2) p->profile_word = qP_word (read, mat, readLen, n);
	p->read = read;
	p->mat = mat;
	p->readLen = readLen;
	p->n = n;
	return p;
}

void init_destroy (s_profile* p) {
	free(p->profile_byte);
	free(p->profile_word);
	free(p);
}

s_align* ssw_align (const s_profile* prof,
                    const int8_t* ref,
                    int32_t refLen,
                    const uint8_t weight_gapO,
                    const uint8_t weight_gapE,
                    const uint8_t flag,
                    const uint16_t filters,
                    const int32_t filterd,
                    const int32_t maskLen) {

	alignment_end* bests = 0, *bests_reverse = 0;
	int32_t word = 0, band_width = 0, readLen = prof->readLen;
	int8_t* read_reverse = 0;
	cigar* path;
	s_align* r = (s_align*)calloc(1, sizeof(s_align));
	r->ref_begin1 = -1;
	r->read_begin1 = -1;
	r->cigar = 0;
	r->cigarLen = 0;
    r->flag = 0;
	if (maskLen < 15) {
		fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
	}

	// Find the alignment scores and ending positions
	if (prof->profile_byte) {
		bests = ssw_hwy::CallSswByte(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
		    prof->profile_byte, 0xFF, prof->bias, maskLen);
		if (prof->profile_word && bests[0].score == 255) {
			free(bests);
			bests = ssw_hwy::CallSswWord(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
			    prof->profile_word, 0xFFFF, maskLen);
			word = 1;
		} else if (bests[0].score == 255) {
			fprintf(stderr, "Please set 2 to the score_size parameter of the function ssw_init, otherwise the alignment results will be incorrect.\n");
			free(r);
			return NULL;
		}
	} else if (prof->profile_word) {
		bests = ssw_hwy::CallSswWord(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
		    prof->profile_word, 0xFFFF, maskLen);
		word = 1;
	} else {
		fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
		free(r);
		return NULL;
	}
	if (bests[0].score <= 0) {
		free(bests);
		goto ssw_align_end;
	}

	r->score1 = bests[0].score;
	r->ref_end1 = bests[0].ref;
	r->read_end1 = bests[0].read;
	if (maskLen >= 15) {
		r->score2 = bests[1].score;
		r->ref_end2 = bests[1].ref;
	} else {
		r->score2 = 0;
		r->ref_end2 = -1;
	}
	free(bests);
	if (flag == 0 || (flag == 2 && r->score1 < filters)) goto ssw_align_end;

	// Find the beginning position of the best alignment.
	read_reverse = seq_reverse(prof->read, r->read_end1);
	if (word == 0) {
		uint8_t* vP_byte = qP_byte(read_reverse, prof->mat, r->read_end1 + 1, prof->n, prof->bias);
		bests_reverse = ssw_hwy::CallSswByte(ref, 1, r->ref_end1 + 1, r->read_end1 + 1,
		    weight_gapO, weight_gapE, vP_byte, (uint8_t)r->score1, prof->bias, maskLen);
		free(vP_byte);
	} else {
		int16_t* vP_word = qP_word(read_reverse, prof->mat, r->read_end1 + 1, prof->n);
		bests_reverse = ssw_hwy::CallSswWord(ref, 1, r->ref_end1 + 1, r->read_end1 + 1,
		    weight_gapO, weight_gapE, vP_word, r->score1, maskLen);
		free(vP_word);
	}
	free(read_reverse);
	r->ref_begin1 = bests_reverse[0].ref;
	r->read_begin1 = r->read_end1 - bests_reverse[0].read;

    if (UNLIKELY(r->score1 > bests_reverse[0].score)) {
		fprintf(stderr, "Warning: The alignment path of one pair of sequences may miss a small part. [ssw.c ssw_align]\n");
        r->flag = 2;
    }
    free(bests_reverse);

	if ((7&flag) == 0 || ((2&flag) != 0 && r->score1 < filters) || ((4&flag) != 0 && (r->ref_end1 - r->ref_begin1 > filterd || r->read_end1 - r->read_begin1 > filterd))) goto ssw_align_end;

	// Generate cigar.
	refLen = r->ref_end1 - r->ref_begin1 + 1;
	readLen = r->read_end1 - r->read_begin1 + 1;
	band_width = abs(refLen - readLen) + 1;
	{
		int32_t full_band = refLen > readLen ? refLen : readLen;
		while (1) {
			path = banded_sw(ref + r->ref_begin1, prof->read + r->read_begin1, refLen, readLen, r->score1, weight_gapO, weight_gapE, band_width, prof->mat, prof->n);
			if (path == 0) break;
			int32_t cigar_score = cigar_alignment_score(path, ref + r->ref_begin1, prof->read + r->read_begin1, prof->mat, prof->n, weight_gapO, weight_gapE);
			if (cigar_score == r->score1) break;
			free(path->seq);
			free(path);
			if (band_width >= full_band) {
				path = 0;
				break;
			}
			band_width = full_band;
		}
	}

    if (path == 0) r->flag = 1;
    else {
		r->cigar = path->seq;
		r->cigarLen = path->length;
		free(path);
	}

ssw_align_end:
	return r;
}

void align_destroy (s_align* a) {
	free(a->cigar);
	free(a);
}

uint32_t* add_cigar (uint32_t* new_cigar, int32_t* p, int32_t* s, uint32_t length, char op) {
	if ((*p) >= (*s)) {
		++(*s);
		kroundup32(*s);
		new_cigar = (uint32_t*)realloc(new_cigar, (*s)*sizeof(uint32_t));
	}
	new_cigar[(*p) ++] = to_cigar_int(length, op);
	return new_cigar;
}

uint32_t* store_previous_m (int8_t choice,
                           uint32_t* length_m,
                           uint32_t* length_x,
                           int32_t* p,
                           int32_t* s,
                           uint32_t* new_cigar) {

	if ((*length_m) && (choice == 2 || !choice)) {
		new_cigar = add_cigar (new_cigar, p, s, (*length_m), '=');
		(*length_m) = 0;
	} else if ((*length_x) && (choice == 1 || !choice)) {
		new_cigar = add_cigar (new_cigar, p, s, (*length_x), 'X');
		(*length_x) = 0;
	}
	return new_cigar;
}

int32_t mark_mismatch (int32_t ref_begin1,
                       int32_t read_begin1,
                       int32_t read_end1,
                       const int8_t* ref,
                       const int8_t* read,
                       int32_t readLen,
                       uint32_t** cigar,
                       int32_t* cigarLen) {

	int32_t mismatch_length = 0, p = 0, i, length, j, s = *cigarLen + 2;
	uint32_t *new_cigar = (uint32_t*)malloc(s*sizeof(uint32_t)), length_m = 0,  length_x = 0;
	char op;

	ref += ref_begin1;
	read += read_begin1;
	if (read_begin1 > 0) new_cigar[p ++] = to_cigar_int(read_begin1, 'S');
	for (i = 0; i < (*cigarLen); ++i) {
		op = cigar_int_to_op((*cigar)[i]);
		length = cigar_int_to_len((*cigar)[i]);
		if (op == 'M') {
			for (j = 0; j < length; ++j) {
				if (*ref != *read) {
					++ mismatch_length;
					new_cigar = store_previous_m (2, &length_m, &length_x, &p, &s, new_cigar);
					++ length_x;
				} else {
					new_cigar = store_previous_m (1, &length_m, &length_x, &p, &s, new_cigar);
					++ length_m;
				}
				++ ref;
				++ read;
			}
		}else if (op == 'I') {
			read += length;
			mismatch_length += length;
			new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);
			new_cigar = add_cigar (new_cigar, &p, &s, length, 'I');
		}else if (op == 'D') {
			ref += length;
			mismatch_length += length;
			new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);
			new_cigar = add_cigar (new_cigar, &p, &s, length, 'D');
		}
	}
	new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);

	length = readLen - read_end1 - 1;
	if (length > 0) new_cigar = add_cigar(new_cigar, &p, &s, length, 'S');

	(*cigarLen) = p;
	free(*cigar);
	(*cigar) = new_cigar;
	return mismatch_length;
}

}  // extern "C"

#endif  // HWY_ONCE
