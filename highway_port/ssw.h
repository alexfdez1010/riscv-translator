/*
 *  ssw.h
 *
 *  Created by Mengyao Zhao on 6/22/10.
 *  Copyright 2010 Boston College. All rights reserved.
 *	Version 1.2.3
 *	Last revision by Mengyao Zhao on 2022-May-24.
 *
 *  Modified: removed platform-specific SIMD includes (emmintrin.h / sse2neon.h).
 *  The profile struct is opaque; no SIMD types are exposed to callers.
 */

#ifndef SSW_H
#define SSW_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif	// __cplusplus

#define MAPSTR "MIDNSHP=X"
#ifndef BAM_CIGAR_SHIFT
#define BAM_CIGAR_SHIFT 4u
#endif

extern const uint8_t encoded_ops[];

/*!	@typedef	structure of the query profile	*/
struct _profile;
typedef struct _profile s_profile;

/*!	@typedef	structure of the alignment result	*/
typedef struct {
	uint16_t score1;
	uint16_t score2;
	int32_t ref_begin1;
	int32_t ref_end1;
	int32_t	read_begin1;
	int32_t read_end1;
	int32_t ref_end2;
	uint32_t* cigar;
	int32_t cigarLen;
    uint16_t flag;
} s_align;

s_profile* ssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size);

void init_destroy (s_profile* p);

s_align* ssw_align (const s_profile* prof,
					const int8_t* ref,
					int32_t refLen,
					const uint8_t weight_gapO,
					const uint8_t weight_gapE,
					const uint8_t flag,
					const uint16_t filters,
					const int32_t filterd,
					const int32_t maskLen);

void align_destroy (s_align* a);

int32_t mark_mismatch (int32_t ref_begin1,
					   int32_t read_begin1,
					   int32_t read_end1,
					   const int8_t* ref,
					   const int8_t* read,
					   int32_t readLen,
					   uint32_t** cigar,
					   int32_t* cigarLen);

static inline uint32_t to_cigar_int (uint32_t length, unsigned char op_letter) {
	return (length << BAM_CIGAR_SHIFT) | (encoded_ops[op_letter]);
}

static inline char cigar_int_to_op(uint32_t cigar_int) {
	return (cigar_int & 0xfU) > 8 ? 'M': MAPSTR[cigar_int & 0xfU];
}

static inline uint32_t cigar_int_to_len (uint32_t cigar_int) {
	return cigar_int >> BAM_CIGAR_SHIFT;
}

#ifdef __cplusplus
}
#endif	// __cplusplus

#endif	// SSW_H
