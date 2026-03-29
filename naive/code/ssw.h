#ifndef SSW_H
#define SSW_H

#include <stdint.h>

typedef struct {
    uint16_t score1;
    int32_t ref_begin1;
    int32_t ref_end1;
    int32_t read_begin1;
    int32_t read_end1;
} s_align;

typedef struct {
    const int8_t *read;
    const int8_t *mat;
    int32_t readLen;
    int32_t n;
} s_profile;

s_profile *ssw_init(const int8_t *read, int32_t readLen,
                    const int8_t *mat, int32_t n);
void init_destroy(s_profile *p);

s_align *ssw_align(const s_profile *prof, const int8_t *ref, int32_t refLen,
                   uint8_t weight_gapO, uint8_t weight_gapE,
                   uint8_t flag, uint16_t filters,
                   int32_t filterd, int32_t maskLen);
void align_destroy(s_align *a);

#endif
