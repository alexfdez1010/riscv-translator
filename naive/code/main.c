#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "ssw.h"
#include "kseq.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

static int fileread(FILE *fp, char *buf, int len) {
    return (int)fread(buf, 1, len, fp);
}

KSEQ_INIT(FILE*, fileread)

/* Nucleotide encoding: A=0 C=1 G=2 T/U=3 other=4 */
static int8_t nt_table[128] = {
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

static void reverse_comple(const char *seq, char *rc)
{
    int32_t end = strlen(seq), start = 0;
    static const int8_t rc_table[128] = {
        4, 4,  4, 4,  4,  4,  4, 4,  4, 4, 4, 4,  4, 4, 4,  4,
        4, 4,  4, 4,  4,  4,  4, 4,  4, 4, 4, 4,  4, 4, 4,  4,
        4, 4,  4, 4,  4,  4,  4, 4,  4, 4, 4, 4,  4, 4, 4,  4,
        4, 4,  4, 4,  4,  4,  4, 4,  4, 4, 4, 4,  4, 4, 4,  4,
        4, 84, 4, 71, 4,  4,  4, 67, 4, 4, 4, 4,  4, 4, 78, 4,
        4, 4,  4, 4,  65, 65, 4, 4,  4, 4, 4, 4,  4, 4, 4,  4,
        4, 84, 4, 71, 4,  4,  4, 67, 4, 4, 4, 4,  4, 4, 78, 4,
        4, 4,  4, 4,  65, 65, 4, 4,  4, 4, 4, 4,  4, 4, 4,  4
    };
    rc[end] = '\0';
    --end;
    while (LIKELY(start < end)) {
        rc[start] = (char)rc_table[(int8_t)seq[end]];
        rc[end]   = (char)rc_table[(int8_t)seq[start]];
        ++start;
        --end;
    }
    if (start == end)
        rc[start] = (char)rc_table[(int8_t)seq[start]];
}

static void print_result(const s_align *a, const kseq_t *ref_seq,
                         const kseq_t *read, int8_t strand)
{
    fprintf(stdout, "target_name: %s\nquery_name: %s\noptimal_alignment_score: %d\t",
            ref_seq->name.s, read->name.s, a->score1);
    if (strand == 0)
        fprintf(stdout, "strand: +\t");
    else
        fprintf(stdout, "strand: -\t");
    if (a->ref_begin1 + 1)
        fprintf(stdout, "target_begin: %d\t", a->ref_begin1 + 1);
    fprintf(stdout, "target_end: %d\t", a->ref_end1 + 1);
    if (a->read_begin1 + 1)
        fprintf(stdout, "query_begin: %d\t", a->read_begin1 + 1);
    fprintf(stdout, "query_end: %d\n\n", a->read_end1 + 1);
}

int main(int argc, char *const argv[])
{
    clock_t start, end;
    float cpu_time;
    int32_t match = 2, mismatch = 2, gap_open = 3, gap_extension = 1;
    int32_t n = 5, filter = 0, reverse = 0, path = 0;
    int32_t s1 = 67108864, s2 = 128;

    /* Minimal argument parsing: positional args are target.fa query.fa.
       Flags: -m N, -x N, -o N, -e N, -r, -c, -f N  */
    int file_start = 1;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') continue;
        for (int j = 1; argv[i][j]; j++) {
            switch (argv[i][j]) {
            case 'm': if (i+1 < argc) { match = atoi(argv[++i]); } goto next;
            case 'x': if (i+1 < argc) { mismatch = atoi(argv[++i]); } goto next;
            case 'o': if (i+1 < argc) { gap_open = atoi(argv[++i]); } goto next;
            case 'e': if (i+1 < argc) { gap_extension = atoi(argv[++i]); } goto next;
            case 'f': if (i+1 < argc) { filter = atoi(argv[++i]); } goto next;
            case 'r': reverse = 1; break;
            case 'c': path = 1; break;
            }
        }
        next:;
    }

    /* Find first non-option positional argument. */
    file_start = 1;
    while (file_start < argc && argv[file_start][0] == '-') {
        if (argv[file_start][1] == 'm' || argv[file_start][1] == 'x' ||
            argv[file_start][1] == 'o' || argv[file_start][1] == 'e' ||
            argv[file_start][1] == 'f')
            file_start += 2;
        else
            file_start += 1;
    }

    if (file_start + 2 > argc) {
        fprintf(stderr, "Usage: ssw_test [options] <target.fasta> <query.fasta>\n");
        return 1;
    }

    /* Build 5x5 DNA scoring matrix (A C G T N). */
    int8_t mata[25];
    int32_t k = 0;
    for (int32_t i = 0; i < 4; i++) {
        for (int32_t j = 0; j < 4; j++)
            mata[k++] = (i == j) ? (int8_t)match : (int8_t)-mismatch;
        mata[k++] = 0; /* N column */
    }
    for (int32_t j = 0; j < 5; j++)
        mata[k++] = 0; /* N row */

    int8_t *ref_num = (int8_t *)malloc(s1);
    int8_t *num     = (int8_t *)malloc(s2);
    char   *read_rc = NULL;
    int8_t *num_rc  = NULL;
    if (reverse) {
        read_rc = (char *)malloc(s2);
        num_rc  = (int8_t *)malloc(s2);
    }

    FILE *read_fp = fopen(argv[file_start + 1], "r");
    if (!read_fp) {
        fprintf(stderr, "Cannot open '%s'\n", argv[file_start + 1]);
        return 1;
    }
    kseq_t *read_seq = kseq_init(read_fp);

    start = clock();
    while (kseq_read(read_seq) >= 0) {
        int32_t readLen = read_seq->seq.l;
        int32_t maskLen = readLen / 2;
        int32_t m;

        while (readLen >= s2) {
            ++s2; kroundup32(s2);
            num = (int8_t *)realloc(num, s2);
            if (reverse) {
                read_rc = (char *)realloc(read_rc, s2);
                num_rc  = (int8_t *)realloc(num_rc, s2);
            }
        }
        for (m = 0; m < readLen; m++)
            num[m] = nt_table[(int)read_seq->seq.s[m]];

        s_profile *p = ssw_init(num, readLen, mata, n);
        s_profile *p_rc = NULL;
        if (reverse) {
            reverse_comple(read_seq->seq.s, read_rc);
            for (m = 0; m < readLen; m++)
                num_rc[m] = nt_table[(int)read_rc[m]];
            p_rc = ssw_init(num_rc, readLen, mata, n);
        }

        FILE *ref_fp = fopen(argv[file_start], "r");
        kseq_t *ref_seq = kseq_init(ref_fp);

        while (kseq_read(ref_seq) >= 0) {
            int32_t refLen = ref_seq->seq.l;
            int8_t flag = path ? 2 : 0;

            while (refLen > s1) {
                ++s1; kroundup32(s1);
                ref_num = (int8_t *)realloc(ref_num, s1);
            }
            for (m = 0; m < refLen; m++)
                ref_num[m] = nt_table[(int)ref_seq->seq.s[m]];

            s_align *result = ssw_align(p, ref_num, refLen,
                                        gap_open, gap_extension,
                                        flag, filter, 0, maskLen);
            s_align *result_rc = NULL;
            if (reverse)
                result_rc = ssw_align(p_rc, ref_num, refLen,
                                      gap_open, gap_extension,
                                      flag, filter, 0, maskLen);

            if (!result) {
                fprintf(stderr, "Warning: Alignment failed.\nref: %s\nread: %s\n\n",
                        ref_seq->name.s, read_seq->name.s);
            } else if (result_rc && result_rc->score1 > result->score1 &&
                       result_rc->score1 >= filter) {
                print_result(result_rc, ref_seq, read_seq, 1);
            } else if (result->score1 > 0 && result->score1 >= filter) {
                print_result(result, ref_seq, read_seq, 0);
            } else if (result->score1 <= 0) {
                fprintf(stderr, "No identical residue.\nref: %s\nread: %s\n\n",
                        ref_seq->name.s, read_seq->name.s);
            }

            if (result_rc) align_destroy(result_rc);
            if (result) align_destroy(result);
        }

        if (p_rc) init_destroy(p_rc);
        init_destroy(p);
        kseq_destroy(ref_seq);
        fclose(ref_fp);
    }
    end = clock();
    cpu_time = ((float)(end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr, "CPU time: %f seconds\n", cpu_time);

    if (num_rc) { free(num_rc); free(read_rc); }
    kseq_destroy(read_seq);
    fclose(read_fp);
    free(num);
    free(ref_num);
    return 0;
}
