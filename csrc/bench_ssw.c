/*
 * bench_ssw.c — Benchmark harness for the Striped Smith-Waterman library.
 *
 * Reads DNA sequences from a FASTA file (simple reader, no zlib),
 * aligns random pairs using the SSW library, measures execution time,
 * and reports results.
 *
 * Build (inside the initial_code/ workspace with RVV toolchain):
 *   ${RISCVCC} -O2 -march=rv64gcv_zba -c -o rvv_ssw.o ssw.c
 *   ${RISCVCC} -O2 -march=rv64gcv_zba -o bench_ssw bench_ssw.c rvv_ssw.o -lm
 *
 * Run:
 *   ${SIMULATOR} ./bench_ssw dataset.fa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ssw.h"

/* ------------------------------------------------------------------ */
/* Portable timing: use clock_gettime on hosted platforms, fall back   */
/* to the RISC-V cycle CSR on bare-metal (riscv64-unknown-elf).       */
/* ------------------------------------------------------------------ */
#if defined(__linux__) || defined(__APPLE__) || defined(_POSIX_TIMERS)
#include <time.h>
static inline uint64_t _bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#elif defined(__riscv)
static inline uint64_t _bench_now_ns(void) {
    uint64_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r"(cycles));
    /* Approximate: assume 1 GHz clock -> 1 cycle ≈ 1 ns */
    return cycles;
}
#else
#include <time.h>
static inline uint64_t _bench_now_ns(void) {
    return (uint64_t)clock() * (1000000000ULL / CLOCKS_PER_SEC);
}
#endif

#ifndef NUM_PAIRS
#define NUM_PAIRS 50
#endif

#ifndef MAX_SEQS
#define MAX_SEQS 500
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

#ifndef MIN_SUBSEQ_LEN
#define MIN_SUBSEQ_LEN 50
#endif

#ifndef MAX_SUBSEQ_LEN
#define MAX_SUBSEQ_LEN 500
#endif

/* ------------------------------------------------------------------ */
/* Nucleotide encoding table                                          */
/* ------------------------------------------------------------------ */

/* A=0, C=1, G=2, T=3, N(other)=4 */
static const int8_t nt_table[128] = {
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

/* ------------------------------------------------------------------ */
/* Simple FASTA reader (no zlib required)                             */
/* ------------------------------------------------------------------ */

typedef struct
{
  char *data;
  size_t len;
} Sequence;

static size_t read_fasta(const char *path, Sequence *seqs, size_t max_seqs)
{
  FILE *fp = fopen(path, "r");
  if (!fp)
  {
    fprintf(stderr, "Cannot open FASTA file: %s\n", path);
    return 0;
  }

  size_t count = 0;
  char line[4096];
  char buf[MAX_SEQ_LEN + 1];
  size_t buf_len = 0;
  int in_seq = 0;

  while (fgets(line, sizeof(line), fp))
  {
    size_t ll = strlen(line);
    while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r'))
      line[--ll] = '\0';

    if (line[0] == '>')
    {
      if (in_seq && buf_len > 0 && count < max_seqs)
      {
        seqs[count].data = (char *)malloc(buf_len + 1);
        memcpy(seqs[count].data, buf, buf_len);
        seqs[count].data[buf_len] = '\0';
        seqs[count].len = buf_len;
        count++;
      }
      buf_len = 0;
      in_seq = 1;
    }
    else if (in_seq)
    {
      for (size_t k = 0; k < ll && buf_len < MAX_SEQ_LEN; ++k)
      {
        if (line[k] >= 'A' && line[k] <= 'Z')
          buf[buf_len++] = line[k];
        else if (line[k] >= 'a' && line[k] <= 'z')
          buf[buf_len++] = (char)(line[k] - 32);
      }
    }
  }

  if (in_seq && buf_len > 0 && count < max_seqs)
  {
    seqs[count].data = (char *)malloc(buf_len + 1);
    memcpy(seqs[count].data, buf, buf_len);
    seqs[count].data[buf_len] = '\0';
    seqs[count].len = buf_len;
    count++;
  }

  fclose(fp);
  return count;
}

/* ------------------------------------------------------------------ */
/* Extract a random subsequence from a long sequence                  */
/* ------------------------------------------------------------------ */

static void extract_subseq(const Sequence *src, char *dst, int32_t *out_len)
{
  size_t min_len = MIN_SUBSEQ_LEN;
  size_t max_len = MAX_SUBSEQ_LEN;
  if (max_len > src->len)
    max_len = src->len;
  if (min_len > max_len)
    min_len = max_len;

  size_t len = min_len + (size_t)(rand() % (int)(max_len - min_len + 1));
  size_t start = 0;
  if (src->len > len)
    start = (size_t)(rand() % (int)(src->len - len));

  memcpy(dst, src->data + start, len);
  dst[len] = '\0';
  *out_len = (int32_t)len;
}

/* ------------------------------------------------------------------ */
/* Convert DNA string to numeric encoding                             */
/* ------------------------------------------------------------------ */

static int8_t *encode_seq(const char *seq, int32_t len)
{
  int8_t *num = (int8_t *)malloc((size_t)len);
  if (!num)
    return NULL;
  for (int32_t i = 0; i < len; ++i)
    num[i] = nt_table[(unsigned char)seq[i]];
  return num;
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
  const char *fasta_path = (argc > 1) ? argv[1] : "10k.fa";
  int32_t match = 2, mismatch = 2;
  uint8_t gap_open = 3, gap_extend = 1;

  srand(42);

  /* Build 5x5 scoring matrix (A, C, G, T, N) */
  int8_t mat[25];
  {
    int l, m, k;
    for (l = k = 0; l < 4; ++l)
    {
      for (m = 0; m < 4; ++m)
        mat[k++] = l == m ? match : -mismatch;
      mat[k++] = 0;
    }
    for (m = 0; m < 5; ++m)
      mat[k++] = 0;
  }

  /* Read FASTA dataset */
  Sequence *seqs = (Sequence *)malloc(MAX_SEQS * sizeof(Sequence));
  if (!seqs)
  {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }
  size_t num_seqs = read_fasta(fasta_path, seqs, MAX_SEQS);
  if (num_seqs == 0)
  {
    fprintf(stderr, "No sequences loaded from %s\n", fasta_path);
    free(seqs);
    return 1;
  }
  printf("Loaded %zu sequence(s) from %s\n", num_seqs, fasta_path);

  int all_ok = 1;
  char *buf_query = (char *)malloc(MAX_SUBSEQ_LEN + 1);
  char *buf_ref = (char *)malloc(MAX_SUBSEQ_LEN + 1);
  if (!buf_query || !buf_ref)
  {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  uint64_t t0 = _bench_now_ns();

  for (int pair = 0; pair < NUM_PAIRS; ++pair)
  {
    /* Pick source sequences (may be the same for single-sequence files) */
    size_t idx1 = (size_t)(rand() % (int)num_seqs);
    size_t idx2 = (size_t)(rand() % (int)num_seqs);

    int32_t query_len, ref_len;
    extract_subseq(&seqs[idx1], buf_query, &query_len);
    extract_subseq(&seqs[idx2], buf_ref, &ref_len);

    int8_t *num_query = encode_seq(buf_query, query_len);
    int8_t *num_ref = encode_seq(buf_ref, ref_len);
    if (!num_query || !num_ref)
    {
      fprintf(stderr, "encode failed for pair %d\n", pair);
      all_ok = 0;
      free(num_query);
      free(num_ref);
      continue;
    }

    /* Create profile and align */
    s_profile *profile = ssw_init(num_query, query_len, mat, 5, 2);
    if (!profile)
    {
      fprintf(stderr, "ssw_init failed for pair %d\n", pair);
      all_ok = 0;
      free(num_query);
      free(num_ref);
      continue;
    }

    int32_t maskLen = query_len > 30 ? 15 : query_len / 2;
    if (maskLen < 15)
      maskLen = 15;
    s_align *result = ssw_align(profile, num_ref, ref_len,
                                gap_open, gap_extend,
                                1, 0, 0, maskLen);
    if (!result)
    {
      fprintf(stderr, "ssw_align failed for pair %d\n", pair);
      all_ok = 0;
    }
    else
    {
      printf("pair %3d: qlen=%4d rlen=%4d score=%d\n",
             pair, query_len, ref_len, result->score1);
      align_destroy(result);
    }

    init_destroy(profile);
    free(num_query);
    free(num_ref);
  }

  uint64_t t1 = _bench_now_ns();
  double elapsed_ns = (double)(t1 - t0);

  printf("\ntime_ns=%.0f\n", elapsed_ns);
  printf("avg_time_ns=%.0f\n", elapsed_ns / NUM_PAIRS);
  printf("status=%s\n", all_ok ? "OK" : "FAIL");

  free(buf_query);
  free(buf_ref);
  for (size_t i = 0; i < num_seqs; ++i)
    free(seqs[i].data);
  free(seqs);

  return all_ok ? 0 : 2;
}
