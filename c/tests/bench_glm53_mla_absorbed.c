/* Synthetic production-shape MLA absorbed-query benchmark. No model needed.
 * This file deliberately does not change the production absorbed-query loop.
 * The row arithmetic below mirrors quant.h:matmul_i4_grouped for S=1;
 * bitwise comparison against mv_rows checks that it stays equivalent. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#include <assert.h>

enum { BH = 64, BR = 512, BI = 256, BGS = 64, BO = BH * BR, BTRIALS = 3 };

static uint32_t bench_random(uint32_t *state) {
    uint32_t v = *state;
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    return *state = v;
}

static void bench_fill(Mat *w, float *x, unsigned pattern) {
    uint32_t state = 0x91e10da5u ^ (0x9e3779b9u * (pattern + 1u));
    const size_t packed = (size_t)BO * (BI / 2);
    const size_t scales = (size_t)BO * (BI / BGS);
    uint8_t *q4 = (uint8_t *)w->q4;
    float *step = (float *)w->s;
    for (size_t i = 0; i < packed; i++) {
        uint32_t r = bench_random(&state);
        q4[i] = pattern == 1 ? (i & 1 ? 0xff : 0x00) : (uint8_t)(r >> 16);
    }
    for (size_t i = 0; i < scales; i++) {
        uint32_t r = bench_random(&state);
        step[i] = (float)(1u + (r % 31u)) * 0.000625f;
    }
    for (int h = 0; h < BH; h++)
        for (int i = 0; i < BI; i++) {
            uint32_t r = bench_random(&state);
            x[(size_t)h * BI + i] = (float)((int)(r % 127u) - 63) * 0.015625f
                                   + (float)h * 0.0001f;
        }
}

static void bench_current(float *out, const Mat *w, const float *x) {
    for (int h = 0; h < BH; h++)
        mv_rows(out + (size_t)h * BR, w, x + (size_t)h * BI, h * BR, BR);
}

/* Same per-row operation sequence as quant.h:matmul_i4_grouped, with S=1.
 * All four groups are complete at BI=256/BGS=64. There is no head or row
 * reduction; changing only row assignment must retain every output bit. */
static inline float bench_grouped_row(const uint8_t *w, const float *scl,
                                      const float *xs, int I, int gs) {
    float a = 0;
    for (int g = 0; g * gs < I; g++) {
        int base = g * gs; int glen = gs;
        if (base + glen > I) glen = I - base;
        float sc = scl[g];
        int i = base;
#ifdef __AVX2__
        const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi32(8);
        __m256 acc = _mm256_setzero_ps();
        for (; i + 16 <= base + glen; i += 16) {
            __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
            __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
            __m128i nib = _mm_unpacklo_epi8(lo, hi);
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), b8));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), w0, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
        }
        a = fmaf(hsum256(acc), sc, a);
#endif
        for (; i < base + glen; i += 2) {
            if (i + 1 < base + glen) {
                uint8_t byte = w[i >> 1];
                a += (xs[i] * (float)((int)(byte & 0xF) - 8) +
                      xs[i + 1] * (float)((int)(byte >> 4) - 8)) * sc;
            } else {
                uint8_t byte = w[i >> 1];
                a += xs[i] * (float)((int)(byte & 0xF) - 8) * sc;
            }
        }
    }
    return a;
}

static void bench_batched(float *out, const Mat *w, const float *x) {
    const int I = w->columns, gs = w->gs;
    const int packed = (I + 1) / 2, groups = (I + gs - 1) / gs;
    #pragma omp parallel for schedule(static)
    for (int flat = 0; flat < BO; flat++) {
        const int h = flat / BR;
        out[flat] = bench_grouped_row(w->q4 + (size_t)flat * packed,
                                      w->s + (size_t)flat * groups,
                                      x + (size_t)h * I, I, gs);
    }
}

/* Diagnostic control: the same row body as BATCHED, but 64 parallel regions.
 * A difference between CURRENT and this control is benchmark codegen, not
 * evidence that team creation caused the entire CURRENT/BATCHED gap. */
static void bench_same_row_64_regions(float *out, const Mat *w, const float *x) {
    const int I = w->columns, gs = w->gs;
    const int packed = (I + 1) / 2, groups = (I + gs - 1) / gs;
    for (int h = 0; h < BH; h++) {
        const float *query = x + (size_t)h * I;
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < BR; row++) {
            const int flat = h * BR + row;
            out[flat] = bench_grouped_row(w->q4 + (size_t)flat * packed,
                                          w->s + (size_t)flat * groups,
                                          query, I, gs);
        }
    }
}

static void bench_parity(Mat *w, float *x, float *current, float *batched,
                         float *same_row) {
    for (unsigned pattern = 0; pattern < 4; pattern++) {
        bench_fill(w, x, pattern);
        bench_current(current, w, x);
        bench_batched(batched, w, x);
        bench_same_row_64_regions(same_row, w, x);
        for (int i = 0; i < BO; i++) {
            if (memcmp(current + i, batched + i, sizeof(float)) ||
                memcmp(current + i, same_row + i, sizeof(float))) {
                uint32_t old_bits, new_bits, control_bits;
                memcpy(&old_bits, current + i, sizeof old_bits);
                memcpy(&new_bits, batched + i, sizeof new_bits);
                memcpy(&control_bits, same_row + i, sizeof control_bits);
                fprintf(stderr, "absorbed parity FAILED pattern=%u head=%d row=%d current=%08x batched=%08x same_row=%08x\n",
                        pattern, i / BR, i % BR, old_bits, new_bits, control_bits);
                exit(1);
            }
            assert(isfinite(current[i]));
        }
    }
    puts("absorbed parity: PASS (4 patterns, all 32768 output bits each, both prototypes)");
}

static double bench_elapsed(void (*fn)(float *, const Mat *, const float *),
                            float *out, const Mat *w, const float *x, int iterations) {
    const double start = now_s();
    for (int i = 0; i < iterations; i++) fn(out, w, x);
    return now_s() - start;
}

static double bench_median3(double a, double b, double c) {
    if (a > b) { double tmp = a; a = b; b = tmp; }
    if (b > c) { double tmp = b; b = c; c = tmp; }
    if (a > b) b = a;
    return b;
}

int main(int argc, char **argv) {
    int parity_only = argc > 1 && !strcmp(argv[1], "--parity-only");
    int iterations = 1024;
    if (argc > 1 && !parity_only) {
        char *end = NULL;
        long parsed = strtol(argv[1], &end, 10);
        if (!end || *end || parsed < 1 || parsed > 100000) {
            fprintf(stderr, "usage: %s [iterations|--parity-only]\n", argv[0]);
            return 2;
        }
        iterations = (int)parsed;
    }
    Mat w = {0};
    w.fmt = 4; w.rows = BO; w.columns = BI; w.gs = BGS;
    w.q4 = malloc((size_t)BO * (BI / 2));
    w.s = malloc((size_t)BO * (BI / BGS) * sizeof(float));
    float *x = malloc((size_t)BH * BI * sizeof(float));
    float *current = malloc((size_t)BO * sizeof(float));
    float *batched = malloc((size_t)BO * sizeof(float));
    float *same_row = malloc((size_t)BO * sizeof(float));
    if (!w.q4 || !w.s || !x || !current || !batched || !same_row) {
        fprintf(stderr, "absorbed benchmark: allocation failed\n");
        return 2;
    }
    bench_parity(&w, x, current, batched, same_row);
    if (!parity_only) {
        bench_fill(&w, x, 0);
        for (int i = 0; i < 4; i++) {
            bench_current(current, &w, x);
            bench_batched(batched, &w, x);
            bench_same_row_64_regions(same_row, &w, x);
        }
        double times_current[BTRIALS], times_batched[BTRIALS], times_same_row[BTRIALS];
        for (int trial = 0; trial < BTRIALS; trial++) {
            if (trial & 1) {
                times_batched[trial] = bench_elapsed(bench_batched, batched, &w, x, iterations);
                times_same_row[trial] = bench_elapsed(bench_same_row_64_regions, same_row, &w, x, iterations);
                times_current[trial] = bench_elapsed(bench_current, current, &w, x, iterations);
            } else {
                times_current[trial] = bench_elapsed(bench_current, current, &w, x, iterations);
                times_same_row[trial] = bench_elapsed(bench_same_row_64_regions, same_row, &w, x, iterations);
                times_batched[trial] = bench_elapsed(bench_batched, batched, &w, x, iterations);
            }
        }
        const double tc = bench_median3(times_current[0], times_current[1], times_current[2]);
        const double tb = bench_median3(times_batched[0], times_batched[1], times_batched[2]);
        const double ts = bench_median3(times_same_row[0], times_same_row[1], times_same_row[2]);
        const int single_iterations = iterations * 16;
        const double single_start = now_s();
        for (int i = 0; i < single_iterations; i++)
            mv_rows(current, &w, x, 0, BR);
        const double single_s = now_s() - single_start;
        if (memcmp(current, batched, (size_t)BO * sizeof(float)) ||
            memcmp(current, same_row, (size_t)BO * sizeof(float))) {
            fprintf(stderr, "absorbed benchmark: timed results changed\n");
            return 1;
        }
        double speedup = tc / tb;
        printf("absorbed benchmark threads=%d heads=%d rows_per_head=%d cols=%d gs=%d "
               "iterations=%d trials=%d statistic=median "
               "current_total_s=%.6f batched_total_s=%.6f speedup=%.4fx "
               "reduction_pct=%.2f current_batch_ms=%.6f batched_batch_ms=%.6f "
               "current_1408_s=%.6f batched_1408_s=%.6f "
               "single_head_us=%.3f current_64_heads_ms=%.6f "
               "same_row_64_regions_total_s=%.6f same_row_vs_batched=%.4fx "
               "parity=bitwise\n",
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               BH, BR, BI, BGS, iterations, BTRIALS, tc, tb, speedup,
               100.0 * (1.0 - tb / tc), 1000.0 * tc / iterations,
               1000.0 * tb / iterations, 1408.0 * tc / iterations,
               1408.0 * tb / iterations, 1e6 * single_s / single_iterations,
               1000.0 * tc / iterations, ts, ts / tb);
    }
    free(same_row); free(batched); free(current); free(x); mat_release(&w);
    return 0;
}
