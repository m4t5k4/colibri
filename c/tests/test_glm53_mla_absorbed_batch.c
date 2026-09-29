/* Production-geometry absorbed-query dispatch, fallback, and profile oracle. */
#define GLM53_ABSORBED_BENCH_NO_MAIN
#include "bench_glm53_mla_absorbed.c"
#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"

#if defined(__AVX2__) && defined(_OPENMP) && !defined(COLI_METAL) && !defined(COLI_VULKAN)
#define EXPECT_ABSORBED_BATCH 1
#else
#define EXPECT_ABSORBED_BATCH 0
#endif

static Mat absorbed_fixture(void) {
    Mat w = {0};
    w.fmt = 4; w.rows = BO; w.columns = BI; w.gs = BGS;
    w.q4 = malloc((size_t)BO * (BI / 2));
    w.s = malloc((size_t)BO * (BI / BGS) * sizeof(float));
    assert(w.q4 && w.s);
    return w;
}

static void test_env(void) {
    unsetenv("GLM53_MLA_ABSORBED_BATCH");
    assert(glm53_mla_absorbed_batch_env() == 0);
    setenv("GLM53_MLA_ABSORBED_BATCH", "0", 1);
    assert(glm53_mla_absorbed_batch_env() == 0);
    setenv("GLM53_MLA_ABSORBED_BATCH", "1", 1);
    assert(glm53_mla_absorbed_batch_env() == 1);
    setenv("GLM53_MLA_ABSORBED_BATCH", "invalid", 1);
    assert(glm53_mla_absorbed_batch_env() == 0);
    unsetenv("GLM53_MLA_ABSORBED_BATCH");
    assert(glm53_mla_qb_rows4_env() == 0);
}

static void test_production_and_profile(Mat *w, float *x, float *reference,
                                        float *candidate) {
    GModel legacy = {0}, fast = {0};
    fast.mla_absorbed_batch = 1;
    legacy.mla_decode_call = 1;
    fast.mla_decode_call = 1;
    legacy.cuda.profile.clock = now_s;
    fast.cuda.profile.clock = now_s;
    for (unsigned pattern = 0; pattern < 4; pattern++) {
        bench_fill(w, x, pattern);
        bench_current(reference, w, x);
        mla_absorbed_project(&legacy, candidate, w, x, BH, BR, BI, 1, 1);
        assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
        assert(legacy.mla_absorbed_batch_calls == 0);
        assert(legacy.mla_absorbed_legacy_calls == pattern + 1u);
        assert(mla_absorbed_queries(candidate, w, x, BH, BR, BI, 1, 1) ==
               EXPECT_ABSORBED_BATCH);
        assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
        mla_absorbed_project(&fast, candidate, w, x, BH, BR, BI, 1, 1);
        assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
        assert(fast.mla_absorbed_batch_calls ==
               (uint64_t)(pattern + 1u) * EXPECT_ABSORBED_BATCH);
        assert(fast.mla_absorbed_legacy_calls ==
               (uint64_t)(pattern + 1u) * !EXPECT_ABSORBED_BATCH);
    }
    assert(legacy.t_mla_absorbed_q > 0 && fast.t_mla_absorbed_q > 0);
    assert(legacy.mla_absorbed_q_calls == 4 * BH);
    assert(fast.mla_absorbed_q_rows == 4 * BO);

    G53DecodeBase start, end;
    g53_decode_capture(&fast, &start); /* four prefill projections excluded */
    bench_fill(w, x, 3);
    mla_absorbed_project(&fast, candidate, w, x, BH, BR, BI, 1, 1);
    g53_decode_capture(&fast, &end);
    g53_decode_accumulate(&fast.decode_total, &end, &start);
    const G53DecodeBase *d = &fast.decode_total;
    assert(d->valid && d->mla_absorbed_q_calls == BH && d->mla_absorbed_q_rows == BO);
    assert(d->mla_absorbed_batch_calls == EXPECT_ABSORBED_BATCH);
    assert(d->mla_absorbed_legacy_calls == !EXPECT_ABSORBED_BATCH);
    assert(d->mla_absorbed_q == end.mla_absorbed_q - start.mla_absorbed_q);
    assert(d->mla_absorbed_q > 0);

    GModel no_profile = {0};
    no_profile.mla_absorbed_batch = 1;
    no_profile.mla_decode_call = 1;
    mla_absorbed_project(&no_profile, candidate, w, x, BH, BR, BI, 1, 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
    assert(no_profile.t_mla_absorbed_q == 0);
    assert(no_profile.mla_absorbed_q_calls == 0);
    assert(no_profile.mla_absorbed_batch_calls == EXPECT_ABSORBED_BATCH);
    assert(no_profile.mla_absorbed_legacy_calls == !EXPECT_ABSORBED_BATCH);

    GModel prefill = {0};
    prefill.mla_absorbed_batch = 1; /* tokens=1 is not sufficient for decode. */
    mla_absorbed_project(&prefill, candidate, w, x, BH, BR, BI, 1, 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
    assert(prefill.mla_absorbed_batch_calls == 0);
    assert(prefill.mla_absorbed_legacy_calls == 1);
}

static void test_fallbacks(Mat *w, float *x, float *reference, float *candidate) {
    bench_fill(w, x, 2);
    bench_current(reference, w, x);

    /* More than one input token never enters the new S=1 dispatch. */
    assert(mla_absorbed_queries(candidate, w, x, BH, BR, BI, 2, 1) == 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);

#ifdef _OPENMP
    /* The new team is not created from an already active OpenMP team. */
    int nested_result = -1;
    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        nested_result = mla_absorbed_queries(candidate, w, x, BH, BR, BI, 1, 1);
    }
    assert(nested_result == 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
#endif

    /* The same bytes with eight scales per row exercise unsupported gs32. */
    Mat gs32 = *w;
    float *steps32 = malloc((size_t)BO * (BI / 32) * sizeof(float));
    assert(steps32);
    for (size_t i = 0; i < (size_t)BO * (BI / 32); i++)
        steps32[i] = (float)(1u + i % 13u) * 0.000625f;
    gs32.gs = 32; gs32.s = steps32;
    bench_current(reference, &gs32, x);
    assert(mla_absorbed_queries(candidate, &gs32, x, BH, BR, BI, 1, 1) == 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
    free(steps32);

    /* A different head count has safe, contiguous rows but uses legacy. */
    Mat two_heads = *w; two_heads.rows = 2 * BR;
    for (int h = 0; h < 2; h++)
        mv_rows(reference + (size_t)h * BR, &two_heads,
                x + (size_t)h * BI, h * BR, BR);
    assert(mla_absorbed_queries(candidate, &two_heads, x, 2, BR, BI, 1, 1) == 0);
    assert(memcmp(reference, candidate, (size_t)2 * BR * sizeof(float)) == 0);

    /* Non-int4 production geometry must remain on the generic f32 matvec. */
    Mat f32 = {0};
    f32.fmt = 0; f32.rows = BO; f32.columns = BI;
    float *weights = calloc((size_t)BO * BI, sizeof(float));
    assert(weights);
    for (int row = 0; row < BO; row++)
        weights[(size_t)row * BI + row % BI] = (float)((row % 11) + 1) * 0.001f;
    f32.f = weights;
    for (int h = 0; h < BH; h++)
        mv_rows(reference + (size_t)h * BR, &f32,
                x + (size_t)h * BI, h * BR, BR);
    assert(mla_absorbed_queries(candidate, &f32, x, BH, BR, BI, 1, 1) == 0);
    assert(memcmp(reference, candidate, (size_t)BO * sizeof(float)) == 0);
    mat_release(&f32);
}

int main(void) {
    test_env();
    Mat w = absorbed_fixture();
    float *x = malloc((size_t)BH * BI * sizeof(float));
    float *reference = malloc((size_t)BO * sizeof(float));
    float *candidate = malloc((size_t)BO * sizeof(float));
    assert(x && reference && candidate);
    test_production_and_profile(&w, x, reference, candidate);
    test_fallbacks(&w, x, reference, candidate);
    free(candidate); free(reference); free(x); mat_release(&w);
    puts("glm53 MLA absorbed batch: PASS");
    return 0;
}
