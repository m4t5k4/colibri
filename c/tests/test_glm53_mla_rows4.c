/* Opt-in MLA output dispatch and one-token MLA parity. No checkpoint needed. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#include <assert.h>

#if defined(__AVX2__) && !defined(COLI_METAL) && !defined(COLI_VULKAN)
#define EXPECT_ROWS4 1
#else
#define EXPECT_ROWS4 0
#endif

static Mat test_f32(int rows, int columns) {
    Mat w = {0};
    w.fmt = 0; w.rows = rows; w.columns = columns;
    w.f = calloc((size_t)rows * columns, sizeof(float));
    assert(w.f);
    return w;
}

static Mat test_i4(int rows, int columns, int gs) {
    Mat w = {0};
    w.fmt = 4; w.rows = rows; w.columns = columns; w.gs = gs;
    const size_t rb = (size_t)(columns + 1) / 2;
    const size_t ng = (size_t)(columns + gs - 1) / gs;
    uint8_t *q = malloc((size_t)rows * rb);
    float *s = malloc((size_t)rows * ng * sizeof(float));
    assert(q && s);
    for (size_t i = 0; i < (size_t)rows * rb; i++)
        q[i] = (uint8_t)((i * 73u + 39u) & 255u);
    for (size_t i = 0; i < (size_t)rows * ng; i++)
        s[i] = (float)((int)(i % 17u) - 8) * 0.00125f;
    w.q4 = q; w.s = s;
    return w;
}

static void projection_case(int rows, int columns, int gs, int fmt,
                            int enabled, int expected) {
    Mat w = fmt == 4 ? test_i4(rows, columns, gs) : test_f32(rows, columns);
    float *x = malloc((size_t)columns * sizeof(float));
    float *reference = malloc((size_t)rows * sizeof(float));
    float *candidate = malloc((size_t)rows * sizeof(float));
    assert(x && reference && candidate);
    for (int i = 0; i < columns; i++) x[i] = (float)((i * 11) % 29 - 14) * 0.03125f;
    if (fmt == 0) {
        float *f = (float *)w.f;
        for (int i = 0; i < rows * columns; i++)
            f[i] = (float)((i * 7) % 19 - 9) * 0.0078125f;
    }
    mv(reference, &w, x);
    assert(mv_mla_out(candidate, &w, x, enabled) == expected);
    assert(memcmp(reference, candidate, (size_t)rows * sizeof(float)) == 0);
    if (fmt == 4 && rows == 4096 && columns == 2048 && gs == 64) {
        float *ko = malloc((size_t)rows * sizeof(float));
        assert(ko);
        mv_kda_ko(ko, &w, x);
        assert(memcmp(reference, ko, (size_t)rows * sizeof(float)) == 0);
        free(ko);
    }
    free(candidate); free(reference); free(x);
    mat_release(&w);
}

static void one_token_mla_case(int profiled) {
    Cfg c = {0};
    c.hidden = 8; c.n_heads = 3; c.q_lora = 64; c.kv_lora = 64;
    c.qk_nope = 64; c.v_head = 64;
    c.index_nh = 1; c.index_hd = 1; c.index_topk = 1; c.index_kpool = 1;
    c.eps = 1e-6f;
    GLayer l = {0};
    l.qa = test_f32(64, 8); l.qb = test_f32(3 * 64, 64);
    l.kva = test_f32(64, 8); l.kvb_kt = test_f32(3 * 64, 64);
    l.iwq = test_f32(1, 64); l.iwk = test_f32(1, 8);
    l.ikpg = test_f32(1, 8); l.iwp = test_f32(1, 8);
    l.kvb_v = test_f32(3 * 64, 64); l.o = test_i4(8, 3 * 64, 64);
    ((float *)l.qa.f)[0] = 1.0f;
    ((float *)l.qb.f)[0] = 1.0f;
    ((float *)l.kva.f)[0] = 1.0f;
    ((float *)l.kvb_kt.f)[0] = 1.0f;
    ((float *)l.iwq.f)[0] = 1.0f;
    ((float *)l.iwk.f)[0] = 1.0f;
    ((float *)l.iwp.f)[0] = 1.0f;
    ((float *)l.kvb_v.f)[0] = 1.0f;
    float norm[64], index_norm[1] = {1}, index_bias[1] = {0}, ape[1] = {0};
    for (int i = 0; i < 64; i++) norm[i] = 1.0f;
    l.qa_ln = norm; l.kva_ln = norm; l.ik_nw = index_norm;
    l.ik_nb = index_bias; l.ikpa = ape;
    float latent[64] = {0}, keys[1] = {0}, gates[1] = {0};
    GLayerState st = {0};
    st.latent = latent; st.ikeys = keys; st.igates = gates;
    float x[8] = {0.75f, -0.25f, 0.125f, 0, 0, 0, 0, 0};
    float generic[8], fast[8];
    GModel baseline = {0}, experimental = {0};
    experimental.mla_out_rows4 = 1;
#ifdef COLI_CUDA
    if (profiled) experimental.cuda.profile.clock = now_s;
#else
    (void)profiled;
#endif
    mla_layer(&baseline, &c, &l, x, 1, generic, &st, 0);
    mla_layer(&experimental, &c, &l, x, 1, fast, &st, 0);
    assert(memcmp(generic, fast, sizeof(generic)) == 0);
#ifdef COLI_CUDA
    if (profiled) {
        assert(experimental.t_mla_qa > 0);
        assert(experimental.t_mla_qnorm > 0);
        assert(experimental.t_mla_qb > 0);
        double query_child = experimental.t_mla_qa + experimental.t_mla_qnorm +
                             experimental.t_mla_qb;
        assert(experimental.t_mla_query_path >= query_child);
        assert(experimental.t_mla_query_path - query_child < 0.01);
        assert(experimental.t_mla_query_path > 0);
        assert(experimental.t_mla_latent_path > 0);
        assert(experimental.t_mla_absorbed_q > 0);
        assert(experimental.t_mla_index_path > 0);
        assert(experimental.mla_absorbed_q_calls == 3);
        assert(experimental.mla_absorbed_q_rows == 3 * 64);
        double child = experimental.t_mla_query_path + experimental.t_mla_latent_path +
                       experimental.t_mla_absorbed_q + experimental.t_mla_index_path;
        assert(experimental.t_mla_proj >= child);
        assert(experimental.t_mla_proj - child < 0.01);
        G53DecodeBase start, end;
        g53_decode_capture(&experimental, &start); /* one prefill MLA already ran */
        mla_layer(&experimental, &c, &l, x, 1, fast, &st, 0);
        assert(memcmp(generic, fast, sizeof(generic)) == 0);
        g53_decode_capture(&experimental, &end);
        g53_decode_accumulate(&experimental.decode_total, &end, &start);
        const G53DecodeBase *d = &experimental.decode_total;
        assert(d->valid && d->mla_absorbed_q_calls == 3 && d->mla_absorbed_q_rows == 3 * 64);
        assert(d->mla_query_path == end.mla_query_path - start.mla_query_path);
        assert(d->mla_qa == end.mla_qa - start.mla_qa);
        assert(d->mla_qnorm == end.mla_qnorm - start.mla_qnorm);
        assert(d->mla_qb == end.mla_qb - start.mla_qb);
        assert(d->mla_latent_path == end.mla_latent_path - start.mla_latent_path);
        assert(d->mla_absorbed_q == end.mla_absorbed_q - start.mla_absorbed_q);
        assert(d->mla_index_path == end.mla_index_path - start.mla_index_path);
        assert(d->mla_proj == end.mla_proj - start.mla_proj);
        assert(d->mla_query_path > 0 && d->mla_latent_path > 0);
        assert(d->mla_qa > 0 && d->mla_qnorm > 0 && d->mla_qb > 0);
        assert(d->mla_query_path >= d->mla_qa + d->mla_qnorm + d->mla_qb);
        assert(d->mla_query_path - d->mla_qa - d->mla_qnorm - d->mla_qb < 0.01);
        assert(d->mla_absorbed_q > 0 && d->mla_index_path > 0);
        assert(experimental.t_kda_proj == 0 && experimental.t_kda_qkv == 0);
    } else
#endif
    {
        assert(experimental.t_mla_qa == 0);
        assert(experimental.t_mla_qnorm == 0);
        assert(experimental.t_mla_qb == 0);
        assert(experimental.t_mla_query_path == 0);
        assert(experimental.t_mla_latent_path == 0);
        assert(experimental.t_mla_absorbed_q == 0);
        assert(experimental.t_mla_index_path == 0);
        assert(experimental.mla_absorbed_q_calls == 0);
        assert(experimental.mla_absorbed_q_rows == 0);
    }
    assert(baseline.mla_out_generic_calls == 1);
    const int calls = profiled ? 2 : 1;
    assert(experimental.mla_out_rows4_calls == (uint64_t)calls * EXPECT_ROWS4);
    assert(experimental.mla_out_generic_calls == (uint64_t)calls * !EXPECT_ROWS4);
    mat_release(&l.qa); mat_release(&l.qb); mat_release(&l.kva);
    mat_release(&l.kvb_kt); mat_release(&l.iwq); mat_release(&l.iwk);
    mat_release(&l.ikpg); mat_release(&l.iwp); mat_release(&l.kvb_v);
    mat_release(&l.o);
}

#ifndef GLM53_MLA_NO_MAIN
int main(void) {
    unsetenv("GLM53_MLA_OUT_ROWS4");
    assert(!glm53_mla_out_rows4_env());
    setenv("GLM53_MLA_OUT_ROWS4", "0", 1);
    assert(!glm53_mla_out_rows4_env());
    setenv("GLM53_MLA_OUT_ROWS4", "1", 1);
    assert(glm53_mla_out_rows4_env());
    unsetenv("GLM53_MLA_OUT_ROWS4");
    projection_case(4096, 2048, 64, 4, 0, 0); /* default/off */
    projection_case(4096, 2048, 64, 4, 1, EXPECT_ROWS4);
    projection_case(5, 64, 64, 4, 1, 0);     /* row tail */
    projection_case(8, 64, 32, 4, 1, 0);     /* unsupported MLA group */
    projection_case(8, 64, 64, 0, 1, 0);     /* non-int4 */
    projection_case(8, 70, 64, 4, 1, 0);    /* input/group tail */
    one_token_mla_case(0);
    puts("glm53 MLA output rows4: PASS");
    return 0;
}
#endif
