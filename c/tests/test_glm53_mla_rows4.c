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

static void one_token_mla_case(void) {
    Cfg c = {0};
    c.hidden = 8; c.n_heads = 1; c.q_lora = 64; c.kv_lora = 64;
    c.qk_nope = 64; c.v_head = 64;
    c.index_nh = 1; c.index_hd = 1; c.index_topk = 1; c.index_kpool = 1;
    c.eps = 1e-6f;
    GLayer l = {0};
    l.qa = test_f32(64, 8); l.qb = test_f32(64, 64);
    l.kva = test_f32(64, 8); l.kvb_kt = test_f32(64, 64);
    l.iwq = test_f32(1, 64); l.iwk = test_f32(1, 8);
    l.ikpg = test_f32(1, 8); l.iwp = test_f32(1, 8);
    l.kvb_v = test_f32(64, 64); l.o = test_i4(8, 64, 64);
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
    mla_layer(&baseline, &c, &l, x, 1, generic, &st, 0);
    mla_layer(&experimental, &c, &l, x, 1, fast, &st, 0);
    assert(memcmp(generic, fast, sizeof(generic)) == 0);
    assert(baseline.mla_out_generic_calls == 1);
    assert(experimental.mla_out_rows4_calls == EXPECT_ROWS4);
    assert(experimental.mla_out_generic_calls == !EXPECT_ROWS4);
    mat_release(&l.qa); mat_release(&l.qb); mat_release(&l.kva);
    mat_release(&l.kvb_kt); mat_release(&l.iwq); mat_release(&l.iwk);
    mat_release(&l.ikpg); mat_release(&l.iwp); mat_release(&l.kvb_v);
    mat_release(&l.o);
}

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
    one_token_mla_case();
    puts("glm53 MLA output rows4: PASS");
    return 0;
}
