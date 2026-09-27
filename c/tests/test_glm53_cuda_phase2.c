/* Phase-2 production FFN tests with a fake asynchronous CUDA backend. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#define GLM53_CUDA_TEST_HOOK
#include <assert.h>
#include "../glm53_cuda.h"
static int cpu_miss_during_issue;
static void glm53_cuda_test_cpu_miss_pending(const G53Cuda *g) {
    if (g->group_pending[0] || g->group_pending[1]) cpu_miss_during_issue++;
    else assert(g->failed); /* first issue can fail before any device launches */
}
#include "../glm53.c"
#define G53_ORACLE_NO_MAIN
#include "test_glm53_cuda_multidev_oracle.c"

typedef struct {
    GModel m;
    GLayer layer;
    LCache cache;
    Slot slots[ORACLE_EXPERTS];
    float zero[ORACLE_D * ORACLE_I];
    float router[ORACLE_EXPERTS * ORACLE_D];
    float bias[ORACLE_EXPERTS];
} FixtureModel;
static void model_fixture(FixtureModel *f, const char *order) {
    memset(f, 0, sizeof(*f));
    f->m.c.hidden = ORACLE_D;
    f->m.c.moe_inter = f->m.c.dense_inter = ORACLE_I;
    f->m.c.n_layers = 1;
    f->m.c.n_experts = ORACLE_EXPERTS;
    f->m.c.topk = 3;
    f->m.c.routed_scale = 1;
    f->m.c.swiglu_limit = 0.5f;
    f->m.layer_end = 1;
    f->m.streaming = 1;
    f->m.ecache = &f->cache;
    f->cache.s = f->slots;
    f->cache.n = f->cache.cap = ORACLE_EXPERTS;
    f->layer.rg = (Mat){.fmt=0, .f=f->zero, .rows=ORACLE_I, .columns=ORACLE_D};
    f->layer.ru = f->layer.rg;
    f->layer.rd = (Mat){.fmt=0, .f=f->zero, .rows=ORACLE_D, .columns=ORACLE_I};
    f->layer.router = f->router;
    f->bias[0] = 3; f->bias[1] = 2; f->bias[2] = 1;
    f->layer.rbias = f->bias;
    for (int e = 0; e < ORACLE_EXPERTS; e++) {
        f->slots[e].eid = e;
        for (int p = 0; p < 6; p++) f->slots[e].piece[p] = opieces[e][p];
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPU", "9", 1);
    setenv("COLI_GPUS", order, 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    g53_cuda_init(&f->m.cuda, 1, ORACLE_EXPERTS, ORACLE_D, ORACLE_I, 1);
    f->m.cuda.budget = 2 * f->m.cuda.expert_bytes;
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&f->m.cuda, 0, e, 2);
        g53_cuda_promote(&f->m.cuda, 0, e, opieces[e]);
        assert(f->m.cuda.experts[e].owner == e);
    }
}
static void model_fixture_close(FixtureModel *f) {
    g53_cuda_close(&f->m.cuda);
    assert(live == 0 && live_device[0] == 0 && live_device[1] == 0);
    if (f->m.ehit) { free(f->m.ehit[0]); free(f->m.ehit); }
}
static void case_group(const char *order, int first_complete,
                       int fail_issue, int fail_take, int nonfinite) {
    FixtureModel f;
    model_fixture(&f, order);
    float x[ORACLE_D], serial[ORACLE_D] = {0}, got[ORACLE_D] = {0};
    for (int d = 0; d < ORACLE_D; d++) x[d] = ((d * 7) % 19 - 9) * 0.09f;
    /* Force only this reference call down the untouched phase-1 serial path.
     * The tensors retain their physical owners (0 and 1). */
    f.m.cuda.ndev = 1;
    ffn_layer(&f.m, &f.layer, 0, x, 1, serial);
    f.m.cuda.ndev = 2;
    int issue_before = group_issue_calls, take_before = group_take_calls;
    fake_completion_first = first_complete;
    fake_completion_count = 0;
    cpu_miss_during_issue = 0;
    fail_group_issue_device = fail_issue;
    fail_group_take_device = fail_take;
    nonfinite_group_device = nonfinite;
    if (nonfinite >= 0) setenv("GLM53_CUDA_DIAG_GROUP", "1", 1);
    ffn_layer(&f.m, &f.layer, 0, x, 1, got);
    if (nonfinite >= 0) unsetenv("GLM53_CUDA_DIAG_GROUP");
    compare_projection("phase-2 vs serial phase-1 FFN", got, serial, ORACLE_D);
    assert(group_issue_calls - issue_before == (fail_issue == 0 ? 1 : 2));
    assert(group_take_calls - take_before == (fail_issue == 0 ? 0 : fail_issue < 0 ? 2 : 1));
    assert(cpu_miss_during_issue == (fail_issue == 0 ? 0 : 1));
    assert(!f.m.cuda.group_pending[0] && !f.m.cuda.group_pending[1]);
    if (fail_issue < 0) {
        assert(fake_completion_count == 2);
        assert(fake_completion_order[0] == first_complete);
        assert(fake_completion_order[1] == 1 - first_complete);
    }
    if (fail_issue >= 0 || fail_take >= 0 || nonfinite >= 0) {
        assert(f.m.cuda.failed && f.m.cuda.errors == 1);
        assert(f.m.cuda.fallback >= 3); /* all contributions recomputed on CPU */
        if (nonfinite >= 0) assert(f.m.cuda.executed == 3); /* replay is diagnostic */
    } else {
        assert(!f.m.cuda.failed && f.m.cuda.errors == 0);
        assert(f.m.cuda.device_executed[0] >= 1 && f.m.cuda.device_executed[1] >= 1);
        /* Repeated whole-expert promotion and eviction after both takes. */
        g53_cuda_heat(&f.m.cuda, 0, 2, 10);
        g53_cuda_promote(&f.m.cuda, 0, 2, opieces[2]);
        assert(f.m.cuda.experts[2].w[0] && f.m.cuda.resident == 2);
        g53_cuda_heat(&f.m.cuda, 0, 0, 20);
        g53_cuda_promote(&f.m.cuda, 0, 0, opieces[0]);
        assert(f.m.cuda.experts[0].w[0] && f.m.cuda.resident == 2);
        float again[ORACLE_D] = {0};
        fake_completion_count = 0;
        ffn_layer(&f.m, &f.layer, 0, x, 1, again);
        compare_projection("phase-2 after eviction", again, serial, ORACLE_D);
    }
    fail_group_issue_device = fail_group_take_device = nonfinite_group_device = -1;
    fake_completion_first = -1;
    model_fixture_close(&f);
}
static void case_shutdown_pending(void) {
    FixtureModel f;
    model_fixture(&f, "0,1");
    float x[ORACLE_D] = {0};
    int rows[1] = {1}, before = group_take_calls;
    for (int di = 0; di < 2; di++) {
        G53CudaExpert *e = &f.m.cuda.experts[di];
        ColiCudaTensor *gate[1] = {e->w[0]}, *up[1] = {e->w[1]}, *down[1] = {e->w[2]};
        assert(g53_cuda_group_issue(&f.m.cuda, di, gate, up, down, rows, 1, x, 0.5f));
    }
    assert(f.m.cuda.group_pending[0] && f.m.cuda.group_pending[1]);
    model_fixture_close(&f);
    assert(group_take_calls - before == 2);
}
static void case_diagnostic_value_types(void) {
    const float values[] = {1.5f, NAN, INFINITY, -INFINITY, -2.0f};
    G53DiagValues s = g53_diag_values(values, sizeof(values) / sizeof(values[0]));
    assert(s.bad == 3 && s.nan_count == 1 && s.pos_inf == 1 && s.neg_inf == 1);
    assert(s.first_bad == 1 && isnan(s.first_value));
    assert(s.has_finite && s.min == -2.0f && s.max == 1.5f && s.max_abs == 2.0f);
}
static void case_scale_pread_and_scan(void) {
    FILE *file = tmpfile();
    assert(file);
    GModel *m = calloc(1, sizeof(*m));
    assert(m);
    m->c.n_experts = 288;
    m->eref = calloc((size_t)18 * m->c.n_experts, sizeof(*m->eref));
    assert(m->eref);
    m->e_len[1] = m->e_len[3] = m->e_len[5] = 4 * sizeof(float);
    unsigned char pad[16] = {0};
    assert(fwrite(pad, 1, sizeof(pad), file) == sizeof(pad));
    for (int eid = 0; eid < m->c.n_experts; eid++) {
        ERef *ref = &m->eref[17 * m->c.n_experts + eid];
        for (int k = 0; k < 3; k++) {
            float scales[4] = {0.01f, 0.02f, 0.03f, 0.04f};
            if (eid == 166 && k == 0) scales[1] = INFINITY;
            if (eid == 167 && k == 1) scales[2] = 1e7f;
            int p = 2*k + 1;
            ref->fd[p] = fileno(file);
            ref->off[p] = (int64_t)sizeof(pad) +
                          ((int64_t)eid * 3 + k) * (int64_t)sizeof(scales);
            assert(fwrite(scales, 1, sizeof(scales), file) == sizeof(scales));
        }
    }
    assert(fflush(file) == 0);
    uint8_t raw[4 * sizeof(float)];
    G53DiagValues s = g53_scale_pread(m, 17, 166, 0, fileno(file), raw);
    assert(s.bad == 1 && s.pos_inf == 1 && s.first_bad == 1);
    Slot slot = {.eid = 166};
    Mat mats[3] = {0};
    uint8_t mapped[3][4 * sizeof(float)];
    for (int k = 0; k < 3; k++) {
        g53_scale_pread(m, 17, 166, k, fileno(file), mapped[k]);
        slot.piece[2*k + 1] = mapped[k];
        mats[k].s = (float *)mapped[k];
    }
    assert(g53_scale_file_audit(m, 17, 166, &slot, mats));
    mapped[1][0] ^= 1;
    assert(!g53_scale_file_audit(m, 17, 166, &slot, mats));
    assert(g53_scale_layer_scan(m, 17) == 2);
    free(m->eref); free(m);
    fclose(file);
}
int main(void) {
    fixture();
    case_diagnostic_value_types();
    case_scale_pread_and_scan();
    case_group("0,1", 1, -1, -1, -1); /* device 1 completes first */
    case_group("1,0", 0, -1, -1, -1); /* physical device 0 completes first */
    case_group("0,1", 0, -1, 1, -1); /* second take fails after first succeeds */
    case_group("0,1", 1, -1, 0, -1); /* first take fails; second still drains */
    case_group("0,1", 1, 1, -1, -1); /* second issue fails; first still drains */
    case_group("0,1", 0, 0, -1, -1); /* first issue fails before dispatch */
    case_group("0,1", 0, -1, -1, 1); /* nonfinite result never publishes */
    case_shutdown_pending();
    puts("PASS GLM53 phase-2 groups: completion order, misses, failure, eviction, teardown");
    return 0;
}
