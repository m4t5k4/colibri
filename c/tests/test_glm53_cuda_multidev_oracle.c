/* Synthetic gs64 expert oracle, shared by the fake and real CUDA backends.
 * No production checkpoint is needed. All routing/scatter is synchronous. */
#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include "../compat.h"

enum { ORACLE_D = 128, ORACLE_I = 64, ORACLE_N = ORACLE_D * ORACLE_I,
       ORACLE_EXPERTS = 4 };
static unsigned char ow[ORACLE_EXPERTS][3][ORACLE_N / 2];
static float os[ORACLE_EXPERTS][3][ORACLE_N / 64];
static uint8_t *opieces[ORACLE_EXPERTS][6];

static void fixture(void) {
    for (int e = 0; e < ORACLE_EXPERTS; e++)
        for (int k = 0; k < 3; k++) {
            opieces[e][2*k] = ow[e][k];
            opieces[e][2*k+1] = (uint8_t *)os[e][k];
            for (int j = 0; j < ORACLE_N / 2; j++)
                ow[e][k][j] = (unsigned char)(j * 37 + e * 53 + k * 19);
            for (int j = 0; j < ORACLE_N / 64; j++)
                os[e][k][j] = 0.007f * (1 + (j * 3 + e * 5 + k) % 9);
        }
}
static void cpu_expert(int e, float *y, const float *x) {
    float gate[ORACLE_I], up[ORACLE_I];
    project(gate, x, ow[e][0], os[e][0], ORACLE_D, ORACLE_I);
    project(up, x, ow[e][1], os[e][1], ORACLE_D, ORACLE_I);
    clamp_ref(gate, up, ORACLE_I, 0.5f);
    project(y, gate, ow[e][2], os[e][2], ORACLE_I, ORACLE_D);
}
static void add_weighted(float *sum, const float *y, float weight) {
    for (int j = 0; j < ORACLE_D; j++) sum[j] += weight * y[j];
}
static void routed_oracle(G53Cuda *g, const float *x, int inject_failure,
                          const char *label) {
    const int ids[] = {0, 1, 2}; /* Third expert is a deliberate CPU miss. */
    const float routing[] = {0.35f, 0.45f, 0.20f};
    float got[ORACLE_D] = {0}, want[ORACLE_D] = {0};
    float y[ORACLE_D], reference[ORACLE_D], sg[ORACLE_I], su[ORACLE_I];
    uint64_t before[2] = {g->device_executed[0], g->device_executed[1]};
    int cuda_success = 0, cpu_fallback = 0;
    for (int r = 0; r < 3; r++) {
        int e = ids[r];
        cpu_expert(e, reference, x);
        add_weighted(want, reference, routing[r]);
        if (inject_failure && r == 1) {
#ifdef G53_REAL_CUDA
            /* The real backend's hook counts only calls while enabled. Gate
             * and up succeed; the second device's down matmul fails. */
            setenv("COLI_GPU_FAIL_AFTER", "2", 1);
#else
            fail_mat = mat_calls + 3;
#endif
        }
        int cuda_ok = g53_cuda_run(g, 0, e, y, x, sg, su, 0.5f, clamp_ref);
        if (cuda_ok) {
            cuda_success++;
            compare_projection("resident expert", y, reference, ORACLE_D);
        } else {
            cpu_fallback++;
            /* Failed output is private scratch; replace it before scatter. */
            memcpy(y, reference, sizeof(y));
        }
        add_weighted(got, y, routing[r]);
    }
#ifdef G53_REAL_CUDA
    if (inject_failure) unsetenv("COLI_GPU_FAIL_AFTER");
#endif
    compare_projection(label, got, want, ORACLE_D);
    assert(cuda_success == (inject_failure ? 1 : 2));
    assert(cpu_fallback == (inject_failure ? 2 : 1));
    assert(g->device_executed[0] == before[0] + 1);
    assert(g->device_executed[1] == before[1] + (inject_failure ? 0 : 1));
    if (inject_failure) assert(g->failed && g->errors == 1 && g->failure_stage == 2);
}
static void check_order(const char *order, int first, int second, int fail) {
    G53Cuda g = {0};
    float x[ORACLE_D], y[ORACLE_D], reference[ORACLE_D];
    float sg[ORACLE_I], su[ORACLE_I];
    for (int j = 0; j < ORACLE_D; j++) x[j] = (j % 17 - 8) * 0.11f;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPU", "9", 1); /* Nonempty ordered plural form overrides it. */
    setenv("COLI_GPUS", order, 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    g53_cuda_init(&g, 1, ORACLE_EXPERTS, ORACLE_D, ORACLE_I, 1);
    assert(g.active && g.ndev == 2 && g.devices[0] == first && g.devices[1] == second);
    /* Two complete experts fit globally, forcing repeated whole-expert
     * evictions while each physical device retains its own CUDA tensors. */
    g.budget = 2 * g.expert_bytes;
    const int sequence[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1};
    for (int p = 0; p < (int)(sizeof(sequence)/sizeof(sequence[0])); p++) {
        int e = sequence[p];
        g53_cuda_heat(&g, 0, e, 10 + p);
        g53_cuda_promote(&g, 0, e, opieces[e]);
        assert(g.experts[e].w[0] && g.experts[e].owner == p % 2);
        assert(g.devices[g.experts[e].owner] == (p % 2 ? second : first));
        assert(g53_cuda_run(&g, 0, e, y, x, sg, su, 0.5f, clamp_ref));
        cpu_expert(e, reference, x);
        compare_projection("evicted/promoted expert", y, reference, ORACLE_D);
        assert(g.resident <= 2 && g.bytes <= g.budget);
    }
    assert(g.resident == 2 && g.experts[0].owner == 0 && g.experts[1].owner == 1);
    assert(g.used[0] == g.expert_bytes && g.used[1] == g.expert_bytes);
    assert(g.device_executed[0] == 5 && g.device_executed[1] == 5);
    routed_oracle(&g, x, fail, fail ? "failure-after-success weighted oracle"
                                   : "two-device weighted oracle");
    g53_cuda_close(&g);
    g53_cuda_close(&g); /* Adapter/model teardown may both request close. */
#ifndef G53_REAL_CUDA
    assert(live == 0 && live_device[0] == 0 && live_device[1] == 0);
    fail_mat = 0;
#endif
    printf("PASS order=%s owner0=%d owner1=%d failure=%d\n", order, first, second, fail);
}
#ifndef G53_ORACLE_NO_MAIN
int main(void) {
    unsetenv("COLI_GPU_FAIL_AFTER");
#ifdef G53_REAL_CUDA
    if (coli_cuda_available_device_count() < 2) {
        puts("SKIP GLM53 CUDA multi-device oracle: fewer than two visible CUDA devices");
        return 0;
    }
#endif
    fixture();
    check_order("0,1", 0, 1, 1);
    check_order("1,0", 1, 0, 0);
    puts("PASS GLM53 CUDA two-device numerical parity, order, eviction, fallback, cleanup");
    return 0;
}
#endif
