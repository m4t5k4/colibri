/* Production ffn_layer scatter/fallback with two fake CUDA devices.
 * The second expert fails on its second row, after its first row succeeded. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#define G53_ORACLE_NO_MAIN
#include "test_glm53_cuda_multidev_oracle.c"

static void release_hits(GModel *m) {
    if (m->ehit) {
        for (int i = 0; i < m->c.n_layers; i++) free(m->ehit[i]);
        free(m->ehit);
    }
}
int main(void) {
    fixture();
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPU", "9", 1);
    setenv("COLI_GPUS", "0,1", 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    float zero_g[ORACLE_D * ORACLE_I] = {0};
    float zero_d[ORACLE_D * ORACLE_I] = {0};
    float router[ORACLE_EXPERTS * ORACLE_D] = {0};
    float bias[ORACLE_EXPERTS] = {3, 2, 1, 0};
    GLayer layer = {0};
    layer.rg = (Mat){.fmt=0, .f=zero_g, .rows=ORACLE_I, .columns=ORACLE_D};
    layer.ru = layer.rg;
    layer.rd = (Mat){.fmt=0, .f=zero_d, .rows=ORACLE_D, .columns=ORACLE_I};
    layer.router = router;
    layer.rbias = bias;
    Slot slots[ORACLE_EXPERTS] = {0};
    for (int e = 0; e < ORACLE_EXPERTS; e++) {
        slots[e].eid = e;
        for (int p = 0; p < 6; p++) slots[e].piece[p] = opieces[e][p];
    }
    LCache cache = {.s=slots, .n=ORACLE_EXPERTS, .cap=ORACLE_EXPERTS};
    GModel cpu = {0}, gpu = {0};
    cpu.c.hidden = ORACLE_D;
    cpu.c.moe_inter = ORACLE_I;
    cpu.c.dense_inter = ORACLE_I;
    cpu.c.n_layers = 1;
    cpu.c.n_experts = ORACLE_EXPERTS;
    cpu.c.topk = 3;
    cpu.c.first_dense = 0;
    cpu.c.routed_scale = 1;
    cpu.c.swiglu_limit = 0.5f;
    cpu.layer_end = 1;
    cpu.streaming = 1;
    cpu.ecache = &cache;
    gpu = cpu;
    g53_cuda_init(&gpu.cuda, 1, ORACLE_EXPERTS, ORACLE_D, ORACLE_I, 1);
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&gpu.cuda, 0, e, 2);
        g53_cuda_promote(&gpu.cuda, 0, e, opieces[e]);
        assert(gpu.cuda.experts[e].owner == e);
    }
    float x[2 * ORACLE_D], expected[2 * ORACLE_D] = {0}, got[2 * ORACLE_D] = {0};
    for (int j = 0; j < 2 * ORACLE_D; j++) x[j] = ((j * 7) % 19 - 9) * 0.09f;
    ffn_layer(&cpu, &layer, 0, x, 2, expected);
    /* First expert: two complete rows (6 calls). Second expert: first row
     * completes, second row fails at down (6th call after those 6). */
    fail_mat = mat_calls + 12;
    ffn_layer(&gpu, &layer, 0, x, 2, got);
    compare_projection("production ffn_layer scatter after late failure",
                       got, expected, 2 * ORACLE_D);
    assert(gpu.cuda.failed && gpu.cuda.errors == 1 && gpu.cuda.failure_stage == 2);
    assert(gpu.cuda.device_executed[0] == 2 && gpu.cuda.device_executed[1] == 1);
    assert(gpu.cuda.fallback == 4); /* two rows each for failed + cold expert */
    g53_cuda_close(&gpu.cuda);
    assert(live == 0 && live_device[0] == 0 && live_device[1] == 0);
    release_hits(&cpu);
    release_hits(&gpu);
    puts("PASS production ffn_layer: late second-device fault, CPU fallback, no missing/double scatter");
    return 0;
}
