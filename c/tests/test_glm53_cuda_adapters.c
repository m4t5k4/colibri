/* Actual adapter teardown while another model owns the CUDA backend.
 * The source contract test pins init exclusively to the full-model wrapper;
 * no checkpoint or NVIDIA hardware is required for this ownership regression. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"

static void check_teardown(int ndev) {
    init_calls = shutdown_calls = 0;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPU", "0", 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    unsetenv("COLI_GPUS");
    if (ndev == 2) setenv("COLI_GPUS", "0,1", 1);
    GModel full = {0};
    g53_cuda_init(&full.cuda, 1, 2, 64, 64, 1);
    assert(full.cuda.active && init_calls == 1 && shutdown_calls == 0);
    unsigned char weights[64*64/2];
    float scales[64], x[64], y[64], sg[64], su[64];
    memset(weights, 0x99, sizeof(weights));
    for (int i = 0; i < 64; i++) { scales[i] = 0.125f; x[i] = 0.1f; }
    uint8_t *pieces[] = {weights, (uint8_t *)scales, weights, (uint8_t *)scales,
                        weights, (uint8_t *)scales};
    g53_cuda_heat(&full.cuda, 0, 0, 2);
    g53_cuda_promote(&full.cuda, 0, 0, pieces);
    if (ndev == 2) {
        g53_cuda_heat(&full.cuda, 0, 1, 2);
        g53_cuda_promote(&full.cuda, 0, 1, pieces);
    }
    assert(live == 3 * ndev);
    for (int i = 0; i < 3; i++) {
        Glm53SegmentEngine *segment = calloc(1, sizeof(*segment));
        Glm53EdgeEngine *edge = calloc(1, sizeof(*edge));
        assert(segment && edge);
        pthread_mutex_init(&segment->run_lock, NULL);
        /* Same zero-initialized model ownership used by engine_open. */
        assert(!segment->model.cuda.active && !edge->model.cuda.active);
        glm53_segment_engine_destroy(segment);
        glm53_edge_engine_destroy(edge);
        assert(full.cuda.active && init_calls == 1 && shutdown_calls == 0);
        assert(live == 3 * ndev);
        assert(g53_cuda_run(&full.cuda, 0, 0, y, x, sg, su, 0.5f, clamp_ref));
        if (ndev == 2)
            assert(g53_cuda_run(&full.cuda, 0, 1, y, x, sg, su, 0.5f, clamp_ref));
    }
    model_release(&full);
    assert(shutdown_calls == 1 && live == 0);
    model_release(&full); /* repeat release must be inert */
    assert(shutdown_calls == 1);
}
int main(void) {
    check_teardown(1);
    check_teardown(2);
    puts("PASS GLM53 adapter teardown leaves full-model CUDA owner alive");
    return 0;
}
