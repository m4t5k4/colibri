#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"

static double warm_clock(void) {
    static double tick;
    return tick += 0.001;
}

static void warm_devices(G53Cuda *g) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0,1", 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    g53_cuda_init(g, 1, 4, 64, 64, 1);
    g->budget = 2 * g->expert_bytes;
}

int main(void) {
    G53Cuda config = {0};
    config.ndev = 8;
    config.expert_bytes = 14155776;
    config.budget = 48000000000ULL;
    for (int i = 0; i < 8; i++) config.capacity[i] = (size_t)(i == 3 ? 424 : 427) * config.expert_bytes;
    assert(g53_cuda_warm_capacity(&config) == 3390);
    unsetenv("GLM53_CUDA_WARM_RESIDENCY");
    g53_cuda_warm_config(&config);
    assert(!config.warm_enabled && config.warm_requested == 0);
    setenv("GLM53_CUDA_WARM_RESIDENCY", "0", 1);
    g53_cuda_warm_config(&config);
    assert(!config.warm_enabled);
    setenv("GLM53_CUDA_WARM_RESIDENCY", "1", 1);
    unsetenv("GLM53_CUDA_WARM_FRACTION");
    g53_cuda_warm_config(&config);
    assert(config.warm_requested == 1695);
    setenv("GLM53_CUDA_WARM_FRACTION", "0.25", 1);
    g53_cuda_warm_config(&config);
    assert(config.warm_requested == 848);
    setenv("GLM53_CUDA_WARM_FRACTION", "0.75", 1);
    g53_cuda_warm_config(&config);
    assert(config.warm_requested == 2543);
    setenv("GLM53_CUDA_WARM_FRACTION", "1", 1);
    g53_cuda_warm_config(&config);
    assert(config.warm_requested == 3390);
    config.capacity[0] = 0;
    config.budget = 100 * config.expert_bytes;
    g53_cuda_warm_config(&config);
    assert(g53_cuda_warm_capacity(&config) == 100 && config.warm_requested == 100);
    config.budget = 48000000000ULL;
    for (int i = 1; i < 8; i++) config.capacity[i] = config.expert_bytes;
    g53_cuda_warm_config(&config);
    assert(g53_cuda_warm_capacity(&config) == 7 && config.warm_requested == 7);
    setenv("GLM53_CUDA_WARM_FRACTION", "0.50", 1);

    uint32_t row0[4] = {7, 9, 9, 0};
    uint32_t row1[4] = {9, 0, 12, 0};
    uint32_t *counts[2] = {row0, row1};
    G53Cuda rank_model = {.count = 8, .ne = 4};
    int n = -1;
    G53CudaWarmCandidate *ranked = g53_cuda_warm_rank(&rank_model, counts, 0, &n);
    assert(ranked && n == 5);
    assert(ranked[0].layer == 1 && ranked[0].eid == 2);
    assert(ranked[1].layer == 0 && ranked[1].eid == 1);
    assert(ranked[2].layer == 0 && ranked[2].eid == 2);
    assert(ranked[3].layer == 1 && ranked[3].eid == 0);
    assert(ranked[4].layer == 0 && ranked[4].eid == 0);
    free(ranked);

    unsigned char weights[3][2048] = {{0}};
    float scales[3][64] = {{0}};
    uint8_t *pieces[6];
    for (int k = 0; k < 3; k++) {
        pieces[k * 2] = weights[k];
        pieces[k * 2 + 1] = (uint8_t *)scales[k];
    }
    setenv("GLM53_CUDA_PROFILE", "1", 1);
    G53Cuda g = {0};
    warm_devices(&g);
    g53_cuda_profile_enable(&g, warm_clock);
    g.profile.selection_trace = tmpfile();
    assert(g.profile.selection_trace);
    assert(g53_cuda_warm_upload(&g, 0, 0, pieces, warm_clock));
    assert(g53_cuda_warm_upload(&g, 0, 1, pieces, warm_clock));
    g53_cuda_warm_boundary(&g);
    assert(g.experts[0].owner == 0 && g.experts[1].owner == 1);
    assert(g.experts[0].heat == 0 && g.experts[1].heat == 0);
    assert(g.resident == 2 && g.warm_loaded == 2 && g.uploads == 0);
    assert(g.profile.cache_layer[0].promotions == 0 && g.profile.seconds[G53_PROMOTION] == 0);
    assert(g.profile.seconds[G53_UPLOAD] == 0 && g.warm_upload_s > 0);
    g53_cuda_heat(&g, 0, 0, 1);
    assert(g.experts[0].heat == 1 && g.warm_hits == 1);
    g53_cuda_heat(&g, 0, 2, 3);
    g53_cuda_promote(&g, 0, 2, pieces);
    assert(g.resident == 2 && g.uploads == 1 && g.warm_evicted == 1);
    assert(g.warm_evicted_before_first_hit == 1 && !g.experts[1].w[0]);
    assert(g.profile.cache_layer[0].promotions == 1);
    rewind(g.profile.selection_trace);
    int warm = 0, boundary = 0, runtime = 0;
    char line[256];
    while (fgets(line, sizeof(line), g.profile.selection_trace)) {
        warm += line[0] == 'W';
        boundary += line[0] == 'B';
        runtime += line[0] == 'P';
    }
    assert(warm == 2 && boundary == 1 && runtime == 1);
    g53_cuda_close(&g);
    assert(live == 0);

    warm_devices(&g);
    g.capacity[0] = g.expert_bytes;
    g.capacity[1] = 2 * g.expert_bytes;
    g.budget = 3 * g.expert_bytes;
    assert(g53_cuda_warm_capacity(&g) == 3);
    assert(g53_cuda_warm_upload(&g, 0, 0, pieces, warm_clock));
    assert(g53_cuda_warm_upload(&g, 0, 1, pieces, warm_clock));
    assert(g53_cuda_warm_upload(&g, 0, 2, pieces, warm_clock));
    assert(g.experts[0].owner == 0 && g.experts[1].owner == 1 && g.experts[2].owner == 1);
    assert(g.used[0] == g.expert_bytes && g.used[1] == 2 * g.expert_bytes);
    assert(g.warm_loaded == 3 && g.uploads == 0);
    g53_cuda_close(&g);
    assert(live == 0);

    warm_devices(&g);
    assert(g53_cuda_warm_upload(&g, 0, 0, pieces, warm_clock));
    size_t used_before = g.used[0];
    fail_upload = upload_calls + 2;
    assert(!g53_cuda_warm_upload(&g, 0, 1, pieces, warm_clock));
    assert(g.failed && g.errors == 1 && g.warm_failed == 1);
    assert(g.warm_loaded == 1 && g.resident == 1 && g.bytes == used_before);
    assert(g.uploads == 0 && live == 3);
    fail_upload = 0;
    g53_cuda_close(&g);
    assert(live == 0);
    unsetenv("GLM53_CUDA_WARM_RESIDENCY");
    unsetenv("GLM53_CUDA_WARM_FRACTION");
    unsetenv("GLM53_CUDA_PROFILE");
    puts("glm53 cuda warm preload: ok");
    return 0;
}
