#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include <time.h>

static double warm_real_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

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
    unsetenv("GLM53_CUDA_WARM_PARALLEL");
    unsetenv("GLM53_CUDA_WARM_FRACTION");
    g53_cuda_warm_config(&config);
    assert(config.warm_requested == 1695);
    assert(!config.warm_parallel);
    setenv("GLM53_CUDA_WARM_PARALLEL", "0", 1);
    g53_cuda_warm_config(&config);
    assert(!config.warm_parallel);
    setenv("GLM53_CUDA_WARM_PARALLEL", "1", 1);
    g53_cuda_warm_config(&config);
    assert(config.warm_parallel && config.warm_workers == 8);
    unsetenv("GLM53_CUDA_WARM_PARALLEL");
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

    setenv("GLM53_CUDA_WARM_PARALLEL", "1", 1);
    warm_devices(&g);
    g.capacity[0] = 2 * g.expert_bytes;
    g.capacity[1] = 2 * g.expert_bytes;
    g.budget = 4 * g.expert_bytes;
    G53CudaWarmCandidate candidates[4] = {{0, 0, 10}, {0, 1, 9}, {0, 2, 8}, {0, 3, 7}};
    G53CudaWarmPlan plan[4];
    assert(g53_cuda_warm_plan(&g, candidates, 4, plan) == 4);
    assert(g53_cuda_warm_batch_size(plan, 0, 4, g.ndev) == 2);
    assert(g53_cuda_warm_batch_size(plan, 2, 4, g.ndev) == 2);
    G53CudaWarmPlan same_owner[3] = {{0, 0, 0}, {0, 1, 0}, {0, 2, 1}};
    assert(g53_cuda_warm_batch_size(same_owner, 0, 3, g.ndev) == 1);
    assert(g53_cuda_warm_batch_size(same_owner, 1, 3, g.ndev) == 2);
    for (int i = 0; i < 4; i++) {
        assert(plan[i].eid == i && plan[i].owner == i % 2);
    }
    g.profile.selection_trace = tmpfile();
    assert(g.profile.selection_trace);
    fake_upload_delay = 1; fake_upload_slow_device = 0;
    fake_record_upload_completion = 1; fake_upload_complete_count = 0;
    fake_upload_peak = 0;
    for (int batch = 0; batch < 2; batch++) {
        G53CudaPromotionTask tasks[2] = {{0}};
        for (int i = 0; i < 2; i++) {
            G53CudaWarmPlan *p = &plan[2 * batch + i];
            tasks[i].layer = p->layer; tasks[i].eid = p->eid;
            tasks[i].owner = p->owner; tasks[i].device = g.devices[p->owner];
            tasks[i].D = g.D; tasks[i].I = g.I;
            for (int k = 0; k < 6; k++) tasks[i].piece[k] = pieces[k];
        }
        assert(g53_cuda_warm_batch(&g, tasks, 2, warm_real_clock));
    }
    assert(fake_upload_peak == 2 && g.warm_peak_inflight == 2);
    assert(fake_upload_complete_count == 4 && fake_upload_complete_order[0] == 1);
    assert(g.warm_parallel_uploads == 4 && g.warm_batches == 2);
    assert(g.warm_loaded == 4 && g.warm_bytes == 4 * g.expert_bytes);
    assert(g.resident == 4 && g.uploads == 0);
    assert(g.used[0] == 2 * g.expert_bytes && g.used[1] == 2 * g.expert_bytes);
    for (int i = 0; i < 4; i++)
        assert(g.experts[i].owner == i % 2 && g.experts[i].heat == 0);
    g53_cuda_warm_boundary(&g);
    rewind(g.profile.selection_trace);
    for (int i = 0; i < 4; i++) {
        assert(fgets(line, sizeof(line), g.profile.selection_trace));
        int layer, eid, owner, device;
        assert(sscanf(line, "W,%d,%d,%d,%d", &layer, &eid, &owner, &device) == 4);
        assert(layer == 0 && eid == i && owner == i % 2 && device == i % 2);
    }
    assert(fgets(line, sizeof(line), g.profile.selection_trace) && line[0] == 'B');
    fake_upload_delay = 0; fake_upload_slow_device = -1;
    fake_record_upload_completion = 0;
    g53_cuda_close(&g);
    assert(live == 0);

    warm_devices(&g);
    g.budget = 2 * g.expert_bytes;
    G53CudaPromotionTask tasks[2] = {{0}};
    for (int i = 0; i < 2; i++) {
        tasks[i].layer = 0; tasks[i].eid = i;
        tasks[i].owner = i; tasks[i].device = g.devices[i];
        tasks[i].D = g.D; tasks[i].I = g.I;
        for (int k = 0; k < 6; k++) tasks[i].piece[k] = pieces[k];
    }
    fake_fail_upload_device = 1;
    fake_fail_upload_n = fake_device_upload_calls[1] + 2;
    assert(!g53_cuda_warm_batch(&g, tasks, 2, warm_real_clock));
    assert(g.failed && g.errors == 1 && g.warm_failed == 1);
    assert(g.resident == 1 && g.warm_loaded == 1 && g.uploads == 0);
    assert(g.used[0] == g.expert_bytes && g.used[1] == 0 && live == 3);
    fake_fail_upload_device = -1;
    g53_cuda_close(&g);
    assert(live == 0);

    warm_devices(&g);
    g.budget = 2 * g.expert_bytes;
    memset(tasks, 0, sizeof(tasks));
    for (int i = 0; i < 2; i++) {
        tasks[i].layer = 0; tasks[i].eid = i;
        tasks[i].owner = i; tasks[i].device = g.devices[i];
        tasks[i].D = g.D; tasks[i].I = g.I;
        for (int k = 0; k < 6; k++) tasks[i].piece[k] = pieces[k];
    }
    fake_fail_upload_device = 0;
    fake_fail_upload_n = fake_device_upload_calls[0] + 1;
    assert(!g53_cuda_warm_batch(&g, tasks, 2, warm_real_clock));
    assert(g.failed && g.warm_failed == 1 && g.warm_loaded == 0);
    assert(g.resident == 0 && g.bytes == 0 && g.uploads == 0 && live == 0);
    assert(g.used[0] == 0 && g.used[1] == 0);
    fake_fail_upload_device = -1;
    g53_cuda_close(&g);

    unsetenv("GLM53_CUDA_WARM_PARALLEL");
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
    unsetenv("GLM53_CUDA_WARM_PARALLEL");
    unsetenv("GLM53_CUDA_PROFILE");
    puts("glm53 cuda warm preload: ok");
    return 0;
}
