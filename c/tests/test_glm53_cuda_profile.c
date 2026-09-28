#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include "../compat.h"

static int clock_calls;
static double fake_clock(void) { return ++clock_calls; }

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPU", "0", 1);
    setenv("COLI_GPUS", "", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    unsetenv("GLM53_CUDA_PROFILE");
    G53Cuda g = {0};
    g53_cuda_init(&g, 1, 2, 64, 64, 1);
    g53_cuda_profile_enable(&g, fake_clock);
    assert(!g.profile.clock);
    uint8_t weights[2048]; float scales[64], x[64], y[64], reference[64], sg[64], su[64];
    memset(weights, 0x97, sizeof(weights));
    for (int i = 0; i < 64; i++) { scales[i] = 0.125f; x[i] = (float)i / 64; }
    uint8_t *pieces[] = {weights, (uint8_t *)scales, weights, (uint8_t *)scales,
                        weights, (uint8_t *)scales};
    g.budget = g.expert_bytes;
    g53_cuda_heat(&g, 0, 0, 2); g53_cuda_promote(&g, 0, 0, pieces);
    assert(g53_cuda_run(&g, 0, 0, reference, x, sg, su, 0.5f, clamp_ref));
    assert(clock_calls == 0);
    setenv("GLM53_CUDA_PROFILE", "1", 1);
    g53_cuda_profile_enable(&g, fake_clock);
    assert(g53_cuda_run(&g, 0, 0, y, x, sg, su, 0.5f, clamp_ref));
    assert(!memcmp(reference, y, sizeof(y)));
    assert(g.profile.seconds[G53_GATE] == 1 && g.profile.seconds[G53_UP] == 1);
    assert(g.profile.seconds[G53_CLAMP] == 1 && g.profile.seconds[G53_DOWN] == 1);
    g53_cuda_heat(&g, 0, 1, 4); g53_cuda_promote(&g, 0, 1, pieces);
    assert(g.profile.evictions == 1 && g.uploads == 2 && g.resident == 1);
    assert(g.profile.seconds[G53_UPLOAD] == 1 && g.profile.seconds[G53_EVICT] == 1);
    assert(g.profile.seconds[G53_PROMOTION] == 5);
    fail_mat = mat_calls + 2;
    assert(!g53_cuda_run(&g, 0, 1, y, x, sg, su, 0.5f, clamp_ref));
    assert(g.profile.seconds[G53_GATE] == 2 && g.profile.seconds[G53_UP] == 2);
    assert(g.profile.seconds[G53_CLAMP] == 1 && g.profile.seconds[G53_DOWN] == 1);
    assert(g.executed == 2 && g.errors == 1);
    g53_cuda_close(&g); /* final snapshot precedes freeing, no eviction increments */
    assert(live == 0);
    G53Cuda cache = {0};
    g53_cuda_init(&cache, 1, 3, 64, 64, 1);
    g53_cuda_profile_enable(&cache, fake_clock);
    cache.profile.decode = 1;
    cache.budget = cache.expert_bytes;
    g53_cuda_heat(&cache, 0, 0, 2); g53_cuda_promote(&cache, 0, 0, pieces);
    g53_cuda_heat(&cache, 0, 0, 1); /* resident selection after upload */
    cache.profile.decode_tokens++;
    g53_cuda_heat(&cache, 0, 1, 5); g53_cuda_promote(&cache, 0, 1, pieces);
    cache.profile.decode_tokens++;
    g53_cuda_heat(&cache, 0, 0, 10); g53_cuda_promote(&cache, 0, 0, pieces);
    assert(cache.profile.cache_layer[0].hit == 1);
    assert(cache.profile.cache_layer[0].miss == 3);
    assert(cache.profile.cache_layer[0].hit_rows == 1);
    assert(cache.profile.cache_layer[0].miss_rows == 17);
    assert(cache.profile.cache_layer[0].promotions == 3);
    assert(cache.profile.cache_layer[0].evictions == 2);
    assert(cache.profile.cache_layer[0].repromotions == 1);
    assert(cache.profile.cache_layer[0].dead_on_arrival == 1);
    assert(cache.profile.cache_expert[0].promotions == 2);
    assert(cache.profile.repromotions == 1 && cache.profile.dead_on_arrival == 1);
    assert(cache.profile.eviction_record_count == 2);
    g53_cuda_close(&cache);
    assert(live == 0);
    puts("PASS GLM53 profiler: disabled clock, unchanged output, intervals, eviction and failure accounting");
    return 0;
}
