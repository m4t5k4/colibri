/* Fake only the device callbacks; use production stage and lifetime logic. */
#define _POSIX_C_SOURCE 200809L
#define COLI_CUDA
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#include "../glm53.c"
#include "../backend_cuda_lifetime.h"
#include <assert.h>
#include <stdio.h>

static ColiCudaLifetime lifetime;
static int inits, shutdowns, fail_init;
static int init(const int *devices, int count) {
    assert(devices && count > 0);
    if (fail_init) return 0;
    inits++; return 1;
}
static void shutdown_backend(void) { shutdowns++; }
int coli_cuda_acquire(const int *devices, int count) {
    return coli_cuda_lifetime_acquire(&lifetime, devices, count, init);
}
void coli_cuda_release(void) {
    coli_cuda_lifetime_release(&lifetime, shutdown_backend);
}

int main(void) {
    _Static_assert(sizeof(ColiGlm53StagePlan) == 12, "stage ABI");
    ColiGlm53StagePlan plan = {12, 1, 0}, parsed;
    Glm53SegmentEngine *ea = calloc(1, sizeof(*ea));
    Glm53SegmentEngine *eb = calloc(1, sizeof(*eb));
    assert(ea && eb);
    pthread_mutex_init(&ea->run_lock, NULL);
    pthread_mutex_init(&eb->run_lock, NULL);
    ColiGlm53CudaStage a, b;
    unsigned char storage[17];
    assert(coli_glm53_stage_plan_parse(&plan, 12, &parsed) == 1);
    plan.version = 2;
    assert(coli_glm53_stage_plan_parse(&plan, 12, &parsed) == -1);
    ColiSegmentEngineOptions options = {.model_dir = "unused-no-checkpoint",
        .resource_plan = &plan, .resource_plan_size = 12};
    ColiSegmentCapabilities capabilities;
    void *impl = NULL;
    char error[256];
    assert(glm53_segment_engine_open(&impl, &capabilities, &options,
                                     error, sizeof(error)) != 0);
    assert(!impl && !lifetime.users);
    plan.version = 1; plan.struct_size = 8;
    assert(coli_glm53_stage_plan_parse(&plan, 12, &parsed) == -1);
    plan.struct_size = 12;
    for (size_t n = 0; n < 12; n++)
        assert(coli_glm53_stage_plan_parse(&plan, n, &parsed) == -1);
    plan.struct_size = 16;
    memcpy(storage + 1, &plan, 12);
    assert(coli_glm53_stage_plan_parse(storage + 1, 12, &parsed) == -1);
    assert(coli_glm53_stage_plan_parse(storage + 1, 16, &parsed) == 1);
    plan.struct_size = UINT32_MAX;
    assert(coli_glm53_stage_plan_parse(&plan, 12, &parsed) == -1);
    plan.struct_size = 12;
    memcpy(storage + 1, &plan, 12);
    assert(coli_glm53_stage_plan_parse(storage + 1, 16, &parsed) == 1);
    assert(coli_glm53_stage_plan_parse(NULL, 1, &parsed) == -1);
    puts("A-E: version/size/truncation/extension/unaligned parsing: OK");

    unsetenv("COLI_GPU"); setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0,1", 1);
    for (int reverse = 0; reverse < 2; reverse++) {
        int before_init = inits, before_shutdown = shutdowns;
        plan.cuda_device_ordinal = 0;
        assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
        plan.cuda_device_ordinal = 1;
        assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == 1);
        assert(a.cuda_device_ordinal == 0 && b.cuda_device_ordinal == 1);
        assert(lifetime.users == 2 && inits == before_init + 1);
        assert(lifetime.count == 2 && lifetime.devices[1] == 1);
        coli_cuda_lifetime_raw_shutdown(&lifetime, shutdown_backend);
        assert(shutdowns == before_shutdown);
        ColiGlm53CudaStage *first = reverse ? &b : &a;
        ColiGlm53CudaStage *survivor = reverse ? &a : &b;
        coli_glm53_cuda_stage_close(first);
        assert(lifetime.users == 1 && survivor->lease_live);
        assert(survivor->cuda_device_ordinal == (reverse ? 0 : 1));
        assert(shutdowns == before_shutdown);
        coli_glm53_cuda_stage_close(survivor);
        assert(!lifetime.users && shutdowns == before_shutdown + 1);
        coli_glm53_cuda_stage_close(survivor);
        assert(shutdowns == before_shutdown + 1);
    }
    puts("F-G, J-M: distinct owners, shared lease, both close orders: OK");
    setenv("COLI_GPUS", "2,4,6", 1);
    plan.cuda_device_ordinal = 4;
    assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
    assert(a.cuda_device_ordinal == 4 && lifetime.count == 3);
    assert(lifetime.devices[0] == 2 && lifetime.devices[2] == 6);
    plan.cuda_device_ordinal = 1;
    assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == -1);
    assert(!b.lease_live && b.cuda_device_ordinal == -1);
    setenv("COLI_GPUS", "0,1", 1); plan.cuda_device_ordinal = 0;
    assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == -1);
    assert(lifetime.users == 1);
    coli_glm53_cuda_stage_close(&a);
    puts("H-I: owner is an ordinal in {2,4,6}; mismatched process list fails: OK");
    fail_init = 1;
    assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == -1);
    assert(!a.lease_live && !lifetime.users);
    fail_init = 0;
    unsetenv("COLI_CUDA"); setenv("COLI_GPUS", "invalid", 1);
    int before = inits;
    assert(coli_glm53_cuda_stage_open(&a, NULL, 0) == 0);
    assert(!a.lease_live && a.cuda_device_ordinal == -1 && inits == before);
    coli_glm53_cuda_stage_close(&a);
    puts("N: no plan preserves unplanned behavior; no owner or backend acquire: OK");
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0,1", 1);
    plan.cuda_device_ordinal = 0;
    assert(coli_glm53_cuda_stage_open(&ea->cuda_stage, &plan, 12) == 1);
    plan.cuda_device_ordinal = 1;
    assert(coli_glm53_cuda_stage_open(&eb->cuda_stage, &plan, 12) == 1);
    glm53_segment_engine_destroy(ea);
    assert(eb->cuda_stage.lease_live && lifetime.users == 1);
    glm53_segment_engine_destroy(eb);
    assert(!lifetime.users);
    puts("Production Segment engine teardown preserves survivor and releases final lease: OK");
    puts("GLM53 CUDA stage tests: PASS");
    return 0;
}
