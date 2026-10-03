/* Stage ownership only. No expert tier, KDA state, or execution hooks. */
#ifndef COLIBRI_GLM53_CUDA_H
#define COLIBRI_GLM53_CUDA_H
#include "segment_adapters.h"
#include "cuda_device_config.h"
#include <string.h>

typedef struct {
    int cuda_device_ordinal; /* meaningful only while lease_live is true */
    int lease_live;         /* engine ownership flag, not a lifetime counter */
} ColiGlm53CudaStage;

/* 0 absent, 1 valid, -1 malformed. Copy only the known prefix: the pointer
 * need not be aligned, and neither external nor declared sizes drive a copy. */
static inline int coli_glm53_stage_plan_parse(const void *data, size_t bytes,
                                            ColiGlm53StagePlan *plan) {
    if (!data) return bytes ? -1 : 0;
    if (bytes < sizeof(*plan)) return -1;
    memcpy(plan, data, sizeof(*plan));
    if (plan->version != COLI_GLM53_STAGE_PLAN_VERSION ||
        plan->struct_size < sizeof(*plan) || plan->struct_size > bytes ||
        plan->cuda_device_ordinal < 0) return -1;
    return 1;
}

/* Uses the existing process list, never a singleton containing the owner.
 * No CUDA calls occur for unplanned engines, including in CPU/Vulkan builds. */
static inline int coli_glm53_cuda_stage_open(ColiGlm53CudaStage *stage,
                                            const void *data, size_t bytes) {
    ColiGlm53StagePlan plan;
    stage->cuda_device_ordinal = -1;
    stage->lease_live = 0;
    int parsed = coli_glm53_stage_plan_parse(data, bytes, &plan);
    if (parsed <= 0) return parsed;
#ifdef COLI_CUDA
    const char *enabled = getenv("COLI_CUDA");
    int devices[COLI_CUDA_MAX_DEVICES];
    if (!enabled || !atoi(enabled)) return -1;
    int count = coli_cuda_configured_devices(devices), found = 0;
    for (int i = 0; i < count; i++)
        if (devices[i] == plan.cuda_device_ordinal) found = 1;
    if (!found || !coli_cuda_acquire(devices, count)) return -1;
    stage->cuda_device_ordinal = plan.cuda_device_ordinal;
    stage->lease_live = 1;
    return 1;
#else
    return -1;
#endif
}
/* Caller frees all stage-owned resources before this final lease release. */
static inline void coli_glm53_cuda_stage_close(ColiGlm53CudaStage *stage) {
#ifdef COLI_CUDA
    if (stage->lease_live) coli_cuda_release();
#endif
    stage->lease_live = 0;
    stage->cuda_device_ordinal = -1;
}
#endif
