/* Stage ownership and one persistent residual-stream row. No execution hooks. */
#ifndef COLIBRI_GLM53_CUDA_H
#define COLIBRI_GLM53_CUDA_H
#include "segment_adapters.h"
#include "cuda_device_config.h"
#include <string.h>

typedef struct {
    int cuda_device_ordinal; /* meaningful only while lease_live is true */
    int lease_live;         /* engine ownership flag, not a lifetime counter */
    void *wire;            /* one decode row, allocated on the logical owner */
    size_t wire_bytes;
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
    stage->wire = NULL;
    stage->wire_bytes = 0;
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
    if (stage->wire) coli_cuda_pipe_free(stage->cuda_device_ordinal, stage->wire);
#endif
    stage->wire = NULL;
    stage->wire_bytes = 0;
#ifdef COLI_CUDA
    if (stage->lease_live) coli_cuda_release();
#endif
    stage->lease_live = 0;
    stage->cuda_device_ordinal = -1;
}

/* Config-authoritative wire geometry; no weight/tensor-header geometry.
 * Check both size_t byte arithmetic and Segment's uint32_t state_width.
 * Failure retains the lease so the caller can destroy model resources first.
 * The caller must close the stage after failure. No plan means no resource. */
static inline int coli_glm53_cuda_stage_wire_create(ColiGlm53CudaStage *stage,
                                                   size_t hc_mult, size_t hidden) {
    if (!stage->lease_live) return 0;
    if (stage->wire) return -1; /* create once; never replace a live resource */
    size_t floats;
    if (!hc_mult || !hidden || hc_mult > SIZE_MAX / hidden) goto fail;
    floats = hc_mult * hidden;
    if (floats > SIZE_MAX / sizeof(float) || floats > UINT32_MAX) goto fail;
#ifdef COLI_CUDA
    stage->wire = coli_cuda_pipe_alloc(stage->cuda_device_ordinal,
                                       floats * sizeof(float));
    if (!stage->wire) goto fail;
    stage->wire_bytes = floats * sizeof(float);
    return 1;
#endif
fail:
    return -1;
}

/* Synchronous copies using the existing pipeline API. Subtraction avoids
 * bytes+offset overflow. The stage owner and capacity stay authoritative. */
static inline int coli_glm53_cuda_stage_upload(const ColiGlm53CudaStage *stage,
                            const void *src, size_t bytes, size_t offset) {
    if (!stage->lease_live || !stage->wire || offset > stage->wire_bytes ||
        bytes > stage->wire_bytes - offset || (bytes && !src)) return 0;
    if (!bytes) return 1;
#ifdef COLI_CUDA
    return coli_cuda_pipe_upload(stage->cuda_device_ordinal,
        (unsigned char *)stage->wire + offset, src, bytes);
#else
    return 0;
#endif
}
static inline int coli_glm53_cuda_stage_download(const ColiGlm53CudaStage *stage,
                            void *dst, size_t bytes, size_t offset) {
    if (!stage->lease_live || !stage->wire || offset > stage->wire_bytes ||
        bytes > stage->wire_bytes - offset || (bytes && !dst)) return 0;
    if (!bytes) return 1;
#ifdef COLI_CUDA
    return coli_cuda_pipe_download(stage->cuda_device_ordinal,
        (const unsigned char *)stage->wire + offset, dst, bytes);
#else
    return 0;
#endif
}

/* Caller-owned transport context, serialized by its caller. One bounce can be
 * reused across stage pairs; no stage references or process-global resources.
 * Its own full-process lease keeps pinned free safe even if stages close first. */
typedef struct {
    void *bounce;
    size_t bytes;
    int lease_live;
} ColiGlm53CudaHandoff;

static inline int coli_glm53_cuda_stage_live(const ColiGlm53CudaStage *stage) {
    if (!stage || !stage->lease_live || !stage->wire || !stage->wire_bytes ||
        stage->cuda_device_ordinal < 0) return 0;
#ifdef COLI_CUDA
    int count = coli_cuda_device_count();
    for (int i = 0; i < count; i++)
        if (coli_cuda_device_at(i) == stage->cuda_device_ordinal) return 1;
#endif
    return 0;
}
static inline void coli_glm53_cuda_handoff_close(ColiGlm53CudaHandoff *handoff) {
    if (!handoff) return;
#ifdef COLI_CUDA
    if (handoff->bounce) coli_cuda_host_free(handoff->bounce);
#endif
    handoff->bounce = NULL;
    handoff->bytes = 0;
#ifdef COLI_CUDA
    if (handoff->lease_live) coli_cuda_release();
#endif
    handoff->lease_live = 0;
}
/* Open a fresh/closed context. Capacity is exactly the source's existing wire,
 * without new geometry or multiplication. All live leases necessarily share
 * the backend's single ordered process set; enumerate it, never reselect it. */
static inline int coli_glm53_cuda_handoff_open(ColiGlm53CudaHandoff *handoff,
                  const ColiGlm53CudaStage *src, const ColiGlm53CudaStage *dst) {
    if (!handoff) return 0;
    memset(handoff, 0, sizeof(*handoff));
    if (!coli_glm53_cuda_stage_live(src) || !coli_glm53_cuda_stage_live(dst)) return 0;
    if (src->wire_bytes % sizeof(float) || src->wire_bytes / sizeof(float) > UINT32_MAX) return 0;
#ifdef COLI_CUDA
    int devices[COLI_CUDA_MAX_DEVICES], count = coli_cuda_device_count();
    if (count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    for (int i = 0; i < count; i++) devices[i] = coli_cuda_device_at(i);
    if (!coli_cuda_acquire(devices, count)) return 0;
    handoff->lease_live = 1;
    handoff->bounce = coli_cuda_host_alloc(src->wire_bytes);
    if (!handoff->bounce) { coli_glm53_cuda_handoff_close(handoff); return 0; }
    handoff->bytes = src->wire_bytes;
    return 1;
#else
    return 0;
#endif
}
/* Synchronous D2H then H2D. Zero bytes is a validated no-op. Failure leaves
 * all leases/resources intact; H2D failure may leave destination bytes changed.
 * There is no retry, peer copy, stream, or per-transfer allocation. */
static inline int coli_glm53_cuda_handoff(ColiGlm53CudaHandoff *handoff,
            const ColiGlm53CudaStage *src, const ColiGlm53CudaStage *dst, size_t bytes) {
    if (!handoff || !handoff->lease_live || !handoff->bounce || !handoff->bytes ||
        !coli_glm53_cuda_stage_live(src) || !coli_glm53_cuda_stage_live(dst) ||
        bytes > src->wire_bytes || bytes > dst->wire_bytes || bytes > handoff->bytes) return 0;
    if (!bytes) return 1;
    if (!coli_glm53_cuda_stage_download(src, handoff->bounce, bytes, 0)) return 0;
    return coli_glm53_cuda_stage_upload(dst, handoff->bounce, bytes, 0);
}
#endif
