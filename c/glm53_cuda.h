/* Stage ownership, session KDA coherence and persistent decode staging. */
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

/* Session resources, never part of the stage workspace. The parent stage lease
 * must outlive these buffers. State and window are one coherence unit. */
typedef enum {
    G53_KDA_HOST, G53_KDA_DEVICE, G53_KDA_BOTH, G53_KDA_UNKNOWN
} ColiGlm53KdaAuthority;
typedef struct {
    void *state, *window;
    size_t state_bytes, window_bytes;
    ColiGlm53KdaAuthority authority;
} ColiGlm53CudaKdaLayer;

static inline int coli_glm53_cuda_size_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return 0;
    *out = a * b;
    return 1;
}
static inline int coli_glm53_cuda_kda_geometry(size_t heads, size_t dim,
        size_t proj, size_t kernel, size_t *state_bytes, size_t *window_bytes) {
    size_t p, state, window;
    if (!heads || !dim || !kernel ||
        !coli_glm53_cuda_size_mul(heads, dim, &p) || p != proj ||
        !coli_glm53_cuda_size_mul(p, dim, &state) ||
        !coli_glm53_cuda_size_mul(state, sizeof(float), &state) ||
        !coli_glm53_cuda_size_mul(3, proj, &window) ||
        !coli_glm53_cuda_size_mul(window, kernel, &window) ||
        !coli_glm53_cuda_size_mul(window, sizeof(float), &window)) return 0;
    *state_bytes = state; *window_bytes = window;
    return 1;
}
static inline void coli_glm53_cuda_kda_close(const ColiGlm53CudaStage *stage,
                                           ColiGlm53CudaKdaLayer *layer) {
#ifdef COLI_CUDA
    if (layer->state) coli_cuda_pipe_free(stage->cuda_device_ordinal, layer->state);
    if (layer->window) coli_cuda_pipe_free(stage->cuda_device_ordinal, layer->window);
#else
    (void)stage;
#endif
    memset(layer, 0, sizeof(*layer));
}
/* Fresh record only. No initialization upload: the host starts authoritative. */
static inline int coli_glm53_cuda_kda_open(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaLayer *layer, size_t heads, size_t dim,
        size_t proj, size_t kernel) {
    size_t sb, wb;
    if (!stage || !stage->lease_live || stage->cuda_device_ordinal < 0 ||
        layer->state || layer->window ||
        !coli_glm53_cuda_kda_geometry(heads, dim, proj, kernel, &sb, &wb)) return 0;
#ifdef COLI_CUDA
    layer->state = coli_cuda_pipe_alloc(stage->cuda_device_ordinal, sb);
    if (layer->state) layer->window = coli_cuda_pipe_alloc(stage->cuda_device_ordinal, wb);
    if (!layer->state || !layer->window) {
        coli_glm53_cuda_kda_close(stage, layer); return 0;
    }
    layer->state_bytes = sb; layer->window_bytes = wb;
    layer->authority = G53_KDA_HOST;
    return 1;
#else
    return 0;
#endif
}
/* Only a COMPLETE trusted host state+window overwrite may recover UNKNOWN.
 * CPU recurrence must first ensure_host; partial writes cannot call this. */
static inline void coli_glm53_cuda_kda_host_written(ColiGlm53CudaKdaLayer *layer) {
    layer->authority = G53_KDA_HOST;
}
static inline int coli_glm53_cuda_kda_prepare(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaLayer *layer, const float *state, const float *window) {
    if (!stage || !stage->lease_live || !layer->state || !layer->window ||
        !layer->state_bytes || !layer->window_bytes || !state || !window) return 0;
    if (layer->authority == G53_KDA_BOTH || layer->authority == G53_KDA_DEVICE) return 1;
    if (layer->authority != G53_KDA_HOST) return 0;
#ifdef COLI_CUDA
    if (!coli_cuda_pipe_upload(stage->cuda_device_ordinal, layer->state, state, layer->state_bytes) ||
        !coli_cuda_pipe_upload(stage->cuda_device_ordinal, layer->window, window, layer->window_bytes)) return 0;
    layer->authority = G53_KDA_BOTH;
    return 1;
#else
    return 0;
#endif
}
static inline int coli_glm53_cuda_kda_ensure_host(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaLayer *layer, float *state, float *window) {
    if (layer->authority == G53_KDA_HOST || layer->authority == G53_KDA_BOTH) return 1;
    if (layer->authority != G53_KDA_DEVICE || !stage || !stage->lease_live ||
        !layer->state || !layer->window || !layer->state_bytes ||
        !layer->window_bytes || !state || !window) return 0;
#ifdef COLI_CUDA
    /* Call-local, pageable staging. Never publish half of a recurrent pair. */
    void *s = malloc(layer->state_bytes), *w = malloc(layer->window_bytes);
    int ok = s && w &&
        coli_cuda_pipe_download(stage->cuda_device_ordinal, layer->state, s, layer->state_bytes) &&
        coli_cuda_pipe_download(stage->cuda_device_ordinal, layer->window, w, layer->window_bytes);
    if (ok) {
        memcpy(state, s, layer->state_bytes); memcpy(window, w, layer->window_bytes);
        layer->authority = G53_KDA_BOTH;
    }
    free(s); free(w);
    return ok;
#else
    return 0;
#endif
}

/* One session workspace, reused under its engine run_lock. It borrows the
 * stage lease; neither the stage wire nor another session owns these bytes. */
typedef struct {
    float *device, *q, *k, *v, *decay, *beta, *raw_out;
    float *next_window;
    size_t proj_bytes, window_bytes, device_bytes;
} ColiGlm53CudaKdaStaging;
static inline int coli_glm53_cuda_kda_staging_geometry(size_t heads, size_t dim,
        size_t proj, size_t kernel, size_t *device_bytes, size_t *window_bytes) {
    size_t state_bytes, floats;
    if (heads > 65535 || dim > 256 ||
        !coli_glm53_cuda_kda_geometry(heads, dim, proj, kernel, &state_bytes, window_bytes) ||
        !coli_glm53_cuda_size_mul(5, proj, &floats) || heads > SIZE_MAX - floats ||
        !coli_glm53_cuda_size_mul(floats + heads, sizeof(float), device_bytes)) return 0;
    return 1; /* q/k/v/decay/raw_out = 5*P; beta = H */
}
static inline void coli_glm53_cuda_kda_staging_close(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaStaging *work) {
#ifdef COLI_CUDA
    if (work->device) coli_cuda_pipe_free(stage->cuda_device_ordinal, work->device);
#else
    (void)stage;
#endif
    free(work->next_window);
    memset(work, 0, sizeof(*work));
}
static inline int coli_glm53_cuda_kda_staging_open(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaStaging *work, size_t heads, size_t dim, size_t proj, size_t kernel) {
    size_t db, wb;
    if (!stage || !stage->lease_live || stage->cuda_device_ordinal < 0 ||
        work->device || work->next_window ||
        !coli_glm53_cuda_kda_staging_geometry(heads, dim, proj, kernel, &db, &wb)) return 0;
#ifdef COLI_CUDA
    work->next_window = (float *)malloc(wb);
    if (work->next_window) work->device = (float *)coli_cuda_pipe_alloc(stage->cuda_device_ordinal, db);
    if (!work->device) { coli_glm53_cuda_kda_staging_close(stage, work); return 0; }
    work->device_bytes = db; work->window_bytes = wb;
    work->proj_bytes = proj * sizeof(float); /* checked by geometry above */
    work->q = work->device; work->k = work->q + proj; work->v = work->k + proj;
    work->decay = work->v + proj; work->beta = work->decay + proj;
    work->raw_out = work->beta + heads;
    return 1;
#else
    return 0;
#endif
}
/* mixed/decay/beta/core are host buffers; all backend numeric arguments are
 * slices of the persistent device block. Caller has prepared BOTH/DEVICE and
 * computed ShortConv into next_window, leaving the canonical window untouched.
 * A rejected launch preserves valid authority. Accepted mutation is UNKNOWN
 * until completion, output readback and complete window publication succeed. */
static inline int coli_glm53_cuda_kda_recur(const ColiGlm53CudaStage *stage,
        ColiGlm53CudaKdaLayer *layer, ColiGlm53CudaKdaStaging *work,
        float *window, const float *mixed, const float *decay, const float *beta,
        float *core, int heads, int dim) {
    size_t db, wb;
    if (!stage || !stage->lease_live || !layer->state || !layer->window ||
        (layer->authority != G53_KDA_BOTH && layer->authority != G53_KDA_DEVICE) ||
        !work->device || !work->next_window || !window || !mixed || !decay || !beta || !core ||
        heads < 1 || dim < 1 ||
        !coli_glm53_cuda_kda_staging_geometry((size_t)heads, (size_t)dim,
            work->proj_bytes / sizeof(float), 1, &db, &wb) ||
        db != work->device_bytes || layer->state_bytes != work->proj_bytes * (size_t)dim ||
        layer->window_bytes != work->window_bytes) return 0;
#ifdef COLI_CUDA
    int owner = stage->cuda_device_ordinal;
    size_t p = work->proj_bytes / sizeof(float);
    if (!coli_cuda_pipe_upload(owner, work->q, mixed, work->proj_bytes) ||
        !coli_cuda_pipe_upload(owner, work->k, mixed + p, work->proj_bytes) ||
        !coli_cuda_pipe_upload(owner, work->v, mixed + 2*p, work->proj_bytes) ||
        !coli_cuda_pipe_upload(owner, work->decay, decay, work->proj_bytes) ||
        !coli_cuda_pipe_upload(owner, work->beta, beta, (size_t)heads * sizeof(float))) return 0;
    if (!coli_cuda_pipe_kda_recur(owner, (float *)layer->state, work->raw_out,
            work->q, work->k, work->v, work->decay, work->beta, heads, dim, dim, 1e-6f)) return 0;
    layer->authority = G53_KDA_UNKNOWN;
    if (!coli_cuda_pipe_sync(owner) ||
        !coli_cuda_pipe_download(owner, work->raw_out, core, work->proj_bytes) ||
        !coli_cuda_pipe_upload(owner, layer->window, work->next_window, work->window_bytes)) return 0;
    memcpy(window, work->next_window, work->window_bytes);
    layer->authority = G53_KDA_DEVICE;
    return 1;
#else
    return 0;
#endif
}

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
