/* Single-device, synchronous hot-expert tier. No host pointers survive upload.
 * Caller serializes placement/execution; disk workers never call this API.
 * Only the full-model CLI/SERVE loader may init this process-global backend;
 * Segment/Edge/range loaders leave G53Cuda zero-initialized and inactive.
 * Like qwen36_tier.c's Qwen3.8 streaming mode, promotion occurs while streamed
 * bytes are live and owns copies independent of RAM slots. Here the copy is
 * synchronous (no staging queue) and execution keeps GLM53's host clamp.
 * See docs/glm53-flash.md for the placement/residency comparison. */
#ifndef GLM53_CUDA_H
#define GLM53_CUDA_H
#include "backend_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef struct {
    ColiCudaTensor *w[3];
    uint64_t heat;
} G53CudaExpert;
/* Host wall-clock envelopes, using the engine's existing monotonic clock.
 * No clock calls or environment lookups on hot paths when clock is NULL. */
enum { G53_UPLOAD, G53_GATE, G53_UP, G53_CLAMP, G53_DOWN, G53_FALLBACK,
       G53_PROMOTION, G53_EVICT, G53_PROFILE_TIMES };
typedef struct {
    double (*clock)(void);
    double seconds[G53_PROFILE_TIMES], forward_s, decode_forward_s;
    double disk_s, attn_s, ffn_s, head_s;
    uint64_t tokens, decode_tokens, forwards, evictions, fallback_compute_rows;
    int decode;
} G53CudaProfile;
typedef struct {
    G53CudaExpert *experts;
    int device, D, I, ne, count, active, failed;
    size_t budget, bytes, expert_bytes;
    unsigned resident;
    uint64_t executed, fallback, uploads, errors;
    G53CudaProfile profile;
} G53Cuda;

static double g53_cuda_profile_now(const G53Cuda *g) {
    return g->profile.clock ? g->profile.clock() : 0;
}
static void g53_cuda_profile_add(G53Cuda *g, int field, double start) {
    if (g->profile.clock) g->profile.seconds[field] += g->profile.clock() - start;
}
static void g53_cuda_profile_enable(G53Cuda *g, double (*clock)(void)) {
    const char *env = getenv("GLM53_CUDA_PROFILE");
    if (g->active && env && !strcmp(env, "1")) g->profile.clock = clock;
}
static void g53_cuda_profile_report(const G53Cuda *g, const char *phase) {
    const G53CudaProfile *p = &g->profile;
    if (!p->clock) return;
    int full = g->expert_bytes && g->budget >= g->expert_bytes &&
               g->bytes > g->budget - g->expert_bytes;
    fprintf(stderr, "[glm53-cuda-profile] phase=%s tokens=%llu decode_tokens=%llu forwards=%llu"
        " cuda_rows=%llu fallback_rows=%llu fallback_compute_rows=%llu uploads=%llu evictions=%llu errors=%llu"
        " resident=%u vram_bytes=%zu budget_bytes=%zu tier_full=%d"
        " upload_s=%.6f gate_s=%.6f up_s=%.6f clamp_s=%.6f down_s=%.6f"
        " fallback_compute_s=%.6f promotion_s=%.6f eviction_s=%.6f cuda_expert_s=%.6f"
        " disk_s=%.6f attn_s=%.6f ffn_s=%.6f head_s=%.6f forward_s=%.6f decode_forward_s=%.6f\n",
        phase, (unsigned long long)p->tokens, (unsigned long long)p->decode_tokens,
        (unsigned long long)p->forwards, (unsigned long long)g->executed,
        (unsigned long long)g->fallback, (unsigned long long)p->fallback_compute_rows,
        (unsigned long long)g->uploads, (unsigned long long)p->evictions,
        (unsigned long long)g->errors, g->resident, g->bytes, g->budget, full,
        p->seconds[G53_UPLOAD], p->seconds[G53_GATE], p->seconds[G53_UP],
        p->seconds[G53_CLAMP], p->seconds[G53_DOWN], p->seconds[G53_FALLBACK],
        p->seconds[G53_PROMOTION], p->seconds[G53_EVICT],
        p->seconds[G53_GATE]+p->seconds[G53_UP]+p->seconds[G53_CLAMP]+p->seconds[G53_DOWN],
        p->disk_s, p->attn_s, p->ffn_s, p->head_s, p->forward_s, p->decode_forward_s);
}

static void g53_cuda_stats(const G53Cuda *g) {
    if (g->active) fprintf(stderr,
        "[glm53-cuda] resident=%u vram_bytes=%zu executed=%llu fallback=%llu uploads=%llu errors=%llu\n",
        g->resident, g->bytes, (unsigned long long)g->executed,
        (unsigned long long)g->fallback, (unsigned long long)g->uploads,
        (unsigned long long)g->errors);
}
static void g53_cuda_drop(G53Cuda *g, G53CudaExpert *e) {
    if (!e->w[0]) return;
    for (int k = 0; k < 3; k++) {
        g->bytes -= coli_cuda_tensor_vram(e->w[k]);
        coli_cuda_tensor_free(e->w[k]); e->w[k] = NULL;
    }
    g->resident--;
}
static void g53_cuda_close(G53Cuda *g) {
    if (!g->active) return;
    g53_cuda_stats(g);
    g53_cuda_profile_report(g, "final"); /* before teardown, not an eviction */
    for (int i = 0; i < g->count; i++) g53_cuda_drop(g, &g->experts[i]);
    free(g->experts);
    coli_cuda_shutdown();
    memset(g, 0, sizeof(*g));
}
static void g53_cuda_init(G53Cuda *g, int nl, int ne, int D, int I, int streaming) {
    const char *enabled = getenv("COLI_CUDA");
    if (!enabled || strcmp(enabled, "1")) return;
    const char *dev = getenv("COLI_GPU"), *many = getenv("COLI_GPUS");
    char *end;
    long ordinal = dev ? strtol(dev, &end, 10) : 0;
    if ((dev && (!*dev || *end || ordinal < 0 || ordinal > INT_MAX)) ||
        (many && *many) || !streaming || D % 64 || I % 64) {
        fprintf(stderr, "GLM53 CUDA requires int4-gs64 experts and one COLI_GPU ordinal (no COLI_GPUS)\n");
        exit(1);
    }
    g->device = (int)ordinal;
    if (!coli_cuda_init(&g->device, 1)) {
        fprintf(stderr, "GLM53 CUDA initialization failed\n"); exit(1);
    }
    size_t available = 0, total = 0;
    if (!coli_cuda_mem_info(g->device, &available, &total)) {
        coli_cuda_shutdown(); fprintf(stderr, "GLM53 CUDA memory query failed\n"); exit(1);
    }
    /* Decimal GB, matching CUDA_EXPERT_GB. Reserve 2 GB for runtime/scratch. */
    g->budget = available > 2000000000ULL ? available - 2000000000ULL : 0;
    const char *budget = getenv("CUDA_EXPERT_GB");
    if (budget && strcmp(budget, "auto")) {
        double gb = strtod(budget, &end);
        if (!*budget || *end || !isfinite(gb) || gb < 0) {
            coli_cuda_shutdown(); fprintf(stderr, "invalid CUDA_EXPERT_GB\n"); exit(1);
        }
        if (gb * 1e9 < (double)g->budget) g->budget = (size_t)(gb * 1e9);
    }
    g->D = D; g->I = I; g->ne = ne; g->count = nl * ne;
    g->expert_bytes = 3 * (coli_cuda_alloc_footprint((size_t)D * I / 2) +
                          coli_cuda_alloc_footprint((size_t)D * I / 64 * sizeof(float)));
    g->experts = calloc((size_t)g->count, sizeof(*g->experts));
    if (!g->experts) { coli_cuda_shutdown(); fprintf(stderr, "OOM CUDA expert table\n"); exit(1); }
    g->active = 1;
    fprintf(stderr, "[glm53-cuda] device=%d budget_bytes=%zu gs=64 host_clamp=1\n", g->device, g->budget);
    g53_cuda_stats(g);
}
/* Called once per routed expert union, with its actual selected-row count. */
static void g53_cuda_heat(G53Cuda *g, int layer, int eid, int rows) {
    if (g->active) g->experts[layer * g->ne + eid].heat += (uint64_t)rows;
}
/* A miss executes on host this time. Promote only after its host buffers are
 * available, and only once selected at least twice. No RAM cache ownership. */
static void g53_cuda_promote_impl(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    if (!g->active || g->failed || g->expert_bytes > g->budget) return;
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (e->w[0] || e->heat < 2) return;
    if (g->bytes > g->budget - g->expert_bytes) {
        G53CudaExpert *victim = NULL;
        for (int j = 0; j < g->count; j++)
            if (g->experts[j].w[0] && (!victim || g->experts[j].heat < victim->heat))
                victim = &g->experts[j];
        if (!victim || e->heat <= victim->heat) return;
        double evict_start = g53_cuda_profile_now(g);
        g53_cuda_drop(g, victim);
        if (g->profile.clock) g->profile.evictions++;
        g53_cuda_profile_add(g, G53_EVICT, evict_start);
    }
    ColiCudaTensor *w[3] = {NULL, NULL, NULL};
    int ok = 1;
    double upload_start = g53_cuda_profile_now(g);
    for (int k = 0; k < 3 && ok; k++)
        ok = coli_cuda_tensor_upload_g(&w[k], pieces[k*2], (const float *)pieces[k*2+1],
                                      4, k == 2 ? g->I : g->D,
                                      k == 2 ? g->D : g->I, g->device, 64);
    g53_cuda_profile_add(g, G53_UPLOAD, upload_start);
    if (!ok) {
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(w[k]);
        g->errors++; g->failed = 1; /* avoid repeated failed allocations */
        fprintf(stderr, "[glm53-cuda] upload failed; host fallback enabled\n");
        return;
    }
    for (int k = 0; k < 3; k++) { e->w[k] = w[k]; g->bytes += coli_cuda_tensor_vram(w[k]); }
    g->resident++; g->uploads++;
}
static void g53_cuda_promote(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    double start = g53_cuda_profile_now(g);
    g53_cuda_promote_impl(g, layer, eid, pieces);
    g53_cuda_profile_add(g, G53_PROMOTION, start);
}
/* Each existing matmul already copies y back synchronously. This measures
 * the whole API call, NOT kernel-only time, and adds no device synchronization. */
static int g53_cuda_profile_matmul(G53Cuda *g, G53CudaExpert *e, int k,
                                  float *y, const float *x) {
    double start = g53_cuda_profile_now(g);
    int ok = coli_cuda_matmul(&e->w[k], y, x, NULL, NULL, 4, 1,
                             k == 2 ? g->I : g->D, k == 2 ? g->D : g->I, g->device, 64);
    g53_cuda_profile_add(g, k == 0 ? G53_GATE : k == 1 ? G53_UP : G53_DOWN, start);
    return ok;
}
/* Output is private scratch: the engine scatters only after the whole expert
 * succeeds. Existing CPU clamping is supplied to preserve exact semantics. */
static int g53_cuda_run(G53Cuda *g, int layer, int eid, float *y, const float *x,
                       float *sg, float *su, float limit,
                       void (*clamp)(float *, const float *, int, float)) {
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (g->failed || !e->w[0]) return 0;
    int ok = g53_cuda_profile_matmul(g, e, 0, sg, x) &&
             g53_cuda_profile_matmul(g, e, 1, su, x);
    if (ok) {
        double start = g53_cuda_profile_now(g);
        clamp(sg, su, g->I, limit);
        g53_cuda_profile_add(g, G53_CLAMP, start);
        ok = g53_cuda_profile_matmul(g, e, 2, y, sg);
    }
    if (!ok) {
        g->errors++; g->failed = 1;
        fprintf(stderr, "[glm53-cuda] execution failed; host fallback enabled\n");
    } else g->executed++;
    return ok;
}
#endif
