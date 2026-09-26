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
typedef struct {
    G53CudaExpert *experts;
    int device, D, I, ne, count, active, failed;
    size_t budget, bytes, expert_bytes;
    unsigned resident;
    uint64_t executed, fallback, uploads, errors;
} G53Cuda;

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
static void g53_cuda_promote(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    if (!g->active || g->failed || g->expert_bytes > g->budget) return;
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (e->w[0] || e->heat < 2) return;
    if (g->bytes > g->budget - g->expert_bytes) {
        G53CudaExpert *victim = NULL;
        for (int j = 0; j < g->count; j++)
            if (g->experts[j].w[0] && (!victim || g->experts[j].heat < victim->heat))
                victim = &g->experts[j];
        if (!victim || e->heat <= victim->heat) return;
        g53_cuda_drop(g, victim);
    }
    ColiCudaTensor *w[3] = {NULL, NULL, NULL};
    int ok = 1;
    for (int k = 0; k < 3 && ok; k++)
        ok = coli_cuda_tensor_upload_g(&w[k], pieces[k*2], (const float *)pieces[k*2+1],
                                      4, k == 2 ? g->I : g->D,
                                      k == 2 ? g->D : g->I, g->device, 64);
    if (!ok) {
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(w[k]);
        g->errors++; g->failed = 1; /* avoid repeated failed allocations */
        fprintf(stderr, "[glm53-cuda] upload failed; host fallback enabled\n");
        return;
    }
    for (int k = 0; k < 3; k++) { e->w[k] = w[k]; g->bytes += coli_cuda_tensor_vram(w[k]); }
    g->resident++; g->uploads++;
}
/* Output is private scratch: the engine scatters only after the whole expert
 * succeeds. Existing CPU clamping is supplied to preserve exact semantics. */
static int g53_cuda_run(G53Cuda *g, int layer, int eid, float *y, const float *x,
                       float *sg, float *su, float limit,
                       void (*clamp)(float *, const float *, int, float)) {
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (g->failed || !e->w[0]) return 0;
    int ok = coli_cuda_matmul(&e->w[0], sg, x, NULL, NULL, 4, 1, g->D, g->I, g->device, 64) &&
             coli_cuda_matmul(&e->w[1], su, x, NULL, NULL, 4, 1, g->D, g->I, g->device, 64);
    if (ok) {
        clamp(sg, su, g->I, limit);
        ok = coli_cuda_matmul(&e->w[2], y, sg, NULL, NULL, 4, 1, g->I, g->D, g->device, 64);
    }
    if (!ok) {
        g->errors++; g->failed = 1;
        fprintf(stderr, "[glm53-cuda] execution failed; host fallback enabled\n");
    } else g->executed++;
    return ok;
}
#endif
