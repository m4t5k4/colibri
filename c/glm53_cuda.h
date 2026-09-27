/* Multi-device, synchronous hot-expert tier. No host pointers survive upload.
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
#include <errno.h>

typedef struct {
    ColiCudaTensor *w[3];
    uint64_t heat;
    int owner; /* index in G53Cuda.devices; all three tensors share it */
} G53CudaExpert;
/* Host wall-clock envelopes, using the engine's existing monotonic clock.
 * No clock calls or environment lookups on hot paths when clock is NULL. */
enum { G53_UPLOAD, G53_GATE, G53_UP, G53_CLAMP, G53_DOWN, G53_FALLBACK,
       G53_PROMOTION, G53_EVICT, G53_GROUP_ISSUE, G53_GROUP_TAKE, G53_PROFILE_TIMES };
typedef struct {
    uint64_t hit, miss, hit_rows, miss_rows, promotions, evictions;
    uint64_t repromotions, dead_on_arrival, age_sum, age_min, age_max, age_unknown;
    uint64_t incoming_heat_sum, victim_heat_sum;
} G53CudaLayerCache;
typedef struct {
    uint64_t last_use_tick, upload_tick;
    unsigned promotions, hits_since_upload;
    unsigned char selected;
} G53CudaExpertCache;
typedef struct {
    uint64_t tick, incoming_heat, victim_heat, victim_residence_age, victim_last_use_distance;
    int incoming_layer, incoming_eid, victim_layer, victim_eid;
} G53CudaEvictionRecord;
typedef struct {
    double (*clock)(void);
    double seconds[G53_PROFILE_TIMES], forward_s, decode_forward_s;
    double disk_s, attn_s, ffn_s, head_s;
    uint64_t tokens, decode_tokens, forwards, evictions, fallback_compute_rows;
    G53CudaLayerCache *cache_layer;
    G53CudaExpertCache *cache_expert;
    FILE *selection_trace, *eviction_records;
    uint64_t selection_tick, repromotions, dead_on_arrival, eviction_record_count;
    int decode;
} G53CudaProfile;
typedef struct {
    G53CudaExpert *experts;
    int device, D, I, ne, count, active, failed;
    int ndev, devices[COLI_CUDA_MAX_DEVICES];
    size_t capacity[COLI_CUDA_MAX_DEVICES], used[COLI_CUDA_MAX_DEVICES];
    uint64_t device_executed[COLI_CUDA_MAX_DEVICES];
    unsigned char group_pending[COLI_CUDA_MAX_DEVICES];
    int failure_stage;
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
    if (!g->active || !env || strcmp(env, "1")) return;
    if (!g->profile.clock) {
        g->profile.cache_layer = calloc((size_t)(g->count / g->ne), sizeof(*g->profile.cache_layer));
        g->profile.cache_expert = calloc((size_t)g->count, sizeof(*g->profile.cache_expert));
        if (g->profile.cache_expert) for (int i = 0; i < g->count; i++)
            if (g->experts[i].w[0]) {
                /* A late profiler start cannot know whether an earlier upload
                 * had a hit. Do not call its eventual eviction dead-on-arrival. */
                g->profile.cache_expert[i].promotions = 1;
                g->profile.cache_expert[i].hits_since_upload = 1;
            }
        g->profile.eviction_records = tmpfile();
        const char *path = getenv("GLM53_CUDA_TRACE");
        if (path && *path) {
            g->profile.selection_trace = fopen(path, "w");
            if (g->profile.selection_trace)
                fprintf(g->profile.selection_trace,
                        "# S,tick,decode_token,layer,eid,rows,resident_before | A,tick,layer,eid | P,tick,layer,eid,owner_index,device_id | E,tick,in_layer,in_eid,victim_layer,victim_eid,in_heat,victim_heat,victim_residence_age,victim_last_use_distance\n");
            else fprintf(stderr, "[glm53-cuda-cache] cannot open trace %s: %s\n", path, strerror(errno));
        }
        if (!g->profile.cache_layer || !g->profile.cache_expert)
            fprintf(stderr, "[glm53-cuda-cache] cache counters unavailable (allocation failed)\n");
        if (!g->profile.eviction_records)
            fprintf(stderr, "[glm53-cuda-cache] eviction record spool unavailable\n");
    }
    g->profile.clock = clock;
}
static void g53_cuda_cache_report(const G53Cuda *g) {
    const G53CudaProfile *p = &g->profile;
    if (!p->clock || !p->cache_layer || !p->cache_expert) return;
    uint64_t unique_selected = 0, unique_resident = 0, unique_repromoted = 0;
    for (int i = 0; i < g->count; i++) {
        const G53CudaExpertCache *e = &p->cache_expert[i];
        unique_selected += !!e->selected;
        unique_resident += e->promotions > 0;
        unique_repromoted += e->promotions > 1;
    }
    fprintf(stderr, "[glm53-cuda-cache] unique_selected=%llu unique_ever_resident=%llu unique_promoted_more_than_once=%llu re_promotions=%llu dead_on_arrival=%llu eviction_records=%llu\n",
            (unsigned long long)unique_selected, (unsigned long long)unique_resident,
            (unsigned long long)unique_repromoted, (unsigned long long)p->repromotions,
            (unsigned long long)p->dead_on_arrival,
            (unsigned long long)p->eviction_record_count);
    for (int layer = 0; layer < g->count / g->ne; layer++) {
        const G53CudaLayerCache *s = &p->cache_layer[layer];
        if (!(s->hit || s->miss || s->promotions || s->evictions)) continue;
        uint64_t selected = 0, resident = 0, repromoted = 0;
        for (int eid = 0; eid < g->ne; eid++) {
            const G53CudaExpertCache *e = &p->cache_expert[layer * g->ne + eid];
            selected += !!e->selected;
            resident += e->promotions > 0;
            repromoted += e->promotions > 1;
        }
        fprintf(stderr, "[glm53-cuda-cache-layer] layer=%d hit=%llu miss=%llu hit_rows=%llu miss_rows=%llu promotions=%llu evictions=%llu re_promotions=%llu dead_on_arrival=%llu unique_selected=%llu unique_resident=%llu unique_repromoted=%llu victim_last_use_min=%llu victim_last_use_max=%llu victim_last_use_mean=%.1f victim_last_use_unknown=%llu incoming_heat_mean=%.1f victim_heat_mean=%.1f\n",
                layer, (unsigned long long)s->hit, (unsigned long long)s->miss,
                (unsigned long long)s->hit_rows, (unsigned long long)s->miss_rows,
                (unsigned long long)s->promotions, (unsigned long long)s->evictions,
                (unsigned long long)s->repromotions, (unsigned long long)s->dead_on_arrival,
                (unsigned long long)selected, (unsigned long long)resident,
                (unsigned long long)repromoted,
                (unsigned long long)(s->evictions > s->age_unknown ? s->age_min : 0),
                (unsigned long long)s->age_max,
                s->evictions > s->age_unknown ? (double)s->age_sum / (s->evictions - s->age_unknown) : 0,
                (unsigned long long)s->age_unknown,
                s->evictions ? (double)s->incoming_heat_sum / s->evictions : 0,
                s->evictions ? (double)s->victim_heat_sum / s->evictions : 0);
    }
}
static void g53_cuda_profile_report(const G53Cuda *g, const char *phase) {
    const G53CudaProfile *p = &g->profile;
    if (!p->clock) return;
    int full = g->expert_bytes && g->budget >= g->expert_bytes &&
               g->bytes > g->budget - g->expert_bytes;
    if (g->ndev > 1 && g->expert_bytes && !full) {
        /* Aggregate free bytes can be stranded across devices. */
        full = 1;
        for (int i = 0; i < g->ndev; i++)
            if (g->capacity[i] >= g->expert_bytes &&
                g->used[i] <= g->capacity[i] - g->expert_bytes) full = 0;
    }
    fprintf(stderr, "[glm53-cuda-profile] phase=%s tokens=%llu decode_tokens=%llu forwards=%llu"
        " cuda_rows=%llu fallback_rows=%llu fallback_compute_rows=%llu uploads=%llu evictions=%llu errors=%llu"
        " resident=%u vram_bytes=%zu budget_bytes=%zu tier_full=%d"
        " upload_s=%.6f gate_s=%.6f up_s=%.6f clamp_s=%.6f down_s=%.6f"
        " fallback_compute_s=%.6f promotion_s=%.6f eviction_s=%.6f group_issue_s=%.6f group_take_s=%.6f cuda_expert_s=%.6f"
        " disk_s=%.6f attn_s=%.6f ffn_s=%.6f head_s=%.6f forward_s=%.6f decode_forward_s=%.6f\n",
        phase, (unsigned long long)p->tokens, (unsigned long long)p->decode_tokens,
        (unsigned long long)p->forwards, (unsigned long long)g->executed,
        (unsigned long long)g->fallback, (unsigned long long)p->fallback_compute_rows,
        (unsigned long long)g->uploads, (unsigned long long)p->evictions,
        (unsigned long long)g->errors, g->resident, g->bytes, g->budget, full,
        p->seconds[G53_UPLOAD], p->seconds[G53_GATE], p->seconds[G53_UP],
        p->seconds[G53_CLAMP], p->seconds[G53_DOWN], p->seconds[G53_FALLBACK],
        p->seconds[G53_PROMOTION], p->seconds[G53_EVICT],
        p->seconds[G53_GROUP_ISSUE], p->seconds[G53_GROUP_TAKE],
        p->seconds[G53_GATE]+p->seconds[G53_UP]+p->seconds[G53_CLAMP]+p->seconds[G53_DOWN]
            +p->seconds[G53_GROUP_ISSUE]+p->seconds[G53_GROUP_TAKE],
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
    if (g->group_pending[e->owner]) {
        /* Defensive lifetime guard. The FFN path normally drains before it
         * can promote/evict, but teardown and future callers must also be safe. */
        coli_cuda_expert_group_take(g->devices[e->owner]);
        g->group_pending[e->owner] = 0;
        if (!g->failed) {
            g->failed = 1; g->errors++;
            fprintf(stderr, "[glm53-cuda] outstanding group discarded device=%d stage=eviction\n",
                    g->devices[e->owner]);
        }
    }
    for (int k = 0; k < 3; k++) {
        size_t bytes = coli_cuda_tensor_vram(e->w[k]);
        g->bytes -= bytes;
        g->used[e->owner] -= bytes;
        coli_cuda_tensor_free(e->w[k]); e->w[k] = NULL;
    }
    g->resident--;
}
static void g53_cuda_device_stats(const G53Cuda *g);
static void g53_cuda_close(G53Cuda *g) {
    if (!g->active) return;
    for (int i = 0; i < g->ndev; i++) if (g->group_pending[i]) {
        const float *result = coli_cuda_expert_group_take(g->devices[i]);
        g->group_pending[i] = 0;
        if (!result && !g->failed) {
            g->failed = 1; g->errors++;
            fprintf(stderr, "[glm53-cuda] group drain failed device=%d stage=shutdown\n",
                    g->devices[i]);
        }
    }
    g53_cuda_stats(g);
    g53_cuda_device_stats(g);
    g53_cuda_profile_report(g, "final"); /* before teardown, not an eviction */
    g53_cuda_cache_report(g);
    for (int i = 0; i < g->count; i++) g53_cuda_drop(g, &g->experts[i]);
    if (g->profile.selection_trace) fclose(g->profile.selection_trace);
    if (g->profile.eviction_records) fclose(g->profile.eviction_records);
    free(g->profile.cache_layer);
    free(g->profile.cache_expert);
    free(g->experts);
    coli_cuda_shutdown();
    memset(g, 0, sizeof(*g));
}
/* Nonempty plural form takes precedence, matching the existing Qwen tier.
 * Reject duplicates/empty entries instead of silently changing list order. */
static int g53_cuda_devices(const char *many, const char *single, int *devices) {
    const char *p = many && *many ? many : single ? single : "0";
    int plural = many && *many, n = 0;
    for (;;) {
        char *end;
        errno = 0;
        long d = strtol(p, &end, 10);
        if (end == p || errno == ERANGE || d < 0 || d > INT_MAX ||
            n == COLI_CUDA_MAX_DEVICES || (*end && (!plural || *end != ','))) return 0;
        for (int i = 0; i < n; i++) if (devices[i] == d) return 0;
        devices[n++] = (int)d;
        if (!*end) return n;
        p = end + 1;
    }
}
/* Summary only at teardown; the aggregate line above keeps its existing
 * cadence and format for dashboards and profile parsers. */
static void g53_cuda_device_stats(const G53Cuda *g) {
    if (g->ndev < 2) return;
    for (int i = 0; i < g->ndev; i++) {
        unsigned resident = 0;
        for (int j = 0; j < g->count; j++)
            resident += g->experts[j].w[0] && g->experts[j].owner == i;
        fprintf(stderr,
            "[glm53-cuda-device] device=%d ceiling_bytes=%zu resident=%u vram_bytes=%zu executed=%llu\n",
            g->devices[i], g->capacity[i] < g->budget ? g->capacity[i] : g->budget,
            resident, g->used[i], (unsigned long long)g->device_executed[i]);
    }
}
static void g53_cuda_init(G53Cuda *g, int nl, int ne, int D, int I, int streaming) {
    const char *enabled = getenv("COLI_CUDA");
    if (!enabled || strcmp(enabled, "1")) return;
    const char *dev = getenv("COLI_GPU"), *many = getenv("COLI_GPUS");
    char *end;
    g->ndev = g53_cuda_devices(many, dev, g->devices);
    if (!g->ndev || !streaming || D % 64 || I % 64) {
        fprintf(stderr, "GLM53 CUDA requires int4-gs64 experts and valid COLI_GPU/COLI_GPUS ordinals\n");
        exit(1);
    }
    g->device = g->devices[0]; /* retain the single-device form */
    if (!coli_cuda_init(g->devices, g->ndev)) {
        fprintf(stderr, "GLM53 CUDA initialization failed\n"); exit(1);
    }
    /* Backend initialization is all-or-nothing; it unwinds its own failures. */
    if (coli_cuda_device_count() != g->ndev) {
        coli_cuda_shutdown(); fprintf(stderr, "GLM53 CUDA device count mismatch\n"); exit(1);
    }
    /* Decimal GB. Shared total cap; reserve 2 GB on EACH device for scratch.
     * Do not rigidly divide the cap: a small device must not strand allowance. */
    g->budget = 0;
    for (int i = 0; i < g->ndev; i++) {
        size_t available = 0, total = 0;
        if (!coli_cuda_mem_info(g->devices[i], &available, &total)) {
            fprintf(stderr, "[glm53-cuda] initialization failed device=%d stage=mem_info\n",
                    g->devices[i]);
            coli_cuda_shutdown(); exit(1);
        }
        g->capacity[i] = available > 2000000000ULL ? available - 2000000000ULL : 0;
        g->budget += g->capacity[i];
    }
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
    fprintf(stderr, "[glm53-cuda] device=%d budget_bytes=%zu gs=64 host_clamp=1 group_clamp=%d devices=",
            g->device, g->budget, g->ndev > 1);
    for (int i = 0; i < g->ndev; i++)
        fprintf(stderr, "%s%d(usable_bytes=%zu,capacity=%zu)", i ? "," : "", g->devices[i],
                g->capacity[i] < g->budget ? g->capacity[i] : g->budget,
                (g->capacity[i] < g->budget ? g->capacity[i] : g->budget) / g->expert_bytes);
    fputc('\n', stderr);
    g53_cuda_stats(g);
}
/* Called once per routed expert union, with its actual selected-row count. */
static void g53_cuda_heat(G53Cuda *g, int layer, int eid, int rows) {
    if (!g->active) return;
    int index = layer * g->ne + eid;
    G53CudaExpert *e = &g->experts[index];
    if (g->profile.clock && g->profile.cache_layer && g->profile.cache_expert && rows > 0) {
        G53CudaProfile *p = &g->profile;
        G53CudaLayerCache *s = &p->cache_layer[layer];
        G53CudaExpertCache *ec = &p->cache_expert[index];
        int resident = e->w[0] != NULL;
        uint64_t tick = ++p->selection_tick;
        ec->selected = 1;
        if (resident) {
            s->hit++; s->hit_rows += (uint64_t)rows;
            ec->last_use_tick = tick; ec->hits_since_upload++;
        } else { s->miss++; s->miss_rows += (uint64_t)rows; }
        if (p->selection_trace)
            fprintf(p->selection_trace, "S,%llu,%llu,%d,%d,%d,%d\n",
                    (unsigned long long)tick,
                    (unsigned long long)(p->decode ? p->decode_tokens + 1 : 0),
                    layer, eid, rows, resident);
    }
    e->heat += (uint64_t)rows;
}
static int g53_cuda_place(const G53Cuda *g) {
    int owner = -1;
    for (int i = 0; i < g->ndev; i++)
        if (g->capacity[i] >= g->expert_bytes &&
            g->used[i] <= g->capacity[i] - g->expert_bytes &&
            (owner < 0 || g->used[i] < g->used[owner])) owner = i;
    return owner;
}
/* A miss executes on host this time. Promote only after its host buffers are
 * available, and only once selected at least twice. No RAM cache ownership. */
static void g53_cuda_promote_impl(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    if (!g->active || g->failed || g->expert_bytes > g->budget) return;
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (e->w[0] || e->heat < 2) return;
    int owner = g53_cuda_place(g);
    if (owner < 0 || g->bytes > g->budget - g->expert_bytes) {
        G53CudaExpert *victim = NULL;
        for (int j = 0; j < g->count; j++)
            if (g->experts[j].w[0] && (!victim || g->experts[j].heat < victim->heat))
                victim = &g->experts[j];
        if (!victim || e->heat <= victim->heat) return;
        double evict_start = g53_cuda_profile_now(g);
        if (g->profile.clock && g->profile.cache_layer && g->profile.cache_expert) {
            G53CudaProfile *p = &g->profile;
            int vi = (int)(victim - g->experts), vl = vi / g->ne, ve = vi % g->ne;
            G53CudaExpertCache *vc = &p->cache_expert[vi];
            G53CudaLayerCache *ls = &p->cache_layer[vl];
            uint64_t age = vc->last_use_tick ? p->selection_tick - vc->last_use_tick : UINT64_MAX;
            uint64_t residence_age = p->selection_tick - vc->upload_tick;
            G53CudaEvictionRecord rec = {p->selection_tick, e->heat, victim->heat,
                                         residence_age, age,
                                         layer, eid, vl, ve};
            ls->evictions++;
            ls->incoming_heat_sum += e->heat;
            ls->victim_heat_sum += victim->heat;
            if (age == UINT64_MAX) ls->age_unknown++;
            else {
                if (ls->evictions - ls->age_unknown == 1 || age < ls->age_min) ls->age_min = age;
                if (age > ls->age_max) ls->age_max = age;
                ls->age_sum += age;
            }
            if (vc->promotions && !vc->hits_since_upload) {
                ls->dead_on_arrival++; p->dead_on_arrival++;
            }
            if (p->eviction_records && fwrite(&rec, sizeof(rec), 1, p->eviction_records) == 1)
                p->eviction_record_count++;
            if (p->selection_trace)
                fprintf(p->selection_trace, "E,%llu,%d,%d,%d,%d,%llu,%llu,%llu,%lld\n",
                        (unsigned long long)rec.tick, rec.incoming_layer, rec.incoming_eid,
                        rec.victim_layer, rec.victim_eid,
                        (unsigned long long)rec.incoming_heat,
                        (unsigned long long)rec.victim_heat,
                        (unsigned long long)rec.victim_residence_age,
                        age == UINT64_MAX ? -1LL : (long long)age);
        }
        g53_cuda_drop(g, victim);
        /* Eviction changes the least-allocated device. The freed whole slot
         * guarantees at least one eligible owner. */
        owner = g53_cuda_place(g);
        if (g->profile.clock) g->profile.evictions++;
        g53_cuda_profile_add(g, G53_EVICT, evict_start);
    }
    ColiCudaTensor *w[3] = {NULL, NULL, NULL};
    int ok = 1;
    double upload_start = g53_cuda_profile_now(g);
    int failed_k = -1;
    for (int k = 0; k < 3 && ok; k++) {
        ok = coli_cuda_tensor_upload_g(&w[k], pieces[k*2], (const float *)pieces[k*2+1],
                                      4, k == 2 ? g->I : g->D,
                                      k == 2 ? g->D : g->I, g->devices[owner], 64);
        if (!ok) failed_k = k;
    }
    g53_cuda_profile_add(g, G53_UPLOAD, upload_start);
    if (!ok) {
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(w[k]);
        g->errors++; g->failed = 1; /* avoid repeated failed allocations */
        fprintf(stderr, "[glm53-cuda] upload failed device=%d stage=%s; host fallback enabled\n",
                g->devices[owner], (const char *const[]){"gate", "up", "down"}[failed_k]);
        return;
    }
    e->owner = owner;
    for (int k = 0; k < 3; k++) {
        e->w[k] = w[k];
        size_t bytes = coli_cuda_tensor_vram(w[k]);
        g->bytes += bytes; g->used[owner] += bytes;
    }
    g->resident++; g->uploads++;
    if (g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "P,%llu,%d,%d,%d,%d\n",
                (unsigned long long)g->profile.selection_tick, layer, eid,
                owner, g->devices[owner]);
    if (g->profile.clock && g->profile.cache_layer && g->profile.cache_expert) {
        G53CudaProfile *p = &g->profile;
        G53CudaExpertCache *ec = &p->cache_expert[layer * g->ne + eid];
        p->cache_layer[layer].promotions++;
        if (ec->promotions) { p->repromotions++; p->cache_layer[layer].repromotions++; }
        ec->promotions++;
        ec->hits_since_upload = 0;
        ec->last_use_tick = 0;
        ec->upload_tick = p->selection_tick;
    }
}
static void g53_cuda_promote(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    double start = g53_cuda_profile_now(g);
    if (g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "A,%llu,%d,%d\n",
                (unsigned long long)g->profile.selection_tick, layer, eid);
    g53_cuda_promote_impl(g, layer, eid, pieces);
    g53_cuda_profile_add(g, G53_PROMOTION, start);
}
/* Each existing matmul already copies y back synchronously. This measures
 * the whole API call, NOT kernel-only time, and adds no device synchronization. */
static int g53_cuda_profile_matmul(G53Cuda *g, G53CudaExpert *e, int k,
                                  float *y, const float *x) {
    double start = g53_cuda_profile_now(g);
    int ok = coli_cuda_matmul(&e->w[k], y, x, NULL, NULL, 4, 1,
                             k == 2 ? g->I : g->D, k == 2 ? g->D : g->I, g->devices[e->owner], 64);
    if (!ok) g->failure_stage = k;
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
        fprintf(stderr, "[glm53-cuda] execution failed device=%d stage=%s; host fallback enabled\n",
                g->devices[e->owner], (const char *const[]){"gate", "up", "down"}[g->failure_stage]);
    } else { g->executed++; g->device_executed[e->owner]++; }
    return ok;
}
/* One in-flight group per device, matching the shared backend contract.
 * Results belong to backend pinned staging until that device is issued again. */
static int g53_cuda_group_issue(G53Cuda *g, int di, ColiCudaTensor *const *gate,
                               ColiCudaTensor *const *up, ColiCudaTensor *const *down,
                               const int *rows, int count, const float *x, float limit) {
    if (g->failed || g->group_pending[di]) return 0;
    double start = g53_cuda_profile_now(g);
    int ok = coli_cuda_expert_group_issue_clamped(gate, up, down, rows, count, x, limit);
    g53_cuda_profile_add(g, G53_GROUP_ISSUE, start);
    if (!ok) {
        if (!g->failed) {
            g->failed = 1; g->errors++;
            fprintf(stderr, "[glm53-cuda] group failed device=%d stage=issue; host fallback enabled\n",
                    g->devices[di]);
        }
        return 0;
    }
    g->group_pending[di] = 1;
    return 1;
}
static const float *g53_cuda_group_take(G53Cuda *g, int di) {
    if (!g->group_pending[di]) return NULL;
    double start = g53_cuda_profile_now(g);
    const float *result = coli_cuda_expert_group_take(g->devices[di]);
    g53_cuda_profile_add(g, G53_GROUP_TAKE, start);
    g->group_pending[di] = 0;
    if (!result && !g->failed) {
        g->failed = 1; g->errors++;
        fprintf(stderr, "[glm53-cuda] group failed device=%d stage=take; host fallback enabled\n",
                g->devices[di]);
    }
    return result;
}
#endif
