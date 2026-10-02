/* Multi-device hot-expert tier. No host pointers survive upload.
 * The model thread owns placement/publication; upload workers own private tensors.
 * Only the full-model CLI/SERVE loader may init this process-global backend;
 * Segment/Edge/range loaders leave G53Cuda zero-initialized and inactive.
 * Like qwen36_tier.c's Qwen3.8 streaming mode, promotion occurs while streamed
 * bytes are live and owns copies independent of RAM slots. Here the copy is
 * synchronous per device (no staging queue) and execution keeps GLM53's host clamp.
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
#include <stdint.h>
#include <pthread.h>

typedef struct G53CudaPromotionFlight G53CudaPromotionFlight;
typedef struct G53Cuda G53Cuda;
static void g53_cuda_join_pending(G53Cuda *g, int reason);
static void g53_cuda_warm_report(const G53Cuda *g, const char *phase);
enum { G53_JOIN_NEXT_FFN, G53_JOIN_LAST_LAYER, G53_JOIN_PROFILE,
       G53_JOIN_CLOSE, G53_JOIN_BEFORE_DISPATCH, G53_JOIN_POLICY_OR_DROP,
       G53_JOIN_REASONS };

typedef struct {
    ColiCudaTensor *w[3];
    uint64_t heat;
    int owner; /* index in G53Cuda.devices; all three tensors share it */
    unsigned char warm_state; /* 0 other, 1 warm unhit, 2 warm hit, 3/4 evicted */
} G53CudaExpert;
typedef struct { int layer, eid; uint32_t count; } G53CudaWarmCandidate;
typedef struct { int layer, eid, owner; } G53CudaWarmPlan;
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
    uint64_t promotion_batches, concurrent_promotions, serial_promotions;
    uint64_t eviction_batch_boundaries, batch_experts, batch_devices;
    uint64_t flush_end_boundary, flush_owner_busy, flush_capacity_or_eviction;
    uint64_t flush_batch_full, flush_failure_or_disabled, flush_staging_failure;
    uint64_t flush_other_serial;
    uint64_t heat_noop_passthrough, batch_size_hist[COLI_CUDA_MAX_DEVICES + 1];
    uint64_t deferred_end_batches, deferred_end_experts, deferred_already_complete_at_join;
    uint64_t deferred_join_reason[G53_JOIN_REASONS];
    double deferred_dispatch_s, deferred_lifetime_s, deferred_join_wait_s, deferred_hidden_s;
    double promotion_plan_s, promotion_dispatch_wait_s, promotion_join_s;
    double promotion_batch_wall_s, promotion_worker_s[COLI_CUDA_MAX_DEVICES];
    double promotion_expert_upload_min_s, promotion_expert_upload_max_s;
    uint64_t late_join_opportunities;
    double late_join_prelude_s, late_join_current_wait_s, late_join_hideable_upper_s;
    double late_join_router_s, late_join_topk_s, late_join_shared_s;
    double late_join_union_s, late_join_misc_s;
    uint64_t late_join_actual_opportunities, late_join_already_complete;
    double late_join_join_wait_s, late_join_hidden_s;
    double late_join_router_inflight_s, late_join_topk_inflight_s;
    double late_join_shared_inflight_s, late_join_union_inflight_s;
} G53CudaProfile;
typedef struct {
    int armed;
    double prelude_begin, prelude_end;
    double router_begin, router_end, topk_begin, topk_end;
    double shared_begin, shared_end, union_begin, union_end;
} G53CudaLateWindow;
struct G53Cuda {
    G53CudaExpert *experts;
    int device, D, I, ne, count, active, failed;
    int ndev, devices[COLI_CUDA_MAX_DEVICES];
    size_t capacity[COLI_CUDA_MAX_DEVICES], used[COLI_CUDA_MAX_DEVICES];
    uint64_t device_executed[COLI_CUDA_MAX_DEVICES];
    unsigned char group_pending[COLI_CUDA_MAX_DEVICES];
    int failure_stage;
    size_t budget, bytes, expert_bytes;
    uint64_t heat_min, heat_margin;
    int parallel_promote;
    int promote_overlap;
    int promote_late_join;
    int decode_call;
    G53CudaPromotionFlight *promotion_flight;
    G53CudaLateWindow late_window; /* model-thread-only profiling around the late barrier */
    unsigned resident;
    uint64_t executed, fallback, uploads, errors;
    int warm_enabled, warm_parallel, warm_workers;
    int warm_first_forward_resident, warm_first_decode_resident;
    int warm_teardown;
    size_t warm_requested, warm_candidates, warm_loaded, warm_failed, warm_bytes;
    unsigned warm_placed[COLI_CUDA_MAX_DEVICES];
    uint64_t warm_hits, warm_misses, warm_evicted, warm_evicted_before_first_hit;
    size_t warm_batches, warm_parallel_uploads, warm_peak_inflight;
    double warm_fraction, warm_read_s, warm_upload_s, warm_upload_wall_s;
    double warm_worker_wait_s, warm_publish_s, warm_total_s;
    double warm_device_upload_s[COLI_CUDA_MAX_DEVICES];
    G53CudaProfile profile;
};

static double g53_cuda_profile_now(const G53Cuda *g) {
    return g->profile.clock ? g->profile.clock() : 0;
}
static void g53_cuda_profile_add(G53Cuda *g, int field, double start) {
    if (g->profile.clock) g->profile.seconds[field] += g->profile.clock() - start;
}
/* Counterfactual only: the real join has already finished before this CPU work. */
static void g53_cuda_profile_late_join(G53Cuda *g, double wait, double prelude,
                                       double router, double topk, double shared,
                                       double union_time) {
    G53CudaProfile *p = &g->profile;
    if (!p->clock) return;
    p->late_join_opportunities++;
    p->late_join_prelude_s += prelude;
    /* The old entry-join estimate is only measurable when joining there. */
    if (!g->promote_late_join) {
        p->late_join_current_wait_s += wait;
        p->late_join_hideable_upper_s += wait < prelude ? wait : prelude;
    }
    p->late_join_router_s += router;
    p->late_join_topk_s += topk;
    p->late_join_shared_s += shared;
    p->late_join_union_s += union_time;
    double misc = prelude - router - topk - shared - union_time;
    p->late_join_misc_s += misc > 0 ? misc : 0;
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
                        "# W,layer,eid,owner_index,device_id | F,layer,eid,owner_index,device_id,stage | B,warm_loaded,resident_before_inference | S,tick,decode_token,layer,eid,rows,resident_before | A,tick,layer,eid | P,tick,layer,eid,owner_index,device_id | E,tick,in_layer,in_eid,victim_layer,victim_eid,in_heat,victim_heat,victim_residence_age,victim_last_use_distance | U,tick,layer,eid,device_id,upload_seconds,ok\n");
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
static void g53_cuda_profile_report(G53Cuda *g, const char *phase) {
    g53_cuda_join_pending(g, G53_JOIN_PROFILE);
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
    fprintf(stderr, "[glm53-cuda-promotion] phase=%s batches=%llu batch_experts=%llu batch_devices=%llu concurrent=%llu serial=%llu eviction_boundaries=%llu planning_s=%.6f worker_dispatch_wait_sum_s=%.6f join_wait_s=%.6f batch_wall_s=%.6f expert_upload_sum_s=%.6f expert_upload_min_s=%.6f expert_upload_max_s=%.6f\n",
            phase, (unsigned long long)p->promotion_batches,
            (unsigned long long)p->batch_experts, (unsigned long long)p->batch_devices,
            (unsigned long long)p->concurrent_promotions,
            (unsigned long long)p->serial_promotions,
            (unsigned long long)p->eviction_batch_boundaries,
            p->promotion_plan_s, p->promotion_dispatch_wait_s, p->promotion_join_s,
            p->promotion_batch_wall_s, p->seconds[G53_UPLOAD],
            p->promotion_expert_upload_min_s, p->promotion_expert_upload_max_s);
    fprintf(stderr, "[glm53-cuda-promotion-overlap] phase=%s deferred_end_batches=%llu deferred_end_experts=%llu deferred_dispatch_s=%.6f deferred_lifetime_s=%.6f deferred_join_wait_s=%.6f deferred_hidden_s=%.6f deferred_already_complete_at_join=%llu join_next_ffn=%llu forced_last_layer=%llu forced_profile=%llu forced_close=%llu forced_before_dispatch=%llu forced_policy_or_drop=%llu\n",
            phase, (unsigned long long)p->deferred_end_batches,
            (unsigned long long)p->deferred_end_experts, p->deferred_dispatch_s,
            p->deferred_lifetime_s, p->deferred_join_wait_s, p->deferred_hidden_s,
            (unsigned long long)p->deferred_already_complete_at_join,
            (unsigned long long)p->deferred_join_reason[G53_JOIN_NEXT_FFN],
            (unsigned long long)p->deferred_join_reason[G53_JOIN_LAST_LAYER],
            (unsigned long long)p->deferred_join_reason[G53_JOIN_PROFILE],
            (unsigned long long)p->deferred_join_reason[G53_JOIN_CLOSE],
            (unsigned long long)p->deferred_join_reason[G53_JOIN_BEFORE_DISPATCH],
            (unsigned long long)p->deferred_join_reason[G53_JOIN_POLICY_OR_DROP]);
    fprintf(stderr, "[glm53-cuda-late-join-opportunity] phase=%s late_join_opportunities=%llu late_join_prelude_s=%.6f late_join_current_wait_s=%.6f late_join_hideable_upper_s=%.6f router_s=%.6f topk_s=%.6f shared_s=%.6f union_s=%.6f misc_s=%.6f\n",
            phase, (unsigned long long)p->late_join_opportunities,
            p->late_join_prelude_s, p->late_join_current_wait_s,
            p->late_join_hideable_upper_s, p->late_join_router_s,
            p->late_join_topk_s, p->late_join_shared_s,
            p->late_join_union_s, p->late_join_misc_s);
    fprintf(stderr, "[glm53-cuda-late-join-actual] phase=%s late_join_enabled=%d late_join_actual_opportunities=%llu late_join_join_wait_s=%.6f late_join_hidden_s=%.6f late_join_already_complete=%llu prelude_inflight_s=%.6f router_inflight_s=%.6f topk_inflight_s=%.6f shared_inflight_s=%.6f union_inflight_s=%.6f\n",
            phase, g->promote_late_join,
            (unsigned long long)p->late_join_actual_opportunities,
            p->late_join_join_wait_s, p->late_join_hidden_s,
            (unsigned long long)p->late_join_already_complete,
            p->late_join_hidden_s, p->late_join_router_inflight_s,
            p->late_join_topk_inflight_s, p->late_join_shared_inflight_s,
            p->late_join_union_inflight_s);
    fprintf(stderr, "[glm53-cuda-promotion-flush] phase=%s flush_end_boundary=%llu flush_owner_busy=%llu flush_capacity_or_eviction=%llu flush_batch_full=%llu flush_failure_or_disabled=%llu flush_staging_failure=%llu flush_other_serial=%llu heat_noop_passthrough=%llu\n",
            phase, (unsigned long long)p->flush_end_boundary,
            (unsigned long long)p->flush_owner_busy,
            (unsigned long long)p->flush_capacity_or_eviction,
            (unsigned long long)p->flush_batch_full,
            (unsigned long long)p->flush_failure_or_disabled,
            (unsigned long long)p->flush_staging_failure,
            (unsigned long long)p->flush_other_serial,
            (unsigned long long)p->heat_noop_passthrough);
    fprintf(stderr, "[glm53-cuda-promotion-batch-size] phase=%s", phase);
    for (int n = 1; n <= g->ndev; n++)
        fprintf(stderr, " size%d=%llu", n, (unsigned long long)p->batch_size_hist[n]);
    fputc('\n', stderr);
    for (int i = 0; i < g->ndev; i++)
        fprintf(stderr, "[glm53-cuda-promotion-device] phase=%s device=%d worker_active_s=%.6f\n",
                phase, g->devices[i], p->promotion_worker_s[i]);
}

static void g53_cuda_stats(G53Cuda *g) {
    g53_cuda_join_pending(g, G53_JOIN_PROFILE);
    if (g->active) fprintf(stderr,
        "[glm53-cuda] resident=%u vram_bytes=%zu executed=%llu fallback=%llu uploads=%llu errors=%llu\n",
        g->resident, g->bytes, (unsigned long long)g->executed,
        (unsigned long long)g->fallback, (unsigned long long)g->uploads,
        (unsigned long long)g->errors);
}
static void g53_cuda_drop(G53Cuda *g, G53CudaExpert *e) {
    g53_cuda_join_pending(g, G53_JOIN_POLICY_OR_DROP);
    if (!e->w[0]) return;
    if (!g->warm_teardown && (e->warm_state == 1 || e->warm_state == 2)) {
        g->warm_evicted++;
        if (e->warm_state == 1) g->warm_evicted_before_first_hit++;
        e->warm_state += 2;
    }
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
    g53_cuda_join_pending(g, G53_JOIN_CLOSE);
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
    g53_cuda_warm_report(g, "final");
    g->warm_teardown = 1;
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
static int g53_cuda_heat_setting(const char *value, uint64_t fallback,
                                 int allow_zero, uint64_t *result) {
    if (!value) { *result = fallback; return 1; }
    if (!*value) return 0;
    uint64_t parsed = 0;
    for (const char *p = value; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        unsigned digit = (unsigned)(*p - '0');
        if (parsed > (UINT64_MAX - digit) / 10) return 0;
        parsed = parsed * 10 + digit;
    }
    if (!allow_zero && !parsed) return 0;
    *result = parsed;
    return 1;
}
static int g53_cuda_warm_cmp(const void *a, const void *b) {
    const G53CudaWarmCandidate *x = a, *y = b;
    if (x->count != y->count) return x->count > y->count ? -1 : 1;
    if (x->layer != y->layer) return x->layer < y->layer ? -1 : 1;
    return x->eid < y->eid ? -1 : x->eid > y->eid;
}
static G53CudaWarmCandidate *g53_cuda_warm_rank(const G53Cuda *g, uint32_t *const *counts,
                                                int first_dense, int *n) {
    *n = 0;
    if (!counts || first_dense < 0 || first_dense > g->count / g->ne) return NULL;
    G53CudaWarmCandidate *ranked = malloc((size_t)g->count * sizeof(*ranked));
    if (!ranked) return NULL;
    for (int l = first_dense; l < g->count / g->ne; l++) {
        if (!counts[l]) continue;
        for (int e = 0; e < g->ne; e++) if (counts[l][e])
            ranked[(*n)++] = (G53CudaWarmCandidate){l, e, counts[l][e]};
    }
    qsort(ranked, (size_t)*n, sizeof(*ranked), g53_cuda_warm_cmp);
    return ranked;
}
static size_t g53_cuda_warm_capacity(const G53Cuda *g) {
    if (!g->expert_bytes) return 0;
    size_t slots = 0;
    for (int i = 0; i < g->ndev; i++) slots += g->capacity[i] / g->expert_bytes;
    size_t shared = g->budget / g->expert_bytes;
    return slots < shared ? slots : shared;
}
static size_t g53_cuda_warm_plan(const G53Cuda *g, const G53CudaWarmCandidate *ranked,
                                 size_t target, G53CudaWarmPlan *plan) {
    size_t used[COLI_CUDA_MAX_DEVICES], bytes = g->bytes, n = 0;
    memcpy(used, g->used, sizeof(used));
    for (; n < target; n++) {
        if (g->expert_bytes > g->budget || bytes > g->budget - g->expert_bytes) break;
        int owner = -1;
        for (int i = 0; i < g->ndev; i++)
            if (g->capacity[i] >= g->expert_bytes &&
                used[i] <= g->capacity[i] - g->expert_bytes &&
                (owner < 0 || used[i] < used[owner])) owner = i;
        if (owner < 0) break;
        plan[n] = (G53CudaWarmPlan){ranked[n].layer, ranked[n].eid, owner};
        used[owner] += g->expert_bytes;
        bytes += g->expert_bytes;
    }
    return n;
}
static int g53_cuda_warm_batch_size(const G53CudaWarmPlan *plan, size_t at,
                                    size_t planned, int ndev) {
    int busy[COLI_CUDA_MAX_DEVICES] = {0}, n = 0;
    while (at < planned && n < ndev && !busy[plan[at].owner]) {
        busy[plan[at].owner] = 1;
        at++; n++;
    }
    return n;
}
static void g53_cuda_warm_config(G53Cuda *g) {
    const char *enabled = getenv("GLM53_CUDA_WARM_RESIDENCY");
    if (!enabled || !strcmp(enabled, "0")) return;
    if (strcmp(enabled, "1")) {
        fprintf(stderr, "invalid GLM53_CUDA_WARM_RESIDENCY (expected 0 or 1)\n"); exit(1);
    }
    g->warm_enabled = 1;
    const char *value = getenv("GLM53_CUDA_WARM_FRACTION");
    g->warm_fraction = 0.50;
    if (value) {
        char *end;
        errno = 0;
        double parsed = strtod(value, &end);
        if (!*value || *end || errno == ERANGE || !isfinite(parsed) || parsed <= 0 || parsed > 1) {
            fprintf(stderr, "invalid GLM53_CUDA_WARM_FRACTION (expected >0 and <=1)\n"); exit(1);
        }
        g->warm_fraction = parsed;
    }
    g->warm_requested = (size_t)floor((double)g53_cuda_warm_capacity(g) * g->warm_fraction + 0.5);
    const char *parallel = getenv("GLM53_CUDA_WARM_PARALLEL");
    if (parallel && strcmp(parallel, "0") && strcmp(parallel, "1")) {
        fprintf(stderr, "invalid GLM53_CUDA_WARM_PARALLEL (expected 0 or 1)\n"); exit(1);
    }
    g->warm_parallel = parallel && !strcmp(parallel, "1") && g->ndev > 1;
    g->warm_workers = g->warm_parallel ? g->ndev : 1;
    g->warm_first_forward_resident = -1;
    g->warm_first_decode_resident = -1;
}
static void g53_cuda_warm_report(const G53Cuda *g, const char *phase) {
    if (!g->warm_enabled) return;
    size_t ever_hit = 0;
    for (int i = 0; i < g->count; i++)
        ever_hit += g->experts[i].warm_state == 2 || g->experts[i].warm_state == 4;
    fprintf(stderr, "[glm53-cuda-warm] phase=%s requested=%zu candidates=%zu loaded=%zu failed=%zu bytes=%zu read_s=%.6f upload_s=%.6f total_s=%.6f warm_hits=%llu warm_misses=%llu warm_evicted=%llu warm_evicted_before_first_hit=%llu warm_never_hit=%zu resident_at_first_forward=%d resident_at_first_decode=%d parallel=%d workers=%d batches=%zu parallel_uploads=%zu peak_inflight=%zu upload_wall_s=%.6f worker_wait_sum_s=%.6f publish_s=%.6f\n",
            phase, g->warm_requested, g->warm_candidates, g->warm_loaded, g->warm_failed,
            g->warm_bytes, g->warm_read_s, g->warm_upload_s, g->warm_total_s,
            (unsigned long long)g->warm_hits, (unsigned long long)g->warm_misses,
            (unsigned long long)g->warm_evicted,
            (unsigned long long)g->warm_evicted_before_first_hit,
            g->warm_loaded - ever_hit, g->warm_first_forward_resident, g->warm_first_decode_resident,
            g->warm_parallel, g->warm_workers, g->warm_batches, g->warm_parallel_uploads,
            g->warm_peak_inflight, g->warm_upload_wall_s, g->warm_worker_wait_s, g->warm_publish_s);
    for (int i = 0; i < g->ndev; i++)
        fprintf(stderr, "[glm53-cuda-warm-device] phase=%s device=%d placed=%u bytes=%zu upload_s=%.6f\n",
                phase, g->devices[i], g->warm_placed[i],
                (size_t)g->warm_placed[i] * g->expert_bytes, g->warm_device_upload_s[i]);
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
    if (!g53_cuda_heat_setting(getenv("GLM53_CUDA_HEAT_MIN"), 2, 0, &g->heat_min) ||
        !g53_cuda_heat_setting(getenv("GLM53_CUDA_HEAT_MARGIN"), 1, 1, &g->heat_margin)) {
        coli_cuda_shutdown();
        fprintf(stderr, "invalid GLM53_CUDA_HEAT_MIN or GLM53_CUDA_HEAT_MARGIN\n");
        exit(1);
    }
    /* Decimal GB. Shared total cap. By default reserve 2 GB on EACH
     * device for scratch, matching the historical GLM53 behaviour.
     * GLM53_CUDA_RESERVE_GB may lower or raise that per-device reserve.
     * Do not rigidly divide the cap: a small device must not strand allowance. */
    double reserve_gb = 2.0;
    const char *reserve = getenv("GLM53_CUDA_RESERVE_GB");
    if (reserve && *reserve) {
        char *reserve_end = NULL;
        double parsed = strtod(reserve, &reserve_end);
        if (*reserve_end || !isfinite(parsed) || parsed < 0.0 || parsed > 64.0) {
            coli_cuda_shutdown();
            fprintf(stderr, "invalid GLM53_CUDA_RESERVE_GB\n");
            exit(1);
        }
        reserve_gb = parsed;
    }
    size_t reserve_bytes = (size_t)(reserve_gb * 1e9);

    fprintf(stderr,
            "[glm53-cuda] expert_reserve_gb=%.3f reserve_bytes=%zu\n",
            reserve_gb, reserve_bytes);

    g->budget = 0;
    for (int i = 0; i < g->ndev; i++) {
        size_t available = 0, total = 0;
        if (!coli_cuda_mem_info(g->devices[i], &available, &total)) {
            fprintf(stderr, "[glm53-cuda] initialization failed device=%d stage=mem_info\n",
                    g->devices[i]);
            coli_cuda_shutdown(); exit(1);
        }
        g->capacity[i] = available > reserve_bytes ? available - reserve_bytes : 0;
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
    g53_cuda_warm_config(g);
    g->active = 1;
    g->parallel_promote = g->ndev > 1 && getenv("GLM53_CUDA_PARALLEL_PROMOTE") &&
                          !strcmp(getenv("GLM53_CUDA_PARALLEL_PROMOTE"), "1");
    g->promote_overlap = g->parallel_promote && getenv("GLM53_CUDA_PROMOTE_OVERLAP") &&
                         !strcmp(getenv("GLM53_CUDA_PROMOTE_OVERLAP"), "1");
    g->promote_late_join = g->promote_overlap && getenv("GLM53_CUDA_PROMOTE_LATE_JOIN") &&
                           !strcmp(getenv("GLM53_CUDA_PROMOTE_LATE_JOIN"), "1");
    fprintf(stderr, "[glm53-cuda] device=%d budget_bytes=%zu gs=64 host_clamp=1 group_clamp=%d heat_min=%llu heat_margin=%llu devices=",
            g->device, g->budget, g->ndev > 1,
            (unsigned long long)g->heat_min, (unsigned long long)g->heat_margin);
    for (int i = 0; i < g->ndev; i++)
        fprintf(stderr, "%s%d(usable_bytes=%zu,capacity=%zu)", i ? "," : "", g->devices[i],
                g->capacity[i] < g->budget ? g->capacity[i] : g->budget,
                (g->capacity[i] < g->budget ? g->capacity[i] : g->budget) / g->expert_bytes);
    fputc('\n', stderr);
    fprintf(stderr, "[glm53-cuda] parallel_promotion=%d promotion_overlap=%d promotion_late_join=%d\n",
            g->parallel_promote, g->promote_overlap, g->promote_late_join);
    g53_cuda_stats(g);
}
/* Called once per routed expert union, with its actual selected-row count. */
static void g53_cuda_heat(G53Cuda *g, int layer, int eid, int rows) {
    g53_cuda_join_pending(g, G53_JOIN_NEXT_FFN);
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
    if (e->w[0] && (e->warm_state == 1 || e->warm_state == 2)) {
        g->warm_hits++;
        e->warm_state = 2;
    } else if (!e->w[0] && (e->warm_state == 3 || e->warm_state == 4)) {
        g->warm_misses++;
    }
}
static int g53_cuda_place(const G53Cuda *g) {
    int owner = -1;
    for (int i = 0; i < g->ndev; i++)
        if (g->capacity[i] >= g->expert_bytes &&
            g->used[i] <= g->capacity[i] - g->expert_bytes &&
            (owner < 0 || g->used[i] < g->used[owner])) owner = i;
    return owner;
}
static void g53_cuda_publish_ex(G53Cuda *g, int layer, int eid, int owner,
                                ColiCudaTensor *const w[3], int warm) {
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    e->owner = owner;
    for (int k = 0; k < 3; k++) {
        e->w[k] = w[k];
        size_t bytes = coli_cuda_tensor_vram(w[k]);
        g->bytes += bytes; g->used[owner] += bytes;
    }
    g->resident++;
    if (warm) {
        e->warm_state = 1;
        g->warm_loaded++;
        g->warm_placed[owner]++;
        g->warm_bytes += g->expert_bytes;
    } else g->uploads++;
    if (warm && g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "W,%d,%d,%d,%d\n",
                layer, eid, owner, g->devices[owner]);
    if (!warm && g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "P,%llu,%d,%d,%d,%d\n",
                (unsigned long long)g->profile.selection_tick, layer, eid,
                owner, g->devices[owner]);
    if (!warm && g->profile.clock && g->profile.cache_layer && g->profile.cache_expert) {
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
static void g53_cuda_publish(G53Cuda *g, int layer, int eid, int owner,
                             ColiCudaTensor *const w[3]) {
    g53_cuda_publish_ex(g, layer, eid, owner, w, 0);
}
static int g53_cuda_warm_upload(G53Cuda *g, int layer, int eid,
                                uint8_t *const *pieces, double (*clock)(void)) {
    if (!g->active || g->failed || g->expert_bytes > g->budget ||
        g->bytes > g->budget - g->expert_bytes) return 0;
    int owner = g53_cuda_place(g);
    if (owner < 0) return 0;
    ColiCudaTensor *w[3] = {NULL, NULL, NULL};
    int failed_k = -1;
    double start = clock();
    for (int k = 0; k < 3; k++) {
        if (!coli_cuda_tensor_upload_g(&w[k], pieces[k*2], (const float *)pieces[k*2+1],
                                       4, k == 2 ? g->I : g->D,
                                       k == 2 ? g->D : g->I, g->devices[owner], 64)) {
            failed_k = k;
            break;
        }
    }
    if (failed_k < 0 && !coli_cuda_tensor_upload_complete(g->devices[owner])) failed_k = 3;
    double elapsed = clock() - start;
    g->warm_upload_s += elapsed;
    g->warm_upload_wall_s += elapsed;
    g->warm_device_upload_s[owner] += elapsed;
    if (failed_k >= 0) {
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(w[k]);
        g->warm_failed++;
        g->errors++;
        g->failed = 1;
        fprintf(stderr, "[glm53-cuda] warm upload failed device=%d stage=%s; host fallback enabled\n",
                g->devices[owner], (const char *const[]){"gate", "up", "down", "complete"}[failed_k]);
        if (g->profile.selection_trace)
            fprintf(g->profile.selection_trace, "F,%d,%d,%d,%d,%d\n",
                    layer, eid, owner, g->devices[owner], failed_k);
        return 0;
    }
    g53_cuda_publish_ex(g, layer, eid, owner, w, 1);
    return 1;
}
static void g53_cuda_warm_boundary(G53Cuda *g) {
    if (g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "B,%zu,%u\n", g->warm_loaded, g->resident);
}
/* A miss executes on host this time. Promote only after its host buffers are
 * available, and only once selected at least twice. No RAM cache ownership. */
static void g53_cuda_promote_impl(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    if (!g->active || g->failed || g->expert_bytes > g->budget) return;
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (e->w[0] || e->heat < g->heat_min) return;
    int owner = g53_cuda_place(g);
    if (owner < 0 || g->bytes > g->budget - g->expert_bytes) {
        G53CudaExpert *victim = NULL;
        for (int j = 0; j < g->count; j++)
            if (g->experts[j].w[0] && (!victim || g->experts[j].heat < victim->heat))
                victim = &g->experts[j];
        if (!victim || e->heat <= victim->heat ||
            e->heat - victim->heat <= g->heat_margin) return;
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
    g53_cuda_publish(g, layer, eid, owner, w);
}
static void g53_cuda_promote(G53Cuda *g, int layer, int eid, uint8_t *const *pieces) {
    g53_cuda_join_pending(g, G53_JOIN_POLICY_OR_DROP);
    double start = g53_cuda_profile_now(g);
    uint64_t uploads_before = g->uploads, errors_before = g->errors;
    if (g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "A,%llu,%d,%d\n",
                (unsigned long long)g->profile.selection_tick, layer, eid);
    g53_cuda_promote_impl(g, layer, eid, pieces);
    if (g->profile.clock && (g->uploads != uploads_before || g->errors != errors_before))
        g->profile.serial_promotions++;
    g53_cuda_profile_add(g, G53_PROMOTION, start);
}

/* The batch owns copied host bytes and unpublished tensors. Only the model
 * thread chooses owners or changes residency; a worker touches one backend
 * DeviceContext, then synchronizes its default upload stream before return. */
typedef struct {
    int layer, eid, owner, failed_k;
    uint8_t *host, *piece[6];
    ColiCudaTensor *w[3];
    int ok, device, D, I;
    double (*clock)(void);
    double dispatched, completed, start_wait, active_s;
} G53CudaPromotionTask;
typedef struct {
    G53CudaPromotionTask task[COLI_CUDA_MAX_DEVICES];
    size_t planned_used[COLI_CUDA_MAX_DEVICES], planned_bytes;
    int n, owner_busy[COLI_CUDA_MAX_DEVICES];
} G53CudaPromotionBatch;
static void g53_cuda_upload_task(G53CudaPromotionTask *t) {
    double start = t->clock ? t->clock() : 0;
    t->start_wait = t->clock ? start - t->dispatched : 0;
    t->ok = 1; t->failed_k = -1;
    for (int k = 0; k < 3; k++) {
        if (!coli_cuda_tensor_upload_g(&t->w[k], t->piece[2*k],
                                       (const float *)t->piece[2*k+1], 4,
                                       k == 2 ? t->I : t->D,
                                       k == 2 ? t->D : t->I, t->device, 64)) {
            t->ok = 0; t->failed_k = k; break;
        }
    }
    /* cudaMemcpy from pageable RAM need not finish device DMA before return.
     * The group's nonblocking stream does not inherit default-stream order. */
    if (!coli_cuda_tensor_upload_complete(t->device) && t->ok) {
        t->ok = 0; t->failed_k = 3;
    }
    t->completed = t->clock ? t->clock() : 0;
    t->active_s = t->clock ? t->completed - start : 0;
}
/* Warm tasks have distinct owners and stable per-owner host slots. Only the
 * model thread publishes, in ranked order, after all device streams drain. */
static int g53_cuda_warm_batch(G53Cuda *g, G53CudaPromotionTask *task, int n,
                                double (*clock)(void)) {
    if (n < 1 || n > g->ndev) return 0;
    double start = clock();
    for (int i = 0; i < n; i++) {
        task[i].clock = clock;
        task[i].dispatched = start;
    }
    if (n == 1) g53_cuda_upload_task(&task[0]);
    else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n) schedule(static, 1)
#endif
        for (int i = 0; i < n; i++) g53_cuda_upload_task(&task[i]);
    }
    double joined = clock();
    g->warm_batches++;
    if (n > 1) g->warm_parallel_uploads += (size_t)n;
    g->warm_upload_wall_s += joined - start;
    for (int i = 0; i < n; i++) {
        G53CudaPromotionTask *t = &task[i];
        g->warm_upload_s += t->active_s; /* sum of worker durations, not wall time */
        g->warm_device_upload_s[t->owner] += t->active_s;
        g->warm_worker_wait_s += t->start_wait;
        size_t overlapping = 0;
        double from = t->dispatched + t->start_wait;
        for (int j = 0; j < n; j++)
            overlapping += task[j].dispatched + task[j].start_wait <= from &&
                           task[j].completed > from;
        if (overlapping > g->warm_peak_inflight) g->warm_peak_inflight = overlapping;
    }
    int first_bad = n;
    for (int i = 0; i < n; i++) if (!task[i].ok) { first_bad = i; break; }
    double publish_start = clock();
    for (int i = 0; i < first_bad; i++) {
        G53CudaPromotionTask *t = &task[i];
        g53_cuda_publish_ex(g, t->layer, t->eid, t->owner, t->w, 1);
        for (int k = 0; k < 3; k++) t->w[k] = NULL;
    }
    g->warm_publish_s += clock() - publish_start;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(task[i].w[k]);
    if (first_bad < n) {
        G53CudaPromotionTask *t = &task[first_bad];
        g->warm_failed++; g->errors++; g->failed = 1;
        fprintf(stderr, "[glm53-cuda] warm upload failed device=%d stage=%s; host fallback enabled\n",
                t->device, (const char *const[]){"gate", "up", "down", "complete"}[t->failed_k]);
        if (g->profile.selection_trace)
            fprintf(g->profile.selection_trace, "F,%d,%d,%d,%d,%d\n",
                    t->layer, t->eid, t->owner, t->device, t->failed_k);
    }
    return first_bad == n;
}
typedef struct { G53CudaPromotionFlight *flight; int index; } G53CudaWorkerArg;
struct G53CudaPromotionFlight {
    G53CudaPromotionBatch batch;
    pthread_t worker[COLI_CUDA_MAX_DEVICES];
    G53CudaWorkerArg arg[COLI_CUDA_MAX_DEVICES];
    int started, finished;
    double dispatched, dispatch_done;
};
static void *g53_cuda_upload_worker(void *arg) {
    G53CudaWorkerArg *a = arg;
    g53_cuda_upload_task(&a->flight->batch.task[a->index]);
    __atomic_add_fetch(&a->flight->finished, 1, __ATOMIC_RELEASE);
    return NULL;
}
typedef enum {
    G53_FLUSH_END_BOUNDARY, G53_FLUSH_OWNER_BUSY, G53_FLUSH_CAPACITY_OR_EVICTION,
    G53_FLUSH_BATCH_FULL, G53_FLUSH_FAILURE_OR_DISABLED, G53_FLUSH_STAGING_FAILURE,
    G53_FLUSH_OTHER_SERIAL
} G53CudaFlushReason;
static void g53_cuda_batch_finish(G53Cuda *g, G53CudaPromotionBatch *b,
                                  double wall, double joined, double join_wait,
                                  double blocked);
static void g53_cuda_batch_flush_reason(G53Cuda *g, G53CudaPromotionBatch *b,
                                         G53CudaFlushReason reason) {
    if (!b->n) return;
    if (g->profile.clock) {
        G53CudaProfile *p = &g->profile;
        switch (reason) {
        case G53_FLUSH_END_BOUNDARY: p->flush_end_boundary++; break;
        case G53_FLUSH_OWNER_BUSY: p->flush_owner_busy++; break;
        case G53_FLUSH_CAPACITY_OR_EVICTION:
            p->flush_capacity_or_eviction++; p->eviction_batch_boundaries++; break;
        case G53_FLUSH_BATCH_FULL: p->flush_batch_full++; break;
        case G53_FLUSH_FAILURE_OR_DISABLED: p->flush_failure_or_disabled++; break;
        case G53_FLUSH_STAGING_FAILURE: p->flush_staging_failure++; break;
        case G53_FLUSH_OTHER_SERIAL: p->flush_other_serial++; break;
        }
    }
    double wall = g53_cuda_profile_now(g);
    for (int i = 0; i < b->n; i++) {
        G53CudaPromotionTask *t = &b->task[i];
        t->clock = g->profile.clock;
        t->dispatched = wall;
    }
    if (b->n == 1) g53_cuda_upload_task(&b->task[0]);
    else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(b->n) schedule(static, 1)
#endif
        for (int i = 0; i < b->n; i++) g53_cuda_upload_task(&b->task[i]);
    }
    double joined = g53_cuda_profile_now(g);
    g53_cuda_batch_finish(g, b, wall, joined, joined - wall, joined - wall);
}
static void g53_cuda_batch_finish(G53Cuda *g, G53CudaPromotionBatch *b,
                                  double wall, double joined, double join_wait,
                                  double blocked) {
    if (g->profile.clock) {
        G53CudaProfile *p = &g->profile;
        p->promotion_batches++;
        p->batch_size_hist[b->n]++;
        p->batch_experts += (uint64_t)b->n;
        p->batch_devices += (uint64_t)b->n;
        if (b->n > 1) p->concurrent_promotions += (uint64_t)b->n;
        else p->serial_promotions++;
        p->promotion_join_s += join_wait;
        p->promotion_batch_wall_s += joined - wall;
        p->seconds[G53_PROMOTION] += blocked;
        for (int i = 0; i < b->n; i++) {
            G53CudaPromotionTask *t = &b->task[i];
            p->promotion_dispatch_wait_s += t->start_wait;
            p->promotion_worker_s[t->owner] += t->active_s;
            p->seconds[G53_UPLOAD] += t->active_s;
            if (!p->promotion_expert_upload_min_s ||
                t->active_s < p->promotion_expert_upload_min_s)
                p->promotion_expert_upload_min_s = t->active_s;
            if (t->active_s > p->promotion_expert_upload_max_s)
                p->promotion_expert_upload_max_s = t->active_s;
            if (p->selection_trace)
                fprintf(p->selection_trace, "U,%llu,%d,%d,%d,%.9f,%d\n",
                        (unsigned long long)p->selection_tick, t->layer, t->eid,
                        t->device, t->active_s, t->ok);
        }
    }
    /* Serial failure semantics: commit only the successful logical prefix.
     * Later workers may have completed speculatively, but never publish. */
    int first_bad = b->n;
    for (int i = 0; i < b->n; i++) if (!b->task[i].ok) { first_bad = i; break; }
    for (int i = 0; i < first_bad; i++) {
        G53CudaPromotionTask *t = &b->task[i];
        g53_cuda_publish(g, t->layer, t->eid, t->owner, t->w);
        for (int k = 0; k < 3; k++) t->w[k] = NULL;
    }
    for (int i = 0; i < b->n; i++) {
        G53CudaPromotionTask *t = &b->task[i];
        for (int k = 0; k < 3; k++) coli_cuda_tensor_free(t->w[k]);
        free(t->host);
    }
    if (first_bad < b->n) {
        G53CudaPromotionTask *t = &b->task[first_bad];
        g->errors++; g->failed = 1;
        fprintf(stderr, "[glm53-cuda] upload failed device=%d stage=%s; host fallback enabled\n",
                t->device, (const char *const[]){"gate", "up", "down", "complete"}[t->failed_k]);
    }
    memset(b, 0, sizeof(*b));
}
static void g53_cuda_batch_flush(G53Cuda *g, G53CudaPromotionBatch *b) {
    g53_cuda_batch_flush_reason(g, b, G53_FLUSH_END_BOUNDARY);
}
/* Union of worker-active intervals, clipped to a model-thread CPU interval.
 * Called only after pthread_join, when every task timestamp is published. */
static double g53_cuda_worker_overlap(const G53CudaPromotionFlight *f,
                                      double from, double to) {
    if (to <= from) return 0;
    double starts[COLI_CUDA_MAX_DEVICES], ends[COLI_CUDA_MAX_DEVICES];
    int n = 0;
    for (int i = 0; i < f->batch.n; i++) {
        const G53CudaPromotionTask *t = &f->batch.task[i];
        double start = t->dispatched + t->start_wait, end = t->completed;
        if (start < from) start = from;
        if (end > to) end = to;
        if (end <= start) continue;
        int j = n++;
        while (j && starts[j-1] > start) {
            starts[j] = starts[j-1]; ends[j] = ends[j-1]; j--;
        }
        starts[j] = start; ends[j] = end;
    }
    double overlap = 0, until = from;
    for (int i = 0; i < n; i++) {
        double begin = starts[i] > until ? starts[i] : until;
        if (ends[i] > begin) overlap += ends[i] - begin;
        if (ends[i] > until) until = ends[i];
    }
    return overlap;
}
static void g53_cuda_join_pending(G53Cuda *g, int reason) {
    G53CudaPromotionFlight *f = g->promotion_flight;
    if (!f) return;
    double wait_start = g53_cuda_profile_now(g);
    int complete = __atomic_load_n(&f->finished, __ATOMIC_ACQUIRE) == f->batch.n;
    for (int i = 0; i < f->started; i++) pthread_join(f->worker[i], NULL);
    double joined = g53_cuda_profile_now(g);
    double wait = joined - wait_start;
    double dispatch = f->dispatch_done - f->dispatched;
    if (g->profile.clock) {
        G53CudaProfile *p = &g->profile;
        p->deferred_join_reason[reason]++;
        p->deferred_already_complete_at_join += complete;
        p->deferred_lifetime_s += joined - f->dispatched;
        p->deferred_join_wait_s += wait;
        /* Union of worker-active intervals clipped to caller CPU work. This
         * excludes both join blocking and gaps where no upload was active. */
        double starts[COLI_CUDA_MAX_DEVICES], ends[COLI_CUDA_MAX_DEVICES];
        int intervals = 0;
        for (int i = 0; i < f->batch.n; i++) {
            G53CudaPromotionTask *t = &f->batch.task[i];
            double start = t->dispatched + t->start_wait;
            double end = t->completed;
            if (start < f->dispatch_done) start = f->dispatch_done;
            if (end > wait_start) end = wait_start;
            if (end <= start) continue;
            int j = intervals++;
            while (j && starts[j-1] > start) {
                starts[j] = starts[j-1]; ends[j] = ends[j-1]; j--;
            }
            starts[j] = start; ends[j] = end;
        }
        double hidden = 0, until = 0;
        for (int i = 0; i < intervals; i++) {
            double begin = starts[i] > until ? starts[i] : until;
            if (ends[i] > begin) hidden += ends[i] - begin;
            if (ends[i] > until) until = ends[i];
        }
        p->deferred_hidden_s += hidden;
        if (g->late_window.armed && reason == G53_JOIN_NEXT_FFN) {
            const G53CudaLateWindow *w = &g->late_window;
            p->late_join_actual_opportunities++;
            p->late_join_join_wait_s += wait;
            p->late_join_already_complete += complete;
            p->late_join_hidden_s += g53_cuda_worker_overlap(f, w->prelude_begin,
                                                               w->prelude_end);
            p->late_join_router_inflight_s += g53_cuda_worker_overlap(f, w->router_begin,
                                                                         w->router_end);
            p->late_join_topk_inflight_s += g53_cuda_worker_overlap(f, w->topk_begin,
                                                                       w->topk_end);
            p->late_join_shared_inflight_s += g53_cuda_worker_overlap(f, w->shared_begin,
                                                                         w->shared_end);
            p->late_join_union_inflight_s += g53_cuda_worker_overlap(f, w->union_begin,
                                                                        w->union_end);
        }
    }
    g->late_window.armed = 0;
    g->promotion_flight = NULL;
    g53_cuda_batch_finish(g, &f->batch, f->dispatched, joined,
                          wait, dispatch + wait);
    free(f);
}
/* Only the completed grouped decode FFN may use this path. Its planning and
 * host copies are finished; no model/cache state is touched by a worker. */
static void g53_cuda_batch_dispatch_end(G53Cuda *g, G53CudaPromotionBatch *b) {
    if (!b->n) return;
    g53_cuda_join_pending(g, G53_JOIN_BEFORE_DISPATCH);
    if (!g->promote_overlap || g->failed || !g->active) {
        g53_cuda_batch_flush(g, b); return;
    }
    if (b->n >= g->ndev) {
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_BATCH_FULL); return;
    }
    G53CudaPromotionFlight *f = calloc(1, sizeof(*f));
    if (!f) { g53_cuda_batch_flush(g, b); return; }
    f->batch = *b;
    memset(b, 0, sizeof(*b));
    f->dispatched = g53_cuda_profile_now(g);
    for (int i = 0; i < f->batch.n; i++) {
        G53CudaPromotionTask *t = &f->batch.task[i];
        t->clock = g->profile.clock;
        t->dispatched = f->dispatched;
        f->arg[i] = (G53CudaWorkerArg){f, i};
        if (pthread_create(&f->worker[i], NULL, g53_cuda_upload_worker,
                           &f->arg[i]) != 0) break;
        f->started++;
    }
    f->dispatch_done = g53_cuda_profile_now(g);
    if (f->started != f->batch.n) {
        /* Thread creation refusal is a synchronous end-boundary fallback. */
        for (int i = 0; i < f->started; i++) pthread_join(f->worker[i], NULL);
        for (int i = f->started; i < f->batch.n; i++)
            g53_cuda_upload_task(&f->batch.task[i]);
        if (g->profile.clock) g->profile.flush_end_boundary++;
        double done = g53_cuda_profile_now(g);
        g53_cuda_batch_finish(g, &f->batch, f->dispatched, done,
                              done - f->dispatched, done - f->dispatched);
        free(f);
        return;
    }
    g->promotion_flight = f;
    if (g->profile.clock) {
        G53CudaProfile *p = &g->profile;
        p->flush_end_boundary++;
        p->deferred_end_batches++;
        p->deferred_end_experts += (uint64_t)f->batch.n;
        p->deferred_dispatch_s += f->dispatch_done - f->dispatched;
    }
}
static int g53_cuda_batch_owner(const G53Cuda *g, const G53CudaPromotionBatch *b) {
    int owner = -1;
    for (int i = 0; i < g->ndev; i++)
        if (g->capacity[i] >= g->expert_bytes &&
            b->planned_used[i] <= g->capacity[i] - g->expert_bytes &&
            (owner < 0 || b->planned_used[i] < b->planned_used[owner])) owner = i;
    return owner;
}
static void g53_cuda_batch_reset(const G53Cuda *g, G53CudaPromotionBatch *b) {
    b->planned_bytes = g->bytes;
    for (int i = 0; i < g->ndev; i++) b->planned_used[i] = g->used[i];
}
static void g53_cuda_promote_batched(G53Cuda *g, G53CudaPromotionBatch *b,
                                     int layer, int eid, uint8_t *const *pieces) {
    g53_cuda_join_pending(g, G53_JOIN_POLICY_OR_DROP);
    if (!g->parallel_promote || g->failed || !g->active) {
        g53_cuda_promote(g, layer, eid, pieces); return;
    }
    double plan = g53_cuda_profile_now(g);
    G53CudaExpert *e = &g->experts[layer * g->ne + eid];
    if (g->expert_bytes > g->budget || e->w[0]) {
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_OTHER_SERIAL);
        g53_cuda_promote(g, layer, eid, pieces); return;
    }
    if (e->heat < g->heat_min) {
        if (b->n && g->profile.clock) g->profile.heat_noop_passthrough++;
        g53_cuda_promote(g, layer, eid, pieces); return;
    }
    if (!b->n) g53_cuda_batch_reset(g, b);
    int owner = g53_cuda_batch_owner(g, b);
    if (owner < 0 || b->planned_bytes > g->budget - g->expert_bytes) {
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_CAPACITY_OR_EVICTION);
        g53_cuda_promote(g, layer, eid, pieces); return;
    }
    if (b->owner_busy[owner]) {
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_OWNER_BUSY);
        if (g->failed) { g53_cuda_promote(g, layer, eid, pieces); return; }
        g53_cuda_batch_reset(g, b);
        owner = g53_cuda_batch_owner(g, b);
        if (owner < 0 || g->bytes > g->budget - g->expert_bytes) {
            g53_cuda_promote(g, layer, eid, pieces); return;
        }
    }
    /* Copy while the caller still owns this reusable host-cache slot. */
    size_t wb = (size_t)g->D * (size_t)g->I / 2;
    size_t sb = (size_t)g->D * (size_t)g->I / 64 * sizeof(float);
    size_t host_bytes = 3 * (wb + sb);
    uint8_t *host = malloc(host_bytes);
    if (!host) {
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_STAGING_FAILURE);
        g53_cuda_promote(g, layer, eid, pieces); return;
    }
    G53CudaPromotionTask *t = &b->task[b->n++];
    t->layer = layer; t->eid = eid; t->owner = owner;
    t->device = g->devices[owner]; t->D = g->D; t->I = g->I; t->host = host;
    for (int k = 0; k < 6; k++) {
        size_t n = k & 1 ? sb : wb;
        t->piece[k] = host;
        memcpy(host, pieces[k], n);
        host += n;
    }
    b->owner_busy[owner] = 1;
    b->planned_used[owner] += g->expert_bytes;
    b->planned_bytes += g->expert_bytes;
    if (g->profile.selection_trace)
        fprintf(g->profile.selection_trace, "A,%llu,%d,%d\n",
                (unsigned long long)g->profile.selection_tick, layer, eid);
    if (g->profile.clock) {
        double elapsed = g->profile.clock() - plan;
        g->profile.promotion_plan_s += elapsed;
        g->profile.seconds[G53_PROMOTION] += elapsed;
    }
    if (b->n == g->ndev)
        g53_cuda_batch_flush_reason(g, b, G53_FLUSH_BATCH_FULL);
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
    g53_cuda_join_pending(g, G53_JOIN_NEXT_FFN);
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
    g53_cuda_join_pending(g, G53_JOIN_NEXT_FFN);
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
