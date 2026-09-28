#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include "../compat.h"
#include <time.h>

static double test_clock(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
static void select_devices(const char *list) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", list, 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    unsetenv("GLM53_CUDA_HEAT_MIN");
    unsetenv("GLM53_CUDA_HEAT_MARGIN");
}
int main(void) {
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    enum { D = 64, I = 64, N = D*I };
    uint8_t weights[3][N/2], other_weights[3][N/2], *pieces[6], *other[6];
    float scales[3][N/64], other_scales[3][N/64];
    float x[D], a[2][D], b[D], sg[I], su[I];
    for (int k = 0; k < 3; k++) {
        memset(weights[k], 0x98 + k, sizeof(weights[k]));
        for (int j = 0; j < N/64; j++) scales[k][j] = 0.01f * (k + 1);
        pieces[2*k] = weights[k]; pieces[2*k+1] = (uint8_t *)scales[k];
        memset(other_weights[k], 0x76 + k, sizeof(other_weights[k]));
        for (int j = 0; j < N/64; j++) other_scales[k][j] = 0.015f * (k + 1);
        other[2*k] = other_weights[k]; other[2*k+1] = (uint8_t *)other_scales[k];
    }
    for (int d = 0; d < D; d++) x[d] = (d % 7 - 3) * 0.02f;
    G53CudaPromotionFlight intervals = {0};
    intervals.batch.n = 2;
    intervals.batch.task[0].dispatched = 10;
    intervals.batch.task[0].start_wait = 1;
    intervals.batch.task[0].completed = 15;
    intervals.batch.task[1].dispatched = 10;
    intervals.batch.task[1].start_wait = 2;
    intervals.batch.task[1].completed = 16;
    assert(g53_cuda_worker_overlap(&intervals, 11, 14) == 3);
    assert(g53_cuda_worker_overlap(&intervals, 14, 17) == 2);
    assert(g53_cuda_worker_overlap(&intervals, 16, 17) == 0);
    select_devices("0,1");
    unsetenv("GLM53_CUDA_PARALLEL_PROMOTE");
    G53Cuda serial = {0};
    g53_cuda_init(&serial, 1, 4, D, I, 1);
    assert(!serial.parallel_promote && serial.heat_min == 2 && serial.heat_margin == 1);
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&serial, 0, e, 2);
        g53_cuda_promote(&serial, 0, e, e ? other : pieces);
    }
    int owners[2] = {serial.experts[0].owner, serial.experts[1].owner};
    assert(owners[0] == 0 && owners[1] == 1);
    for (int e = 0; e < 2; e++)
        assert(g53_cuda_run(&serial, 0, e, a[e], x, sg, su, 0.5f, clamp_ref));
    g53_cuda_close(&serial);

    setenv("GLM53_CUDA_PARALLEL_PROMOTE", "1", 1);
    setenv("GLM53_CUDA_PROFILE", "1", 1);
    G53Cuda g = {0}; G53CudaPromotionBatch batch = {0};
    g53_cuda_init(&g, 1, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    fake_upload_delay = 1; fake_upload_peak = 0;
    uint8_t saved_weight = weights[0][0];
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, e ? other : pieces);
        if (!e) {
            assert(!g.experts[e].w[0] && g.resident == 0);
            weights[0][0] ^= 0xff; /* source changes before worker dispatch */
        }
    }
    weights[0][0] = saved_weight;
    assert(!batch.n && !fake_upload_active);
#ifdef _OPENMP
    assert(fake_upload_peak >= 2);
#endif
    assert(g.experts[0].owner == owners[0] && g.experts[1].owner == owners[1]);
    assert(g.uploads == 2 && g.profile.cache_layer[0].promotions == 2);
    assert(g.profile.concurrent_promotions == 2 && g.profile.promotion_batches == 1);
    for (int e = 0; e < 2; e++) {
        assert(g53_cuda_run(&g, 0, e, b, x, sg, su, 0.5f, clamp_ref));
        for (int d = 0; d < D; d++) assert(a[e][d] == b[d]);
    }
    g53_cuda_close(&g);
    fake_upload_delay = 0;

    /* Deferred end boundary: private uploads run during CPU-only work, with
     * no residency or policy publication until the mandatory join. */
    setenv("GLM53_CUDA_PROMOTE_OVERLAP", "1", 1);
    setenv("GLM53_CUDA_PROMOTE_LATE_JOIN", "1", 1);
    g53_cuda_init(&g, 2, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    assert(g.promote_overlap && g.promote_late_join);
    fake_upload_delay = 1;
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, e ? other : pieces);
    }
    /* A two-device full batch was synchronous, so use a three-device tier
     * below for a genuinely deferred two-task end batch. */
    assert(!batch.n && !g.promotion_flight);
    g53_cuda_close(&g);
    select_devices("0,1,2");
    g53_cuda_init(&g, 2, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, e ? other : pieces);
    }
    g53_cuda_batch_dispatch_end(&g, &batch);
    assert(!batch.n && g.promotion_flight && !g.resident && !g.uploads);
    weights[0][0] ^= 0xff; /* reusable host slot changes during upload */
    double cpu_until = test_clock() + 0.008;
    while (test_clock() < cpu_until) { } /* deterministic CPU-only attention stand-in */
    weights[0][0] = saved_weight;
    assert(g.promotion_flight && !g.experts[0].w[0] && !g.experts[1].w[0]);
    g53_cuda_join_pending(&g, G53_JOIN_NEXT_FFN);
    assert(!g.promotion_flight && g.uploads == 2 && g.resident == 2);
    assert(g.experts[0].owner == 0 && g.experts[1].owner == 1);
    assert(g.profile.deferred_end_batches == 1 && g.profile.deferred_end_experts == 2);
    assert(g.profile.deferred_lifetime_s > g.profile.deferred_join_wait_s &&
           g.profile.deferred_hidden_s > 0.004);
    fprintf(stderr, "[glm53-overlap-test] lifetime_s=%.6f join_wait_s=%.6f hidden_s=%.6f\n",
            g.profile.deferred_lifetime_s, g.profile.deferred_join_wait_s,
            g.profile.deferred_hidden_s);
    for (int e = 0; e < 2; e++) {
        assert(g53_cuda_run(&g, 0, e, b, x, sg, su, 0.5f, clamp_ref));
        for (int d = 0; d < D; d++) assert(a[e][d] == b[d]);
    }
    g53_cuda_heat(&g, 1, 0, 2);
    g53_cuda_promote(&g, 1, 0, pieces);
    assert(g.experts[4].owner == 2 && g.resident == 3); /* next-layer owner */
    g53_cuda_close(&g);
    fake_upload_delay = 0;
    setenv("GLM53_CUDA_PROMOTE_LATE_JOIN", "1", 1);
    for (int failing = 0; failing < 2; failing++) {
        memset(fake_device_upload_calls, 0, sizeof(fake_device_upload_calls));
        fake_fail_upload_device = failing; fake_fail_upload_n = 1;
        g53_cuda_init(&g, 2, 4, D, I, 1);
        g53_cuda_profile_enable(&g, test_clock);
        for (int e = 0; e < 2; e++) {
            g53_cuda_heat(&g, 0, e, 2);
            g53_cuda_promote_batched(&g, &batch, 0, e, pieces);
        }
        g53_cuda_batch_dispatch_end(&g, &batch);
        assert(g.promotion_flight && !g.failed && !g.resident);
        double prelude_begin = test_clock();
        cpu_until = test_clock() + 0.002;
        while (test_clock() < cpu_until) { }
        g.late_window = (G53CudaLateWindow){.armed = 1,
            .prelude_begin = prelude_begin, .prelude_end = test_clock()};
        assert(!g.failed); /* CPU-only work did not expose a worker fault. */
        g53_cuda_join_pending(&g, G53_JOIN_NEXT_FFN);
        assert(g.profile.late_join_actual_opportunities == 1 &&
               !g.late_window.armed);
        assert(g.failed && g.errors == 1 && !g.promotion_flight);
        assert(g.resident == (unsigned)failing && g.uploads == (uint64_t)failing);
        assert(!g.experts[1].w[0] && live == 3 * failing);
        assert(!g53_cuda_run(&g, 0, 0, b, x, sg, su, 0.5f, clamp_ref));
        g53_cuda_close(&g); assert(!live && !fake_upload_active);
    }
    unsetenv("GLM53_CUDA_PROMOTE_LATE_JOIN");
    fake_fail_upload_device = -1;
    g53_cuda_init(&g, 2, 4, D, I, 1);
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, pieces);
    }
    g53_cuda_batch_dispatch_end(&g, &batch);
    assert(g.promotion_flight && !g.resident);
    g53_cuda_close(&g); /* joins before tensor/device teardown */
    assert(!live && !fake_upload_active);
    g53_cuda_init(&g, 2, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    g53_cuda_batch_dispatch_end(&g, &batch);
    assert(g.promotion_flight);
    g53_cuda_profile_report(&g, "test");
    assert(!g.promotion_flight && g.resident == 1 &&
           g.profile.deferred_join_reason[G53_JOIN_PROFILE] == 1);
    g53_cuda_close(&g);
    select_devices("0,1");
    unsetenv("GLM53_CUDA_PROMOTE_OVERLAP");

    /* A heat-ineligible attempt is a logical A event, but must leave the
     * independent device-0 upload queued for the device-1 candidate. */
    g53_cuda_init(&g, 1, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    g.profile.selection_trace = tmpfile();
    assert(g.profile.selection_trace);
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    assert(batch.n == 1);
    g53_cuda_heat(&g, 0, 1, 1);
    g53_cuda_promote_batched(&g, &batch, 0, 1, pieces);
    assert(batch.n == 1 && !g.experts[1].w[0] && g.resident == 0);
    g53_cuda_heat(&g, 0, 2, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 2, other);
    assert(!batch.n && g.experts[0].owner == 0 && g.experts[2].owner == 1);
    assert(g.uploads == 2 && g.profile.promotion_batches == 1 &&
           g.profile.batch_size_hist[2] == 1 && !g.profile.batch_size_hist[1]);
    assert(g.profile.heat_noop_passthrough == 1 &&
           g.profile.flush_batch_full == 1 && !g.profile.flush_end_boundary &&
           !g.profile.flush_owner_busy && !g.profile.flush_capacity_or_eviction);
    fflush(g.profile.selection_trace);
    rewind(g.profile.selection_trace);
    char trace_line[256], logical[6] = {0};
    int trace_eid[5] = {0}, trace_n = 0;
    while (fgets(trace_line, sizeof(trace_line), g.profile.selection_trace)) {
        char kind; unsigned long long tick; int layer, eid;
        if (sscanf(trace_line, "%c,%llu,%d,%d", &kind, &tick, &layer, &eid) == 4 &&
            (kind == 'A' || kind == 'P' || kind == 'E')) {
            assert(trace_n < 5 && layer == 0);
            logical[trace_n] = kind; trace_eid[trace_n++] = eid;
        }
    }
    assert(trace_n == 5 && !strcmp(logical, "AAAPP"));
    assert(trace_eid[0] == 0 && trace_eid[1] == 1 && trace_eid[2] == 2 &&
           trace_eid[3] == 0 && trace_eid[4] == 2);
    g53_cuda_close(&g);

    /* A repeated planned owner still flushes; a no-op does not hide it. */
    g53_cuda_init(&g, 1, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    g.capacity[1] = 0;
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    g53_cuda_heat(&g, 0, 1, 1);
    g53_cuda_promote_batched(&g, &batch, 0, 1, pieces);
    g53_cuda_heat(&g, 0, 2, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 2, pieces);
    assert(batch.n == 1 && g.experts[0].w[0] && !g.experts[2].w[0]);
    g53_cuda_batch_flush(&g, &batch);
    assert(g.profile.flush_owner_busy == 1 && g.profile.flush_end_boundary == 1 &&
           g.profile.heat_noop_passthrough == 1 &&
           g.profile.batch_size_hist[1] == 2 && g.uploads == 2);
    assert(g.experts[0].owner == 0 && g.experts[2].owner == 0);
    g53_cuda_close(&g);

    /* Unequal ceilings and a second batch must reproduce serial owner
     * choices, including the tie order after the first three reservations. */
    select_devices("0,1,2");
    fake_free[0] = 2000000000ULL + 6912;
    fake_free[1] = 2000000000ULL + 3 * 6912;
    fake_free[2] = 2000000000ULL + 2 * 6912;
    unsetenv("GLM53_CUDA_PARALLEL_PROMOTE");
    int serial_owner[4];
    g53_cuda_init(&g, 1, 4, D, I, 1);
    for (int e = 0; e < 4; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote(&g, 0, e, pieces);
        serial_owner[e] = g.experts[e].owner;
    }
    assert(serial_owner[0] == 0 && serial_owner[1] == 1 &&
           serial_owner[2] == 2 && serial_owner[3] == 1);
    g53_cuda_close(&g);
    setenv("GLM53_CUDA_PARALLEL_PROMOTE", "1", 1);
    g53_cuda_init(&g, 1, 4, D, I, 1);
    for (int e = 0; e < 4; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, pieces);
    }
    g53_cuda_batch_flush(&g, &batch);
    for (int e = 0; e < 4; e++) assert(g.experts[e].owner == serial_owner[e]);
    assert(g.uploads == 4 && g.resident == 4);
    g53_cuda_close(&g);
    select_devices("0,1");
    fake_free[0] = fake_free[1] = fake_free[2] = 0;

    /* A full global budget forces the pending batch to finish before the
     * original global heat-victim decision is made. */
    g53_cuda_init(&g, 1, 4, D, I, 1);
    g53_cuda_profile_enable(&g, test_clock);
    g.profile.selection_trace = tmpfile();
    assert(g.profile.selection_trace);
    g.budget = g.expert_bytes;
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    assert(batch.n == 1 && !g.experts[0].w[0]);
    g53_cuda_heat(&g, 0, 2, 1);
    g53_cuda_promote_batched(&g, &batch, 0, 2, pieces);
    assert(batch.n == 1 && !g.experts[2].w[0]);
    g53_cuda_heat(&g, 0, 1, 4);
    g53_cuda_promote_batched(&g, &batch, 0, 1, pieces);
    assert(!batch.n && !g.experts[0].w[0] && g.experts[1].w[0]);
    assert(g.profile.eviction_batch_boundaries == 1 && g.profile.evictions == 1);
    assert(g.profile.flush_capacity_or_eviction == 1 &&
           g.profile.batch_size_hist[1] == 1 && !g.profile.flush_end_boundary &&
           g.profile.heat_noop_passthrough == 1);
    fflush(g.profile.selection_trace);
    rewind(g.profile.selection_trace);
    char eviction_trace[7] = {0}; int eviction_eid[6] = {0}, eviction_n = 0;
    while (fgets(trace_line, sizeof(trace_line), g.profile.selection_trace)) {
        char kind; unsigned long long tick; int layer, eid;
        if (sscanf(trace_line, "%c,%llu,%d,%d", &kind, &tick, &layer, &eid) == 4 &&
            (kind == 'A' || kind == 'P' || kind == 'E')) {
            assert(eviction_n < 6 && layer == 0);
            eviction_trace[eviction_n] = kind; eviction_eid[eviction_n++] = eid;
        }
    }
    assert(eviction_n == 6 && !strcmp(eviction_trace, "AAPAEP"));
    assert(eviction_eid[0] == 0 && eviction_eid[1] == 2 &&
           eviction_eid[2] == 0 && eviction_eid[3] == 1 &&
           eviction_eid[4] == 1 && eviction_eid[5] == 1);
    g53_cuda_close(&g);

    /* If the second owner fails, the successful logical prefix remains,
     * while no partial second expert is published. */
    g53_cuda_init(&g, 1, 4, D, I, 1);
    memset(fake_device_upload_calls, 0, sizeof(fake_device_upload_calls));
    fake_fail_upload_device = 1; fake_fail_upload_n = 1;
    for (int e = 0; e < 2; e++) {
        g53_cuda_heat(&g, 0, e, 2);
        g53_cuda_promote_batched(&g, &batch, 0, e, pieces);
    }
    assert(g.failed && g.errors == 1 && g.resident == 1 && g.uploads == 1);
    assert(g.experts[0].w[0] && !g.experts[1].w[0] && !fake_upload_active);
    assert(live == 3 && !live_device[1]);
    assert(!g53_cuda_run(&g, 0, 1, b, x, sg, su, 0.5f, clamp_ref));
    g53_cuda_close(&g); assert(!live);
    fake_fail_upload_device = -1;

    /* The first owner failing also discards a completed, unpublished second
     * owner. Teardown begins only after all upload tasks have joined. */
    g53_cuda_init(&g, 1, 4, D, I, 1);
    memset(fake_device_upload_calls, 0, sizeof(fake_device_upload_calls));
    fake_fail_upload_device = 0; fake_fail_upload_n = 1;
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    g53_cuda_heat(&g, 0, 1, 1);
    g53_cuda_promote_batched(&g, &batch, 0, 1, pieces);
    assert(batch.n == 1 && !g.failed);
    g53_cuda_heat(&g, 0, 2, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 2, pieces);
    assert(g.failed && g.resident == 0 && !g.experts[0].w[0] &&
           !g.experts[1].w[0] && !g.experts[2].w[0] && !live && !fake_upload_active);
    assert(!g53_cuda_run(&g, 0, 2, b, x, sg, su, 0.5f, clamp_ref));
    g53_cuda_close(&g);
    fake_fail_upload_device = -1;

    setenv("GLM53_CUDA_PROMOTE_OVERLAP", "1", 1);
    setenv("GLM53_CUDA_PROMOTE_LATE_JOIN", "1", 1);
    select_devices("0");
    g53_cuda_init(&g, 1, 2, D, I, 1);
    assert(!g.parallel_promote && !g.promote_overlap && !g.promote_late_join);
    g53_cuda_heat(&g, 0, 0, 2);
    g53_cuda_promote_batched(&g, &batch, 0, 0, pieces);
    assert(g.uploads == 1 && !batch.n);
    g53_cuda_close(&g);
    select_devices("0,1");
    unsetenv("GLM53_CUDA_PARALLEL_PROMOTE");
    g53_cuda_init(&g, 1, 2, D, I, 1);
    assert(!g.parallel_promote && !g.promote_overlap && !g.promote_late_join);
    g53_cuda_close(&g);
    setenv("GLM53_CUDA_PARALLEL_PROMOTE", "1", 1);
    unsetenv("GLM53_CUDA_PROMOTE_OVERLAP");
    g53_cuda_init(&g, 1, 2, D, I, 1);
    assert(g.parallel_promote && !g.promote_overlap && !g.promote_late_join);
    g53_cuda_close(&g);
    setenv("GLM53_CUDA_PROMOTE_OVERLAP", "1", 1);
    g53_cuda_init(&g, 1, 2, D, I, 1);
    assert(g.parallel_promote && g.promote_overlap && g.promote_late_join);
    g53_cuda_close(&g);
    unsetenv("GLM53_CUDA_PROMOTE_OVERLAP");
    unsetenv("GLM53_CUDA_PROMOTE_LATE_JOIN");
    puts("PASS GLM53 concurrent promotion planning, lifetime, failure, eviction, one-device fallback");
    return 0;
}
