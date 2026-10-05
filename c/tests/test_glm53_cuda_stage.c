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
static int live_buffers, allocations, fail_alloc;
static int live_pinned, pinned_allocs, pinned_frees, fail_pinned, fail_download, fail_upload;
typedef struct { void *pointer; size_t bytes; int owner; } FakeBuffer;
static FakeBuffer buffers[16];
typedef struct {
    char kind;
    int owner, count, devices[COLI_CUDA_MAX_DEVICES];
    size_t bytes;
    const void *resource;
} Event;
static Event events[256];
static size_t event_count;
static Event *record(char kind) {
    assert(event_count < sizeof(events)/sizeof(*events));
    Event *event = &events[event_count++];
    memset(event, 0, sizeof(*event)); event->kind = kind; return event;
}
static int init(const int *devices, int count) {
    assert(devices && count > 0);
    if (fail_init) return 0;
    inits++; return 1;
}
static void shutdown_backend(void) { assert(!live_buffers && !live_pinned); shutdowns++; }
int coli_cuda_device_count(void) { return lifetime.users ? lifetime.count : 0; }
int coli_cuda_device_at(int index) { return index >= 0 && index < coli_cuda_device_count() ? lifetime.devices[index] : -1; }
void *coli_cuda_host_alloc(size_t bytes) {
    Event *event = record('P'); event->bytes = bytes;
    assert(lifetime.users && bytes);
    if (fail_pinned) return NULL;
    void *p = malloc(bytes); assert(p); event->resource = p;
    live_pinned++; pinned_allocs++; return p;
}
void coli_cuda_host_free(void *p) {
    Event *event = record('Q'); event->resource = p;
    assert(lifetime.users && p && live_pinned);
    free(p); live_pinned--; pinned_frees++;
}
int coli_cuda_acquire(const int *devices, int count) {
    Event *event = record('A');
    assert(count > 0 && count <= COLI_CUDA_MAX_DEVICES);
    event->count = count;
    memcpy(event->devices, devices, (size_t)count * sizeof(*devices));
    return coli_cuda_lifetime_acquire(&lifetime, devices, count, init);
}
void coli_cuda_release(void) {
    record('R');
    coli_cuda_lifetime_release(&lifetime, shutdown_backend);
}
void *coli_cuda_pipe_alloc(int device, size_t bytes) {
    Event *event = record('N'); event->owner = device; event->bytes = bytes;
    assert(lifetime.users && bytes);
    int found = 0;
    for (int i = 0; i < lifetime.count; i++) found |= lifetime.devices[i] == device;
    assert(found);
    allocations++;
    if (fail_alloc) return NULL;
    for (size_t i = 0; i < 16; i++) if (!buffers[i].pointer) {
        buffers[i] = (FakeBuffer){malloc(bytes), bytes, device};
        assert(buffers[i].pointer); event->resource = buffers[i].pointer;
        live_buffers++; return buffers[i].pointer;
    }
    assert(0); return NULL;
}
void coli_cuda_pipe_free(int device, void *pointer) {
    Event *event = record('F'); event->owner = device; event->resource = pointer;
    for (size_t i = 0; i < 16; i++) if (buffers[i].pointer == pointer) {
        assert(lifetime.users && buffers[i].owner == device);
        free(pointer); memset(&buffers[i], 0, sizeof(buffers[i]));
        live_buffers--; return;
    }
    assert(0);
}
static void validate_copy(int device, const void *pointer, size_t bytes) {
    for (size_t i = 0; i < 16; i++) if (buffers[i].pointer) {
        uintptr_t start = (uintptr_t)buffers[i].pointer, p = (uintptr_t)pointer;
        if (p >= start && p - start <= buffers[i].bytes) {
            assert(lifetime.users && buffers[i].owner == device);
            assert(bytes <= buffers[i].bytes - (p - start)); return;
        }
    }
    assert(0);
}
int coli_cuda_pipe_upload(int device, void *dst, const void *src, size_t bytes) {
    Event *event = record('U'); event->owner = device; event->resource = dst; event->bytes = bytes;
    validate_copy(device, dst, bytes); if (fail_upload) return 0; memcpy(dst, src, bytes); return 1;
}
int coli_cuda_pipe_download(int device, const void *src, void *dst, size_t bytes) {
    Event *event = record('D'); event->owner = device; event->resource = src; event->bytes = bytes;
    validate_copy(device, src, bytes); if (fail_download) return 0; memcpy(dst, src, bytes); return 1;
}
int coli_cuda_pipe_kda_shortconv(int device, float *window, float *mixed,
        const float *qkv, const float *conv, int channels, int kernel) {
    (void)device; (void)window; (void)mixed; (void)qkv; (void)conv; (void)channels; (void)kernel;
    assert(0 && "ownership probe must not execute ShortConv");return 0;
}
int coli_cuda_pipe_kda_recur(int device, float *state, float *out, const float *q,
        const float *k, const float *v, const float *decay, const float *beta,
        int heads, int kd, int vd, float eps) {
    (void)device; (void)state; (void)out; (void)q; (void)k; (void)v;
    (void)decay; (void)beta; (void)heads; (void)kd; (void)vd; (void)eps;
    assert(0 && "ownership probe must not execute recurrence"); return 0;
}
int coli_cuda_pipe_sync(int device) {
    (void)device; assert(0 && "ownership probe must not sync recurrence"); return 0;
}
static void roundtrip(ColiGlm53CudaStage *s) {
    float input[16384], output[16384];
    assert(s->wire_bytes == sizeof(input));
    for (size_t i = 0; i < 16384; i++) input[i] = (float)(i + s->cuda_device_ordinal);
    assert(coli_glm53_cuda_stage_upload(s, input, sizeof(input), 0));
    assert(coli_glm53_cuda_stage_download(s, output, sizeof(output), 0));
    assert(!memcmp(input, output, sizeof(input)));
    float value = 42, copied = 0;
    assert(coli_glm53_cuda_stage_upload(s, &value, sizeof(value), s->wire_bytes - sizeof(value)));
    assert(coli_glm53_cuda_stage_download(s, &copied, sizeof(copied), s->wire_bytes - sizeof(value)));
    assert(copied == value);
    size_t before = event_count;
    assert(!coli_glm53_cuda_stage_upload(s, input, sizeof(input), 1));
    assert(!coli_glm53_cuda_stage_download(s, output, 1, SIZE_MAX));
    assert(!coli_glm53_cuda_stage_upload(s, input, SIZE_MAX, 1));
    assert(!coli_glm53_cuda_stage_download(s, NULL, 1, 0));
    assert(coli_glm53_cuda_stage_upload(s, NULL, 0, s->wire_bytes));
    assert(event_count == before); /* rejected/empty copies never reach CUDA */
}
static void resource_checks(void) {
    ColiGlm53StagePlan plan = {12, 1, 0};
    ColiGlm53CudaStage a, b;
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0,1", 1);
    for (int reverse = 0; reverse < 2; reverse++) {
        event_count = 0;
        plan.cuda_device_ordinal = 0;
        assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
        assert(coli_glm53_cuda_stage_wire_create(&a, 4, 4096) == 1);
        plan.cuda_device_ordinal = 1;
        assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == 1);
        assert(coli_glm53_cuda_stage_wire_create(&b, 4, 4096) == 1);
        assert(a.wire != b.wire && live_buffers == 2);
        assert(events[0].kind == 'A' && events[1].kind == 'N' &&
               events[2].kind == 'A' && events[3].kind == 'N');
        assert(events[0].count == 2 && events[0].devices[0] == 0 && events[0].devices[1] == 1);
        assert(events[2].count == 2 && events[2].devices[0] == 0 && events[2].devices[1] == 1);
        assert(events[1].owner == 0 && events[3].owner == 1);
        assert(events[1].bytes == 65536 && events[3].bytes == 65536);
        roundtrip(&a); roundtrip(&b);
        ColiGlm53CudaStage *first = reverse ? &b : &a, *last = reverse ? &a : &b;
        coli_glm53_cuda_stage_close(first);
        assert(events[event_count-2].kind == 'F' && events[event_count-1].kind == 'R');
        assert(events[event_count-2].owner == (reverse ? 1 : 0));
        assert(!first->wire && !first->wire_bytes && !first->lease_live);
        assert(live_buffers == 1 && lifetime.users == 1);
        roundtrip(last);
        coli_glm53_cuda_stage_close(last);
        assert(!live_buffers && !lifetime.users);
        assert(events[event_count-2].kind == 'F' && events[event_count-1].kind == 'R');
    }
    puts("resource A-B,D,H-K: independent 65536-byte owner 0/1 wires; bounded copies and both close orders: OK");
    setenv("COLI_GPUS", "2,4,6", 1); plan.cuda_device_ordinal = 4;
    assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
    assert(coli_glm53_cuda_stage_wire_create(&a, 4, 4096) == 1);
    assert(events[event_count-2].count == 3 && events[event_count-2].devices[0] == 2 &&
           events[event_count-2].devices[1] == 4 && events[event_count-2].devices[2] == 6);
    assert(events[event_count-1].owner == 4 && events[event_count-1].resource == a.wire);
    roundtrip(&a);
    int before_alloc = allocations;
    plan.cuda_device_ordinal = 1;
    assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == -1);
    assert(allocations == before_alloc && lifetime.users == 1);
    fail_alloc = 1; plan.cuda_device_ordinal = 6;
    assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == 1);
    assert(coli_glm53_cuda_stage_wire_create(&b, 4, 4096) == -1);
    assert(!b.wire && !b.wire_bytes && b.lease_live && lifetime.users == 2);
    coli_glm53_cuda_stage_close(&b);
    assert(!b.lease_live && b.cuda_device_ordinal == -1);
    assert(lifetime.users == 1 && live_buffers == 1);
    roundtrip(&a); fail_alloc = 0; coli_glm53_cuda_stage_close(&a);
    fail_alloc = 1;
    int before_shutdown = shutdowns;
    assert(coli_glm53_cuda_stage_open(&b, &plan, 12) == 1);
    assert(coli_glm53_cuda_stage_wire_create(&b, 4, 4096) == -1);
    assert(!b.wire && b.lease_live && lifetime.users == 1);
    assert(shutdowns == before_shutdown);
    coli_glm53_cuda_stage_close(&b);
    assert(!b.lease_live && !lifetime.users);
    assert(shutdowns == before_shutdown + 1);
    fail_alloc = 0;
    puts("resource C,F-G: sparse ordinal 4, invalid owner rejected, allocation failure leaves survivor usable: OK");
    before_alloc = allocations; size_t before_events = event_count;
    assert(coli_glm53_cuda_stage_open(&a, NULL, 0) == 0);
    assert(coli_glm53_cuda_stage_wire_create(&a, 4, 4096) == 0);
    assert(!a.wire && !a.wire_bytes && allocations == before_alloc && event_count == before_events);
    const size_t dims[][2] = {{0,4096},{4,0},{SIZE_MAX,2},{1,SIZE_MAX/sizeof(float)+1},{UINT32_MAX,2}};
    plan.cuda_device_ordinal = 4;
    for (size_t i = 0; i < sizeof(dims)/sizeof(*dims); i++) {
        assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
        assert(coli_glm53_cuda_stage_wire_create(&a, dims[i][0], dims[i][1]) == -1);
        assert(!a.wire && a.lease_live && lifetime.users == 1);
        coli_glm53_cuda_stage_close(&a);
        assert(!a.lease_live && !lifetime.users);
    }
    assert(allocations == before_alloc);
    assert(coli_glm53_cuda_stage_open(&a, &plan, 12) == 1);
    assert(coli_glm53_cuda_stage_wire_create(&a, 4, 4096) == 1);
    roundtrip(&a); coli_glm53_cuda_stage_close(&a);
    puts("resource E,L-M: no-plan inactivity, zero/overflow rejection, fresh reacquire and allocation: OK");
}

static void handoff_checks(void) {
    const int pairs[][2] = {{0,1},{1,0},{2,6},{0,7}};
    for (size_t pair = 0; pair < 4; pair++) {
        event_count = 0;
        setenv("COLI_GPUS", pair == 2 ? "2,4,6" : "0,1,2,3,4,5,6,7", 1);
        ColiGlm53CudaStage src, dst, absent;
        ColiGlm53StagePlan plan = {12,1,pairs[pair][0]};
        assert(coli_glm53_cuda_stage_open(&src, &plan, 12) == 1);
        assert(coli_glm53_cuda_stage_wire_create(&src, 4, 4096) == 1);
        plan.cuda_device_ordinal = pairs[pair][1];
        assert(coli_glm53_cuda_stage_open(&dst, &plan, 12) == 1);
        assert(coli_glm53_cuda_stage_wire_create(&dst, 4, 4096) == 1);
        ColiGlm53CudaHandoff h;
        fail_pinned = 1;
        assert(!coli_glm53_cuda_handoff_open(&h, &src, &dst));
        assert(!h.bounce && !h.bytes && !h.lease_live && lifetime.users == 2 && !live_pinned);
        fail_pinned = 0;
        assert(coli_glm53_cuda_handoff_open(&h, &src, &dst));
        assert(h.bytes == 65536 && live_pinned == 1 && lifetime.users == 3);
        int allocs = pinned_allocs;
        void *bounce = h.bounce;
        unsigned char input[65536], output[65536], initial[65536];
        for (size_t i = 0; i < sizeof(input); i++) input[i] = (unsigned char)(i * 17 + pair);
        memset(initial, 0x5a, sizeof(initial));
        assert(coli_glm53_cuda_stage_upload(&src, input, sizeof(input), 0));
        assert(coli_glm53_cuda_stage_upload(&dst, initial, sizeof(initial), 0));
        for (int repeat = 0; repeat < 3; repeat++) {
            size_t before = event_count;
            assert(coli_glm53_cuda_handoff(&h, &src, &dst, repeat ? 65536 : 123));
            assert(events[before].kind == 'D' && events[before].owner == pairs[pair][0]);
            assert(events[before+1].kind == 'U' && events[before+1].owner == pairs[pair][1]);
            assert(coli_glm53_cuda_stage_download(&dst, output, sizeof(output), 0));
            assert(!memcmp(input, output, repeat ? sizeof(input) : 123));
            if (!repeat) assert(!memcmp(initial+123, output+123, sizeof(input)-123));
        }
        assert(h.bounce == bounce && pinned_allocs == allocs);
        assert(coli_glm53_cuda_stage_open(&absent, NULL, 0) == 0);
        size_t before = event_count;
        assert(coli_glm53_cuda_handoff(&h, &src, &dst, 0));
        assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 65537));
        assert(!coli_glm53_cuda_handoff(&h, &src, &dst, SIZE_MAX));
        assert(!coli_glm53_cuda_handoff(&h, &absent, &dst, 1));
        assert(!coli_glm53_cuda_handoff(&h, &src, &absent, 1));
        assert(!coli_glm53_cuda_handoff(&h, NULL, &dst, 1));
        assert(!coli_glm53_cuda_handoff(NULL, &src, &dst, 1));
        ColiGlm53CudaStage bad = dst;
        bad.cuda_device_ordinal = 99;
        assert(!coli_glm53_cuda_handoff(&h, &src, &bad, 1));
        bad = dst; bad.wire = NULL;
        assert(!coli_glm53_cuda_handoff(&h, &src, &bad, 1));
        bad = dst; bad.wire_bytes = 4;
        assert(!coli_glm53_cuda_handoff(&h, &src, &bad, 5));
        assert(!coli_glm53_cuda_handoff(&h, &bad, &dst, 5));
        h.bytes = 4;
        assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 5)); h.bytes = 65536;
        assert(event_count == before);
        ColiGlm53CudaHandoff rejected;
        bad = src; bad.wire_bytes = SIZE_MAX;
        assert(!coli_glm53_cuda_handoff_open(&rejected, &bad, &dst));
        setenv("COLI_GPUS", "0", 1); plan.cuda_device_ordinal = 0;
        assert(coli_glm53_cuda_stage_open(&bad, &plan, 12) == -1);
        assert(lifetime.users == 3);
        fail_download = 1; before = event_count;
        assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 65536));
        assert(event_count == before + 1 && events[before].kind == 'D');
        fail_download = 0; fail_upload = 1; before = event_count;
        assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 65536));
        assert(event_count == before + 2 && events[before+1].kind == 'U');
        fail_upload = 0;
        assert(src.lease_live && dst.lease_live && lifetime.users == 3);
        assert(coli_glm53_cuda_handoff(&h, &src, &dst, 65536));
        int frees = pinned_frees, shutdown_before = shutdowns;
        if (pair % 2) {
            coli_glm53_cuda_handoff_close(&h);
            assert(events[event_count-2].kind == 'Q' && events[event_count-1].kind == 'R');
            assert(lifetime.users == 2 && shutdowns == shutdown_before);
            roundtrip(&src); roundtrip(&dst);
            coli_glm53_cuda_stage_close(&dst); coli_glm53_cuda_stage_close(&src);
        } else {
            coli_glm53_cuda_stage_close(&src);
            assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 1));
            coli_glm53_cuda_stage_close(&dst);
            assert(lifetime.users == 1 && live_pinned == 1 && shutdowns == shutdown_before);
            before = event_count;
            assert(!coli_glm53_cuda_handoff(&h, &src, &dst, 65536));
            assert(event_count == before);
            coli_glm53_cuda_handoff_close(&h);
            assert(events[event_count-2].kind == 'Q' && events[event_count-1].kind == 'R');
        }
        coli_glm53_cuda_handoff_close(&h);
        assert(pinned_frees == frees + 1 && !live_pinned && !lifetime.users);
        assert(shutdowns == shutdown_before + 1);
        printf("handoff %d -> %d: exact/partial bytes, reuse, failures, teardown/reacquire: OK\n",
               pairs[pair][0], pairs[pair][1]);
    }
    puts("handoff A-O: persistent pinned bounce, synchronous D2H/H2D, safe independent leases: PASS");
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
    assert(coli_glm53_cuda_stage_wire_create(&ea->cuda_stage, 4, 4096) == 1);
    assert(coli_glm53_cuda_stage_wire_create(&eb->cuda_stage, 4, 4096) == 1);
    glm53_segment_engine_destroy(ea);
    assert(eb->cuda_stage.lease_live && lifetime.users == 1);
    roundtrip(&eb->cuda_stage);
    glm53_segment_engine_destroy(eb);
    assert(!lifetime.users);
    puts("Production Segment engine teardown preserves survivor and releases final lease: OK");
    resource_checks();
    handoff_checks();
    puts("GLM53 CUDA stage tests: PASS");
    return 0;
}
