/* CPU-only resource/lifetime oracle using production bookkeeping. */
#include "../backend_cuda_lifetime.h"
#include <assert.h>
#include <stdio.h>

static int init_calls, shutdown_calls, live, resources, fail_init;
static int active_count, active[COLI_CUDA_MAX_DEVICES];
static int fake_init(const int *devices, int count) {
    init_calls++;
    if (fail_init || !devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (live) return count == active_count &&
        !memcmp(devices, active, (size_t)count * sizeof(*devices));
    memcpy(active, devices, (size_t)count * sizeof(*devices));
    active_count = count; live = 1;
    return 1;
}
static void fake_shutdown(void) {
    assert(resources == 0); /* DN objects and tensors were freed first. */
    shutdown_calls++; live = 0; active_count = 0;
}
typedef struct { int dn, tensor, drained; } Owner;
static Owner owner_open(void) {
    Owner o = {1, 1, 0}; assert(live); resources += 2; return o;
}
static void owner_use(const Owner *o) { assert(live && o->dn && o->tensor); }
static void owner_close(ColiCudaLifetime *s, Owner *o) {
    o->drained = 1;
    assert(live && o->drained && o->dn && o->tensor);
    o->dn = 0; resources--; /* free DN before tensors, before release */
    assert(live && !o->dn);
    o->tensor = 0; resources--;
    coli_cuda_lifetime_release(s, fake_shutdown);
}
static void reset(void) {
    assert(!resources && !live);
    init_calls = shutdown_calls = fail_init = 0;
}
static void two_owners(int reverse) {
    ColiCudaLifetime s = {0, 0, 0, {0}}; const int dev[] = {3, 1}, conflict[] = {1, 3};
    reset();
    assert(coli_cuda_lifetime_acquire(&s, dev, 2, fake_init));
    assert(init_calls == 1 && s.users == 1 && !s.raw_live);
    assert(s.count == 2 && s.devices[0] == 3 && s.devices[1] == 1);
    Owner a = owner_open();
    assert(coli_cuda_lifetime_acquire(&s, dev, 2, fake_init));
    Owner b = owner_open();
    assert(init_calls == 1 && s.users == 2);
    assert(!coli_cuda_lifetime_acquire(&s, conflict, 2, fake_init));
    assert(!coli_cuda_lifetime_acquire(&s, dev, 1, fake_init));
    assert(!coli_cuda_lifetime_acquire(&s, NULL, 2, fake_init));
    assert(!coli_cuda_lifetime_acquire(&s, dev, 0, fake_init));
    assert(!coli_cuda_lifetime_acquire(&s, dev, COLI_CUDA_MAX_DEVICES+1, fake_init));
    assert(s.users == 2 && init_calls == 1 && s.devices[0] == 3);
    /* Legacy calls cannot acquire/revoke leased ownership. */
    assert(coli_cuda_lifetime_raw_init(&s, dev, 2, fake_init));
    assert(!coli_cuda_lifetime_raw_init(&s, conflict, 2, fake_init));
    coli_cuda_lifetime_raw_shutdown(&s, fake_shutdown);
    assert(s.users == 2 && init_calls == 1 && !shutdown_calls);
    s.users = UINT_MAX;
    assert(!coli_cuda_lifetime_acquire(&s, dev, 2, fake_init));
    assert(s.users == UINT_MAX && init_calls == 1 && live);
    s.users = 2;
    Owner *first = reverse ? &b : &a, *last = reverse ? &a : &b;
    owner_close(&s, first);
    assert(s.users == 1 && !shutdown_calls);
    owner_use(last);
    owner_close(&s, last);
    assert(!s.users && !s.count && !s.raw_live && !s.devices[0]);
    assert(shutdown_calls == 1 && init_calls == 1 && !live);
    coli_cuda_lifetime_release(&s, fake_shutdown);
    assert(shutdown_calls == 1);
    assert(coli_cuda_lifetime_acquire(&s, conflict, 2, fake_init));
    assert(init_calls == 2 && s.users == 1 && s.devices[0] == 1);
    coli_cuda_lifetime_release(&s, fake_shutdown);
    assert(shutdown_calls == 2 && !s.users && !live);
    printf("close %s first: survivor usable; init=2 shutdown=2 users=0 (two lifetimes)\n",
           reverse ? "B" : "A");
}
int main(void) {
    two_owners(0); two_owners(1);
    reset();
    ColiCudaLifetime s = {0, 0, 0, {0}}; const int dev[] = {0}, conflict[] = {1};
    fail_init = 1;
    assert(!coli_cuda_lifetime_acquire(&s, dev, 1, fake_init));
    assert(init_calls == 1 && !s.users && !s.count && !s.raw_live && !live);
    coli_cuda_lifetime_release(&s, fake_shutdown);
    assert(!shutdown_calls);
    fail_init = 0;
    assert(coli_cuda_lifetime_raw_init(&s, dev, 1, fake_init));
    assert(coli_cuda_lifetime_raw_init(&s, dev, 1, fake_init));
    assert(!coli_cuda_lifetime_raw_init(&s, conflict, 1, fake_init));
    assert(s.raw_live && !s.users);
    int before = init_calls;
    assert(!coli_cuda_lifetime_acquire(&s, dev, 1, fake_init));
    assert(init_calls == before && s.raw_live && live);
    coli_cuda_lifetime_release(&s, fake_shutdown);
    assert(live && !shutdown_calls); /* release is not raw shutdown */
    coli_cuda_lifetime_raw_shutdown(&s, fake_shutdown);
    assert(!live && !s.raw_live && shutdown_calls == 1);
    assert(coli_cuda_lifetime_acquire(&s, dev, 1, fake_init));
    coli_cuda_lifetime_release(&s, fake_shutdown);
    assert(shutdown_calls == 2 && !s.users);
    puts("PASS: failure, zero release, overflow, ordered identity, raw/lease isolation and resource ordering");
    return 0;
}
