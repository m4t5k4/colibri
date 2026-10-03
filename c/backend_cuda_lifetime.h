/* Backend-owned bookkeeping. Callers serialize all helpers. Callbacks are
 * the private raw implementation, never the public locking entry points. */
#ifndef COLIBRI_BACKEND_CUDA_LIFETIME_H
#define COLIBRI_BACKEND_CUDA_LIFETIME_H
#include "backend_cuda.h"
#include <limits.h>
#include <string.h>

typedef struct {
    unsigned users;
    int raw_live, count, devices[COLI_CUDA_MAX_DEVICES];
} ColiCudaLifetime;
typedef int (*ColiCudaLifetimeInit)(const int *, int);
typedef void (*ColiCudaLifetimeShutdown)(void);

static inline int coli_cuda_lifetime_matches(const ColiCudaLifetime *s,
                                             const int *devices, int count) {
    return devices && count == s->count &&
        !memcmp(devices, s->devices, (size_t)count * sizeof(*devices));
}
static inline int coli_cuda_lifetime_acquire(ColiCudaLifetime *s,
        const int *devices, int count, ColiCudaLifetimeInit init) {
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES ||
        s->raw_live || s->users == UINT_MAX) return 0;
    if (s->users) {
        if (!coli_cuda_lifetime_matches(s, devices, count)) return 0;
    } else {
        if (!init(devices, count)) return 0;
        s->count = count;
        memcpy(s->devices, devices, (size_t)count * sizeof(*devices));
    }
    s->users++;
    return 1;
}
static inline void coli_cuda_lifetime_release(ColiCudaLifetime *s,
                                             ColiCudaLifetimeShutdown shutdown) {
    if (!s->users || --s->users) return;
    shutdown();
    memset(s, 0, sizeof(*s));
}
/* Raw init remains idempotent in raw mode. During leases it may only inspect
 * a matching list, without acquiring ownership or invoking the callback. */
static inline int coli_cuda_lifetime_raw_init(ColiCudaLifetime *s,
        const int *devices, int count, ColiCudaLifetimeInit init) {
    if (s->users) return coli_cuda_lifetime_matches(s, devices, count);
    if (!init(devices, count)) return 0;
    s->raw_live = 1;
    return 1;
}
/* Raw shutdown cannot revoke somebody else's lease. */
static inline void coli_cuda_lifetime_raw_shutdown(ColiCudaLifetime *s,
                                              ColiCudaLifetimeShutdown shutdown) {
    if (s->users) return;
    shutdown();
    memset(s, 0, sizeof(*s));
}
#endif
