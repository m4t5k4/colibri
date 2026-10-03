#ifndef COLIBRI_CUDA_DEVICE_CONFIG_H
#define COLIBRI_CUDA_DEVICE_CONFIG_H
#include "backend_cuda.h"
#include <stdlib.h>
#include <limits.h>

/* Shared CLI/Segment process-device selection. Preserve ordered ordinals. */
static inline int parse_cuda_devices(const char *list, int *out) {
    if (!list || !*list) return 0;
    int n = 0; const char *p = list;
    while (*p) {
        char *end = NULL; long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT_MAX || n >= COLI_CUDA_MAX_DEVICES) return 0;
        for (int i = 0; i < n; i++) if (out[i] == (int)v) return 0;
        out[n++] = (int)v; p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (*p++ != ',') return 0;
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) return 0;
    }
    return n;
}
static inline int coli_cuda_configured_devices(int *out) {
    const char *one = getenv("COLI_GPU"), *many = getenv("COLI_GPUS");
    if (one && many) return 0;
    if (many) return parse_cuda_devices(many, out);
    if (one) return parse_cuda_devices(one, out);
    out[0] = 0;
    return 1;
}
#endif
