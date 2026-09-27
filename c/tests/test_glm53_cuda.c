/* Same tier/numerical test with a fake backend (portable) or -DG53_REAL_CUDA.
 * The latter links backend_cuda.o and MUST execute device tensor matmuls. */
#include <assert.h>
#include "../glm53_cuda.h"

/* Report observed errors, not just a token match. The fixed acceptance envelope
 * is provisional until measured on NVIDIA hardware; never learn a tolerance
 * from the same outputs being tested. Nonfinite values always fail. */
static void compare_projection(const char *label, const float *got, const float *want, int n) {
    const double atol = 0.002, rtol = 0.002;
    double max_abs = 0, max_rel = 0, max_scaled = 0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)got[i] - want[i]);
        double rel = err / fmax(fabs((double)want[i]), 1e-12);
        double scaled = err / (atol + rtol * fabs((double)want[i]));
        if (!isfinite(got[i]) || !isfinite(want[i]) || scaled > 1) bad++;
        max_abs = fmax(max_abs, err); max_rel = fmax(max_rel, rel);
        max_scaled = fmax(max_scaled, scaled);
    }
    printf("%s: max_abs=%.9g max_rel=%.9g max_scaled=%.9g atol=%.9g rtol=%.9g bad=%d\n",
           label, max_abs, max_rel, max_scaled, atol, rtol, bad);
    if (bad) exit(1);
}

static void clamp_ref(float *g, const float *u, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float a = fminf(g[i], limit), b = fmaxf(-limit, fminf(u[i], limit));
        g[i] = a / (1 + expf(-a)) * b;
    }
}
static void project(float *y, const float *x, const unsigned char *w,
                    const float *s, int I, int O) {
    for (int o = 0; o < O; o++) {
        float sum = 0;
        for (int i = 0; i < I; i++) {
            int q = (w[((size_t)o * I + i) / 2] >> ((i & 1) * 4)) & 15;
            sum += x[i] * (q - 8) * s[o * (I / 64) + i / 64];
        }
        y[o] = sum;
    }
}
#ifndef G53_REAL_CUDA
struct ColiCudaTensor { unsigned char *w; float *s; int I, O, device; };
static int live, upload_calls, fail_upload, mat_calls, fail_mat, init_calls, shutdown_calls;
static int fake_ndev, fake_devices[COLI_CUDA_MAX_DEVICES], live_device[COLI_CUDA_MAX_DEVICES];
static size_t fake_free[COLI_CUDA_MAX_DEVICES];
static int fail_mem_device = -1;
int coli_cuda_init(const int *d, int n) {
    init_calls++; fake_ndev = n;
    memcpy(fake_devices, d, (size_t)n * sizeof(*d)); return 1;
}
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_available_device_count(void) { return 2; }
void coli_cuda_shutdown(void) { assert(live == 0); shutdown_calls++; }
int coli_cuda_mem_info(int d, size_t *f, size_t *t) {
    if (d == fail_mem_device) return 0;
    *f = *t = fake_free[d] ? fake_free[d] : 8000000000ULL; return 1;
}
size_t coli_cuda_alloc_footprint(size_t b) { return b; }
size_t coli_cuda_tensor_vram(const ColiCudaTensor *t) { return (size_t)t->I * t->O * 9 / 16; }
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if (t) { assert(live_device[t->device] > 0); live_device[t->device]--;
        free(t->w); free(t->s); free(t); live--; }
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                             int fmt, int I, int O, int d, int gs) {
    assert(fmt == 4 && gs == 64);
    int found = 0;
    for (int i = 0; i < fake_ndev; i++) if (fake_devices[i] == d) found = 1;
    assert(found);
    if (++upload_calls == fail_upload) return 0;
    *t = calloc(1, sizeof(**t)); (*t)->I = I; (*t)->O = O; (*t)->device = d;
    (*t)->w = malloc((size_t)I * O / 2); (*t)->s = malloc((size_t)I * O / 16);
    memcpy((*t)->w, w, (size_t)I * O / 2); memcpy((*t)->s, s, (size_t)I * O / 16);
    live++; live_device[d]++; return 1;
}
int coli_cuda_matmul(ColiCudaTensor **t, float *y, const float *x, const void *w,
                     const float *s, int fmt, int S, int I, int O, int d, int gs) {
    (void)w; (void)s; assert(fmt == 4 && S == 1 && gs == 64);
    assert((*t)->device == d);
    assert(*t && (*t)->I == I && (*t)->O == O);
    if (++mat_calls == fail_mat) { y[0] = 12345; return 0; }
    project(y, x, (*t)->w, (*t)->s, I, O); return 1;
}
#endif
#ifndef G53_CUDA_NO_TEST_MAIN
int main(void) {
    enum { D = 128, I = 64, N = D * I };
#ifdef G53_REAL_CUDA
    puts("GLM53 projection validation: real CUDA backend, D=128 I=64 gs=64");
#else
    puts("GLM53 projection validation: fake backend (not NVIDIA accuracy evidence)");
#endif
    unsigned char weights[3][N/2], saved[N/2];
    float scales[3][N/64], x[D], sg[I], su[I], y[D], expected[D], eg[I], eu[I];
    uint8_t *pieces[6];
    for (int k = 0; k < 3; k++) {
        pieces[k*2] = weights[k]; pieces[k*2+1] = (uint8_t *)scales[k];
        for (int j = 0; j < N/2; j++) weights[k][j] = (unsigned char)(j * 37 + k * 19);
        for (int j = 0; j < N/64; j++) scales[k][j] = 0.02f * (1 + j % 7);
    }
    for (int j = 0; j < D; j++) x[j] = (j % 13 - 6) * 0.7f;
    project(sg, x, weights[0], scales[0], D, I);
    project(su, x, weights[1], scales[1], D, I);
    memcpy(eg, sg, sizeof(eg)); memcpy(eu, su, sizeof(eu));
    clamp_ref(sg, su, I, 0.5f);
    project(expected, sg, weights[2], scales[2], I, D);
    G53Cuda g = {0};
    g53_cuda_init(&g, 1, 2, D, I, 1);
    assert(g.active); /* run with COLI_CUDA=1 COLI_GPU=0 */
    g.budget = g.expert_bytes;
    g53_cuda_heat(&g, 0, 0, 1); g53_cuda_promote(&g, 0, 0, pieces);
    assert(g.resident == 0);
    g53_cuda_heat(&g, 0, 0, 1); g53_cuda_promote(&g, 0, 0, pieces);
    assert(g.resident == 1 && g.bytes <= g.budget);
    /* Simulate RAM-slot reuse. Device copies must still produce original y. */
    memcpy(saved, weights[0], sizeof(saved)); memset(weights[0], 0, sizeof(saved));
    /* Compare the two unclamped projections too: saturation must not conceal
     * a layout/scale error that happens to leave the final output unchanged. */
    assert(coli_cuda_matmul(&g.experts[0].w[0], sg, x, NULL, NULL, 4, 1, D, I, g.device, 64));
    assert(coli_cuda_matmul(&g.experts[0].w[1], su, x, NULL, NULL, 4, 1, D, I, g.device, 64));
    compare_projection("gate", sg, eg, I);
    compare_projection("up", su, eu, I);
    assert(g53_cuda_run(&g, 0, 0, y, x, sg, su, 0.5f, clamp_ref));
    compare_projection("clamped-down", y, expected, D);
    memcpy(weights[0], saved, sizeof(saved));
    assert(g.executed == 1);
    g53_cuda_heat(&g, 0, 1, 3); g53_cuda_promote(&g, 0, 1, pieces);
    assert(!g.experts[0].w[0] && g.experts[1].w[0] && g.resident == 1);
#ifndef G53_REAL_CUDA
    fail_mat = mat_calls + 3; /* down fails after gate/up succeeded */
    assert(!g53_cuda_run(&g, 0, 1, y, x, sg, su, 0.5f, clamp_ref));
    assert(g.failed && g.errors == 1);
#endif
    g53_cuda_close(&g);
#ifndef G53_REAL_CUDA
    /* Each possible partial upload must roll back all allocations. */
    for (int k = 1; k <= 3; k++) {
        g53_cuda_init(&g, 1, 2, D, I, 1);
        fail_upload = upload_calls + k;
        g53_cuda_heat(&g, 0, 0, 2); g53_cuda_promote(&g, 0, 0, pieces);
        assert(live == 0 && g.resident == 0 && g.errors == 1);
        g53_cuda_close(&g);
    }
#endif
    puts("PASS GLM53 CUDA tier: gs64 numerics, owned tensors, heat eviction, cleanup");
    return 0;
}
#endif
