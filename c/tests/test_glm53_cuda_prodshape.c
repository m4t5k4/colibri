/* Production geometry with synthetic bounded gs64 fmt4 weights. Two whole
 * experts suffice to exercise distinct device staging without a checkpoint. */
#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include "../compat.h"

enum { PROD_D = 4096, PROD_I = 2048, PROD_E = 2 };
typedef struct {
    unsigned char *w[3];
    float *s[3];
    uint8_t *pieces[6];
} ProdExpert;

static void prod_clamp(float *gate, const float *up, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i] > limit ? limit : gate[i];
        float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
        gate[i] = g / (1.0f + expf(-g)) * u;
    }
}

static void prod_cpu(const ProdExpert *e, const float *x, float *y,
                     float *gate, float *up) {
    project(gate, x, e->w[0], e->s[0], PROD_D, PROD_I);
    project(up, x, e->w[1], e->s[1], PROD_D, PROD_I);
    prod_clamp(gate, up, PROD_I, 0.5f);
    project(y, gate, e->w[2], e->s[2], PROD_I, PROD_D);
}
static void prod_fixture(ProdExpert *e) {
    for (int a = 0; a < PROD_E; a++) for (int k = 0; k < 3; k++) {
        int in = k == 2 ? PROD_I : PROD_D;
        int out = k == 2 ? PROD_D : PROD_I;
        size_t n = (size_t)in * out;
        e[a].w[k] = malloc(n / 2);
        e[a].s[k] = malloc(n / 64 * sizeof(float));
        assert(e[a].w[k] && e[a].s[k]);
        e[a].pieces[2*k] = e[a].w[k];
        e[a].pieces[2*k+1] = (uint8_t *)e[a].s[k];
        for (size_t i = 0; i < n / 2; i++)
            e[a].w[k][i] = (unsigned char)((i * 37 + a * 71 + k * 19) & 255);
        for (size_t i = 0; i < n / 64; i++)
            e[a].s[k][i] = 0.007f * (float)(1 + (i * 3 + a * 7 + k) % 5);
    }
}
static void prod_free(ProdExpert *e) {
    for (int a = 0; a < PROD_E; a++) for (int k = 0; k < 3; k++) {
        free(e[a].w[k]); free(e[a].s[k]);
    }
}
int main(void) {
#ifdef G53_REAL_CUDA
    if (coli_cuda_available_device_count() < 2) {
        puts("SKIP GLM53 production-shape CUDA oracle: fewer than two devices");
        return 0;
    }
#endif
    ProdExpert e[PROD_E] = {0};
    prod_fixture(e);
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0,1", 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    G53Cuda g = {0};
    g53_cuda_init(&g, 1, PROD_E, PROD_D, PROD_I, 1);
    assert(g.active && g.ndev == 2);
    float *x = malloc((size_t)PROD_D * sizeof(float));
    float *serial = malloc((size_t)PROD_E * PROD_D * sizeof(float));
    float *cpu = malloc((size_t)PROD_E * PROD_D * sizeof(float));
    float *gate = malloc((size_t)PROD_I * sizeof(float));
    float *up = malloc((size_t)PROD_I * sizeof(float));
    assert(x && serial && cpu && gate && up);
    for (int d = 0; d < PROD_D; d++) x[d] = (float)((d * 13) % 37 - 18) * 0.045f;
    for (int a = 0; a < PROD_E; a++) {
        g53_cuda_heat(&g, 0, a, 10);
        g53_cuda_promote(&g, 0, a, e[a].pieces);
        assert(g.experts[a].w[0] && g.experts[a].owner == a);
        project(gate, x, e[a].w[0], e[a].s[0], PROD_D, PROD_I);
        project(up, x, e[a].w[1], e[a].s[1], PROD_D, PROD_I);
        int clamped = 0;
        for (int i = 0; i < PROD_I; i++)
            if (gate[i] > 0.5f || fabsf(up[i]) > 0.5f) clamped++;
        printf("production-shape expert=%d clamped_coordinates=%d\n", a, clamped);
        assert(clamped > 0 && clamped < PROD_I);
        assert(g53_cuda_run(&g, 0, a, serial + (size_t)a * PROD_D,
                            x, gate, up, 0.5f, prod_clamp));
        prod_cpu(&e[a], x, cpu + (size_t)a * PROD_D, gate, up);
        compare_projection("production-shape serial CUDA vs CPU",
                           serial + (size_t)a * PROD_D,
                           cpu + (size_t)a * PROD_D, PROD_D);
    }
    const int one[1] = {1};
    for (int di = 0; di < PROD_E; di++) {
        G53CudaExpert *r = &g.experts[di];
        ColiCudaTensor *gw[1] = {r->w[0]}, *uw[1] = {r->w[1]}, *dw[1] = {r->w[2]};
        assert(g53_cuda_group_issue(&g, di, gw, uw, dw, one, 1, x, 0.5f));
    }
    for (int di = 0; di < PROD_E; di++) {
        const float *grouped = g53_cuda_group_take(&g, di);
        assert(grouped);
        compare_projection("production-shape group vs serial CUDA", grouped,
                           serial + (size_t)di * PROD_D, PROD_D);
        compare_projection("production-shape group vs CPU", grouped,
                           cpu + (size_t)di * PROD_D, PROD_D);
    }
    assert(!g.failed && !g.errors);
    g53_cuda_close(&g);
    free(up); free(gate); free(cpu); free(serial); free(x);
    prod_free(e);
    puts("PASS GLM53 production-shape two-device gs64 fmt4 oracle");
    return 0;
}
