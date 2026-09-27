#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"
#include "../compat.h"
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
static int expected_shutdown;
static void check_init_cleanup(void) {
    /* exit(1) is the existing init-error contract; distinguish a leaked backend. */
    if (shutdown_calls != expected_shutdown || live) _Exit(99);
}
#endif

int main(void) {
    int devices[COLI_CUDA_MAX_DEVICES];
    assert(g53_cuda_devices("0,1", "9", devices) == 2);
    assert(devices[0] == 0 && devices[1] == 1);
    assert(g53_cuda_devices("3,1,7", NULL, devices) == 3);
    assert(devices[0] == 3 && devices[1] == 1 && devices[2] == 7);
    assert(g53_cuda_devices("", "4", devices) == 1 && devices[0] == 4);
    assert(g53_cuda_devices(NULL, NULL, devices) == 1 && devices[0] == 0);
    const char *bad[] = {"0,", ",0", "0,,1", "0,0", "-1", "x", "0,1x",
        "999999999999999999999999999999", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"};
    for (size_t i = 0; i < sizeof(bad)/sizeof(*bad); i++)
        assert(!g53_cuda_devices(bad[i], NULL, devices));
    assert(!g53_cuda_devices(NULL, "0,1", devices));
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPU", "9", 1);
    setenv("COLI_GPUS", "0,1", 1);
    /* Four experts total, not four per device (each expert is 6912 bytes). */
    setenv("CUDA_EXPERT_GB", "0.000027648", 1);
    G53Cuda g = {0};
    uint8_t weights[2048]; float scales[64], x[64], y[64], sg[64], su[64];
    memset(weights, 0x99, sizeof(weights));
    for (int i = 0; i < 64; i++) { scales[i] = 0.125f; x[i] = 0.1f; }
    uint8_t *pieces[] = {weights, (uint8_t *)scales, weights, (uint8_t *)scales,
                        weights, (uint8_t *)scales};
    g53_cuda_init(&g, 1, 6, 64, 64, 1);
    assert(fake_ndev == 2 && fake_devices[0] == 0 && fake_devices[1] == 1);
    assert(g.budget == 4 * g.expert_bytes);
    for (int i = 0; i < 5; i++) {
        g53_cuda_heat(&g, 0, i, 2); g53_cuda_promote(&g, 0, i, pieces);
        if (i < 4) {
            assert(g.experts[i].owner == i % 2);
            assert(g53_cuda_run(&g, 0, i, y, x, sg, su, 0.5f, clamp_ref));
        }
    }
    assert(g.resident == 4 && g.bytes == g.budget && !g.experts[4].w[0]);
    assert(g.used[0] == 2 * g.expert_bytes && g.used[1] == g.used[0]);
    g53_cuda_heat(&g, 0, 4, 1); g53_cuda_promote(&g, 0, 4, pieces);
    assert(!g.experts[0].w[0] && g.experts[4].w[0] && g.bytes == g.budget);
    g53_cuda_heat(&g, 0, 5, 3); g53_cuda_promote(&g, 0, 5, pieces);
    assert(!g.experts[1].w[0] && g.experts[5].owner == 1);
    assert(g.used[0] == g.used[1]); /* choose again after eviction */
    assert(live_device[0] && live_device[1]);
    g53_cuda_close(&g); g53_cuda_close(&g);
    assert(!live_device[0] && !live_device[1] && shutdown_calls == 1);

    /* Unequal free memory, whole-expert fragmentation, ordered tie breaking. */
    setenv("COLI_GPUS", "3,1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    fake_free[3] = 2000000000ULL + 6912;
    fake_free[1] = 2000000000ULL + 2 * 6912 + 100;
    g53_cuda_init(&g, 1, 6, 64, 64, 1);
    assert(g.budget == 3 * g.expert_bytes + 100);
    for (int i = 0; i < 4; i++) {
        g53_cuda_heat(&g, 0, i, 2); g53_cuda_promote(&g, 0, i, pieces);
    }
    assert(g.experts[0].owner == 0 && g.experts[1].owner == 1 && g.experts[2].owner == 1);
    assert(g.resident == 3 && g.used[0] <= g.capacity[0] && g.used[1] <= g.capacity[1]);
    g53_cuda_heat(&g, 0, 3, 1); g53_cuda_promote(&g, 0, 3, pieces);
    assert(g.experts[3].owner == 0 && g.resident == 3);
    g53_cuda_close(&g);

    /* Roll back each partial upload on the SECOND device, keep CPU fallback. */
    setenv("COLI_GPUS", "0,1", 1); fake_free[1] = 0;
    for (int k = 1; k <= 3; k++) {
        g53_cuda_init(&g, 1, 2, 64, 64, 1);
        g53_cuda_heat(&g, 0, 0, 2); g53_cuda_promote(&g, 0, 0, pieces);
        fail_upload = upload_calls + k;
        g53_cuda_heat(&g, 0, 1, 2); g53_cuda_promote(&g, 0, 1, pieces);
        assert(g.failed && g.errors == 1 && g.resident == 1 && live == 3);
        assert(g.used[1] == 0 && !live_device[1] && g.bytes == g.expert_bytes);
        assert(!g53_cuda_run(&g, 0, 1, y, x, sg, su, 0.5f, clamp_ref));
        g53_cuda_close(&g); g53_cuda_close(&g); assert(!live);
    }
#ifndef _WIN32
    /* Initialization has already created every context when the second query
     * or budget validation fails. Each error must shut the backend down. */
    for (int mode = 0; mode < 2; mode++) {
        pid_t child = fork(); assert(child >= 0);
        if (!child) {
            expected_shutdown = shutdown_calls + 1;
            assert(!atexit(check_init_cleanup));
            if (mode == 0) fail_mem_device = 1;
            else setenv("CUDA_EXPERT_GB", "invalid", 1);
            g53_cuda_init(&g, 1, 2, 64, 64, 1);
            _Exit(98);
        }
        int status;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    }
#endif
    puts("PASS GLM53 multi-device parsing, placement, total budget, headroom, rollback, cleanup");
    return 0;
}
