/* Production sparse FFN with the existing fake asynchronous CUDA fixture. */
#define main phase2_test_main
#include "test_glm53_cuda_phase2.c"
#undef main

static void run_profile(int enabled, int fault, float *out, uint64_t *executed,
                        uint64_t *fallback, uint64_t *uploads) {
    FixtureModel f;
    model_fixture(&f, "0,1");
    setenv("GLM53_CUDA_PROFILE", "1", 1);
    g53_cuda_profile_enable(&f.m.cuda, now_s);
    f.m.phase3b.enabled = enabled;
    f.m.cuda.decode_call = 1;
    FILE *capture = tmpfile();
    assert(capture);
    int saved = dup(fileno(stderr));
    assert(saved >= 0 && dup2(fileno(capture), fileno(stderr)) >= 0);
    g53_phase3b_report(&f.m, "start");
    float x[ORACLE_D];
    for (int d = 0; d < ORACLE_D; d++) x[d] = ((d * 7) % 19 - 9) * 0.09f;
    fail_group_take_device = fault ? 1 : -1;
    ffn_layer(&f.m, &f.layer, 0, x, 1, out);
    fail_group_take_device = -1;
    f.m.cuda.profile.decode_tokens++;
    g53_phase3b_report(&f.m, "decode");
    g53_phase3b_report(&f.m, "final");
    fflush(stderr);
    assert(dup2(saved, fileno(stderr)) >= 0);
    close(saved);
    rewind(capture);
    char line[4096];
    int profiles = 0;
    while (fgets(line, sizeof(line), capture)) {
        if (strncmp(line, "[glm53-phase3b-profile]", 23)) continue;
        profiles++;
        if (!fault) fputs(line, stdout); /* parser validates real emitted fields */
    }
    assert(profiles == (enabled ? 3 : 0));
    fclose(capture);
    assert(f.m.phase3b.gpu_group_attempts == (uint64_t)enabled);
    assert(f.m.phase3b.gpu_group_successes == (uint64_t)(enabled && !fault));
    assert(f.m.phase3b.sparse_router_s >= 0);
    assert(enabled || f.m.phase3b.sparse_routed_tier_s == 0);
    assert(f.m.kda_tier.calls == 0 && f.m.kda_tier.pushes == 0);
    *executed = f.m.cuda.executed;
    *fallback = f.m.cuda.fallback;
    *uploads = f.m.cuda.uploads;
    model_fixture_close(&f);
    unsetenv("GLM53_CUDA_PROFILE");
}

int main(void) {
    fixture();
    float off[ORACLE_D] = {0}, on[ORACLE_D] = {0};
    uint64_t off_executed, off_fallback, off_uploads;
    uint64_t on_executed, on_fallback, on_uploads;
    for (int fault = 0; fault < 2; fault++) {
        memset(off, 0, sizeof(off)); memset(on, 0, sizeof(on));
        run_profile(0, fault, off, &off_executed, &off_fallback, &off_uploads);
        run_profile(1, fault, on, &on_executed, &on_fallback, &on_uploads);
        assert(!memcmp(off, on, sizeof(off)));
        assert(off_executed == on_executed && off_fallback == on_fallback &&
               off_uploads == on_uploads);
    }
    puts("PASS Phase3B: off/on identical FFN output and CUDA execution counters");
    return 0;
}
