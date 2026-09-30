/* Compile the actual backend here, never link this object into glm53.
 * Only the step and pull entry points are wrapped; all kernels/resources and
 * expert operations are the production implementation. */
#define coli_cuda_kda_step g53_live_original_step
#define coli_cuda_kda_get_state g53_live_original_get_state
#include "../backend_cuda.cu"
#undef coli_cuda_kda_step
#undef coli_cuda_kda_get_state
#include "glm53_kda_live_test_api.h"
#include <vector>
#include <cstring>

static ColiCudaKda *armed, *failed;
static G53LiveEvidence evidence;
static std::vector<float> prior_state, prior_window, failing_x;

static bool observe_next(ColiCudaKda *a, int generation,
        std::vector<float> &state, std::vector<float> &window) {
    KdaDeviceScope device(a->device);
    return device.entered &&
        kda_cuda_ok(cudaMemcpyAsync(state.data(), a->state[1 - generation],
            state.size() * sizeof(float), cudaMemcpyDeviceToHost, a->stream),
            "live_observe_next_state", a->device) &&
        kda_cuda_ok(cudaMemcpyAsync(window.data(), a->window[1 - generation],
            window.size() * sizeof(float), cudaMemcpyDeviceToHost, a->stream),
            "live_observe_next_window", a->device) &&
        kda_cuda_ok(cudaStreamSynchronize(a->stream), "live_observe_wait", a->device) &&
        device.restore();
}
static unsigned long long buffer_digest(const std::vector<float> &values) {
    const unsigned char *bytes = (const unsigned char *)values.data();
    unsigned long long h = 14695981039346656037ULL;
    for (size_t i = 0; i < values.size() * sizeof(float); i++)
        h = (h ^ bytes[i]) * 1099511628211ULL;
    return h;
}

extern "C" void g53_live_clear(void) {
    armed = failed = nullptr;
    evidence = {};
    prior_state.clear(); prior_window.clear(); failing_x.clear();
}
extern "C" int g53_live_arm(ColiCudaKda *object) {
    if (!object || armed || failed) return 0;
    armed = object;
    return 1;
}
extern "C" G53LiveEvidence g53_live_evidence(void) { return evidence; }
extern "C" int g53_live_failure_input(float *state, float *window, float *x) {
    if (!evidence.injected || !state || !window || !x) return 0;
    std::memcpy(state, prior_state.data(), prior_state.size() * sizeof(float));
    std::memcpy(window, prior_window.data(), prior_window.size() * sizeof(float));
    std::memcpy(x, failing_x.data(), failing_x.size() * sizeof(float));
    return 1;
}
extern "C" int coli_cuda_kda_get_state(ColiCudaKda *a, float *s, float *w) {
    int ok = g53_live_original_get_state(a, s, w);
    if (a == failed && ok) {
        evidence.recovery_pull_exact =
            !std::memcmp(s, prior_state.data(), prior_state.size() * sizeof(float)) &&
            !std::memcmp(w, prior_window.data(), prior_window.size() * sizeof(float));
        failed = nullptr; /* Observe the production recovery pull only once. */
    }
    return ok;
}
extern "C" int coli_cuda_kda_step(ColiCudaKda *a, float *out, const float *x,
        const float *decay, const float *beta, const float *gate, float eps) {
    if (a != armed) return g53_live_original_step(a, out, x, decay, beta, gate, eps);
    armed = nullptr; /* Exactly one injected failure. */
    prior_state.resize((size_t)a->H * a->D * a->D);
    prior_window.resize((size_t)3 * a->P * a->K);
    failing_x.assign(x, x + a->hidden);
    if (!g53_live_original_get_state(a, prior_state.data(), prior_window.data()))
        return 0;
    int generation = a->committed;
    std::vector<float> next_before_state(prior_state.size()), next_before_window(prior_window.size());
    if (!observe_next(a, generation, next_before_state, next_before_window)) return 0;
    bool ok = kda_step(a, out, x, decay, beta, gate,
                       KDA_FAIL_AFTER_RECURRENCE, eps);
    std::vector<float> s(prior_state.size()), w(prior_window.size());
    if (ok || !g53_live_original_get_state(a, s.data(), w.data())) return 0;
    evidence.committed_generation_unchanged = a->committed == generation;
    evidence.committed_state_unchanged =
        !std::memcmp(s.data(), prior_state.data(), s.size() * sizeof(float));
    evidence.committed_window_unchanged =
        !std::memcmp(w.data(), prior_window.data(), w.size() * sizeof(float));
    std::vector<float> next_after_state(prior_state.size()), next_after_window(prior_window.size());
    if (!observe_next(a, generation, next_after_state, next_after_window)) return 0;
    evidence.next_state_mutated =
        std::memcmp(next_after_state.data(), next_before_state.data(), s.size() * sizeof(float)) != 0;
    evidence.next_window_mutated =
        std::memcmp(next_after_window.data(), next_before_window.data(), w.size() * sizeof(float)) != 0;
    evidence.injected = 1; failed = a;
    std::fprintf(stderr, "[test-only-kda-injection] device=%d fault=AFTER_NEXT_BEFORE_COMMIT expected_failures=1\n", a->device);
    std::fprintf(stderr, "[test-only-kda-generation] committed_before=%d committed_after=%d next_state_before_fnv64=%016llx next_state_after_fnv64=%016llx next_window_before_fnv64=%016llx next_window_after_fnv64=%016llx next_state_mutated=%d next_window_mutated=%d\n",
        generation, a->committed, buffer_digest(next_before_state), buffer_digest(next_after_state),
        buffer_digest(next_before_window), buffer_digest(next_after_window),
        evidence.next_state_mutated, evidence.next_window_mutated);
    return 0;
}
