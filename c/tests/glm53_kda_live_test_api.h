/* Private API: linked only by test_glm53_kda_live_lifecycle. */
#ifndef GLM53_KDA_LIVE_TEST_API_H
#define GLM53_KDA_LIVE_TEST_API_H
#include "../backend_cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int injected, committed_generation_unchanged;
    int committed_state_unchanged, committed_window_unchanged;
    int next_state_mutated, next_window_mutated, recovery_pull_exact;
} G53LiveEvidence;
int g53_live_arm(ColiCudaKda *object);
G53LiveEvidence g53_live_evidence(void);
/* Caller-owned copies of the pre-token committed generation and failing input. */
int g53_live_failure_input(float *state, float *window, float *x);
void g53_live_clear(void);
#ifdef __cplusplus
}
#endif
#endif
