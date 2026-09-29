/* Synthetic MLA projection timers with the fake CUDA profiler enabled. */
#define GLM53_MLA_NO_MAIN
#include "test_glm53_mla_rows4.c"
#define G53_CUDA_NO_TEST_MAIN
#include "test_glm53_cuda.c"

int main(void) {
    one_token_mla_case(0, 0);
    one_token_mla_case(0, 1);
    one_token_mla_case(1, 0);
    one_token_mla_case(1, 1);
    puts("glm53 MLA projection profile: PASS");
    return 0;
}
