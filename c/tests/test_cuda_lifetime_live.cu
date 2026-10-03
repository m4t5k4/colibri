/* Checkpoint-free public backend lease regression on two physical GPUs. */
#include "../backend_cuda.h"
#include <cstdio>
#include <cstdlib>

static void check(int ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void ordered_devices(void) {
    check(coli_cuda_device_count() == 2 && coli_cuda_device_at(0) == 0 &&
          coli_cuda_device_at(1) == 1, "active ordered devices {0,1}");
}
static void execute(ColiCudaTensor **tensor, int device, float value) {
    const float weights[] = {2, 3}, x[] = {value, 1};
    float y = -1;
    check(coli_cuda_matmul(tensor, &y, x, weights, nullptr,
                          0, 1, 2, 1, device, 0), "real backend CUDA matmul");
    check(y == 2 * value + 3, "CUDA matmul result");
    size_t count = 0, bytes = 0;
    coli_cuda_stats(device, &count, &bytes);
    check(count == 1 && bytes > 0, "resident tensor on requested GPU");
    std::printf("GPU%d: matmul %.0f (expected %.0f); resident tensor verified\n",
                device, y, 2 * value + 3);
}
static void close_owner(ColiCudaTensor *tensor) {
    /* matmul returns host output after draining its stream. */
    coli_cuda_tensor_free(tensor);
    coli_cuda_release();
}
static void run(int reverse) {
    const int devices[] = {0, 1}, conflict[] = {1, 0};
    ColiCudaTensor *a = nullptr, *b = nullptr, *again = nullptr;
    std::printf("close %s first:\n", reverse ? "B" : "A");
    check(coli_cuda_acquire(devices, 2), "acquire A {0,1}");
    ordered_devices();
    execute(&a, 0, 1);
    check(coli_cuda_acquire(devices, 2), "acquire B {0,1}");
    execute(&b, 1, 2);
    check(!coli_cuda_acquire(conflict, 2), "reject live {1,0}");
    ordered_devices();
    execute(&a, 0, 3);
    execute(&b, 1, 4);
    std::puts("conflicting {1,0} rejected; both owners still execute");
    close_owner(reverse ? b : a);
    ordered_devices();
    execute(reverse ? &a : &b, reverse ? 0 : 1, 5);
    std::puts("PASS: survivor execution after first release");
    close_owner(reverse ? a : b);
    check(coli_cuda_device_count() == 0, "final release shuts down backend");
    check(coli_cuda_acquire(devices, 2), "reacquire {0,1} after shutdown");
    ordered_devices();
    execute(&again, 1, 6);
    close_owner(again);
    check(coli_cuda_device_count() == 0, "reacquired lifetime shuts down");
    std::puts("PASS: reacquire and execution after final shutdown");
}
int main(void) {
    check(coli_cuda_available_device_count() >= 2, "two physical CUDA GPUs required");
    run(0);
    run(1);
    std::puts("PASS: live CUDA backend lifetime, both owner close orders");
    return 0;
}
