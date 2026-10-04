/* Checkpoint-free production stage-wire allocation regression on eight GPUs. */
#define COLI_CUDA
#include "../glm53_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void device_set(const int *devices, int count) {
    check(coli_cuda_device_count() == count, "process device count");
    for (int i = 0; i < count; ++i)
        check(coli_cuda_device_at(i) == devices[i], "ordered process device set");
}
struct Stage {
    ColiGlm53CudaStage owner;
};
static int open(Stage &s, int gpu) {
    ColiGlm53StagePlan plan = {sizeof(plan), COLI_GLM53_STAGE_PLAN_VERSION, gpu};
    int result = coli_glm53_cuda_stage_open(&s.owner, &plan, sizeof(plan));
    if (result == 1) {
        check(s.owner.lease_live && s.owner.cuda_device_ordinal == gpu,
              "stage retains physical owner");
        check(coli_glm53_cuda_stage_wire_create(&s.owner, 4, 4096) == 1,
              "production residual-stream row allocation");
    }
    else check(!s.owner.lease_live, "rejected stage holds no lease");
    return result;
}
static void roundtrip(Stage &s, float value) {
    int gpu = s.owner.cuda_device_ordinal;
    check(s.owner.wire_bytes == 4 * 4096 * sizeof(float), "exact one-row capacity");
    cudaPointerAttributes attributes;
    check(cudaPointerGetAttributes(&attributes, s.owner.wire) == cudaSuccess,
          "query physical allocation");
    check(attributes.type == cudaMemoryTypeDevice && attributes.device == gpu,
          "wire physically owned by expected CUDA ordinal");
    std::vector<float> input(4 * 4096), output(input.size());
    for (size_t i = 0; i < input.size(); ++i) input[i] = value + (float)i;
    check(coli_glm53_cuda_stage_upload(&s.owner, input.data(), s.owner.wire_bytes, 0),
          "wire H2D");
    check(coli_glm53_cuda_stage_download(&s.owner, output.data(), s.owner.wire_bytes, 0),
          "wire D2H");
    check(!std::memcmp(input.data(), output.data(), s.owner.wire_bytes), "exact full-wire round trip");
    std::printf("GPU%d PASS: persistent %zu-byte residual wire physically verified; round trip exact\n",
                gpu, s.owner.wire_bytes);
}
static void close(Stage &s) {
    coli_glm53_cuda_stage_close(&s.owner);
    check(!s.owner.lease_live && !s.owner.wire && !s.owner.wire_bytes,
          "closed stage has no resource or lease");
}
int main() {
    check(!std::getenv("CUDA_VISIBLE_DEVICES"), "unremapped physical CUDA ordinals");
    check(!std::getenv("CUDA_DEVICE_ORDER"), "default CUDA device order");
    check(coli_cuda_available_device_count() == 8, "exactly eight physical GPUs");
    for (int gpu = 0; gpu < 8; ++gpu) {
        char pci[32];
        check(cudaDeviceGetPCIBusId(pci, sizeof(pci), gpu) == cudaSuccess,
              "physical PCI identity");
        std::printf("CUDA GPU%d physical PCI %s\n", gpu, pci);
    }
    unsetenv("COLI_GPU");
    setenv("COLI_CUDA", "1", 1);
    const int all[] = {0,1,2,3,4,5,6,7}, sparse[] = {2,4,6};
    setenv("COLI_GPUS", "0,1,2,3,4,5,6,7", 1);
    Stage stages[8], rejected;
    for (int gpu = 0; gpu < 8; ++gpu) {
        check(open(stages[gpu], gpu) == 1, "each owner accepted");
        device_set(all, 8);
        roundtrip(stages[gpu], gpu + 1);
        device_set(all, 8);
    }
    check(open(rejected, 8) == -1, "outside process set rejected");
    device_set(all, 8);
    for (int gpu = 1; gpu < 7; ++gpu) {
        close(stages[gpu]);
        device_set(all, 8);
    }
    std::puts("PASS: all eight stages shared {0,1,2,3,4,5,6,7}; outside owner rejected");
    close(stages[0]);
    device_set(all, 8);
    roundtrip(stages[7], 10);
    device_set(all, 8);
    std::puts("PASS: close A/GPU0 first; B/GPU7 survivor wire round-trips with full process set");
    close(stages[7]);
    device_set(nullptr, 0);
    check(open(stages[0], 0) == 1 && open(stages[7], 7) == 1,
          "reverse order stages coexist");
    device_set(all, 8);
    roundtrip(stages[0], 11);
    roundtrip(stages[7], 12);
    close(stages[7]);
    device_set(all, 8);
    roundtrip(stages[0], 13);
    device_set(all, 8);
    std::puts("PASS: close B/GPU7 first; A/GPU0 survivor wire round-trips with full process set");
    close(stages[0]);
    device_set(nullptr, 0);
    setenv("COLI_GPUS", "2,4,6", 1);
    check(open(stages[4], 4) == 1, "sparse owner 4 accepted");
    device_set(sparse, 3);
    roundtrip(stages[4], 14);
    check(open(rejected, 1) == -1, "sparse owner 1 rejected");
    device_set(sparse, 3);
    close(stages[4]);
    device_set(nullptr, 0);
    std::puts("PASS: sparse {2,4,6}, owner 4 wire round-trips; owner 1 rejected");
    setenv("COLI_GPUS", "0,1,2,3,4,5,6,7", 1);
    check(open(stages[7], 7) == 1, "reacquire full set");
    device_set(all, 8);
    roundtrip(stages[7], 15);
    device_set(all, 8);
    close(stages[7]);
    device_set(nullptr, 0);
    std::puts("PASS: reacquire {0..7} and round-trip after all stages close");
    std::puts("PASS: live GLM53 eight-GPU stage-wire regression");
}
