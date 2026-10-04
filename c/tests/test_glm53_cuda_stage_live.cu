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
static void transfer(ColiGlm53CudaHandoff &h, Stage &src, Stage &dst, int pattern) {
    cudaPointerAttributes attributes;
    check(cudaPointerGetAttributes(&attributes, h.bounce) == cudaSuccess &&
          attributes.type == cudaMemoryTypeHost, "bounce is CUDA page-locked host memory");
    check(h.bytes == 65536, "exact persistent bounce capacity");
    std::vector<float> input(16384), output(16384);
    for (size_t i = 0; i < input.size(); i++) input[i] = (float)(pattern * 16384 + i);
    check(coli_glm53_cuda_stage_upload(&src.owner, input.data(), h.bytes, 0), "handoff source fill");
    check(coli_glm53_cuda_handoff(&h, &src.owner, &dst.owner, h.bytes), "production pinned handoff");
    check(coli_glm53_cuda_stage_download(&dst.owner, output.data(), h.bytes, 0), "handoff destination read");
    check(!std::memcmp(input.data(), output.data(), h.bytes), "exact handoff contents");
    std::printf("PASS: GPU%d -> pinned host -> GPU%d, %zu bytes exact\n",
                src.owner.cuda_device_ordinal, dst.owner.cuda_device_ordinal, h.bytes);
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
    ColiGlm53CudaHandoff handoff;
    check(coli_glm53_cuda_handoff_open(&handoff, &stages[0].owner, &stages[7].owner), "open persistent handoff");
    void *bounce = handoff.bounce;
    for (int gpu = 0; gpu < 7; gpu++) transfer(handoff, stages[gpu], stages[gpu+1], gpu+1);
    transfer(handoff, stages[0], stages[7], 8);
    check(handoff.bounce == bounce, "same bounce reused for every stage pair");
    const int alternating[][2] = {{0,7},{7,0},{2,6},{6,2},{0,1},{1,0}};
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int pair = 0; pair < 6; ++pair) {
            transfer(handoff, stages[alternating[pair][0]], stages[alternating[pair][1]],
                     20 + repeat * 6 + pair);
            check(handoff.bounce == bounce, "alternating transfers reuse pinned pointer");
        }
    }
    std::puts("PASS: alternating 0->7, 7->0, 2->6, 6->2, 0->1, 1->0 repeated three times");
    device_set(all, 8);
    for (int gpu = 1; gpu < 7; ++gpu) {
        close(stages[gpu]);
        device_set(all, 8);
    }
    std::puts("PASS: all eight stages shared {0,1,2,3,4,5,6,7}; outside owner rejected");
    transfer(handoff, stages[0], stages[7], 9);
    close(stages[0]);
    device_set(all, 8);
    roundtrip(stages[7], 10);
    device_set(all, 8);
    std::puts("PASS: close A/GPU0 first; B/GPU7 survivor wire round-trips with full process set");
    close(stages[7]);
    device_set(all, 8);
    check(!coli_glm53_cuda_handoff(&handoff, &stages[0].owner, &stages[7].owner, 65536),
          "closed stages rejected while handoff keeps backend alive");
    check(handoff.bounce == bounce && handoff.lease_live,
          "stages-first teardown preserves pinned resource and handoff lease");
    std::puts("PASS: stages close first; handoff keeps full backend alive; dead-stage transfer rejected");
    coli_glm53_cuda_handoff_close(&handoff);
    device_set(nullptr, 0);
    check(open(stages[0], 0) == 1 && open(stages[7], 7) == 1,
          "reverse order stages coexist");
    device_set(all, 8);
    roundtrip(stages[0], 11);
    roundtrip(stages[7], 12);
    check(coli_glm53_cuda_handoff_open(&handoff, &stages[7].owner, &stages[0].owner), "reverse handoff open");
    transfer(handoff, stages[7], stages[0], 10);
    coli_glm53_cuda_handoff_close(&handoff);
    check(!handoff.bounce && !handoff.bytes && !handoff.lease_live,
          "handoff close clears resources and lease");
    device_set(all, 8);
    roundtrip(stages[0], 16);
    roundtrip(stages[7], 17);
    std::puts("PASS: handoff closes first; both stage wires survive exactly");
    close(stages[7]);
    device_set(all, 8);
    roundtrip(stages[0], 13);
    device_set(all, 8);
    std::puts("PASS: close B/GPU7 first; A/GPU0 survivor wire round-trips with full process set");
    close(stages[0]);
    coli_glm53_cuda_handoff_close(&handoff);
    device_set(nullptr, 0);
    setenv("COLI_GPUS", "2,4,6", 1);
    check(open(stages[4], 4) == 1, "sparse owner 4 accepted");
    check(open(stages[2], 2) == 1 && open(stages[6], 6) == 1, "sparse transfer stages");
    check(coli_glm53_cuda_handoff_open(&handoff, &stages[2].owner, &stages[6].owner), "sparse handoff open");
    transfer(handoff, stages[2], stages[6], 11);
    close(stages[2]); close(stages[6]);
    device_set(sparse, 3);
    roundtrip(stages[4], 14);
    check(open(rejected, 1) == -1, "sparse owner 1 rejected");
    device_set(sparse, 3);
    close(stages[4]);
    coli_glm53_cuda_handoff_close(&handoff);
    device_set(nullptr, 0);
    std::puts("PASS: sparse {2,4,6}, owner 4 wire round-trips; owner 1 rejected");
    setenv("COLI_GPUS", "0,1,2,3,4,5,6,7", 1);
    check(open(stages[7], 7) == 1, "reacquire full set");
    check(open(stages[0], 0) == 1, "reacquire source stage");
    check(coli_glm53_cuda_handoff_open(&handoff, &stages[0].owner, &stages[7].owner), "recreate bounce");
    transfer(handoff, stages[0], stages[7], 12);
    close(stages[0]);
    device_set(all, 8);
    roundtrip(stages[7], 15);
    device_set(all, 8);
    close(stages[7]);
    coli_glm53_cuda_handoff_close(&handoff);
    device_set(nullptr, 0);
    std::puts("PASS: reacquire {0..7} and round-trip after all stages close");
    std::puts("PASS: live GLM53 eight-GPU stage-wire regression");
}
