/* Opt-in, checkpoint-free production-sized KDA mirror test on eight GPUs.
 * No KDA arithmetic. Synthetic sessions use absolute layer slots and the same
 * resource helpers as GSession; the CPU fake test covers adapter lifecycles. */
#define COLI_CUDA
#include "../glm53_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct Geometry { size_t heads, dim, proj, kernel; };
static const Geometry production = {64,128,8192,4};
struct Layer {
    ColiGlm53CudaKdaLayer device = {};
    std::vector<float> state, window;
};
struct Session {
    Layer layer[32];
    const ColiGlm53CudaStage *stage = nullptr; /* borrowed, never another lease */
};
static bool owned_kda(int i) { return i >= 7 && i < 13 && (i == 7 || i == 11); }
static void pattern(std::vector<float> &v, int seed) {
    for (size_t i=0; i<v.size(); ++i) v[i] = (float)(seed * 512 + i % 257);
}
static void physical(const ColiGlm53CudaStage &stage, const ColiGlm53CudaKdaLayer &d) {
    for (void *p : {d.state, d.window}) {
        cudaPointerAttributes a;
        check(cudaPointerGetAttributes(&a,p) == cudaSuccess, "query state/window attributes");
        check(a.type == cudaMemoryTypeDevice && a.device == stage.cuda_device_ordinal,
              "session resource allocated on logical owner CUDA ordinal");
    }
}
static void open_session(Session &s, const ColiGlm53CudaStage &stage, int seed) {
    s.stage = &stage;
    for (int i=0; i<32; ++i) {
        Layer &l=s.layer[i];
        if (!owned_kda(i)) { check(!l.device.state && !l.device.window, "unowned/full layer empty"); continue; }
        check(coli_glm53_cuda_kda_open(&stage,&l.device,production.heads,production.dim,
              production.proj,production.kernel), "allocate session KDA pair");
        check(l.device.state_bytes == 4194304 && l.device.window_bytes == 393216,
              "production pair totals 4587520 bytes");
        check(l.device.authority == G53_KDA_HOST, "fresh device contents invalid");
        physical(stage,l.device);
        l.state.resize(l.device.state_bytes/sizeof(float));
        l.window.resize(l.device.window_bytes/sizeof(float));
        pattern(l.state,seed+i); pattern(l.window,seed+i+1);
        check(coli_glm53_cuda_kda_prepare(&stage,&l.device,l.state.data(),l.window.data()), "prepare H2D");
    }
}
static void roundtrip(Session &s, int seed) {
    for (int i=7; i<13; ++i) if (owned_kda(i)) {
        Layer &l=s.layer[i]; void *st=l.device.state, *win=l.device.window;
        for (int n=0; n<3; ++n)
            check(coli_glm53_cuda_kda_prepare(s.stage,&l.device,l.state.data(),l.window.data()), "BOTH prepare no-op");
        check(l.device.state==st && l.device.window==win, "persistent pointer identity");
        std::vector<float> expected_state(l.state.size()), expected_window(l.window.size());
        pattern(expected_state,seed+i); pattern(expected_window,seed+i+1);
        /* Test-only future device-write seam, using existing synchronous copies. */
        check(coli_cuda_pipe_upload(s.stage->cuda_device_ordinal,st,expected_state.data(),l.device.state_bytes), "test device state write");
        check(coli_cuda_pipe_upload(s.stage->cuda_device_ordinal,win,expected_window.data(),l.device.window_bytes), "test device window write");
        l.device.authority=G53_KDA_DEVICE;
        std::fill(l.state.begin(),l.state.end(),-1.f); std::fill(l.window.begin(),l.window.end(),-2.f);
        check(coli_glm53_cuda_kda_prepare(s.stage,&l.device,l.state.data(),l.window.data()), "DEVICE prepare keeps current device");
        check(coli_glm53_cuda_kda_ensure_host(s.stage,&l.device,l.state.data(),l.window.data()), "transactional D2H");
        check(!std::memcmp(l.state.data(),expected_state.data(),l.device.state_bytes) &&
              !std::memcmp(l.window.data(),expected_window.data(),l.device.window_bytes), "full-size exact byte comparison");
        check(l.device.authority==G53_KDA_BOTH && l.device.state==st && l.device.window==win, "coherence and pointers persistent");
    }
}
static void close_session(Session &s) {
    for (Layer &l : s.layer) coli_glm53_cuda_kda_close(s.stage,&l.device);
    s.stage=nullptr;
}
static void open_stage(ColiGlm53CudaStage &stage, int owner) {
    ColiGlm53StagePlan plan={sizeof(plan),COLI_GLM53_STAGE_PLAN_VERSION,owner};
    check(coli_glm53_cuda_stage_open(&stage,&plan,sizeof(plan))==1, "planned stage lease");
    check(coli_glm53_cuda_stage_wire_create(&stage,4,4096)==1, "stage wire under same lease");
}
static void process_set(const int *devices, int count) {
    check(coli_cuda_device_count()==count, "full process count unchanged");
    for (int i=0; i<count; ++i) check(coli_cuda_device_at(i)==devices[i], "ordered process set unchanged");
}
int main() {
    check(!std::getenv("CUDA_VISIBLE_DEVICES") && !std::getenv("CUDA_DEVICE_ORDER"), "unremapped CUDA ordinals");
    check(coli_cuda_available_device_count()==8, "eight physical GPUs");
    unsetenv("COLI_GPU"); setenv("COLI_CUDA","1",1); setenv("COLI_GPUS","0,1,2,3,4,5,6,7",1);
    const int all[]={0,1,2,3,4,5,6,7}, sparse[]={2,4,6};
    ColiGlm53CudaStage stages[8]={};
    for (int gpu=0; gpu<8; ++gpu) open_stage(stages[gpu],gpu);
    for (int gpu=0; gpu<8; ++gpu) {
        Session a,b; open_session(a,stages[gpu],10); open_session(b,stages[gpu],30);
        for (int i=7; i<13; ++i) if (owned_kda(i)) {
            check(a.layer[i].device.state != b.layer[i].device.state &&
                  a.layer[i].device.window != b.layer[i].device.window, "two sessions have distinct recurrence/window pointers");
        }
        roundtrip(a,50); /* Check B physically, even though its authority is BOTH. */
        for (int i=7; i<13; ++i) if (owned_kda(i)) {
            Layer &l=b.layer[i]; std::vector<float> read(l.state.size());
            check(coli_cuda_pipe_download(gpu,l.device.state,read.data(),l.device.state_bytes), "B independent device contents");
            check(!std::memcmp(read.data(),l.state.data(),l.device.state_bytes), "A did not overwrite B");
            read.resize(l.window.size());
            check(coli_cuda_pipe_download(gpu,l.device.window,read.data(),l.device.window_bytes), "B independent window contents");
            check(!std::memcmp(read.data(),l.window.data(),l.device.window_bytes), "A did not overwrite B window");
        }
        close_session(a); process_set(all,8); roundtrip(b,70); close_session(b);
        std::printf("GPU%d PASS: two sessions, two absolute KDA layers each, 4587520 bytes/layer, exact round trips and survivor\n",gpu);
    }
    for (int gpu=7; gpu>=0; --gpu) coli_glm53_cuda_stage_close(&stages[gpu]);
    check(!coli_cuda_device_count(), "final stage releases backend");
    setenv("COLI_GPUS","2,4,6",1); ColiGlm53CudaStage sparse_stage={}; open_stage(sparse_stage,4);
    process_set(sparse,3); Session s; open_session(s,sparse_stage,90); roundtrip(s,100);
    close_session(s); coli_glm53_cuda_stage_close(&sparse_stage);
    check(!coli_cuda_device_count(), "sparse teardown");
    setenv("COLI_GPUS","0,1,2,3,4,5,6,7",1); open_stage(stages[0],0);
    Session again; open_session(again,stages[0],110); roundtrip(again,120); close_session(again);
    coli_glm53_cuda_stage_close(&stages[0]); check(!coli_cuda_device_count(), "reacquire teardown");
    std::puts("PASS: sparse {2,4,6} owner 4 physically verified; complete teardown and reacquire");
    return 0;
}
