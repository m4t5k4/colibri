/* Opt-in, checkpoint-free CUDA ShortConv. Usage: [process-list] [owner].
 * Examples: 0,1,2,3,4,5,6,7 ; 2,4,6 4. No owner means test every configured ordinal. */
#include "../cuda_device_config.h"
#include "cuda_kda_shortconv_ref.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok,const char *what) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1); }
}
static void fault(bool enabled) {
#ifdef _WIN32
    check(!_putenv_s("COLI_GPU_FAIL_AFTER",enabled?"0":""),"fault environment");
#else
    check(!(enabled?setenv("COLI_GPU_FAIL_AFTER","0",1):unsetenv("COLI_GPU_FAIL_AFTER")),"fault environment");
#endif
}
struct Buffers {
    int owner,channels,kernel;
    size_t n;
    float *window,*mixed,*qkv,*conv;
    float *alloc(size_t floats) {
        float *p=(float *)coli_cuda_pipe_alloc(owner,floats*sizeof(float));check(p,"caller device allocation");
        cudaPointerAttributes a;
        check(cudaPointerGetAttributes(&a,p)==cudaSuccess && a.type==cudaMemoryTypeDevice && a.device==owner,
              "all four pointers physically belong to requested ordinal");return p;
    }
    Buffers(int device,int c,int k):owner(device),channels(c),kernel(k),n((size_t)c*k) {
        window=alloc(n);mixed=alloc((size_t)c);qkv=alloc((size_t)c);conv=alloc(n);
    }
    ~Buffers() { for (float *p : {window,mixed,qkv,conv}) coli_cuda_pipe_free(owner,p); }
    Buffers(const Buffers &)=delete;
    Buffers &operator=(const Buffers &)=delete;
};
static void upload(int owner,float *p,const std::vector<float> &v) {
    check(coli_cuda_pipe_upload(owner,p,v.data(),v.size()*sizeof(float)),"explicit caller H2D");
}
static void download(int owner,const float *p,std::vector<float> &v) {
    check(coli_cuda_pipe_download(owner,p,v.data(),v.size()*sizeof(float)),"explicit caller D2H");
}
static float compare(const std::vector<float> &got,const std::vector<float> &ref) {
    float worst=0;
    for (size_t i=0; i<ref.size(); i++) {
        float error=fabsf(got[i]-ref[i])/(1.f+fabsf(ref[i]));
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || error>1e-6f) {
            std::fprintf(stderr,"mixed[%zu] gpu=%.9g cpu=%.9g scaled error=%.9g\n",i,got[i],ref[i],error);
            check(false,"strict 1e-6 absolute/relative mixed oracle");
        }
        worst=std::max(worst,error);
    }
    return worst;
}
static int launch(Buffers &b,int owner,int channels,int kernel) {
    return coli_cuda_pipe_kda_shortconv(owner,b.window,b.mixed,b.qkv,b.conv,channels,kernel);
}
static void no_mutation(Buffers &b) {
    std::vector<float> initial(b.n),conv(b.n),qkv((size_t)b.channels),sentinel((size_t)b.channels,-17.5f);
    std::vector<float> got_w(b.n),got_m((size_t)b.channels);
    kda_shortconv_initial(initial.data(),conv.data(),b.channels,b.kernel);
    for (int c=0; c<b.channels; c++) qkv[c]=kda_shortconv_input(c,0);
    upload(b.owner,b.window,initial);upload(b.owner,b.mixed,sentinel);upload(b.owner,b.qkv,qkv);upload(b.owner,b.conv,conv);
    float *args[]={b.window,b.mixed,b.qkv,b.conv};
    for (int i=0; i<4; i++) {
        float *saved=args[i];args[i]=nullptr;
        check(!coli_cuda_pipe_kda_shortconv(b.owner,args[0],args[1],args[2],args[3],b.channels,b.kernel),"NULL pointer rejected");
        args[i]=saved;
    }
    check(!launch(b,b.owner,0,b.kernel),"zero channels rejected");
    check(!launch(b,b.owner,-1,b.kernel),"negative channels rejected");
    check(!launch(b,b.owner,b.channels,0),"zero kernel rejected");
    check(!launch(b,b.owner,b.channels,-1),"negative kernel rejected");
    check(!launch(b,-1,b.channels,b.kernel),"invalid physical ordinal rejected");
    int absent=1;
    for (;;) {
        bool found=false;
        for (int i=0; i<coli_cuda_device_count(); i++) found |= coli_cuda_device_at(i)==absent;
        if (!found) break;
        absent++;
    }
    check(!launch(b,absent,b.channels,b.kernel),"unconfigured physical ordinal rejected (including sparse owner/index distinction)");
    if ((size_t)INT_MAX > SIZE_MAX/sizeof(float)/(size_t)INT_MAX)
        check(!launch(b,b.owner,INT_MAX,INT_MAX),"size_t geometry overflow rejected");
    fault(true);check(!launch(b,b.owner,b.channels,b.kernel),"existing injected failure before launch");fault(false);
    check(coli_cuda_pipe_sync(b.owner),"caller sync after rejections");
    download(b.owner,b.window,got_w);download(b.owner,b.mixed,got_m);
    check(!std::memcmp(initial.data(),got_w.data(),b.n*sizeof(float)) &&
          !std::memcmp(sentinel.data(),got_m.data(),sentinel.size()*sizeof(float)),"all rejected calls preserve exact window/mixed bytes");
    kda_shortconv_ref(initial.data(),sentinel.data(),qkv.data(),conv.data(),b.channels,b.kernel);
    check(launch(b,b.owner,b.channels,b.kernel),"next call succeeds after clearing fault");
    check(coli_cuda_pipe_sync(b.owner),"caller proves completion after cleared fault");
    download(b.owner,b.window,got_w);download(b.owner,b.mixed,got_m);
    check(!std::memcmp(initial.data(),got_w.data(),b.n*sizeof(float)),"post-fault next window exact");compare(got_m,sentinel);
}
static void ordering_probe(int owner) {
    Buffers b(owner,1,4);
    std::vector<float> window={1,2,4,8},qkv={16},conv={0.125f,-0.25f,0.5f,0.0625f},mixed(1),expected={2,4,8,16};
    upload(owner,b.window,window);upload(owner,b.qkv,qkv);upload(owner,b.conv,conv);
    check(launch(b,owner,1,4),"ordering probe launch");check(coli_cuda_pipe_sync(owner),"ordering probe caller sync");
    download(owner,b.window,window);download(owner,b.mixed,mixed);
    check(!std::memcmp(window.data(),expected.data(),4*sizeof(float)),"shift before convolution; current occupies final tap");
    compare(mixed,std::vector<float>{4.25f/(1.f+expf(-4.25f))});
}
static void sequence(int owner,int channels,int kernel) {
    enum { STEPS=64 };
    Buffers b(owner,channels,kernel);no_mutation(b);
    std::vector<float> initial(b.n),conv(b.n),cpu_window(b.n),gpu_window(b.n),first_window(b.n);
    std::vector<float> qkv((size_t)channels),cpu_mixed((size_t)channels),gpu_mixed((size_t)channels);
    std::vector<float> trace((size_t)STEPS*channels),first_trace(trace.size());
    std::vector<float> initial_mixed((size_t)channels,-17.5f);
    kda_shortconv_initial(initial.data(),conv.data(),channels,kernel);float worst=0;
    for (int repeat=0; repeat<2; repeat++) {
        cpu_window=initial;upload(owner,b.window,initial);upload(owner,b.conv,conv);
        upload(owner,b.mixed,initial_mixed);
        for (int step=0; step<STEPS; step++) {
            for (int c=0; c<channels; c++) qkv[c]=kda_shortconv_input(c,step);
            kda_shortconv_ref(cpu_window.data(),cpu_mixed.data(),qkv.data(),conv.data(),channels,kernel);
            upload(owner,b.qkv,qkv);check(launch(b,owner,channels,kernel),"resident ShortConv launch");
            check(coli_cuda_pipe_sync(owner),"explicit caller completion proof");
            download(owner,b.mixed,gpu_mixed);download(owner,b.window,gpu_window);
            check(!std::memcmp(cpu_window.data(),gpu_window.data(),b.n*sizeof(float)),"full window bitwise exact every step");
            worst=std::max(worst,compare(gpu_mixed,cpu_mixed));
            std::copy(gpu_mixed.begin(),gpu_mixed.end(),trace.begin()+(size_t)step*channels);
        }
        if (!repeat) { first_window=gpu_window;first_trace=trace; }
        else check(!std::memcmp(first_window.data(),gpu_window.data(),b.n*sizeof(float)) &&
                   !std::memcmp(first_trace.data(),trace.data(),trace.size()*sizeof(float)),"same-device repeated window/mixed trace bitwise deterministic");
    }
    std::printf("GPU%d channels=%d K=%d, 64 steps x2; window exact/deterministic; max scaled mixed error %.3g: PASS\n",owner,channels,kernel,worst);
}
int main(int argc,char **argv) {
    check(argc<=3,"usage: test_cuda_kda_shortconv_live [process-list] [physical owner]");
    int devices[COLI_CUDA_MAX_DEVICES];
    int count=argc>=2?parse_cuda_devices(argv[1],devices):coli_cuda_configured_devices(devices);
    check(count>0,"complete explicit/configured process list");
    int only=-1;
    if (argc==3) {
        char *end=nullptr;errno=0;long value=std::strtol(argv[2],&end,10);
        check(!errno && end!=argv[2] && !*end && value>=0 && value<=INT_MAX,"physical owner ordinal");only=(int)value;
        bool found=false;for (int i=0; i<count; i++) found |= devices[i]==only;
        check(found,"owner must be a physical ordinal present in process list");
    }
    fault(false);
    for (int cycle=0; cycle<2; cycle++) {
        check(coli_cuda_acquire(devices,count),"acquire/reacquire complete process list");
        check(coli_cuda_device_count()==count,"backend process count");
        for (int i=0; i<count; i++) {
            int owner=coli_cuda_device_at(i);check(owner==devices[i],"ordinal is not configured-list index");
            if (only>=0 && owner!=only) continue;
            char pci[32];check(cudaDeviceGetPCIBusId(pci,sizeof(pci),owner)==cudaSuccess,"PCI physical identity");
            std::printf("cycle %d physical CUDA ordinal %d PCI %s\n",cycle,owner,pci);
            ordering_probe(owner);sequence(owner,1,1);sequence(owner,2,2);sequence(owner,7,4);
            sequence(owner,257,4);sequence(owner,24576,4);sequence(owner,7,7);
        }
        /* All caller buffers have been freed by their destructors under this lease. */
        coli_cuda_release();check(!coli_cuda_device_count(),"final release before reacquire");
    }
    std::puts("CUDA KDA ShortConv live: PASS; post-launch hardware-fault recovery remains caller policy");return 0;
}
