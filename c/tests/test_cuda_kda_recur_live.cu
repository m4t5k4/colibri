/* Opt-in, checkpoint-free recurrence-only oracle. Usage: [0,1,...] or existing
 * COLI_GPU/COLI_GPUS configuration. Rig: all 0..7, then a separate 2,4,6 run.
 * Tests own all transfers/resources; the primitive only launches its kernel. */
#include "../cuda_device_config.h"
#include "cuda_kda_recur_ref.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void check(bool ok,const char *what) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",what); std::exit(1); }
}
static void fault(bool enabled) {
#ifdef _WIN32
    check(!_putenv_s("COLI_GPU_FAIL_AFTER",enabled?"0":""),"fault environment");
#else
    check(!(enabled?setenv("COLI_GPU_FAIL_AFTER","0",1):unsetenv("COLI_GPU_FAIL_AFTER")),"fault environment");
#endif
}
struct Buffers {
    int device,h,kd,vd;
    size_t ns,nk,nv;
    float *s,*out,*q,*k,*v,*decay,*beta;
    float *alloc(size_t n) {
        float *p=(float *)coli_cuda_pipe_alloc(device,n*sizeof(float)); check(p,"device allocation");
        cudaPointerAttributes a;
        check(cudaPointerGetAttributes(&a,p)==cudaSuccess && a.type==cudaMemoryTypeDevice && a.device==device,
              "every numeric buffer belongs to configured CUDA ordinal");
        return p;
    }
    Buffers(int device_,int h_,int kd_,int vd_):device(device_),h(h_),kd(kd_),vd(vd_) {
        nk=(size_t)h*kd; nv=(size_t)h*vd; ns=nk*vd;
        s=alloc(ns); out=alloc(nv); q=alloc(nk); k=alloc(nk);
        v=alloc(nv); decay=alloc(nk); beta=alloc((size_t)h);
    }
    ~Buffers() {
        for (float *p : {s,out,q,k,v,decay,beta}) coli_cuda_pipe_free(device,p);
    }
    Buffers(const Buffers &)=delete;
    Buffers &operator=(const Buffers &)=delete;
};
static void upload(int device,float *p,const std::vector<float> &v) {
    check(coli_cuda_pipe_upload(device,p,v.data(),v.size()*sizeof(float)),"explicit test upload");
}
static void download(int device,const float *p,std::vector<float> &v) {
    check(coli_cuda_pipe_download(device,p,v.data(),v.size()*sizeof(float)),"explicit test download");
}
static float compare(const std::vector<float> &got,const std::vector<float> &ref,const char *what) {
    float worst=0;
    for (size_t i=0; i<ref.size(); ++i) {
        float error=fabsf(got[i]-ref[i])/(1.f+fabsf(ref[i]));
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || error>1e-5f) {
            std::fprintf(stderr,"%s[%zu] gpu=%.9g cpu=%.9g scaled error=%.9g\n",what,i,got[i],ref[i],error);
            check(false,"strict absolute/relative FP32 oracle");
        }
        worst=std::max(worst,error);
    }
    return worst;
}
static int recur(Buffers &b,int device,int h,int kd,int vd,float eps) {
    return coli_cuda_pipe_kda_recur(device,b.s,b.out,b.q,b.k,b.v,b.decay,b.beta,h,kd,vd,eps);
}
static void no_mutation(Buffers &b) {
    std::vector<float> initial(b.ns),sentinel(b.nv,-77.5f),got_s(b.ns),got_y(b.nv);
    kda_recur_initial(initial.data(),initial.size(),0);
    upload(b.device,b.s,initial); upload(b.device,b.out,sentinel);
    float *args[]={b.s,b.out,b.q,b.k,b.v,b.decay,b.beta};
    for (int i=0; i<7; ++i) {
        float *saved=args[i]; args[i]=nullptr;
        check(!coli_cuda_pipe_kda_recur(b.device,args[0],args[1],args[2],args[3],args[4],args[5],args[6],
              b.h,b.kd,b.vd,1e-6f),"NULL numeric pointer rejected before launch");
        args[i]=saved;
    }
    check(!recur(b,-1,b.h,b.kd,b.vd,1e-6f),"missing ordinal/context rejected");
    check(!recur(b,b.device,0,b.kd,b.vd,1e-6f),"zero heads rejected");
    check(!recur(b,b.device,65536,b.kd,b.vd,1e-6f),"head limit rejected");
    check(!recur(b,b.device,b.h,0,b.vd,1e-6f),"zero key dimension rejected");
    check(!recur(b,b.device,b.h,b.kd,0,1e-6f),"zero value dimension rejected");
    check(!recur(b,b.device,b.h,257,b.vd,1e-6f),"key limit rejected");
    check(!recur(b,b.device,b.h,b.kd,257,1e-6f),"value limit rejected");
    for (float eps : {0.f,-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        check(!recur(b,b.device,b.h,b.kd,b.vd,eps),"invalid epsilon rejected");
    fault(true); check(!recur(b,b.device,b.h,b.kd,b.vd,1e-6f),"injected failure before launch"); fault(false);
    check(coli_cuda_pipe_sync(b.device),"completion after rejected operations");
    download(b.device,b.s,got_s); download(b.device,b.out,got_y);
    check(!std::memcmp(got_s.data(),initial.data(),b.ns*sizeof(float)) &&
          !std::memcmp(got_y.data(),sentinel.data(),b.nv*sizeof(float)),"rejected/injected calls leave state/output byte-exact");
}
static void sequence(int device,int heads,int kd,int vd,int zero) {
    enum { STEPS=64 };
    Buffers b(device,heads,kd,vd);
    no_mutation(b);
    std::vector<float> initial(b.ns),cpu(b.ns),got_state(b.ns),first_state(b.ns);
    std::vector<float> q(b.nk),k(b.nk),v(b.nv),decay(b.nk),beta((size_t)heads),ref_out(b.nv),got_out(b.nv);
    std::vector<float> first_trace((size_t)STEPS*b.nv),trace(first_trace.size());
    kda_recur_initial(initial.data(),b.ns,zero); float max_out=0,max_state=0;
    for (int repeat=0; repeat<2; ++repeat) {
        cpu=initial; upload(device,b.s,initial);
        for (int step=0; step<STEPS; ++step) {
            kda_recur_inputs(q.data(),k.data(),v.data(),decay.data(),beta.data(),heads,kd,vd,step);
            kda_recur_ref(cpu.data(),ref_out.data(),q.data(),k.data(),v.data(),decay.data(),beta.data(),heads,kd,vd,1e-6f);
            upload(device,b.q,q); upload(device,b.k,k); upload(device,b.v,v);
            upload(device,b.decay,decay); upload(device,b.beta,beta);
            check(recur(b,device,heads,kd,vd,1e-6f),"resident KDA recurrence launch");
            download(device,b.out,got_out); /* Existing synchronous download waits. */
            max_out=std::max(max_out,compare(got_out,ref_out,"raw out"));
            std::copy(got_out.begin(),got_out.end(),trace.begin()+(size_t)step*b.nv);
        }
        download(device,b.s,got_state); max_state=std::max(max_state,compare(got_state,cpu,"final state"));
        if (!repeat) { first_trace=trace; first_state=got_state; }
        else check(!std::memcmp(first_trace.data(),trace.data(),trace.size()*sizeof(float)) &&
                   !std::memcmp(first_state.data(),got_state.data(),b.ns*sizeof(float)),"GPU repeated trace/state bitwise deterministic");
    }
    std::printf("GPU%d H=%d KD=%d VD=%d %s: state=%zu bytes, 64 steps x2; max scaled out %.3g/state %.3g; deterministic PASS\n",
        device,heads,kd,vd,zero?"zero":"nonzero",b.ns*sizeof(float),max_out,max_state);
}
int main(int argc,char **argv) {
    check(argc<=2,"usage: test_cuda_kda_recur_live [comma-separated CUDA ordinals]");
    int devices[COLI_CUDA_MAX_DEVICES];
    int count=argc==2?parse_cuda_devices(argv[1],devices):coli_cuda_configured_devices(devices);
    check(count>0,"valid explicit/configured process device set");
    fault(false);
    check(coli_cuda_acquire(devices,count),"acquire complete configured process set");
    check(coli_cuda_device_count()==count,"configured device count");
    for (int i=0; i<count; ++i) {
        int device=coli_cuda_device_at(i); check(device==devices[i],"ordinal is not list index");
        char pci[32]; check(cudaDeviceGetPCIBusId(pci,sizeof(pci),device)==cudaSuccess,"physical PCI identity");
        std::printf("CUDA ordinal %d PCI %s\n",device,pci);
        for (int zero=0; zero<=1; ++zero) {
            sequence(device,2,8,8,zero); sequence(device,3,17,11,zero);
            sequence(device,64,128,128,zero);
        }
    }
    /* Every Buffers destructor has run before releasing this sole test lease. */
    coli_cuda_release(); check(!coli_cuda_device_count(),"final backend release");
    std::puts("CUDA KDA recurrence live: PASS (asynchronous post-launch failure recovery remains caller policy)");
    return 0;
}
