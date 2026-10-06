/* Opt-in checkpoint-free same-input oracle. No validation readbacks occur
 * in production. Usage: [complete process list]; tests every physical ordinal. */
#include "../cuda_device_config.h"
#include "cuda_kda_post_ref.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void check(bool ok,const char *what){if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
static void fault(bool enabled){
#ifdef _WIN32
    check(!_putenv_s("COLI_GPU_FAIL_AFTER",enabled?"0":""),"fault environment");
#else
    check(!(enabled?setenv("COLI_GPU_FAIL_AFTER","0",1):unsetenv("COLI_GPU_FAIL_AFTER")),"fault environment");
#endif
}
static float *alloc(int owner,size_t n){
    float *p=(float *)coli_cuda_pipe_alloc(owner,n*4);check(p,"device allocation");
    cudaPointerAttributes a;check(cudaPointerGetAttributes(&a,p)==cudaSuccess && a.type==cudaMemoryTypeDevice && a.device==owner,"physical owner");return p;
}
static void sequence(int owner,int h,int d){
    size_t n=(size_t)h*d,ns=n*d;
    std::vector<float> state(ns,0.f),q(n),k(n),v(n),decay(n,-0.13f),beta(h,0.6f),x(d),norm(d),core(n),gate(n),got(n);
    std::vector<float> wa((size_t)d*d,0.f),wb(n*d,0.f);
    for(int j=0;j<d;j++){x[j]=(j%13-6)*0.05f;norm[j]=0.8f+(j%7)*0.03f;wa[(size_t)j*d+j]=0.7f;}
    for(size_t j=0;j<n;j++){q[j]=((int)(j%17)-8)*0.037f;k[j]=((int)(j%19)-9)*0.041f;v[j]=((int)(j%23)-11)*0.053f;wb[j*d+(j%d)]=((int)(j%9)-4)*0.13f;}
    float *sd=alloc(owner,ns),*qd=alloc(owner,n),*kd=alloc(owner,n),*vd=alloc(owner,n),*dd=alloc(owner,n),*bd=alloc(owner,h);
    float *cd=alloc(owner,n),*gd=alloc(owner,n),*nd=alloc(owner,d),*xd=alloc(owner,d),*low=alloc(owner,d);
    ColiCudaTensor *ga=nullptr,*gb=nullptr;
    check(coli_cuda_tensor_upload(&ga,wa.data(),nullptr,0,d,d,owner),"gate A immutable upload");
    check(coli_cuda_tensor_upload(&gb,wb.data(),nullptr,0,d,(int)n,owner),"gate B immutable upload");
    check(coli_cuda_tensor_device(ga)==owner && coli_cuda_tensor_device(gb)==owner,"gate tensor physical owner");
    check(coli_cuda_pipe_upload(owner,sd,state.data(),ns*4) && coli_cuda_pipe_upload(owner,qd,q.data(),n*4) &&
        coli_cuda_pipe_upload(owner,kd,k.data(),n*4) && coli_cuda_pipe_upload(owner,vd,v.data(),n*4) &&
        coli_cuda_pipe_upload(owner,dd,decay.data(),n*4) && coli_cuda_pipe_upload(owner,bd,beta.data(),(size_t)h*4) &&
        coli_cuda_pipe_upload(owner,xd,x.data(),(size_t)d*4) && coli_cuda_pipe_upload(owner,nd,norm.data(),(size_t)d*4),"caller inputs");
    check(coli_cuda_pipe_kda_recur(owner,sd,cd,qd,kd,vd,dd,bd,h,d,d,1e-6f) &&
        coli_cuda_pipe_gemm(ga,low,xd,1) && coli_cuda_pipe_gemm(gb,gd,low,1),"actual device recurrence/gate");
    check(coli_cuda_pipe_sync(owner) && coli_cuda_pipe_download(owner,cd,core.data(),n*4) &&
        coli_cuda_pipe_download(owner,gd,gate.data(),n*4),"validation-only same-input copies");
    check(!coli_cuda_pipe_kda_post(owner,nullptr,gd,nd,h,d,0.003f) &&
        !coli_cuda_pipe_kda_post(owner,cd,nullptr,nd,h,d,0.003f) &&
        !coli_cuda_pipe_kda_post(owner,cd,gd,nullptr,h,d,0.003f) &&
        !coli_cuda_pipe_kda_post(owner,cd,gd,nd,0,d,0.003f) &&
        !coli_cuda_pipe_kda_post(owner,cd,gd,nd,h,257,0.003f) &&
        !coli_cuda_pipe_kda_post(owner,cd,gd,nd,h,d,NAN) &&
        !coli_cuda_pipe_kda_post(-1,cd,gd,nd,h,d,0.003f),"prelaunch rejections");
    fault(true);check(!coli_cuda_pipe_kda_post(owner,cd,gd,nd,h,d,0.003f),"injected prelaunch rejection");fault(false);
    check(coli_cuda_pipe_download(owner,cd,got.data(),n*4) && !std::memcmp(got.data(),core.data(),n*4),"rejections preserve core bytes");
    kda_post_ref(core.data(),gate.data(),norm.data(),h,d,0.003f);
    check(coli_cuda_pipe_kda_post(owner,cd,gd,nd,h,d,0.003f) && coli_cuda_pipe_sync(owner) &&
        coli_cuda_pipe_download(owner,cd,got.data(),n*4),"post caller completion");
    float worst=0.f;
    for(size_t j=0;j<n;j++){float e=fabsf(got[j]-core[j])/(1.f+fabsf(core[j]));check(std::isfinite(got[j]) && e<=1e-5f,"strict scaled same-input post oracle");if(e>worst)worst=e;}
    std::printf("GPU%d post H=%d D=%d actual recurrence/gate inputs, scaled error %.9g: PASS\n",owner,h,d,worst);
    coli_cuda_tensor_free(ga);coli_cuda_tensor_free(gb);
    for(float *p:{sd,qd,kd,vd,dd,bd,cd,gd,nd,xd,low})coli_cuda_pipe_free(owner,p);
}
int main(int argc,char **argv){
    check(argc<=2,"usage: test_cuda_kda_post_live [complete process list]");
    int devices[COLI_CUDA_MAX_DEVICES];int count=argc==2?parse_cuda_devices(argv[1],devices):coli_cuda_configured_devices(devices);
    check(count>0,"configured process list");
    fault(false);
    for(int cycle=0;cycle<2;cycle++){
        check(coli_cuda_acquire(devices,count),"full process lease/reacquire");
        for(int i=0;i<count;i++){sequence(devices[i],1,1);sequence(devices[i],2,3);sequence(devices[i],3,17);sequence(devices[i],64,128);}
        coli_cuda_release();
    }
    return 0;
}
