from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_cuda_kda_shortconv_source import function
ROOT=Path(__file__).resolve().parents[1]
class PostSource(unittest.TestCase):
    def test_scope_and_order(self):
        s=(ROOT/"backend_cuda.cu").read_text()
        op=s[s.index("/* GLM53 output suffix only;"):s.index("/* Full-K, oldest-first history")]
        for text in ("cudaMemcpy","cudaMalloc","cudaFree","Synchronize","coli_cuda_acquire","coli_cuda_release","pipe_gemm","pipe_kda_recur","pipe_kda_shortconv"):
            self.assertNotIn(text,op)
        for text in ("volatile float product","square += product","sqrtf(square / dim + eps)","<<<heads, 256>>>","cudaGetLastError()"):
            self.assertIn(text,op)
        entry=function(s,'extern "C" int coli_cuda_pipe_kda_post(')
        for text in ("fault_injected()","!core_dev","!gate_dev","!onorm_dev","heads > 65535","dim > 256","!std::isfinite(eps)","eps <= 0.f","select_ctx(ctx)"):
            self.assertLess(entry.index(text),entry.index("<<<"))
    def test_optional_wrapper(self):
        s=(ROOT/"backend_loader.c").read_text()
        self.assertIn("RESOLVE_OPT(pipe_kda_post, fn_pipe_kda_post)",s)
        self.assertNotIn("RESOLVE(pipe_kda_post,",s)
        wrapper=function(s,"int coli_cuda_pipe_kda_post(")
        src="""
#include <assert.h>
typedef int (*fn)(int,float*,const float*,const float*,int,int,float);
static struct { int available; fn pipe_kda_post; } g_cuda;
static int calls;
static int fake(int owner,float *c,const float *g,const float *n,int h,int d,float eps){
assert(owner==4 && c && g && n && h==2 && d==3 && eps==0.003f);calls++;return 1;}
"""+wrapper+"""
int main(void){float c=9,g=2,n=1;g_cuda.available=1;
assert(!coli_cuda_pipe_kda_post(4,&c,&g,&n,2,3,0.003f) && c==9 && !calls && g_cuda.available);
g_cuda.pipe_kda_post=fake;g_cuda.available=0;
assert(!coli_cuda_pipe_kda_post(4,&c,&g,&n,2,3,0.003f) && !calls);
g_cuda.available=1;assert(coli_cuda_pipe_kda_post(4,&c,&g,&n,2,3,0.003f) && calls==1);}
"""
        self.compile_run(src,"cc",[])
    def compile_run(self,src,compiler,flags):
        if not shutil.which(compiler):self.skipTest("compiler unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/("test.cpp" if compiler=="g++" else "test.c");p.write_text(src)
            binary=Path(tmp)/"test"
            subprocess.run([compiler,"-O2","-Wall","-Wextra","-Werror",*flags,str(p),"-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
    def test_actual_kernel_host_body(self):
        kernel=function((ROOT/"backend_cuda.cu").read_text(),"__global__ void pipe_kda_post_kernel(")
        src='#include "'+(ROOT/"tests/cuda_kda_post_ref.h").as_posix()+'"\n'+"""
#include <cassert>
#include <vector>
#define __global__
#define __shared__ static
#define __syncthreads() ((void)0)
struct Dim { unsigned x; };static Dim blockIdx,blockDim,threadIdx;
"""+kernel+"""
int main(){const int shapes[][2]={{1,1},{2,3},{3,17},{64,128},{2,256}};blockDim.x=256;
for(const auto &s:shapes){int h=s[0],d=s[1];size_t n=(size_t)h*d;
std::vector<float> core(n),gate(n),norm(d),ref(n);
for(int j=0;j<d;j++)norm[j]=0.7f+(j%7)*0.09f;
for(size_t j=0;j<n;j++){core[j]=((int)(j%23)-11)*0.043f;gate[j]=((int)(j%31)-15)*0.37f;}
if(n>1){gate[0]=-100.f;gate[n-1]=100.f;}
ref=core;kda_post_ref(ref.data(),gate.data(),norm.data(),h,d,0.003f);
for(int head=0;head<h;head++){blockIdx.x=head;
for(unsigned lane=0;lane<256;lane++){threadIdx.x=lane;pipe_kda_post_kernel(core.data(),gate.data(),norm.data(),d,0.003f);}}
for(size_t j=0;j<n;j++)assert(isfinite(core[j]) && fabsf(core[j]-ref[j])<=1e-5f*(1.f+fabsf(ref[j])));
}}
"""
        self.compile_run(src,"g++",["-std=c++11","-ffp-contract=off"])
    def test_actual_entry_prelaunch_rejections(self):
        entry=function((ROOT/"backend_cuda.cu").read_text(),'extern "C" int coli_cuda_pipe_kda_post(')
        entry=entry.replace("pipe_kda_post_kernel<<<heads, 256>>>","fake_launch")
        src="""
#include <cassert>
#include <cmath>
#include <cstdint>
struct DeviceContext {};static DeviceContext ctx;static bool fail;static int launches;
static bool fault_injected(){return fail;}
static DeviceContext *find_ctx(int d){return d==4?&ctx:nullptr;}
static bool select_ctx(DeviceContext *p){return p!=nullptr;}
static int cudaGetLastError(){return 0;}
static int cuda_ok(int e,const char *){return e==0;}
static void fake_launch(float *,const float *,const float *,int,float){launches++;}
"""+entry+"""
int main(){float core=9,gate=2,norm=1;
assert(!coli_cuda_pipe_kda_post(4,nullptr,&gate,&norm,2,3,0.003f));
assert(!coli_cuda_pipe_kda_post(4,&core,nullptr,&norm,2,3,0.003f));
assert(!coli_cuda_pipe_kda_post(4,&core,&gate,nullptr,2,3,0.003f));
for(int h:{-1,0,65536})assert(!coli_cuda_pipe_kda_post(4,&core,&gate,&norm,h,3,0.003f));
for(int d:{-1,0,257})assert(!coli_cuda_pipe_kda_post(4,&core,&gate,&norm,2,d,0.003f));
for(float e:{0.f,-1.f,NAN,INFINITY})assert(!coli_cuda_pipe_kda_post(4,&core,&gate,&norm,2,3,e));
assert(!coli_cuda_pipe_kda_post(1,&core,&gate,&norm,2,3,0.003f));
fail=true;assert(!coli_cuda_pipe_kda_post(4,&core,&gate,&norm,2,3,0.003f));
assert(core==9 && !launches);fail=false;
assert(coli_cuda_pipe_kda_post(4,&core,&gate,&norm,2,3,0.003f) && launches==1 && core==9);}
"""
        src="#include <initializer_list>\n"+src
        self.compile_run(src,"g++",["-std=c++11"])
    def test_integration_contract(self):
        s=(ROOT/"glm53_cuda.h").read_text();op=function(s,"static inline int coli_glm53_cuda_kda_decode(")
        order=[op.index(x) for x in ("coli_cuda_pipe_kda_shortconv(","authority = G53_KDA_UNKNOWN","coli_cuda_pipe_kda_recur(","coli_cuda_pipe_gemm(weights->kga","coli_cuda_pipe_gemm(weights->kgb","coli_cuda_pipe_kda_post(","coli_cuda_pipe_gemm(weights->ko","coli_cuda_pipe_sync(","coli_cuda_pipe_download(","authority = G53_KDA_DEVICE")]
        self.assertEqual(order,sorted(order));self.assertEqual(op.count("coli_cuda_pipe_sync("),1)
        self.assertEqual(op.count("coli_cuda_pipe_download("),1)
        self.assertIn("work->input_x, out, work->input_bytes",op)
        for x in ("alloc(","free(","acquire(","release("):self.assertNotIn(x,op)
        layer=function((ROOT/"glm53.c").read_text(),"static int kda_layer(")
        for name in ("gate","normed","core"):
            self.assertIn("float *"+name+" = gpu ? NULL : malloc",layer)
        self.assertIn("if (!gpu) mm(out, &l->ko",layer)
        setup=function((ROOT/"glm53.c").read_text(),"static int glm53_kda_weights_open(")
        for shape in (".kga, c->kda_hd, c->hidden", ".kgb, c->kda_proj, c->kda_hd", ".ko, c->hidden, c->kda_proj"):
            self.assertIn(shape,setup)
        for text in ("coli_cuda_tensor_bytes", "coli_cuda_tensor_vram", "w->onorm_bytes", "m->layer[i].onorm"):
            self.assertIn(text,setup)
if __name__=="__main__":unittest.main()
