"""Model/toolkit-free guards for the additive full-K ShortConv seam."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(text, marker):
    begin = text.index(marker)
    start = text.index("{", begin)
    depth = 1
    at = start + 1
    while depth:
        depth += (text[at] == "{") - (text[at] == "}")
        at += 1
    return text[begin:at]


class ShortConvSource(unittest.TestCase):
    def test_device_only_order_and_validation(self):
        source = (ROOT / "backend_cuda.cu").read_text()
        op = source[source.index("/* Full-K, oldest-first history"):
                    source.index("/* Recurrence only; caller owns every device buffer.")]
        for forbidden in ("cudaMemcpy", "cudaMalloc", "cudaFree", "malloc(", "free(",
                          "cudaDeviceSynchronize", "cudaStreamSynchronize", "coli_cuda_pipe_sync",
                          "coli_cuda_acquire", "coli_cuda_release", "dn_conv_kernel", "ColiCudaDn",
                          "__syncthreads", "__expf", "half", "double"):
            self.assertNotIn(forbidden, op)
        self.assertIn("(size_t)blockIdx.x * blockDim.x + threadIdx.x", op)
        self.assertIn("base = channel * (size_t)kernel", op)
        self.assertIn("tap < kernel - 1", op)
        self.assertIn("history[tap] = history[tap + 1]", op)
        self.assertLess(op.index("history[kernel - 1] = qkv[channel]"), op.index("float sum = 0.f"))
        self.assertIn("tap = 0; tap < kernel; tap++", op)
        self.assertIn("volatile float product", op)
        self.assertLess(op.index("sum += product"), op.index("sum / (1.f + expf(-sum))"))
        entry = op[op.index('extern "C" int coli_cuda_pipe_kda_shortconv'):]
        for required in ("fault_injected()", "!window_dev", "!mixed_dev", "!qkv_dev", "!conv_w_dev",
                         "channels < 1", "kernel < 1", "SIZE_MAX / sizeof(float) / (size_t)kernel",
                         "select_ctx(find_ctx(device))"):
            self.assertLess(entry.index(required), entry.index("<<<"))
        self.assertIn("(channels - 1) / 256 + 1", entry)
        self.assertIn("<<<blocks, 256>>>", entry)
        self.assertIn("cudaGetLastError()", entry)
        header = (ROOT / "backend_cuda.h").read_text()
        self.assertIn("COLI_CUDA_DLLEXPORT int coli_cuda_pipe_kda_shortconv", header)
        for text in ("FULL K samples", "K=1 is supported", "launch accepted", "preserves window/mixed"):
            self.assertIn(text, header)

    def test_optional_loader(self):
        loader = (ROOT / "backend_loader.c").read_text()
        self.assertIn("RESOLVE_OPT(pipe_kda_shortconv, fn_pipe_kda_shortconv)", loader)
        self.assertNotIn("RESOLVE(pipe_kda_shortconv,", loader)
        wrapper = function(loader, "int coli_cuda_pipe_kda_shortconv(")
        self.assertIn("!g_cuda.available || !g_cuda.pipe_kda_shortconv", wrapper)
        self.assertIn("return 0", wrapper)
        self.assertIn("return g_cuda.pipe_kda_shortconv(device, window_dev, mixed_dev, qkv_dev,", wrapper)

    def test_actual_wrapper_missing_symbol_and_forwarding(self):
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("C compiler unavailable for extracted production wrapper")
        loader = (ROOT / "backend_loader.c").read_text()
        wrapper = function(loader, "int coli_cuda_pipe_kda_shortconv(")
        source = r"""
#include <assert.h>
#include <stddef.h>
typedef int (*fn)(int,float*,float*,const float*,const float*,int,int);
static struct { int available; fn pipe_kda_shortconv; } g_cuda;
static int calls;
static float window=3,mixed=7,input=2,conv=5;
static int fake(int device,float *w,float *m,const float *q,const float *c,int channels,int kernel) {
    assert(device==4 && w==&window && m==&mixed && q==&input && c==&conv && channels==1 && kernel==1);
    calls++;return 1;
}
""" + wrapper + r"""
int main(void) {
    g_cuda.available=1;
    assert(!coli_cuda_pipe_kda_shortconv(4,&window,&mixed,&input,&conv,1,1));
    assert(window==3 && mixed==7 && !calls && g_cuda.available); /* older usable DLL stays available */
    g_cuda.pipe_kda_shortconv=fake;g_cuda.available=0;
    assert(!coli_cuda_pipe_kda_shortconv(4,&window,&mixed,&input,&conv,1,1) && !calls);
    g_cuda.available=1;
    assert(coli_cuda_pipe_kda_shortconv(4,&window,&mixed,&input,&conv,1,1) && calls==1);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrapper.c"
            binary = Path(tmp) / "wrapper"
            path.write_text(source)
            subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_extracted_kernel_body_on_host(self):
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            self.skipTest("C++ compiler unavailable for host kernel-body check")
        kernel = function((ROOT / "backend_cuda.cu").read_text(),
                          "__global__ void pipe_kda_shortconv_kernel(")
        # Compile the actual body with CPU stand-ins for CUDA lane coordinates.
        # This tests arithmetic/indexing, not CUDA compilation or GPU semantics.
        source = '#include "' + (ROOT / "tests/cuda_kda_shortconv_ref.h").as_posix() + '"\n' + r"""
#include <cassert>
#include <vector>
#define __global__
struct Dim { unsigned x; };
static Dim blockIdx,blockDim,threadIdx;
""" + kernel + r"""
int main(void) {
    const int geometries[][2]={{1,1},{2,2},{7,4},{257,4},{24576,4},{7,7}};
    blockDim.x=256;
    for (const auto &g:geometries) {
        int channels=g[0],k=g[1];size_t n=(size_t)channels*k;
        std::vector<float> actual(n),expected(n),conv(n),qkv(channels),out(channels),ref(channels);
        kda_shortconv_initial(actual.data(),conv.data(),channels,k);expected=actual;
        for (int step=0;step<64;step++) {
            for (int c=0;c<channels;c++) qkv[c]=kda_shortconv_input(c,step);
            kda_shortconv_ref(expected.data(),ref.data(),qkv.data(),conv.data(),channels,k);
            for (unsigned lane=0;lane<(unsigned)((channels-1)/256+1)*256;lane++) {
                blockIdx.x=lane/256;threadIdx.x=lane%256;
                pipe_kda_shortconv_kernel(actual.data(),out.data(),qkv.data(),conv.data(),channels,k);
            }
            assert(!memcmp(actual.data(),expected.data(),n*sizeof(float)));
            assert(!memcmp(out.data(),ref.data(),(size_t)channels*sizeof(float)));
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.cpp"
            binary = Path(tmp) / "body"
            path.write_text(source)
            subprocess.run([cxx, "-std=c++11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                            str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_unchanged_donors_and_glm_scope(self):
        git = shutil.which("git")
        if not git or not (ROOT.parent / ".git").exists():
            self.skipTest("Git checkout unavailable for unchanged-source audit")
        def base(path):
            return subprocess.check_output([git, "-c", "safe.directory=" + str(ROOT.parent),
                                            "show", "HEAD:c/" + path], cwd=ROOT.parent, text=True)
        current = (ROOT / "backend_cuda.cu").read_text()
        original = base("backend_cuda.cu")
        start = "#define DN_MAX_HEADS"
        last = 'extern "C" int coli_cuda_dn_step('
        for text in (current, original):
            self.assertIn(start, text)
        self.assertEqual(current[current.index(start):current.index(last)] + function(current, last),
                         original[original.index(start):original.index(last)] + function(original, last))
        def recur(text):
            return text[text.index("/* Recurrence only; caller owns every device buffer."):
                        text.index('extern "C" int coli_cuda_pipe_rmsnorm(')]
        self.assertEqual(recur(current), recur(original))
        self.assertEqual((ROOT / "delta_attention.h").read_text(), base("delta_attention.h"))
        integration = (ROOT / "glm53_cuda.h").read_text()
        self.assertIn("coli_cuda_pipe_kda_shortconv(owner,", integration)
        self.assertNotIn("coli_cuda_dn_", integration)
        for path in ("test_cuda_kda_recur_ref.c", "cuda_kda_recur_ref.h",
                     "test_cuda_kda_recur_source.py", "test_cuda_kda_recur_live.cu"):
            self.assertEqual((ROOT / "tests" / path).read_text(), base("tests/" + path), path)

    def test_live_contract_and_dependency_wiring(self):
        live = (ROOT / "tests/test_cuda_kda_shortconv_live.cu").read_text()
        for text in ("STEPS=64", "repeat<2", "cycle<2", "first_window", "first_trace",
                     "cudaPointerGetAttributes", "coli_cuda_device_at(i)", "only", "fault(true)",
                     "next call succeeds after clearing fault", "error>1e-6f", "full window bitwise exact every step",
                     "sequence(owner,1,1)", "sequence(owner,2,2)", "sequence(owner,7,4)",
                     "sequence(owner,257,4)", "sequence(owner,24576,4)", "sequence(owner,7,7)"):
            self.assertIn(text, live)
        makefile = (ROOT / "Makefile").read_text()
        self.assertRegex(makefile, r"TEST_EXCLUDE.*test_cuda_kda_shortconv_live")
        own = makefile[makefile.index("AUTODEP_OWN_FLAGS ="):makefile.index("ENGINE_RULES :=")]
        self.assertIn("test_cuda_kda_shortconv_live", own)
        self.assertIn("tests/test_cuda_kda_shortconv_ref$(EXE): tests/test_cuda_kda_shortconv_ref.c\n", makefile)
        # Existing post-#1901 fake fixtures keep their Vulkan object linkage.
        for target in ("test_glm53_cuda_stage", "test_glm53_cuda_kda_state", "test_glm53_cuda_kda_recur_integration"):
            rule = re.search(r"tests/" + target + r"\$\(EXE\):[^\n]*\n\t[^\n]*", makefile).group()
            self.assertIn("$(VK_OBJ)", rule.splitlines()[0])
            self.assertIn("$(VK_OBJ)", rule.splitlines()[1])


if __name__ == "__main__":
    unittest.main()
