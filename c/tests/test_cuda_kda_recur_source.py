"""Model/toolkit-free guards for the recurrence-only backend seam."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class KdaRecurSource(unittest.TestCase):
    def test_resident_scope_and_arithmetic(self):
        source = (ROOT / "backend_cuda.cu").read_text()
        primitive = source[source.index("__device__ static float pipe_kda_product"):
                           source.index('extern "C" int coli_cuda_pipe_rmsnorm(')]
        for forbidden in ("cudaMemcpy", "cudaMalloc", "cudaFree", "cudaDeviceSynchronize",
                          "coli_cuda_acquire", "coli_cuda_release", "dn_head_kernel",
                          "double", "norm_w", "out_proj"):
            self.assertNotIn(forbidden, primitive)
        self.assertIn("float qs = norm_eps, ks = norm_eps", primitive)
        self.assertIn("query_scale / sqrtf(qs)", primitive)
        self.assertIn("expf(log_decay[key_base + t])", primitive)
        self.assertLess(primitive.index("*cell = pipe_kda_product(*cell, alpha[i])"),
                        primitive.index("memory +="))
        self.assertLess(primitive.index("memory +="), primitive.index("float residual"))
        self.assertLess(primitive.index("*cell += correction"), primitive.index("result +="))
        self.assertIn("(size_t)h * kdim * vdim + t", primitive)
        entry = primitive[primitive.index('extern "C" int coli_cuda_pipe_kda_recur'):]
        self.assertLess(entry.index("fault_injected"), entry.index("<<<"))
        self.assertIn("select_ctx(find_ctx(device))", entry)
        self.assertIn("cudaGetLastError()", entry)
        self.assertIn("kdim > 256", entry)
        self.assertIn("vdim > 256", entry)
        self.assertIn("SIZE_MAX / sizeof(float)", entry)

    def test_optional_loader_and_shared_backend_boundary(self):
        loader = (ROOT / "backend_loader.c").read_text()
        self.assertIn("RESOLVE_OPT(pipe_kda_recur, fn_pipe_kda_recur)", loader)
        self.assertNotIn("RESOLVE(pipe_kda_recur,", loader)
        wrapper = loader[loader.index("int coli_cuda_pipe_kda_recur("):
                         loader.index("int coli_cuda_pipe_rmsnorm(")]
        self.assertIn("!g_cuda.available || !g_cuda.pipe_kda_recur", wrapper)
        self.assertIn("return 0", wrapper)
        integration = (ROOT / "glm53_cuda.h").read_text()
        self.assertIn("coli_cuda_pipe_kda_recur(owner,", integration)
        self.assertNotIn("coli_cuda_dn_", integration)
        header = (ROOT / "backend_cuda.h").read_text()
        self.assertIn("COLI_CUDA_DLLEXPORT int coli_cuda_pipe_kda_recur", header)

    def test_live_contract_and_opt_in(self):
        live = (ROOT / "tests/test_cuda_kda_recur_live.cu").read_text()
        for text in ("STEPS=64", "repeat<2", "first_trace", "first_state", "cudaPointerGetAttributes",
                     "coli_cuda_device_at(i)", "sequence(device,3,17,11", "sequence(device,64,128,128",
                     "fault(true)", "1e-5f", "no_mutation(b)"):
            self.assertIn(text, live)
        makefile = (ROOT / "Makefile").read_text()
        self.assertIn("test_glm53_cuda_kda_state_live test_cuda_kda_recur_live", makefile)
        self.assertIn("-ffp-contract=off $<", makefile)


if __name__ == "__main__":
    unittest.main()
