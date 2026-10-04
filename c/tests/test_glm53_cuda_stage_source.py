"""Ownership-only ABI and integration guards."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class StageSource(unittest.TestCase):
    def test_abi_and_parser(self):
        header = (ROOT / "segment_adapters.h").read_text()
        source = (ROOT / "glm53_cuda.h").read_text()
        self.assertIn("COLI_GLM53_STAGE_PLAN_VERSION 1u", header)
        self.assertIn("int32_t cuda_device_ordinal", header)
        self.assertIn("memcpy(plan, data, sizeof(*plan))", source)
        self.assertNotIn("(ColiGlm53StagePlan *)", source)
        self.assertIn("plan->struct_size > bytes", source)
        self.assertIn("coli_cuda_acquire(devices, count)", source)
        self.assertNotIn("coli_cuda_shutdown", source)
        self.assertNotIn("ColiCudaLifetime", source)

    def test_engine_wiring(self):
        source = (ROOT / "glm53.c").read_text()
        start = source.index("static int glm53_segment_engine_open")
        end = source.index("static int glm53_segment_session_create", start)
        engine = source[start:end]
        self.assertLess(engine.index("coli_glm53_cuda_stage_open"),
                        engine.index("model_load_range"))
        self.assertLess(engine.index("model_release"),
                        engine.index("coli_glm53_cuda_stage_close"))
        self.assertIn("options->resource_plan_size", engine)
        self.assertLess(engine.index("model_load_range"),
                        engine.index("coli_glm53_cuda_stage_wire_create"))
        self.assertLess(engine.index("coli_glm53_cuda_stage_wire_create"),
                        engine.index("*engine_impl = engine"))
        self.assertIn("engine->model.c.hc_mult", engine)
        self.assertIn("engine->model.c.hidden", engine)
        self.assertNotIn("coli_cuda_shutdown", engine)
        cli = (ROOT / "colibri.c").read_text()
        self.assertIn("coli_cuda_configured_devices(g_cuda_devices)", cli)

    def test_resource_order_and_scope(self):
        source = (ROOT / "glm53_cuda.h").read_text()
        close = source[source.index("static inline void coli_glm53_cuda_stage_close"):
                       source.index("static inline int coli_glm53_cuda_stage_wire_create")]
        self.assertLess(close.index("coli_cuda_pipe_free"), close.index("coli_cuda_release"))
        self.assertLess(close.index("stage->wire = NULL"), close.index("coli_cuda_release"))
        self.assertIn("coli_cuda_pipe_alloc(stage->cuda_device_ordinal", source)
        self.assertIn("bytes > stage->wire_bytes - offset", source)
        self.assertNotIn("cudaMalloc", source)
        self.assertNotIn("cuda_runtime", source)
        self.assertNotIn("safetensors", source)
        header = (ROOT / "backend_cuda.h").read_text()
        loader = (ROOT / "backend_loader.c").read_text()
        for operation in ("alloc", "free", "upload", "download"):
            self.assertIn("coli_cuda_pipe_" + operation, header)
            self.assertIn("coli_cuda_pipe_" + operation, loader)

    def test_pinned_handoff_contract(self):
        source = (ROOT / "glm53_cuda.h").read_text()
        transport = source[source.index("} ColiGlm53CudaHandoff;"):]
        self.assertNotIn("cudaMemcpyPeer", transport)
        self.assertNotIn("coli_cuda_pipe_peer_copy", transport)
        self.assertNotIn("malloc(", transport)
        self.assertNotIn("cudaStream", transport)
        self.assertLess(transport.index("coli_cuda_host_free"), transport.index("coli_cuda_release"))
        copy = transport[transport.index("static inline int coli_glm53_cuda_handoff("):]
        self.assertLess(copy.index("coli_glm53_cuda_stage_download"), copy.index("coli_glm53_cuda_stage_upload"))
        self.assertNotIn("alloc(", copy)
        backend = (ROOT / "backend_cuda.cu").read_text()
        self.assertIn("cudaHostAllocPortable", backend)
        pinned = backend[backend.index('extern "C" void *coli_cuda_host_alloc'):
                         backend.index('extern "C" void coli_cuda_pipe_free')]
        self.assertIn("cudaFreeHost", pinned)
        live = (ROOT / "tests/test_glm53_cuda_stage_live.cu").read_text()
        self.assertIn("cudaMemoryTypeHost", live)
        self.assertIn("coli_glm53_cuda_handoff", live)

    def test_kda_session_ownership_and_coherence(self):
        source = (ROOT / "glm53.c").read_text()
        header = (ROOT / "glm53_cuda.h").read_text()
        layer = source[source.index("float *kda_state;"):source.index("} GLayerState;")]
        self.assertIn("ColiGlm53CudaKdaLayer cuda_kda", layer)
        stage = header[header.index("typedef struct {"):header.index("} ColiGlm53CudaStage;")]
        self.assertNotIn("kda", stage)
        attach = source[source.index("static int glm53_kda_attach"):source.index("static void session_close")]
        self.assertIn("s->layer[i]", attach)
        self.assertIn("if (c->is_full[i]) continue", attach)
        self.assertNotIn("coli_cuda_acquire", attach)
        self.assertNotIn("coli_cuda_release", attach)
        close = source[source.index("static void session_close"):source.index("static float *run_layers")]
        self.assertLess(close.index("coli_glm53_cuda_kda_close"), close.index("free(st->kda_state)"))
        pull = header[header.index("static inline int coli_glm53_cuda_kda_ensure_host"):header.index("/* 0 absent")]
        self.assertLess(pull.index("coli_cuda_pipe_download"), pull.index("memcpy(state, s"))
        self.assertIn("if (ok)", pull)
        self.assertNotIn("authority = G53_KDA_DEVICE", source + header)
        run = source[source.index("static float *run_layers"):source.index("static void mat_release")]
        self.assertLess(run.index("glm53_kda_ensure_host"), run.index("kda_layer(c"))
        self.assertLess(run.index("kda_layer(c"), run.index("coli_glm53_cuda_kda_host_written"))
        restore = source[source.index("static int glm53_segment_session_restore_unlocked"):
                         source.index("/* These wrappers own the lock")]
        self.assertLess(restore.index("coli_segment_spans_restore"), restore.index("glm53_kda_host_written"))
        for op in ("run", "snapshot", "restore"):
            wrapper = source[source.index("static int glm53_segment_session_" + op + "("):]
            wrapper = wrapper[:wrapper.index("\n}")]
            self.assertLess(wrapper.index("pthread_mutex_lock"), wrapper.index(op + "_unlocked"))
            self.assertLess(wrapper.index(op + "_unlocked"), wrapper.index("pthread_mutex_unlock"))

if __name__ == "__main__":
    unittest.main()
