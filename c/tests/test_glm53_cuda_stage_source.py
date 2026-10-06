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
        self.assertIn("G53_KDA_UNKNOWN", header)
        self.assertIn("authority = G53_KDA_DEVICE", header)
        run = source[source.index("static float *run_layers"):source.index("static void mat_release")]
        self.assertNotIn("glm53_kda_ensure_host(m, s)", run)
        layer_run = source[source.index("static int kda_layer"):source.index("/* ---------- MLA")]
        self.assertLess(layer_run.index("coli_glm53_cuda_kda_ensure_host"), layer_run.index("coli_kda_step(core"))
        self.assertLess(layer_run.index("coli_kda_step(core"), layer_run.index("coli_glm53_cuda_kda_host_written"))
        restore = source[source.index("static int glm53_segment_session_restore_unlocked"):
                         source.index("/* These wrappers own the lock")]
        self.assertLess(restore.index("coli_segment_spans_restore"), restore.index("glm53_kda_host_written"))
        for op in ("run", "snapshot", "restore"):
            wrapper = source[source.index("static int glm53_segment_session_" + op + "("):]
            wrapper = wrapper[:wrapper.index("\n}")]
            self.assertLess(wrapper.index("pthread_mutex_lock"), wrapper.index(op + "_unlocked"))
            self.assertLess(wrapper.index(op + "_unlocked"), wrapper.index("pthread_mutex_unlock"))

    def test_decode_transaction_scope(self):
        source = (ROOT / "glm53.c").read_text()
        header = (ROOT / "glm53_cuda.h").read_text()
        layer = source[source.index("static int kda_layer"):source.index("/* ---------- MLA")]
        self.assertIn("gpu = tokens == 1 && s->cuda_stage", layer)
        self.assertNotIn("next_window", source + header)
        self.assertNotIn("glm53_kda_shortconv(", layer)
        self.assertIn("&l->cuda_kda_weights", layer)
        self.assertIn("if (st->cuda_kda.authority != G53_KDA_UNKNOWN)", layer)
        recur = header[header.index("static inline int coli_glm53_cuda_kda_decode("):header.index("/* 0 absent")]
        positions = [recur.index(text) for text in (
            "coli_cuda_pipe_kda_shortconv(", "authority = G53_KDA_UNKNOWN", "coli_cuda_pipe_kda_recur(",
            "coli_cuda_pipe_sync(", "coli_cuda_pipe_download(", "authority = G53_KDA_DEVICE")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("heads, dim, dim, 1e-6f", recur)
        self.assertEqual(recur.count("coli_cuda_pipe_sync("), 1)
        self.assertEqual(recur.count("coli_cuda_pipe_upload("), 2)
        self.assertNotIn("memcpy(", recur)
        self.assertNotIn("pipe_upload(owner, layer->window", recur)
        self.assertNotIn("work->projected_qkv,", recur[:recur.index("coli_cuda_pipe_kda_shortconv(")])
        self.assertIn("work->mixed_qkv + p, work->mixed_qkv + 2*p", recur)
        backend = (ROOT / "backend_cuda.cu").read_text()
        self.assertIn("pipe_kda_shortconv_kernel<<<blocks, 256>>>", backend)
        self.assertIn("pipe_kda_recur_kernel<<<heads, threads>>>", backend)
        # Both launches omit a stream argument, on the same calling thread/device.

        for forbidden in ("alloc(", "free(", "acquire(", "release(", "stage->wire", "coli_cuda_dn_", "cudaStream"):
            self.assertNotIn(forbidden, recur)
        session = source[source.index("float *kda_scratch;"):source.index("} GSession;")]
        self.assertIn("ColiGlm53CudaKdaStaging kda_staging", session)
        geometry = header[header.index("static inline int coli_glm53_cuda_kda_staging_geometry"):
                          header.index("static inline void coli_glm53_cuda_kda_staging_close")]
        self.assertIn("coli_glm53_cuda_size_mul(8, proj", geometry)
        self.assertIn("heads > SIZE_MAX - floats", geometry)
        self.assertIn("coli_glm53_cuda_size_mul(hidden + floats + heads", geometry)
        run = source[source.index("static float *run_layers"):source.index("static void mat_release")]
        self.assertIn("g_vk_chain && !s->cuda_stage", run)
        self.assertNotIn("glm53_kda_ensure_host(m, s)", run)
        self.assertLess(run.index("G53_KDA_UNKNOWN"), run.index("for (int i = begin; i < end; i++) {"))
        restore = source[source.index("static void glm53_state_restore"):source.index("static void slot_reset")]
        self.assertLess(restore.index("glm53_kda_pin_complete"), restore.index("memcpy("))
        self.assertIn("COMPLETE trusted host state+window overwrite", header)

    def test_static_convolution_ownership(self):
        source = (ROOT / "glm53.c").read_text()
        header = (ROOT / "glm53_cuda.h").read_text()
        model_layer = source[source.index("Mat kq, kk"):source.index("} GLayer;")]
        session_layer = source[source.index("float *kda_state;"):source.index("} GLayerState;")]
        session = source[source.index("float *kda_scratch;"):source.index("} GSession;")]
        self.assertIn("ColiGlm53CudaKdaWeights cuda_kda_weights", model_layer)
        self.assertNotIn("CudaKdaWeights", session_layer + session)
        engine = source[source.index("static int glm53_segment_engine_open"):
                        source.index("static int glm53_segment_session_create")]
        self.assertLess(engine.index("model_load_range"), engine.index("glm53_kda_weights_open"))
        self.assertLess(engine.index("glm53_kda_weights_open"), engine.index("*engine_impl = engine"))
        release = source[source.index("static void model_release"):source.index("/*", source.index("    free(m->layer);"))]
        self.assertLess(release.index("coli_glm53_cuda_kda_weights_close"), release.index("free((void *)vectors"))
        weights = header[header.index("/* Immutable model/layer weights"):header.index("/* One session device workspace")]
        self.assertNotIn("coli_cuda_acquire", weights)
        self.assertNotIn("coli_cuda_release", weights)
        self.assertIn("coli_cuda_pipe_upload(stage->cuda_device_ordinal", weights)
        close = source[source.index("static void session_close"):source.index("static float *run_layers")]
        self.assertNotIn("weights_close", close)

    def test_resident_qkv_projection_scope(self):
        source = (ROOT / "glm53.c").read_text()
        header = (ROOT / "glm53_cuda.h").read_text()
        layer = source[source.index("static int kda_layer"):source.index("/* ---------- MLA")]
        for name in ("q", "k", "v", "qkv"):
            self.assertIn("float *" + name + " = gpu ? NULL : malloc", layer)
        project = header[header.index("static inline int coli_glm53_cuda_kda_project_qkv"):
                         header.index("/* Prepared BOTH/DEVICE pair")]
        self.assertEqual(project.count("coli_cuda_pipe_upload("), 1)
        self.assertEqual(project.count("coli_cuda_pipe_gemm("), 3)
        for forbidden in ("sync(", "download(", "authority", "malloc(", "acquire(", "stage->wire"):
            self.assertNotIn(forbidden, project)
        self.assertIn("work->input_x, x, work->input_bytes", project)
        self.assertLess(layer.index("coli_glm53_cuda_kda_project_qkv"), layer.index("mm(low, &l->kfa"))
        self.assertIn("} else {\n        mm(q, &l->kq", layer)
        upload = source[source.index("static int glm53_kda_projection_open"):source.index("static int glm53_kda_attach")]
        for token in ("mat->rows != c->kda_proj", "mat->columns != c->hidden", "mat->fmt", "mat->gs",
                      "mat->q4", "mat->s", "coli_cuda_tensor_upload_g", "coli_cuda_tensor_device",
                      "coli_cuda_tensor_bytes", "coli_cuda_tensor_vram"):
            self.assertIn(token, upload)
        backend = (ROOT / "backend_cuda.cu").read_text()
        gemm = backend[backend.index('extern "C" int coli_cuda_pipe_gemm('):
                       backend.index('extern "C" int coli_cuda_pipe_peer_copy(')]
        self.assertIn("quant_matmul<<<grid,256>>>", gemm)
        self.assertNotIn("Synchronize", gemm)
        self.assertNotIn("cudaMemcpy", gemm)

if __name__ == "__main__":
    unittest.main()
