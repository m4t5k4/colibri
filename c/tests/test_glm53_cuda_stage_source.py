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
        self.assertNotIn("coli_cuda_shutdown", engine)
        cli = (ROOT / "colibri.c").read_text()
        self.assertIn("coli_cuda_configured_devices(g_cuda_devices)", cli)

if __name__ == "__main__":
    unittest.main()
