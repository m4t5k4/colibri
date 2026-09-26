"""Ownership contract complementary to the fake-backend teardown test."""
from pathlib import Path
import re
import unittest

SRC = (Path(__file__).resolve().parents[1] / "glm53.c").read_text()


def body(name):
    return re.search(r"^static [^\n]+\b" + name + r"\([^;{]*\{[\s\S]*?^\}", SRC, re.M).group()


class Glm53CudaOwnershipTests(unittest.TestCase):
    def test_only_full_model_loader_initializes_cuda(self):
        self.assertNotIn("g53_cuda", body("model_load_range"))
        self.assertNotIn('getenv("COLI_CUDA")', body("model_load_range"))
        self.assertIn("g53_cuda_init", body("model_load"))
        self.assertEqual(SRC.count("g53_cuda_init("), 1)
        self.assertNotIn("coli_cuda_init(", SRC)
        self.assertNotIn("coli_cuda_shutdown(", SRC)

    def test_adapters_use_zero_initialized_range_models_and_cpu_contract(self):
        for family in ("segment", "edge"):
            source = body(f"glm53_{family}_engine_open")
            self.assertIn("calloc(1, sizeof(*engine))", source)
            self.assertIn("model_load_range(&engine->model", source)
            self.assertNotIn("model_load(", source)
            self.assertIn("/f32/cpu-v1", source)
            self.assertIn(f"COLI_{family.upper()}_CAP_CPU", source)

    def test_cli_and_serve_keep_full_model_loader(self):
        self.assertIn("model_load(&served, snap)", SRC)
        self.assertIn("model_load(&model, dir)", SRC)

    def test_host_clamped_swiglu_is_preserved(self):
        self.assertIn("c->swiglu_limit, swiglu_clamped)", SRC)
        self.assertNotIn("coli_cuda_expert_mlp(", SRC)


if __name__ == "__main__":
    unittest.main()
