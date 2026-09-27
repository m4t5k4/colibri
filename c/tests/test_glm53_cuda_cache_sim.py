"""Offline cache comparisons use only selection events, never model weights."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools" / "glm53_cuda_cache_sim.py"
SPEC = importlib.util.spec_from_file_location("glm53_cuda_cache_sim", PATH)
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


class CacheSimulationTests(unittest.TestCase):
    def test_current_lru_and_static_on_decode_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("""# selection and eviction events
S,1,0,0,0,2,0
S,2,1,0,0,1,1
S,3,1,0,1,2,0
E,3,0,1,0,0,3,2,1,1
S,4,2,0,1,2,1
S,5,2,0,0,1,0
""")
            result = SIM.compare(trace, 1)
        self.assertEqual(result["tokens"], 2)
        self.assertEqual(result["total_rows"], 6)
        self.assertEqual(result["current_rows"], 3)
        self.assertEqual(result["lru_rows"], 3)
        self.assertEqual(result["static_cold_rows"], 2)
        self.assertEqual(result["static_preloaded_rows"], 4)

    def test_requires_decode_and_positive_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("S,1,0,0,0,2,0\n")
            with self.assertRaisesRegex(ValueError, "no decode"):
                SIM.compare(trace, 1)
            with self.assertRaisesRegex(ValueError, "capacity"):
                SIM.compare(trace, 0)


if __name__ == "__main__":
    unittest.main()
