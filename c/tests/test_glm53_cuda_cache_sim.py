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
    def test_exact_two_device_owner_replay_and_counterfactuals(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("""S,1,1,0,0,2,0
A,1,0,0
P,1,0,0,0,0
U,1,0,0,0,0.018700000,1
S,2,2,0,1,2,0
A,2,0,1
P,2,0,1,1,1
S,3,3,0,2,3,0
A,3,0,2
E,3,0,2,0,0,3,2,2,-1
P,3,0,2,0,0
S,4,4,0,0,3,0
A,4,0,0
E,4,0,0,0,1,5,2,2,-1
P,4,0,0,1,1
S,5,5,0,0,1,1
""")
            result = SIM.replay(SIM.events(trace), (1, 1))
        states = result["states"]
        current = states["current_t2_m0"]
        self.assertEqual(result["selection_snapshot_mismatches"], 0)
        self.assertEqual(result["upload_event_mismatches"], 0)
        self.assertEqual(result["eviction_event_mismatches"], 0)
        self.assertEqual((current.hit_rows, current.promotions, current.evictions,
                          current.repromotions, current.dead_on_arrival),
                         (1, 4, 2, 1, 2))
        self.assertEqual(current.upload_events[-1], ((0, 0), 1))
        self.assertEqual(states["threshold_3"].promotions, 2)
        self.assertEqual(states["margin_1"].evictions, 0)
        self.assertEqual(states["threshold_3_margin_1"].promotions, 2)

    def test_selection_group_delays_promotion_and_legacy_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("""S,1,1,0,0,2,0
S,2,1,0,1,2,0
S,3,2,0,0,1,1
S,4,2,0,1,1,0
""")
            result = SIM.replay(SIM.events(trace), (1,))
        self.assertEqual(result["selection_snapshot_mismatches"], 0)
        self.assertEqual(result["states"]["current_t2_m0"].hit_rows, 1)
        self.assertEqual(result["states"]["current_t2_m0"].promotions, 1)
        self.assertIsNone(result["upload_event_mismatches"])

    def test_least_allocated_owner_and_tie_break(self):
        state = SIM.PolicyState((2, 1), 2, 0)
        for key in ((0, 0), (0, 1), (0, 2)):
            state.select(key, 2, True)
            state.attempt(key)
        self.assertEqual([owner for _, owner in state.upload_events], [0, 1, 0])
        state.select((0, 3), 3, True)
        state.attempt((0, 3))
        self.assertEqual(state.eviction_events, [((0, 3), (0, 0))])
        self.assertEqual(state.upload_events[-1], ((0, 3), 0))

    def test_initial_resident_hit_is_not_repromoted_after_same_group_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("""S,1,1,0,0,2,0
A,1,0,0
P,1,0,0,0,0
S,2,2,0,1,4,0
S,3,2,0,0,1,1
A,3,0,1
E,3,0,1,0,0,4,3,2,0
P,3,0,1,0,0
""")
            result = SIM.replay(SIM.events(trace), (1,))
        current = result["states"]["current_t2_m0"]
        self.assertEqual(current.promotions, 2)
        self.assertEqual(current.evictions, 1)
        self.assertEqual(result["upload_event_mismatches"], 0)
        self.assertEqual(result["eviction_event_mismatches"], 0)
        self.assertEqual(result["selection_snapshot_mismatches"], 0)


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
