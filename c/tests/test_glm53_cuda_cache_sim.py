"""Offline cache comparisons use only selection events, never model weights."""
import importlib.util
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
            result = SIM.replay(SIM.events(trace), (1, 1), heat_margin=0)
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
        self.assertEqual(states["current_t2_m1"].evictions, 0)
        self.assertEqual(states["threshold_3_margin_1"].promotions, 2)

    def test_selection_group_delays_promotion_and_legacy_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("""S,1,1,0,0,2,0
S,2,1,0,1,2,0
S,3,2,0,0,1,1
S,4,2,0,1,1,0
""")
            result = SIM.replay(SIM.events(trace), (1,), heat_margin=0)
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
            result = SIM.replay(SIM.events(trace), (1,), heat_margin=0)
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


class HistoricalWarmTests(unittest.TestCase):
    TRACE = """S,1,0,0,0,2,0
A,1,0,0
P,1,0,0,0,0
S,2,0,0,1,2,0
A,2,0,1
P,2,0,1,1,1
S,3,1,0,0,1,1
S,4,1,0,2,4,0
A,4,0,2
E,4,0,2,0,1,4,2,2,-1
P,4,0,2,1,1
S,5,2,0,2,1,1
S,6,2,0,1,1,0
A,6,0,1
"""

    def test_usage_order_headers_and_future_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage"
            path.write_text(f"-1 2 4\n-2 1 {SIM.glm53_engine_id()}\n0 2 1\n0 3 9\n0 1 9\n1 0 9\n")
            dims, counts = SIM.read_usage(path, 2, 4)
            self.assertEqual(dims, (2, 4))
            self.assertEqual(SIM.ranked_usage(counts), [(0, 1), (0, 3), (1, 0), (0, 2)])
            self.assertEqual(SIM.ranked_usage(counts)[0], (0, 1))
            path.write_text("-1 2 4\n-2 1 17\n0 1 9\n")
            with self.assertRaisesRegex(ValueError, "engine"):
                SIM.read_usage(path, 2, 4)
            path.write_text(f"-1 3 4\n-2 1 {SIM.glm53_engine_id()}\n0 1 9\n")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                SIM.read_usage(path, 2, 4)
            path.write_text(f"-1 2 4\n-2 2 {SIM.glm53_engine_id()}\n0 1 9\n")
            with self.assertRaisesRegex(ValueError, "format"):
                SIM.read_usage(path, 2, 4)

    def test_fractions_and_least_used_capacity(self):
        self.assertEqual([SIM.warm_limit(3390, f) for f in SIM.FRACTIONS],
                         [0, 848, 1695, 2543, 3390])
        state = SIM.PolicyState((2, 1), 2, 1, shared_capacity=3)
        keys = [(0, n) for n in range(5)]
        state.preload(keys)
        self.assertEqual(state.warm_initial_owners, {keys[0]: 0, keys[1]: 1, keys[2]: 0})
        self.assertEqual(state.used, [2, 1])
        self.assertEqual(state.heat, {})
        candidates = [(0, n) for n in range(6)]
        placed = []
        for fraction in SIM.FRACTIONS:
            trial = SIM.PolicyState((2, 2), 2, 1, shared_capacity=4)
            trial.preload(candidates[:SIM.warm_limit(4, fraction)])
            placed.append(len(trial.warm_initial))
            self.assertTrue(all(used <= cap for used, cap in zip(trial.used, trial.capacities)))
        self.assertEqual(placed, [0, 1, 2, 3, 4])

    def test_margin_and_unhit_warm_eviction(self):
        key0, key1 = (0, 0), (0, 1)
        margin1 = SIM.PolicyState((1,), 2, 1)
        margin0 = SIM.PolicyState((1,), 2, 0)
        for state in (margin1, margin0):
            state.preload([key0])
            state.select(key0, 1, True, 1)
            state.select(key1, 1, True, 1)
            state.attempt(key1, True)
            self.assertIn(key0, state.resident)
            state.select(key1, 1, True, 2)
            state.attempt(key1, True)
        self.assertIn(key0, margin1.resident)
        self.assertIn(key1, margin0.resident)
        stale = SIM.PolicyState((1,), 2, 1)
        stale.preload([key0])
        stale.select(key1, 2, True, 1)
        stale.attempt(key1, True)
        self.assertEqual(stale.warm_evicted_before_hit, {key0})
        self.assertEqual(stale.heat[key0], 0)

    def test_full_trace_validation_and_decode_accounting(self):
        trace = list(SIM.events(self._trace()))
        warm_sets = {"baseline": [], ("historical", 0): [],
                     ("oracle_not_deployable", 0): [],
                     ("historical", 0.5): [(0, 0)],
                     ("oracle_not_deployable", 0.5): [(0, 2)]}
        states, check = SIM.replay_warm(trace, (1, 1), 2, 2, 1, warm_sets)
        self.assertEqual([check[key] for key in (
            "selection_mismatches", "attempt_mismatches", "upload_owner_mismatches",
            "eviction_mismatches", "final_resident_mismatches")], [0] * 5)
        self.assertEqual(states["baseline"].windows["full"]["fallback_rows"], 9)
        self.assertEqual(states["baseline"].windows["decode"]["fallback_rows"], 5)
        self.assertEqual(states[("historical", 0)].windows, states["baseline"].windows)
        self.assertEqual(states[("historical", 0)].resident, states["baseline"].resident)
        warm = states[("historical", 0.5)]
        self.assertIn((0, 0), warm.warm_hit_experts)
        self.assertEqual(warm.first_hit_tokens[0], 1)
        self.assertEqual(warm.decode_start_warm, 1)
        self.assertNotEqual(warm_sets[("historical", 0.5)], warm_sets[("oracle_not_deployable", 0.5)])

    def test_trace_boundary_failure_and_deterministic_csv(self):
        bad = list(SIM.events(self._trace()))
        bad[0] = ("S", (1, 2, 0, 0, 2, 0))
        with self.assertRaisesRegex(ValueError, "boundary"):
            SIM.replay_warm(bad, (1, 1), 2, 2, 1, {"baseline": []})
        states, _ = SIM.replay_warm(list(SIM.events(self._trace())), (1, 1), 2, 2, 1,
                                    {"baseline": [], ("historical", 0.5): [(0, 0)],
                                     ("oracle_not_deployable", 0.5): [(0, 2)]})
        sets = {"baseline": [], ("historical", 0.5): [(0, 0)],
                ("oracle_not_deployable", 0.5): [(0, 2)]}
        rows = SIM.warm_rows(states, sets, {(0, 0): 9, (0, 2): 1}, 2, [0.5])
        def output():
            stream = io.StringIO(newline="")
            columns = list(rows[0])
            for row in rows[1:]:
                columns.extend(key for key in row if key not in columns)
            writer = csv.DictWriter(stream, columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            return stream.getvalue()
        self.assertEqual(output(), output())
        self.assertIn("oracle_not_deployable", output())

    def test_command_output_and_device_log_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            trace = folder / "trace.csv"
            trace.write_text(self.TRACE)
            usage = folder / "usage"
            usage.write_text(f"-1 1 3\n-2 1 {SIM.glm53_engine_id()}\n0 0 10\n0 1 1\n0 2 9\n")
            log = folder / "run.log"
            log.write_text("\n".join(
                f"[glm53-cuda-device] device={i} ceiling_bytes={SIM.EXPERT_BYTES} resident=1"
                for i in range(2)))
            csv_path = folder / "result.csv"
            args = SimpleNamespace(trace=trace, warm_usage=usage, n_layers=1, n_experts=3,
                                   first_dense=0, warm_fractions="0,0.25,0.50,0.75,1.00",
                                   warm_experts=None, capacity=2, device_capacities=None,
                                   device_log=log, heat_min=2, heat_margin=1,
                                   decode_tokens=2, csv=csv_path)
            outputs = []
            for _ in range(2):
                text = io.StringIO()
                with redirect_stdout(text):
                    SIM.run_warm(args)
                outputs.append((text.getvalue(), csv_path.read_bytes()))
            self.assertEqual(outputs[0], outputs[1])
            data = list(csv.DictReader(io.StringIO(outputs[0][1].decode())))
            self.assertEqual(len(data), 22)  # baseline and two policies per fraction, full/decode
            self.assertEqual(data[0]["runtime_uploads"], "3")
            self.assertEqual(data[0]["warm_experts"], "0")
            self.assertIn("historical_oracle_uploads_avoided_ratio", data[0])
            historical = next(row for row in data if row["policy"] == "historical" and
                              row["warm_fraction"] == "0.50" and row["window"] == "decode")
            oracle = next(row for row in data if row["policy"] == "oracle_not_deployable" and
                          row["warm_fraction"] == "0.50" and row["window"] == "decode")
            self.assertEqual(historical["historical_count_coverage"], "0.500000")
            self.assertEqual(oracle["historical_count_coverage"], "0.450000")
            log.write_text(log.read_text() + log.read_text())
            with self.assertRaisesRegex(ValueError, "non-duplicated"):
                SIM.capacities_from_log(log)

    def _trace(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.csv"
        path.write_text(self.TRACE)
        return path


if __name__ == "__main__":
    unittest.main()
