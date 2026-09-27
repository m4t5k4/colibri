import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

PATH = Path(__file__).resolve().parents[1] / "tools" / "glm53_cuda_profile.py"
spec = importlib.util.spec_from_file_location("glm53_profile", PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def line(phase, **changes):
    fields = dict.fromkeys(profile.COUNTS + profile.TIMES + profile.GAUGES, 0)
    fields.update(changes)
    return "[glm53-cuda-profile] phase=" + phase + " " + " ".join(f"{k}={v}" for k, v in fields.items())


class ProfileTests(unittest.TestCase):
    def test_deltas_exclude_prefill_final_and_filling_token(self):
        rows = list(profile.decode_rows([
            "[glm53-cuda] resident=0\n", line("start"),
            line("prefill", tokens=1, forwards=1, forward_s=2, uploads=1),
            line("decode", tokens=2, decode_tokens=1, forwards=2, forward_s=5,
                 uploads=3, tier_full=1),
            line("decode", tokens=3, decode_tokens=2, forwards=3, forward_s=6,
                 uploads=4, evictions=1, tier_full=1),
            line("final", tokens=3, decode_tokens=2, forwards=3, forward_s=6,
                 uploads=4, evictions=1, tier_full=1)]))
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["forward_s"] for r in rows], [3, 1])
        self.assertEqual([r["uploads"] for r in rows], [2, 1])
        self.assertEqual(profile.warm_window(rows, 1), [])
        self.assertEqual(profile.warm_window(rows, 2), [])

    def test_faulted_window_is_not_warm(self):
        rows = list(profile.decode_rows([line("start", tier_full=1),
            line("decode", tokens=1, decode_tokens=1, forwards=1, tier_full=1,
                 cuda_rows=2, fallback_rows=1, errors=1)]))
        self.assertEqual((rows[0]["cuda_rows"], rows[0]["fallback_rows"]), (2, 1))
        self.assertEqual(profile.warm_window(rows, 1), [])

    def test_missing_baseline_and_counter_regression_fail(self):
        for lines in ([line("decode")], [line("start", tokens=2), line("prefill", tokens=1)]):
            with self.assertRaises(ValueError):
                list(profile.decode_rows(lines))

    def test_prefill_interrupts_warm_window_even_if_full(self):
        rows = list(profile.decode_rows([line("start", tier_full=1),
            line("decode", tokens=1, decode_tokens=1, forwards=1, tier_full=1),
            line("prefill", tokens=2, decode_tokens=1, forwards=2, tier_full=1),
            line("decode", tokens=3, decode_tokens=2, forwards=3, tier_full=1)]))
        self.assertEqual(profile.warm_window(rows, 2), [])

    def test_not_full_but_unchanging_working_set_is_steady(self):
        rows = list(profile.decode_rows([line("start"),
            line("decode", tokens=1, decode_tokens=1, forwards=1, tier_full=0,
                 uploads=2, evictions=1),
            line("decode", tokens=2, decode_tokens=2, forwards=2, tier_full=0,
                 uploads=2, evictions=1),
            line("decode", tokens=3, decode_tokens=3, forwards=3, tier_full=0,
                 uploads=2, evictions=1)]))
        self.assertEqual(profile.warm_window(rows, 2), rows[-2:])
        self.assertEqual(profile.warm_window(rows, 3), [])

    def test_cli_exposes_churn_deltas_without_a_partition_ratio(self):
        lines = [line("start", tier_full=0),
                 line("decode", tokens=1, decode_tokens=1, forwards=1,
                      tier_full=0, cuda_rows=2, fallback_rows=1),
                 line("decode", tokens=2, decode_tokens=2, forwards=2,
                      tier_full=0, cuda_rows=3, fallback_rows=2)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            output = io.StringIO()
            with patch("sys.argv", ["profile", str(path), "--warm-window", "2"]), redirect_stdout(output):
                profile.main()
        result = output.getvalue()
        for field in ("uploads_delta", "evictions_delta", "errors_delta"):
            self.assertIn(field, result)
        self.assertIn("Steady working-set candidate", result)
        self.assertIn("tier=not_full", result)
        self.assertNotIn("ratio", result)
        self.assertNotIn("share", result)


if __name__ == "__main__":
    unittest.main()
