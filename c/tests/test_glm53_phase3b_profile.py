import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p3', ROOT / 'tools/glm53_phase3b_profile.py')
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)


def line(phase, token=0, **updates):
    fields = dict.fromkeys(p3.METRICS, 0)
    fields.update(updates)
    return p3.PREFIX + f'phase={phase} decode_tokens={token} ' + ' '.join(
        f'{k}={v}' for k, v in fields.items())


class ParserTests(unittest.TestCase):
    def test_prefill_decode_final(self):
        logs = [line('start'), line('prefill', host_cache_misses=10),
                line('decode', 1, host_cache_misses=12, ffn_sparse_s=2),
                line('decode', 2, host_cache_misses=15, ffn_sparse_s=5),
                line('final', 2, host_cache_misses=15, ffn_sparse_s=5)]
        rows = p3.decode_rows(logs)
        self.assertEqual([r['host_cache_misses_delta'] for r in rows], [2, 3])
        self.assertEqual([r['ffn_sparse_s_delta'] for r in rows], [2, 3])
        self.assertEqual(len(rows[0]), len(p3.METRICS) + 1)

    def test_rejections(self):
        cases = [([], 'baseline'), ([line('decode', 1)], 'baseline'),
                 ([line('start')], 'final'),
                 ([line('start'), line('decode', 2)], 'increment'),
                 ([line('start'), line('start')], 'interleaved'),
                 ([line('start', host_cache_hits=2), line('decode', 1)], 'decreasing'),
                 ([line('start', sparse_shared_s=2), line('decode', 1)], 'decreasing'),
                 ([line('start'), line('final'), line('decode', 1)], 'after final'),
                 ([line('start'), line('decode', 1, sparse_shared_s='nan')], 'nonfinite')]
        for logs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                p3.decode_rows(logs)

    def test_other_formats_ignored(self):
        self.assertEqual(p3.decode_rows(['[glm53-cuda-profile] phase=decode',
                                        line('start'), line('final')]), [])

    def test_exact_large_integer_bytes(self):
        value = 2**60
        rows = p3.decode_rows([line('start', kda_h2d_bytes=value),
                              line('decode', 1, kda_h2d_bytes=value + 7),
                              line('final', 1, kda_h2d_bytes=value + 7)])
        self.assertEqual(rows[0]['kda_h2d_bytes_delta'], 7)


class SourceTests(unittest.TestCase):
    def test_opt_in_and_outer_clock_reuse(self):
        source = (ROOT / 'glm53.c').read_text()
        self.assertEqual(source.count('getenv("GLM53_PHASE3B_PROFILE")'), 1)
        self.assertIn('m->phase3b.enabled = m->cuda.profile.clock && phase3b_env', source)
        self.assertIn('if (!m->phase3b.enabled) return;', source)
        outer = source.split('m->t_ffn += phase_s;', 1)[1].split('} else {', 1)[0]
        self.assertIn('ffn_dense_s += phase_s', outer)
        self.assertIn('ffn_sparse_s += phase_s', outer)
        self.assertNotIn('now_s()', outer)
        self.assertIn('g53_phase3b_report(m, "start")', source)
        self.assertIn('g53_phase3b_report(m, "final")', source)


if __name__ == '__main__':
    unittest.main()
