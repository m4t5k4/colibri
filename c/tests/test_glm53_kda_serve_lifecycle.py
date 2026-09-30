"""Protocol/parser checks only; no model or CUDA claims."""
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import glm53_kda_serve_lifecycle as h


class ServeLifecycleTests(unittest.TestCase):
    def request(self, limited=1):
        e = h.Engine.__new__(h.Engine)
        wire = b"ECHO 1 2 0 nan 0\n\nX\n"
        for tid, piece in enumerate((b"a\nb", b"\xc3\xa9", b"", b"!")):
            wire += f"DATA 1 {len(piece)} -0.1 1 {tid} -0.1\n".encode() + piece + b"\n"
        wire += f"DONE 1 STAT 4 1.0 0 0 8 {limited}\n".encode()
        e.p = SimpleNamespace(stdout=io.BytesIO(wire), stdin=io.BytesIO())
        e.raw = io.BytesIO(); e.mode = 0; e.log = Path("unused")
        result = e.request(1, 0, b"prompt")
        self.assertEqual(e.raw.getvalue(), wire)
        self.assertEqual(result["tokens"], [0, 1, 2, 3])
        self.assertEqual(bytes.fromhex(result["data_hex"]), b"a\nb\xc3\xa9!")
        return result

    def test_byte_counted_framing_and_limited_decode_count(self):
        self.assertEqual(self.request()["decode_forwards"], 3)

    def test_stop_requires_last_emitted_token_forward(self):
        self.assertEqual(self.request(0)["decode_forwards"], 4)

    def test_semantic_comparison_uses_ids_and_bytes(self):
        result = self.request()
        self.assertTrue(h.same(result, dict(result)))
        self.assertFalse(h.same(result, dict(result, tokens=[9, 1, 2, 3])))

    def test_healthy_device_banner_not_an_error(self):
        h.unexpected_errors("[CUDA] device 0: RTX 3070, 8 GB VRAM\n")
        for line in ("[CUDA] expert launch: invalid resource handle",
                     "[glm53-kda-cuda] step failed layer=0",
                     "[glm53-cuda] errors=1"):
            with self.assertRaises(AssertionError):
                h.unexpected_errors(line)

    def test_final_health_and_coverage(self):
        text = "[CUDA] device 0: RTX 3070\n[glm53-cuda] errors=0\n"
        text += "".join(f"REUSE {i} {2 if i < 7 else 0} 10\n" for i in range(3, 9))
        text += "".join(f"[glm53-kda-cuda-layer] layer={i} loaded=1\n" for i in range(34))
        text += "[glm53-kda-cuda] kda_cuda_calls=102 kda_cuda_errors=0 kda_cuda_fallbacks=0 state_pushes=2 state_pulls=1 invalidations=3\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log"
            path.write_text(text)
            self.assertEqual(h.check_log(path, 1, [{"decode_forwards": 3}])["kda"]["kda_cuda_calls"], "102")
            path.write_text(text.replace("REUSE 3 2", "REUSE 3 0"))
            with self.assertRaises(AssertionError):
                h.check_log(path, 1, [{"decode_forwards": 3}])


if __name__ == "__main__":
    unittest.main()
