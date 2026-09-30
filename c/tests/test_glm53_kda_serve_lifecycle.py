"""Protocol/parser checks only; no model or CUDA claims."""
import io
from pathlib import Path
import tempfile
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

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
        e.reader = h.PipeReader(e.p.stdout, e.raw)
        e.read_timeout = 1; e.request_timeout = 1; e.deadline = None
        e.last_frame = None; e.request_id = None; e.input_log = io.BytesIO()
        result = e.request(1, 0, b"prompt")
        e.reader.thread.join(timeout=1)
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

    def test_short_writes(self):
        class Short(io.BytesIO):
            def write(self, data):
                return super().write(data[:2])
        stream = Short()
        h.write_all(stream, b"SUBMIT\npayload\n")
        self.assertEqual(stream.getvalue(), b"SUBMIT\npayload\n")

    def test_eof_diagnostics_preserve_natural_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), model=Path("unused"),
                usage=Path("unused"), binary=Path(sys.executable),
                read_timeout=1, request_timeout=1, startup_timeout=1)
            # Engine's production binary launch is verified separately; exercise
            # cleanup against a real process, without a model or CUDA.
            e = h.Engine.__new__(h.Engine)
            e.mode=0; e.log=args.output/"stderr"; e.stderr=e.log.open("wb")
            e.raw=(args.output/"stdout").open("wb")
            e.input_log=(args.output/"stdin").open("wb")
            e.failure_path=args.output/"failure.json"; e.failure=None
            e.startup_frames=["READY","STAT","EMAP"]; e.last_frame="EMAP 42 288 ..."
            e.request_id=1; e.closed=False; e.read_timeout=1; e.deadline=None
            e.p=subprocess.Popen([sys.executable,"-c","import sys; sys.stderr.write('original error\\n'); sys.exit(7)"],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=e.stderr,bufsize=0)
            e.reader=h.PipeReader(e.p.stdout,e.raw)
            with self.assertRaises(AssertionError):
                e.line()
            e.abort()
            self.assertEqual(e.p.returncode,7)
            self.assertEqual(e.failure["original_returncode"],7)
            self.assertFalse(e.failure["stdin_closed"])
            self.assertEqual(e.failure["startup_frame_count"],3)
            self.assertIn("original error",e.failure["stderr_tail"])
            self.assertEqual(e.failure["cleanup_action"],"none: child already exited")

    def test_read_timeout(self):
        reader=h.PipeReader(io.BytesIO(b""),io.BytesIO())
        reader.thread.join(timeout=1)
        reader.items.get(timeout=1)  # Remove EOF to model an open, silent pipe.
        start=time.monotonic()
        with self.assertRaises(h.queue.Empty):
            reader.line(start+0.02)
        self.assertLess(time.monotonic()-start,1)

    def test_real_pipe_startup_and_submit(self):
        child = r'''
import sys
sys.stdout.buffer.write(b"\x01\x01READY\x01\x01\nSTAT 0 0 0 0\nEMAP 42 288 " + b"00" * (42 * 288) + b"\n")
sys.stdout.buffer.flush()
header = sys.stdin.buffer.readline()
assert header == b"SUBMIT 1 0 6 16 0 1 logprobs=1\n", header
assert sys.stdin.buffer.read(7) == b"prompt\n"
for i in range(4):
    sys.stdout.buffer.write(("DATA 1 1 -0.1 1 %d -0.1\nX\n" % i).encode())
sys.stdout.buffer.write(b"DONE 1 STAT 4 1 0 0 8 1\n")
sys.stdout.buffer.flush()
assert sys.stdin.buffer.read() == b""
'''
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=Path(d),model=Path("model"),usage=Path("usage"),
                binary=Path("glm53"),read_timeout=2,request_timeout=2,startup_timeout=2)
            original=subprocess.Popen
            def spawn(argv, **kwargs):
                self.assertEqual(argv,["glm53","1024"])
                self.assertEqual(kwargs["bufsize"],0)
                return original([sys.executable,"-c",child],**kwargs)
            with mock.patch.object(h.subprocess,"Popen",side_effect=spawn):
                e=h.Engine(args,0)
                try:
                    self.assertEqual(e.startup_frames,["READY","STAT","EMAP"])
                    self.assertFalse(e.p.stdin.closed)
                    self.assertEqual(e.request(1,0,b"prompt")["tokens"],[0,1,2,3])
                    e.close()
                finally:
                    if not e.closed:
                        e.abort()

    def test_live_child_timeout_diagnostics(self):
        child = r'''
import sys
sys.stdout.buffer.write(b"\x01\x01READY\x01\x01\nSTAT 0 0 0 0\nEMAP 1 1 00\n")
sys.stdout.buffer.flush()
sys.stdin.buffer.read()  # Intentionally silent until the writer closes.
'''
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=Path(d),model=Path("model"),usage=Path("usage"),
                binary=Path("glm53"),read_timeout=2,request_timeout=0.03,startup_timeout=2)
            original=subprocess.Popen
            with mock.patch.object(h.subprocess,"Popen",side_effect=lambda argv,**kw:
                                   original([sys.executable,"-c",child],**kw)):
                e=h.Engine(args,0)
                try:
                    with self.assertRaises(AssertionError):
                        e.request(1,0,b"prompt")
                finally:
                    e.abort()
            self.assertEqual(e.failure["reason"],"protocol line timeout")
            self.assertIsNone(e.failure["original_returncode"])
            self.assertFalse(e.failure["stdin_closed"])
            self.assertEqual(e.failure["request_id"],1)
            self.assertEqual(e.failure["cleanup_action"],"terminate live child")


if __name__ == "__main__":
    unittest.main()
