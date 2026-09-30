#!/usr/bin/env python3
"""Sequential OFF/ON stock GLM53 SERVE lifecycle validation on the CUDA rig."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import queue
import threading
import time
import subprocess
import stat


def proc_snapshot(pid):
    """Best-effort, read-only Linux evidence; races/errors stay in the report."""
    root = Path(f"/proc/{pid}")
    result = {"pid": pid, "monotonic_s": time.monotonic()}
    def read(path):
        try:
            return path.read_text(errors="replace")
        except OSError as error:
            return {"error": str(error)}
    def link(path):
        try:
            return os.readlink(path)
        except OSError as error:
            return {"error": str(error)}
    for name in ("status", "stat", "wchan"):
        result[name] = read(root / name)
    result["fd"] = {str(i): link(root / "fd" / str(i)) for i in range(3)}
    result["fdinfo"] = {str(i): read(root / "fdinfo" / str(i)) for i in range(3)}
    try:
        result["fd_listing"] = {p.name: link(p) for p in
                                sorted((root / "fd").iterdir(), key=lambda p: int(p.name))}
    except OSError as error:
        result["fd_listing"] = {"error": str(error)}
    try:
        result["task_wchan"] = {p.name: read(p / "wchan") for p in
                                sorted((root / "task").iterdir(), key=lambda p: int(p.name))}
    except OSError as error:
        result["task_wchan"] = {"error": str(error)}
    return result


def pipe_identity(stream):
    if os.name != "posix":
        # Windows CRT fd inspection can wait on another thread's blocking read.
        return {"unavailable": "POSIX pipe metadata only", "stream_type": type(stream).__name__}
    try:
        fd = stream.fileno()
        info = os.fstat(fd)
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError as error:
            target = {"error": str(error)}
        return {"fd": fd, "target": target, "inode": info.st_ino,
                "device": info.st_dev, "blocking": os.get_blocking(fd),
                "is_pipe": stat.S_ISFIFO(info.st_mode),
                "stream_type": type(stream).__name__}
    except (OSError, ValueError) as error:
        return {"error": str(error)}


def unread_pipe_bytes(stream):
    try:
        import array
        import fcntl
        import termios
        value = array.array("i", [0])
        fcntl.ioctl(stream.fileno(), termios.FIONREAD, value, True)
        return value[0]
    except (ImportError, AttributeError, OSError, ValueError) as error:
        return {"error": str(error)}

USAGE_SHA = "7a246a49fc1dc3fb6360c131153c22012ca08949da390455654652d76e18a953"


class PipeReader:
    """Drain the raw pipe independently; deadlines include partial frames."""
    def __init__(self, stream, raw, on_eof=None):
        self.items = queue.Queue()
        self.buffer = bytearray()
        self.eof = False
        self.raw = raw
        self.on_eof = on_eof
        self.terminal = None
        self.thread = threading.Thread(target=self.pump, args=(stream,), daemon=True)
        self.thread.start()

    def pump(self, stream):
        try:
            while True:
                data = stream.read(65536)
                if data is None:
                    # Raw nonblocking streams may return None, not EOF. Do not
                    # silently turn that into the b"" EOF case.
                    raise BlockingIOError("raw stdout read returned None (would block), not EOF")
                if data == b"":
                    self.terminal = {"kind": "zero_byte_read", "requested_bytes": 65536,
                                     "returned_bytes": 0, "pipe": pipe_identity(stream)}
                    if self.on_eof:
                        self.terminal["process_scene"] = self.on_eof()
                    break
                self.raw.write(data)
                self.raw.flush()
                self.items.put(data)
        except Exception as error:
            self.terminal = {"kind": "reader_exception", "error": repr(error)}
            self.items.put(error)
        finally:
            self.items.put(None)

    def more(self, deadline):
        item = self.items.get(timeout=max(0, deadline - time.monotonic()))
        if isinstance(item, Exception):
            raise item
        if item is None:
            self.eof = True
        else:
            self.buffer.extend(item)

    def read(self, count, deadline):
        while len(self.buffer) < count and not self.eof:
            self.more(deadline)
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        return data

    def line(self, deadline):
        while b"\n" not in self.buffer and not self.eof:
            self.more(deadline)
        count = self.buffer.find(b"\n") + 1
        if not count:
            count = len(self.buffer)
        return self.read(count, deadline)


def write_all(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise BrokenPipeError("SUBMIT write made no progress")
        view = view[written:]
    stream.flush()
    return len(data)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Engine:
    def __init__(self, args, mode):
        self.mode = mode
        self.log = args.output / f"{mode}.stderr.log"
        self.stderr = self.log.open("wb")
        self.raw = (args.output / f"{mode}.stdout.bin").open("wb")
        self.input_log = (args.output / f"{mode}.stdin.bin").open("wb")
        self.failure_path = args.output / f"{mode}.failure.json"
        self.failure = None
        self.startup_frames = []
        self.last_frame = None
        self.request_id = None
        self.submitted_bytes = 0
        self.first_request = None
        self.read_timeout = args.read_timeout
        self.request_timeout = args.request_timeout
        self.deadline = time.monotonic() + args.startup_timeout
        env = dict(os.environ, SERVE="1", SERVE_BATCH="1", KV_SLOTS="2",
                   COLI_PIN_SLOTS="4", SNAP=str(args.model),
                   COLI_USAGE=str(args.usage), USAGE_SAVE="0",
                   GLM53_CUDA_WARM_RESIDENCY="0", GLM53_CUDA_KDA=str(mode),
                   GLM53_CUDA_PROFILE="1", GLM53_VERBOSE="1")
        self.p = subprocess.Popen([str(args.binary), "1024"], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.stderr, bufsize=0)
        self.parent_pipes_at_launch = {"stdin": pipe_identity(self.p.stdin),
                                      "stdout": pipe_identity(self.p.stdout)}
        self.reader = PipeReader(self.p.stdout, self.raw, self.process_scene)
        self.closed = False
        try:
            while True:
                frame = self.line()
                self.startup_frames.append(frame.split()[0].decode(errors="replace"))
                if frame == b"\x01\x01READY\x01\x01":
                    self.startup_frames[-1] = "READY"
                    break
            for expected in (b"STAT", b"EMAP"):
                fields = self.line().split()
                assert fields and fields[0] == expected, fields
                self.startup_frames.append(expected.decode())
                if expected == b"EMAP":
                    assert len(fields) == 4 and len(fields[3]) == 2 * int(fields[1]) * int(fields[2]), "bad startup EMAP"
            self.deadline = None
        except BaseException:
            self.abort()
            raise

    def process_scene(self):
        child = proc_snapshot(self.p.pid)
        pipes = {"stdin": pipe_identity(self.p.stdin),
                 "stdout": pipe_identity(self.p.stdout)}
        child_stdout = child["fd"].get("1")
        expected = pipes["stdout"].get("target")
        matches = child_stdout == expected if isinstance(child_stdout, str) and isinstance(expected, str) else None
        return {"child": child,
                "parent_pid": os.getpid(),
                "parent_pipes": pipes, "child_stdout_matches_parent_pipe": matches,
                "stdin_pipe_unread_bytes": unread_pipe_bytes(self.p.stdin)}

    def diagnostics(self, reason):
        if self.failure is not None:
            return self.failure
        before = self.p.poll()
        scene = self.process_scene()  # Always before wait/cleanup signals.
        original = before
        if original is None:
            try:
                original = self.p.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        self.failure = {"reason": reason, "mode": self.mode,
            "request_id": self.request_id, "pid": self.p.pid,
            "poll_before_cleanup": before, "original_returncode": original,
            "original_signal": -original if original is not None and original < 0 else None,
            "stdin_closed": self.p.stdin.closed,
            "startup_frame_count": len(self.startup_frames),
            "startup_frames": self.startup_frames, "last_complete_frame": self.last_frame,
            "stdout_bytes": self.raw.tell(), "buffered_stdout_bytes": len(self.reader.buffer),
            "stderr_tail": self.log.read_bytes()[-6000:].decode(errors="replace")}
        self.input_log.flush()
        input_path = Path(self.input_log.name)
        self.failure.update({"process_scene_before_cleanup": scene,
            "reader_terminal": self.reader.terminal,
            "parent_pipes_at_launch": getattr(self, "parent_pipes_at_launch", None),
            "stdin_capture": {"path": str(input_path), "size": input_path.stat().st_size,
                              "sha256": sha(input_path),
                              "write_all_completed_bytes": self.submitted_bytes,
                              "first_request": self.first_request}})
        self.failure_path.write_text(json.dumps(self.failure, indent=2) + "\n")
        return self.failure

    def fail(self, reason):
        raise AssertionError(json.dumps(self.diagnostics(reason), indent=2))

    def read_deadline(self):
        idle = time.monotonic() + self.read_timeout
        return min(idle, self.deadline) if self.deadline is not None else idle

    def read(self, count):
        try:
            data = self.reader.read(count, self.read_deadline())
        except queue.Empty:
            self.fail("protocol read timeout")
        if len(data) != count:
            self.fail(f"truncated frame: wanted={count} got={len(data)}")
        return data

    def line(self):
        try:
            data = self.reader.line(self.read_deadline())
        except queue.Empty:
            self.fail("protocol line timeout")
        if not data or not data.endswith(b"\n"):
            self.fail("unexpected EOF" if not data else "EOF in partial header")
        if not data.startswith((b"DATA ", b"ECHO ")):
            self.last_frame = data[:240].decode(errors="replace")
        return data.rstrip(b"\r\n")

    def request(self, rid, slot, payload):
        self.request_id = rid
        self.deadline = time.monotonic() + self.request_timeout
        header = f"SUBMIT {rid} {slot} {len(payload)} 16 0 1 logprobs=1\n".encode()
        wire = header + payload + b"\n"
        self.input_log.write(wire)
        self.input_log.flush()
        if self.first_request is None:
            self.first_request = {"bytes": len(wire), "sha256": hashlib.sha256(wire).hexdigest(),
                                  "escaped": repr(wire[:4096]), "escaped_truncated": len(wire) > 4096}
        self.submitted_bytes += write_all(self.p.stdin, wire)
        tokens, pieces = [], []
        while True:
            fields = self.line().split()
            if not fields:
                continue
            if fields[0] in (b"DATA", b"ECHO"):
                assert int(fields[1]) == rid, fields
                body = self.read(int(fields[2]))
                assert self.read(1) == b"\n"
                self.last_frame = f"{fields[0].decode()} request={rid} payload_bytes={len(body)}"
                if fields[0] == b"DATA":
                    assert len(fields) == 7 and fields[4] == b"1", fields
                    assert math.isfinite(float(fields[3])) and math.isfinite(float(fields[6])), fields
                    tokens.append(int(fields[5]))  # Greedy token = top-1 ID.
                    pieces.append(body)
            elif fields[0] == b"ERROR":
                raise AssertionError(fields)
            elif fields[0] == b"DONE":
                assert int(fields[1]) == rid and fields[2] == b"STAT", fields
                emitted, limited = int(fields[3]), int(fields[8])
                assert emitted == len(tokens) and emitted >= 4, fields
                return {"id": rid, "slot": slot, "prompt_hex": payload.hex(),
                        "tokens": tokens, "data_hex": b"".join(pieces).hex(),
                        "emitted": emitted, "limited": limited,
                        "decode_forwards": emitted - 1 if limited else emitted}

    def close(self):
        self.p.stdin.close()
        self.deadline = time.monotonic() + 120
        try:
            while not self.reader.eof:
                self.reader.more(self.deadline)
                self.reader.buffer.clear()
            code = self.p.wait(timeout=max(0, self.deadline - time.monotonic()))
        except (queue.Empty, subprocess.TimeoutExpired):
            self.fail("shutdown timeout")
        self.reader.thread.join(timeout=1)
        self.p.stdout.close()
        self.stderr.close()
        self.raw.close()
        self.input_log.close()
        self.closed = True
        assert code == 0, f"mode={self.mode}: exit={code}; see {self.log}"

    def abort(self):
        if not self.closed:
            report = self.diagnostics("exception before cleanup")
            action = "none: child already exited"
            if self.p.poll() is None:
                action = "terminate live child"
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    action = "kill after terminate timeout"
                    self.p.kill()
                    self.p.wait()
            self.reader.thread.join(timeout=2)
            report["cleanup_action"] = action
            report["returncode_after_cleanup"] = self.p.returncode
            self.failure_path.write_text(json.dumps(report, indent=2) + "\n")
            self.p.stdin.close()
            self.p.stdout.close()
            self.stderr.close()
            self.raw.close()
            self.input_log.close()
            self.closed = True


def same(a, b):
    return all(a[k] == b[k] for k in ("tokens", "data_hex", "emitted", "limited"))


def unexpected_errors(text):
    assert not re.search(r"\[(?:glm53-kda-cuda-error|glm53-cuda-error)\]", text), "backend error diagnostic"
    for line in text.splitlines():
        if "[CUDA]" in line:
            assert re.match(r"^\[CUDA\] device \d+:", line), line
    assert not re.search(r"step failed|state push failed|committed state pull failed|initialization skipped|host fallback enabled", text), "CUDA fallback diagnostic"
    assert all(int(x) == 0 for x in re.findall(r"\berrors=(\d+)", text)), "expert CUDA error"


def check_log(path, mode, results, diagnostic=False):
    text = path.read_text(errors="replace")
    unexpected_errors(text)
    reuse = {int(r): (int(n), int(p)) for r, n, p in
             re.findall(r"^REUSE (\d+) (\d+) (\d+)$", text, re.M)}
    if not diagnostic:
        for rid in (3, 4, 5, 6):
            n, p = reuse[rid]
            assert n > 0 and p - n > 1, f"request {rid}: reuse/prefill unexercised"
        assert reuse[7][0] == reuse[8][0] == 0, "reset was not exercised"
    expert = re.findall(r"^\[glm53-cuda\].*\berrors=(\d+)", text, re.M)
    assert expert and all(int(x) == 0 for x in expert), "expert CUDA error"
    counters = {}
    if mode:
        lines = [x for x in text.splitlines() if "kda_cuda_calls=" in x]
        assert len(lines) == 1, "missing/ambiguous final KDA report"
        counters = dict(re.findall(r"(\w+)=([\d.]+)", lines[0]))
        layers = len(re.findall(r"^\[glm53-kda-cuda-layer\].*loaded=1$", text, re.M))
        assert layers == 34, f"expected model KDA objects: got {layers}"
        expected = layers * sum(x["decode_forwards"] for x in results)
        assert int(counters["kda_cuda_calls"]) == expected, (counters, expected)
        assert int(counters["kda_cuda_errors"]) == int(counters["kda_cuda_fallbacks"]) == 0
        assert all(int(counters[k]) > 0 for k in ("state_pushes", "state_pulls", "invalidations"))
    return {"reuse": reuse, "kda": counters, "expert_errors": expert}


def run(args, mode, reference=None):
    e = Engine(args, mode)
    results = []
    roots = [b"Explain virtual memory, page faults and the TLB in detail.",
             b"Write a C binary search function and explain all edge cases."]
    latest = {}
    try:
        for rid in range(1, 2 if args.diagnostic_one_request else 9):
            slot = (rid - 1) % 2
            if reference is not None:
                payload = bytes.fromhex(reference[rid - 1]["prompt_hex"])
            elif rid <= 2:
                payload = roots[slot]
            elif rid <= 6:
                prior = latest[slot]
                payload = bytes.fromhex(prior["prompt_hex"]) + bytes.fromhex(prior["data_hex"])
                payload += b"\nNow give a concrete example and explain additional limitations."
            else:
                payload = b"Compare TCP and UDP reliability, ordering and typical applications."
            result = e.request(rid, slot, payload)
            unexpected_errors(e.log.read_text(errors="replace"))
            if reference is not None:
                assert same(result, reference[rid - 1]), f"semantic mismatch request={rid}"
            latest[slot] = result
            results.append(result)
            print(f"mode={mode} request={rid} slot={slot} emitted={result['emitted']} semantic=PASS", flush=True)
        if not args.diagnostic_one_request:
            assert same(results[6], results[7]), "reset/fresh-slot reference mismatch"
        e.close()
        health = check_log(e.log, mode, results, args.diagnostic_one_request)
        (args.output / f"{mode}.results.json").write_text(json.dumps(
            {"requests": results, "health": health,
             "startup_frames": e.startup_frames,
             "returncode": e.p.returncode}, indent=2) + "\n")
        return results
    except BaseException:
        e.abort()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--usage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic-one-request", action="store_true",
                   help="Run only OFF request 1; never starts KDA ON")
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--read-timeout", type=float, default=600)
    p.add_argument("--request-timeout", type=float, default=1800)
    args = p.parse_args()
    assert all(x > 0 and math.isfinite(x) for x in
               (args.startup_timeout, args.read_timeout, args.request_timeout))
    for key in ("binary", "model", "usage", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.usage) == USAGE_SHA, "usage snapshot hash mismatch"
    before = {"binary": sha(args.binary), "usage": sha(args.usage)}
    (args.output / "frozen.json").write_text(json.dumps(
        {"hashes": before, "environment": {k: v for k, v in os.environ.items()
         if k.startswith(("OMP_", "COLI_", "CUDA_", "GLM53_")) or k in ("DRAFT", "USAGE_SAVE")}}, indent=2) + "\n")
    reference = run(args, 0)
    assert before == {"binary": sha(args.binary), "usage": sha(args.usage)}
    if args.diagnostic_one_request:
        print("PASS one-request OFF diagnostic; not full lifecycle or KDA ON validation")
        return
    run(args, 1, reference)
    assert before == {"binary": sha(args.binary), "usage": sha(args.usage)}
    print("PASS stock CUDA SERVE lifecycle; hashes unchanged; no injected failures")


if __name__ == "__main__":
    main()
