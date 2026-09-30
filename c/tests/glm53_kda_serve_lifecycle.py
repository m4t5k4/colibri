#!/usr/bin/env python3
"""Sequential OFF/ON stock GLM53 SERVE lifecycle validation on the CUDA rig."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

USAGE_SHA = "7a246a49fc1dc3fb6360c131153c22012ca08949da390455654652d76e18a953"


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
        env = dict(os.environ, SERVE="1", SERVE_BATCH="1", KV_SLOTS="2",
                   COLI_PIN_SLOTS="4", SNAP=str(args.model),
                   COLI_USAGE=str(args.usage), USAGE_SAVE="0",
                   GLM53_CUDA_WARM_RESIDENCY="0", GLM53_CUDA_KDA=str(mode),
                   GLM53_CUDA_PROFILE="1", GLM53_VERBOSE="1")
        self.p = subprocess.Popen([str(args.binary), "1024"], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.stderr)
        self.closed = False
        try:
            while b"READY" not in self.line():
                pass
        except BaseException:
            self.abort()
            raise

    def read(self, count):
        data = self.p.stdout.read(count)
        self.raw.write(data)
        if len(data) != count:
            raise AssertionError(f"mode={self.mode}: truncated frame; see {self.log}")
        return data

    def line(self):
        data = self.p.stdout.readline()
        self.raw.write(data)
        if not data:
            raise AssertionError(f"mode={self.mode}: unexpected EOF; see {self.log}")
        return data.rstrip(b"\r\n")

    def request(self, rid, slot, payload):
        header = f"SUBMIT {rid} {slot} {len(payload)} 16 0 1 logprobs=1\n".encode()
        self.p.stdin.write(header + payload + b"\n")
        self.p.stdin.flush()
        tokens, pieces = [], []
        while True:
            fields = self.line().split()
            if not fields:
                continue
            if fields[0] in (b"DATA", b"ECHO"):
                assert int(fields[1]) == rid, fields
                body = self.read(int(fields[2]))
                assert self.read(1) == b"\n"
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
        while True:
            data = self.p.stdout.read(65536)
            if not data:
                break
            self.raw.write(data)
        code = self.p.wait(timeout=120)
        self.p.stdout.close()
        self.stderr.close()
        self.raw.close()
        self.closed = True
        assert code == 0, f"mode={self.mode}: exit={code}; see {self.log}"

    def abort(self):
        if not self.closed:
            self.p.kill()
            self.p.wait()
            self.p.stdin.close()
            self.p.stdout.close()
            self.stderr.close()
            self.raw.close()
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


def check_log(path, mode, results):
    text = path.read_text(errors="replace")
    unexpected_errors(text)
    reuse = {int(r): (int(n), int(p)) for r, n, p in
             re.findall(r"^REUSE (\d+) (\d+) (\d+)$", text, re.M)}
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
        for rid in range(1, 9):
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
        assert same(results[6], results[7]), "reset/fresh-slot reference mismatch"
        e.close()
        health = check_log(e.log, mode, results)
        (args.output / f"{mode}.results.json").write_text(json.dumps(
            {"requests": results, "health": health}, indent=2) + "\n")
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
    args = p.parse_args()
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
    run(args, 1, reference)
    assert before == {"binary": sha(args.binary), "usage": sha(args.usage)}
    print("PASS stock CUDA SERVE lifecycle; hashes unchanged; no injected failures")


if __name__ == "__main__":
    main()
