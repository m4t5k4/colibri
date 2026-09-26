#!/usr/bin/env python3
"""Real streaming-model CUDA A/B. Requires a CUDA binary and int4-gs64 fixture.

GPU detection cannot pass: successful expert matmuls, VRAM residency and host
fallback must all occur. Use make glm53-cuda-check for tensor-level numerics.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    ref_path = args.fixture / "ref.json"
    if not ref_path.exists():
        print("SKIP: generate an int4 fixture with tools/make_glm53_streaming_pair.py")
        return 2
    ref = json.loads(ref_path.read_text())
    cmd = [str(Path(args.binary).resolve()), "--model", str(args.fixture),
           "--ids", ",".join(map(str, ref["prompt"])), "--greedy", "8"]
    if (args.fixture / "patches.f32").exists():
        cmd += ["--patches", str(args.fixture / "patches.f32"),
                "--grid", "x".join(map(str, ref["grid"]))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("COLI_CUDA", "COLI_GPU", "CUDA_"))}
    env.update(GLM53_BITS="32", GLM53_EXPERT_GB="0.000001", OMP_NUM_THREADS="2")

    def run(**overrides):
        p = subprocess.run(cmd, env={**env, **overrides}, capture_output=True, text=True)
        if p.returncode:
            raise AssertionError(p.stderr)
        tokens = [line for line in p.stdout.splitlines()
                  if line.startswith(("teacher_forcing ", "greedy "))]
        assert tokens, p.stdout
        stats = re.findall(r"resident=(\d+) vram_bytes=(\d+) executed=(\d+) fallback=(\d+) uploads=(\d+) errors=(\d+)", p.stderr)
        return tokens, tuple(map(int, stats[-1])) if stats else None

    cpu, _ = run(COLI_CUDA="0")
    gpu, stats = run(COLI_CUDA="1", COLI_GPU="0", CUDA_EXPERT_GB="0.1")
    assert gpu == cpu, (gpu, cpu)
    assert stats and all(v > 0 for v in stats[:5]) and stats[5] == 0, stats
    zero, stats = run(COLI_CUDA="1", COLI_GPU="0", CUDA_EXPERT_GB="0")
    assert zero == cpu and stats and stats[0] == stats[1] == stats[2] == 0 and stats[3] > 0, stats
    failed, stats = run(COLI_CUDA="1", COLI_GPU="0", CUDA_EXPERT_GB="0.1", COLI_GPU_FAIL_AFTER="2")
    assert failed == cpu and stats and stats[3] > 0 and stats[5] > 0, stats
    print("PASS GLM53 model CUDA: executed tensors, residency, zero-budget and injected-failure fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
