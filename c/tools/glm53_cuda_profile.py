#!/usr/bin/env python3
"""Decode deltas from cumulative GLM53_CUDA_PROFILE=1 snapshots (stdlib only).

A trailing error-free decode window without uploads or evictions is a steady
working-set candidate. Tier fullness remains a separate occupancy gauge.
"""
import argparse
import csv
import math
import sys

COUNTS = ("tokens", "decode_tokens", "forwards", "cuda_rows", "fallback_rows",
          "fallback_compute_rows", "uploads", "evictions", "errors")
TIMES = ("forward_s", "decode_forward_s", "disk_s", "promotion_s", "upload_s",
         "eviction_s", "gate_s", "up_s", "clamp_s", "down_s", "group_issue_s",
         "group_take_s", "cuda_expert_s",
         "fallback_compute_s", "attn_s", "ffn_s", "head_s")
GAUGES = ("resident", "vram_bytes", "budget_bytes", "tier_full")


def decode_rows(lines):
    previous = None
    previous_phase = None
    run = 0
    for line in lines:
        if not line.startswith("[glm53-cuda-profile] "):
            continue
        fields = dict(word.split("=", 1) for word in line.split()[1:])
        phase = fields["phase"]
        current = {k: float(fields.get(k, 0)) if k in TIMES else int(fields[k])
                   for k in COUNTS + TIMES + GAUGES}
        if any(not math.isfinite(v) or v < 0 for v in current.values()):
            raise ValueError("nonfinite or negative profile value")
        if phase == "start":
            previous = current
            previous_phase = phase
            run += 1
            continue
        if previous is None:
            raise ValueError("missing phase=start baseline; capture the complete profile")
        delta = {k: current[k] - previous[k] for k in COUNTS + TIMES}
        if any(v < -1e-6 for v in delta.values()):
            raise ValueError("profile counters decreased; do not interleave model logs")
        if phase == "decode":
            if delta["decode_tokens"] != 1 or delta["forwards"] != 1:
                raise ValueError("expected one decode forward per snapshot")
            yield {"run": run, "token": current["decode_tokens"], **delta,
                   **{k: current[k] for k in GAUGES},
                   "continuation": previous_phase == "decode",
                   "full_before": previous["tier_full"], "total_errors": current["errors"]}
        previous = current
        previous_phase = phase


def warm_window(rows, size):
    """Find a trailing decode span with no errors, uploads, or evictions."""
    tail = []
    for row in rows:
        if tail and (not row["continuation"] or row["run"] != tail[-1]["run"] or row["token"] != tail[-1]["token"] + 1):
            tail = []
        if row["total_errors"] or row["errors"] or row["uploads"] or row["evictions"]:
            tail = []
        else:
            tail.append(row)
    return tail[-size:] if len(tail) >= size else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--warm-window", type=int, default=16)
    args = parser.parse_args()
    if args.warm_window < 1:
        parser.error("--warm-window must be positive")
    with open(args.log, encoding="utf-8", errors="replace") as stream:
        rows = list(decode_rows(stream))
    if not rows:
        parser.error("no decode profile snapshots found")
    columns = ("run", "token", "resident", "vram_bytes", "tier_full",
               "cuda_rows_delta", "fallback_rows_delta", "uploads_delta",
               "evictions_delta", "errors_delta") + tuple(k + "_delta" for k in TIMES)
    writer = csv.DictWriter(sys.stdout, columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        output = {k: row[k] for k in ("run", "token", "resident", "vram_bytes", "tier_full")}
        for key in ("cuda_rows", "fallback_rows", "uploads", "evictions", "errors") + TIMES:
            output[key + "_delta"] = f"{row[key]:.6f}" if key in TIMES else row[key]
        writer.writerow(output)
    warm = warm_window(rows, args.warm_window)
    if not warm:
        print("# No sufficiently long trailing steady working-set decode window.")
    else:
        fullness = "full" if all(r["tier_full"] for r in warm) else "not_full"
        print(f"# Steady working-set candidate: run={warm[0]['run']} tokens={warm[0]['token']}..{warm[-1]['token']}; tier={fullness}.")
        print("# Mean deltas per token: " + " ".join(
            f"{k}_delta={sum(r[k] for r in warm)/len(warm):.6f}"
            for k in ("cuda_rows", "fallback_rows", "uploads", "evictions") + TIMES))
        print("# errors_delta=uploads_delta=evictions_delta=0 throughout this candidate;"
              " tier_full reports occupancy separately.")


if __name__ == "__main__":
    main()
