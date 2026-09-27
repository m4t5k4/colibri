#!/usr/bin/env python3
"""Compare GLM53 CUDA cache policies from GLM53_CUDA_TRACE selection events.

This is a capacity-only, whole-expert simulation. It does not model per-device
stranding, upload time, or CUDA errors. A miss is loaded after its CPU-served
selection, matching the tier's promotion timing for the comparison.
"""
import argparse
from collections import Counter, OrderedDict
from pathlib import Path


def selections(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("S,"):
                continue
            fields = line.rstrip("\n").split(",")
            if len(fields) != 7:
                raise ValueError(f"malformed selection trace: {line.rstrip()}")
            _, tick, token, layer, eid, rows, resident = fields
            yield int(tick), int(token), (int(layer), int(eid)), int(rows), int(resident)


def compare(path, capacity):
    if capacity < 1:
        raise ValueError("capacity must be positive")
    frequency = Counter()
    tokens = set()
    for _, token, key, rows, _ in selections(path):
        if token:
            frequency[key] += rows
            tokens.add(token)
    if not tokens:
        raise ValueError("trace contains no decode selections")
    static = set(sorted(frequency, key=lambda key: (-frequency[key], key))[:capacity])
    lru = OrderedDict()
    static_seen = set()
    total = current = lru_hits = static_cold = static_preloaded = 0
    events = 0
    for _, token, key, rows, resident in selections(path):
        lru_hit = key in lru
        if lru_hit:
            lru.move_to_end(key)
        else:
            lru[key] = None
            if len(lru) > capacity:
                lru.popitem(last=False)
        cold_hit = key in static_seen
        if key in static:
            static_seen.add(key)
        if token:
            events += 1
            total += rows
            current += resident * rows
            lru_hits += lru_hit * rows
            static_cold += cold_hit * rows
            static_preloaded += (key in static) * rows
    return {"tokens": len(tokens), "events": events, "capacity": capacity,
            "total_rows": total, "current_rows": current, "lru_rows": lru_hits,
            "static_cold_rows": static_cold,
            "static_preloaded_rows": static_preloaded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--capacity", type=int, required=True,
                        help="total whole-expert slots (847 on the validated 2x3070 run)")
    args = parser.parse_args()
    result = compare(args.trace, args.capacity)
    print(f"decode_tokens={result['tokens']} selections={result['events']} "
          f"capacity={result['capacity']} routed_rows={result['total_rows']}")
    for label in ("current", "lru", "static_cold", "static_preloaded"):
        hits = result[label + "_rows"]
        print(f"{label}_rows={hits} rows_per_token={hits / result['tokens']:.3f} "
              f"hit_fraction={hits / result['total_rows']:.4f}")
    print("# static_preloaded uses future decode frequencies; it is an upper bound, "
          "not an implementable online policy.")


if __name__ == "__main__":
    main()
