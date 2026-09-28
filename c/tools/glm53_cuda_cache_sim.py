#!/usr/bin/env python3
"""Replay GLM53 CUDA expert selections under small cache-policy changes.

Old S/E traces permit decode-layer-level replay. New A/P events delimit the
attempt phase and record successful upload ownership for validation. This
models equal-size whole experts and successful uploads, not backend time.
"""
import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


UPLOAD_SECONDS = 37.338177 / 1972  # rough accounting, not a speed prediction
POLICIES = (("current_t2_m0", 2, 0),
            ("threshold_3", 3, 0), ("threshold_4", 4, 0),
            ("threshold_5", 5, 0), ("margin_1", 2, 1),
            ("margin_2", 2, 2), ("threshold_3_margin_1", 3, 1))


def events(path):
    lengths = {"S": 7, "A": 4, "P": 6, "E": 10}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.strip().split(",")
            kind = fields[0]
            # Optional per-expert upload timing is diagnostic, not a cache
            # policy event. Keep old and new traces replayable alike.
            if kind == "U" and len(fields) == 7:
                continue
            if kind not in lengths or len(fields) != lengths[kind]:
                raise ValueError(f"malformed trace event on line {number}: {line.rstrip()}")
            try:
                yield kind, tuple(map(int, fields[1:]))
            except ValueError as exc:
                raise ValueError(f"non-integer trace field on line {number}") from exc


def selections(path):
    """Selection-only iterator retained for the original comparison."""
    for kind, values in events(path):
        if kind == "S":
            tick, token, layer, eid, rows, resident = values
            yield tick, token, (layer, eid), rows, resident


@dataclass
class PolicyState:
    capacities: tuple
    threshold: int
    margin: int
    heat: Counter = field(default_factory=Counter)
    resident: dict = field(default_factory=dict)
    used: list = field(init=False)
    promoted: Counter = field(default_factory=Counter)
    hits_since_upload: Counter = field(default_factory=Counter)
    hit_rows: int = 0
    promotions: int = 0
    evictions: int = 0
    repromotions: int = 0
    dead_on_arrival: int = 0
    upload_events: list = field(default_factory=list)
    eviction_events: list = field(default_factory=list)

    def __post_init__(self):
        self.used = [0] * len(self.capacities)

    def select(self, key, rows, decode):
        self.heat[key] += rows
        if key in self.resident:
            self.hits_since_upload[key] += 1
            if decode:
                self.hit_rows += rows

    def attempt(self, key):
        if key in self.resident or self.heat[key] < self.threshold:
            return
        eligible = [i for i, cap in enumerate(self.capacities)
                    if self.used[i] < cap]
        if eligible:
            owner = min(eligible, key=lambda i: (self.used[i], i))
        else:
            if not self.resident:
                return
            # C scans g->experts in layer/eid order, retaining the first tie.
            victim = min(self.resident, key=lambda k: (self.heat[k], k))
            if self.heat[key] <= self.heat[victim] + self.margin:
                return
            self.used[self.resident.pop(victim)] -= 1
            self.evictions += 1
            self.eviction_events.append((key, victim))
            if not self.hits_since_upload[victim]:
                self.dead_on_arrival += 1
            # The runtime places again after dropping the victim.
            eligible = [i for i, cap in enumerate(self.capacities)
                        if self.used[i] < cap]
            owner = min(eligible, key=lambda i: (self.used[i], i))
        if self.promoted[key]:
            self.repromotions += 1
        self.promoted[key] += 1
        self.hits_since_upload[key] = 0
        self.resident[key] = owner
        self.used[owner] += 1
        self.promotions += 1
        self.upload_events.append((key, owner))


def replay(trace, capacities):
    if not capacities or any(cap < 1 for cap in capacities):
        raise ValueError("capacities must be positive")
    states = {name: PolicyState(tuple(capacities), threshold, margin)
              for name, threshold, margin in POLICIES}
    pending = []
    group = None
    observed_rows = total_rows = event_count = snapshot_mismatches = 0
    tokens = set()
    actual_uploads = []
    actual_evictions = []
    attempt_events = 0

    def finish_group():
        nonlocal pending
        # Every policy attempts every selected expert, including an observed
        # hit which would have missed under that counterfactual policy.
        for key, missed_under in pending:
            for name in missed_under:
                states[name].attempt(key)
        pending = []

    for kind, value in trace:
        if kind == "S":
            tick, token, layer, eid, rows, observed_resident = value
            if rows < 0 or token < 0 or observed_resident not in (0, 1):
                raise ValueError("invalid selection rows, token or residency")
            next_group = (token, layer)
            if pending and next_group != group:
                finish_group()
            group = next_group
            key = (layer, eid)
            baseline = states[POLICIES[0][0]]
            snapshot_mismatches += (key in baseline.resident) != bool(observed_resident)
            for state in states.values():
                state.select(key, rows, bool(token))
            pending.append((key, tuple(name for name, state in states.items()
                                       if key not in state.resident)))
            if token:
                tokens.add(token)
                event_count += 1
                total_rows += rows
                observed_rows += rows * observed_resident
        elif kind == "A":
            attempt_events += 1
            if pending:
                finish_group()
        elif kind == "P":
            _, layer, eid, owner, _device_id = value
            actual_uploads.append(((layer, eid), owner))
        elif kind == "E":
            _, incoming_layer, incoming_eid, victim_layer, victim_eid, *_ = value
            actual_evictions.append(((incoming_layer, incoming_eid),
                                     (victim_layer, victim_eid)))
            if pending:
                finish_group()
    finish_group()
    if not tokens or not total_rows:
        raise ValueError("trace contains no decode selections with routed rows")
    baseline = states[POLICIES[0][0]]
    def mismatches(actual, simulated):
        if not actual:
            return None
        return (sum(a != b for a, b in zip(actual, simulated))
                + abs(len(actual) - len(simulated)))
    uploads_for_check = (actual_uploads if len(capacities) > 1 else
                         [key for key, _ in actual_uploads])
    simulated_for_check = (baseline.upload_events if len(capacities) > 1 else
                           [key for key, _ in baseline.upload_events])
    return {"tokens": len(tokens), "events": event_count,
            "total_rows": total_rows, "current_rows": observed_rows,
            "states": states, "selection_snapshot_mismatches": snapshot_mismatches,
            "attempt_events": attempt_events, "actual_upload_events": len(actual_uploads),
            "upload_event_mismatches": mismatches(uploads_for_check, simulated_for_check),
            "eviction_event_mismatches": mismatches(actual_evictions, baseline.eviction_events)}


def compare(path, capacity):
    """Original global LRU/static comparison, retained for existing callers."""
    if capacity < 1:
        raise ValueError("capacity must be positive")
    frequency = Counter()
    tokens = set()
    for _, token, key, rows, _ in selections(path):
        if token:
            frequency[key] += rows
            tokens.add(token)
    if not tokens or not sum(frequency.values()):
        raise ValueError("trace contains no decode selections with routed rows")
    static = set(sorted(frequency, key=lambda key: (-frequency[key], key))[:capacity])
    lru = OrderedDict()
    static_seen = set()
    total = current = lru_hits = static_cold = static_preloaded = count = 0
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
            count += 1
            total += rows
            current += resident * rows
            lru_hits += lru_hit * rows
            static_cold += cold_hit * rows
            static_preloaded += (key in static) * rows
    return {"tokens": len(tokens), "events": count, "capacity": capacity,
            "total_rows": total, "current_rows": current, "lru_rows": lru_hits,
            "static_cold_rows": static_cold,
            "static_preloaded_rows": static_preloaded}


def print_replay(label, result):
    baseline = result["states"][POLICIES[0][0]]
    total = result["total_rows"]
    tokens = result["tokens"]
    print(f"{label}: observed_rows={result['current_rows']} "
          f"simulated_current_rows={baseline.hit_rows} "
          f"timing={'explicit' if result['attempt_events'] else 'inferred'} "
          f"upload_owner_events={result['actual_upload_events']} "
          f"selection_snapshot_mismatches={result['selection_snapshot_mismatches']} "
          f"upload_event_mismatches={result['upload_event_mismatches']} "
          f"eviction_event_mismatches={result['eviction_event_mismatches']}")
    if (baseline.hit_rows != result["current_rows"] or
            result["selection_snapshot_mismatches"] or
            result["upload_event_mismatches"] not in (None, 0) or
            result["eviction_event_mismatches"] not in (None, 0)):
        print("# WARNING: current-policy replay differs from observed trace; "
              "counterfactual results require investigation.")
    print("policy                  threshold margin cuda_rows rows/token hit_frac promotions evictions "
          "re_promotions dead_on_arrival avoided_promotions extra_hit_rows missing_hit_rows "
          "rough_upload_s_saved")
    for name, threshold, margin in POLICIES:
        state = result["states"][name]
        delta = state.hit_rows - baseline.hit_rows
        avoided = baseline.promotions - state.promotions
        print(f"{name:23} {threshold:9} {margin:6} "
              f"{state.hit_rows:9} {state.hit_rows / tokens:10.3f} "
              f"{state.hit_rows / total:8.4f} {state.promotions:10} "
              f"{state.evictions:9} {state.repromotions:13} "
              f"{state.dead_on_arrival:15} {avoided:15} "
              f"{max(delta, 0):10} {max(-delta, 0):12} "
              f"{avoided * UPLOAD_SECONDS:20.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--capacity", type=int, required=True,
                        help="total whole-expert slots (847 on the 2x3070 run)")
    parser.add_argument("--device-capacities", type=str,
                        help="ordered per-device slots, e.g. 424,423")
    args = parser.parse_args()
    old = compare(args.trace, args.capacity)
    print(f"decode_tokens={old['tokens']} selections={old['events']} "
          f"capacity={old['capacity']} routed_rows={old['total_rows']}")
    for label in ("current", "lru", "static_cold", "static_preloaded"):
        hits = old[label + "_rows"]
        print(f"{label}_rows={hits} rows_per_token={hits / old['tokens']:.3f} "
              f"hit_fraction={hits / old['total_rows']:.4f}")
    print_replay("global", replay(events(args.trace), (args.capacity,)))
    if args.device_capacities:
        try:
            capacities = tuple(int(piece) for piece in args.device_capacities.split(","))
        except ValueError as exc:
            parser.error(f"invalid --device-capacities: {exc}")
        if sum(capacities) != args.capacity:
            parser.error("sum of --device-capacities must equal --capacity")
        print_replay("device-aware " + args.device_capacities,
                     replay(events(args.trace), capacities))
    print(f"rough_upload_cost_s={UPLOAD_SECONDS:.6f} "
          "(37.338177 s / 1972 uploads; accounting only, not a performance prediction)")
    print("# static_preloaded uses future decode frequencies and is an upper bound; "
          "replay assumes equal-size experts and successful uploads.")


if __name__ == "__main__":
    main()
