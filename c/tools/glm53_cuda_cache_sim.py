#!/usr/bin/env python3
"""Replay GLM53 CUDA expert selections under small cache-policy changes.

Old S/E traces permit decode-layer-level replay. New A/P events delimit the
attempt phase and record successful upload ownership for validation. This
models equal-size whole experts and successful uploads, not backend time.
"""
import argparse
import csv
import io
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


EXPERT_BYTES = 14_155_776
FRACTIONS = (0, 0.25, 0.5, 0.75, 1)
POLICIES = (("current_t2_m1", 2, 1), ("margin_0", 2, 0),
            ("threshold_3", 3, 0), ("threshold_4", 4, 0),
            ("threshold_5", 5, 0),
            ("margin_2", 2, 2), ("threshold_3_margin_1", 3, 1))


def events(path):
    lengths = {"S": 7, "A": 4, "P": 6, "E": 10, "W": 5, "F": 6, "B": 3}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.strip().split(",")
            kind = fields[0]
            if kind == "U" and len(fields) == 7:
                try:
                    yield kind, (int(fields[1]), int(fields[2]), int(fields[3]),
                                 int(fields[4]), float(fields[5]), int(fields[6]))
                except ValueError as exc:
                    raise ValueError(f"malformed upload event on line {number}") from exc
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


def warm_trace_prefix(trace):
    """Extract preload publications and the transition to normal inference."""
    placed = []
    boundary = None
    failed = False
    inference = False
    for kind, values in trace:
        if kind == "W":
            if inference or boundary is not None or failed:
                raise ValueError("warm publication after inference boundary")
            placed.append(values)
        elif kind == "B":
            if boundary is not None or inference or values != (len(placed), len(placed)):
                raise ValueError("inconsistent warm inference boundary")
            boundary = values
        elif kind == "F":
            if inference or boundary is not None or failed:
                raise ValueError("inconsistent warm failure event")
            failed = True
        elif kind == "S":
            inference = True
            if placed and boundary is None:
                raise ValueError("warm trace lacks inference boundary")
    return placed, boundary


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
    attempts: list = field(default_factory=list)
    warm_initial: set = field(default_factory=set)
    warm_initial_owners: dict = field(default_factory=dict)
    warm_live: set = field(default_factory=set)
    warm_hit_experts: set = field(default_factory=set)
    warm_hit_experts_decode: set = field(default_factory=set)
    warm_evicted_before_hit: set = field(default_factory=set)
    first_hit_tokens: Counter = field(default_factory=Counter)
    windows: dict = field(default_factory=lambda: {name: Counter() for name in ("full", "decode")})
    decode_start_resident: int = -1
    decode_start_warm: int = -1
    shared_capacity: int | None = None

    def __post_init__(self):
        self.used = [0] * len(self.capacities)

    def preload(self, keys):
        for key in keys:
            if len(self.resident) >= (self.shared_capacity or sum(self.capacities)):
                break
            eligible = [i for i, cap in enumerate(self.capacities) if self.used[i] < cap]
            if not eligible:
                break
            owner = min(eligible, key=lambda i: (self.used[i], i))
            self.resident[key] = owner
            self.used[owner] += 1
            self.warm_initial.add(key)
            self.warm_initial_owners[key] = owner
            self.warm_live.add(key)

    def select(self, key, rows, decode, token=0):
        if decode and self.decode_start_resident < 0:
            self.decode_start_resident = len(self.resident)
            self.decode_start_warm = len(self.warm_live)
        self.heat[key] += rows
        for window in ("full", "decode") if decode else ("full",):
            stats = self.windows[window]
            stats["selections"] += 1
            if key in self.resident:
                stats["resident_hits"] += 1
                if key in self.warm_live:
                    stats["warm_hits"] += 1
                else:
                    stats["cold_dynamic_hits"] += 1
            else:
                stats["misses"] += 1
                stats["fallback_rows"] += rows
        if key in self.resident:
            self.hits_since_upload[key] += 1
            if key in self.warm_live and key not in self.warm_hit_experts:
                self.warm_hit_experts.add(key)
                self.first_hit_tokens[token] += 1
            if decode and key in self.warm_live:
                self.warm_hit_experts_decode.add(key)
            if decode:
                self.hit_rows += rows

    def attempt(self, key, decode=False):
        self.attempts.append(key)
        for window in ("full", "decode") if decode else ("full",):
            self.windows[window]["promotion_attempts"] += 1
        if key in self.resident or self.heat[key] < self.threshold:
            return
        eligible = [i for i, cap in enumerate(self.capacities)
                    if self.used[i] < cap]
        if eligible and len(self.resident) < (self.shared_capacity or sum(self.capacities)):
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
            for window in ("full", "decode") if decode else ("full",):
                self.windows[window]["evictions"] += 1
            self.eviction_events.append((key, victim))
            if victim in self.warm_live:
                self.warm_live.remove(victim)
                if victim not in self.warm_hit_experts:
                    self.warm_evicted_before_hit.add(victim)
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
        for window in ("full", "decode") if decode else ("full",):
            self.windows[window]["runtime_uploads"] += 1
        self.upload_events.append((key, owner))


def replay(trace, capacities, heat_min=2, heat_margin=1):
    if not capacities or any(cap < 1 for cap in capacities):
        raise ValueError("capacities must be positive")
    policies = ((f"current_t{heat_min}_m{heat_margin}", heat_min, heat_margin),) + tuple(
        policy for policy in POLICIES if policy[1:] != (heat_min, heat_margin))
    states = {name: PolicyState(tuple(capacities), threshold, margin)
              for name, threshold, margin in policies}
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
        for key, missed_under, decode in pending:
            for name in missed_under:
                states[name].attempt(key, decode)
        pending = []

    for kind, value in trace:
        if kind in ("W", "F", "B"):
            raise ValueError("dynamic replay requires a cold trace; inspect warm placement with warm_trace_prefix")
        if kind == "S":
            tick, token, layer, eid, rows, observed_resident = value
            if rows < 0 or token < 0 or observed_resident not in (0, 1):
                raise ValueError("invalid selection rows, token or residency")
            next_group = (token, layer)
            if pending and next_group != group:
                finish_group()
            group = next_group
            key = (layer, eid)
            baseline = states[policies[0][0]]
            snapshot_mismatches += (key in baseline.resident) != bool(observed_resident)
            for state in states.values():
                state.select(key, rows, bool(token), token)
            pending.append((key, tuple(name for name, state in states.items()
                                       if key not in state.resident), bool(token)))
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
    baseline = states[policies[0][0]]
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
            "states": states, "policies": policies, "selection_snapshot_mismatches": snapshot_mismatches,
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
    baseline = result["states"][result["policies"][0][0]]
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
          "avoided_uploads")
    for name, threshold, margin in result["policies"]:
        state = result["states"][name]
        delta = state.hit_rows - baseline.hit_rows
        avoided = baseline.promotions - state.promotions
        print(f"{name:23} {threshold:9} {margin:6} "
              f"{state.hit_rows:9} {state.hit_rows / tokens:10.3f} "
              f"{state.hit_rows / total:8.4f} {state.promotions:10} "
              f"{state.evictions:9} {state.repromotions:13} "
              f"{state.dead_on_arrival:15} {avoided:15} "
              f"{max(delta, 0):10} {max(-delta, 0):12} "
              f"{avoided:20}")


def glm53_engine_id():
    value = 2166136261
    for byte in b"glm53":
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def read_usage(path, n_layers=None, n_experts=None, first_dense=0):
    """Read the text history; never use the evaluation trace to select candidates."""
    counts = {}
    dimensions = None
    identity = None
    seen_data = False
    with Path(path).open(encoding="ascii") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 3 or any(not part.lstrip("-").isdigit() for part in fields):
                raise ValueError(f"usage line {number}: expected three decimal integers")
            layer, expert, count = map(int, fields)
            if layer == -1:
                if seen_data or dimensions is not None or expert < 1 or count < 1:
                    raise ValueError(f"usage line {number}: invalid dimensions header")
                dimensions = (expert, count)
            elif layer == -2:
                if seen_data or identity is not None or expert != 1 or count != glm53_engine_id():
                    raise ValueError(f"usage line {number}: incompatible format or engine")
                identity = (expert, count)
            elif layer >= 0:
                seen_data = True
                if count < 0 or count > 0xFFFFFFFF or expert < 0 or (layer, expert) in counts:
                    raise ValueError(f"usage line {number}: invalid or duplicate expert count")
                counts[layer, expert] = count
            else:
                raise ValueError(f"usage line {number}: unsupported header")
    if (dimensions is None) != (identity is None):
        raise ValueError("usage: incomplete header")
    if dimensions is None:
        if n_layers is None or n_experts is None:
            raise ValueError("headerless usage requires --n-layers and --n-experts")
        dimensions = (n_layers, n_experts)
    if ((n_layers is not None and n_layers != dimensions[0]) or
            (n_experts is not None and n_experts != dimensions[1])):
        raise ValueError("usage dimensions do not match requested model")
    if first_dense < 0 or first_dense >= dimensions[0]:
        raise ValueError("first dense boundary is outside model layers")
    for layer, expert in counts:
        if not (0 <= layer < dimensions[0] and 0 <= expert < dimensions[1]):
            raise ValueError("usage contains out-of-range layer/expert")
        if layer < first_dense:
            raise ValueError("usage contains a dense-layer expert")
    return dimensions, {key: count for key, count in counts.items() if count}


def ranked_usage(counts):
    return sorted(counts, key=lambda key: (-counts[key], key[0], key[1]))


def warm_limit(capacity, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("warm fraction must be between zero and one")
    return int(capacity * fraction + 0.5)  # half up: 3390 * .25 -> 848


def capacities_from_log(path):
    """Final per-device ceiling lines preserve COLI_GPUS order."""
    pattern = re.compile(r"^\[glm53-cuda-device\] device=(\d+) ceiling_bytes=(\d+)")
    values = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            values.append((int(match[1]), int(match[2]) // EXPERT_BYTES))
    if not values or len({device for device, _ in values}) != len(values):
        raise ValueError("device log needs one complete, non-duplicated [glm53-cuda-device] set")
    return tuple(slots for _, slots in values), tuple(device for device, _ in values)


def event_mismatch(actual, simulated):
    return sum(a != b for a, b in zip(actual, simulated)) + abs(len(actual) - len(simulated))


def replay_warm(trace, capacities, capacity, heat_min, heat_margin, warm_sets, devices=None):
    """Replay one complete profiled run; observed events validate the cold baseline."""
    trace = list(trace)
    if not capacities or any(cap < 1 for cap in capacities) or capacity < 1:
        raise ValueError("positive ordered per-device and shared capacities are required")
    if heat_min < 1 or heat_margin < 0:
        raise ValueError("invalid heat policy")
    if capacity > sum(capacities):
        raise ValueError("shared capacity exceeds physical device slots")
    if any(kind in ("W", "F", "B") for kind, _ in trace):
        raise ValueError("counterfactual baseline trace must start cold; use warm_trace_prefix to inspect warm publications")
    states = {name: PolicyState(tuple(capacities), heat_min, heat_margin,
                                shared_capacity=capacity) for name in warm_sets}
    for name, keys in warm_sets.items():
        states[name].preload(keys)
    if states["baseline"].resident:
        raise ValueError("baseline must start empty")
    observed_uploads, observed_evictions, observed_attempts = [], [], []
    observed_resident = {}
    snapshots = attempts = uploads = evictions = device_mismatches = 0
    pending = []
    group = None
    last_tick = 0
    last_token = 0
    decode_seen = False
    decode_tokens = set()

    def finish_group():
        nonlocal pending
        for key, missed_by, decode in pending:
            for name in missed_by:
                states[name].attempt(key, decode)
        pending = []

    for kind, value in trace:
        if kind == "S":
            tick, token, layer, eid, rows, observed_hit = value
            if (tick != last_tick + 1 or token < 0 or layer < 0 or eid < 0 or
                    rows < 1 or observed_hit not in (0, 1)):
                raise ValueError("invalid or discontinuous selection event")
            last_tick = tick
            if token:
                if token < last_token or (not decode_seen and token != 1):
                    raise ValueError("decode token boundary is ambiguous")
                decode_seen = True
                decode_tokens.add(token)
                last_token = token
            elif decode_seen:
                raise ValueError("prefill selection after decode began")
            next_group = (token, layer)
            if pending and next_group != group:
                finish_group()
            group = next_group
            key = (layer, eid)
            snapshots += (key in states["baseline"].resident) != bool(observed_hit)
            for state in states.values():
                state.select(key, rows, bool(token), token)
            pending.append((key, tuple(name for name, state in states.items()
                                       if key not in state.resident), bool(token)))
        elif kind == "A":
            _, layer, eid = value
            observed_attempts.append((layer, eid))
            attempts += 1
            if pending:
                finish_group()
        elif kind == "P":
            _, layer, eid, owner, device_id = value
            key = (layer, eid)
            if key in observed_resident or owner < 0 or owner >= len(capacities):
                raise ValueError("invalid observed publication")
            if devices is not None:
                device_mismatches += device_id != devices[owner]
            observed_resident[key] = owner
            observed_uploads.append((key, owner))
            uploads += 1
        elif kind == "E":
            _, il, ie, vl, ve, *_ = value
            victim = (vl, ve)
            if victim not in observed_resident:
                raise ValueError("observed eviction has no resident victim")
            del observed_resident[victim]
            observed_evictions.append(((il, ie), victim))
            evictions += 1
            if pending:
                finish_group()
        elif kind == "U" and value[-1] != 1:
            raise ValueError("trace contains a failed upload; warm replay assumes successful uploads")
    finish_group()
    if not decode_seen or decode_tokens != set(range(1, max(decode_tokens) + 1)):
        raise ValueError("trace lacks a contiguous explicit decode window")
    if not attempts or not uploads:
        raise ValueError("warm replay requires a complete A/P event trace")
    baseline = states["baseline"]
    validation = {
        "selection_mismatches": snapshots,
        "attempt_mismatches": event_mismatch(observed_attempts, baseline.attempts),
        "upload_owner_mismatches": event_mismatch(observed_uploads, baseline.upload_events),
        "eviction_mismatches": event_mismatch(observed_evictions, baseline.eviction_events),
        "final_resident_mismatches": len(set(observed_resident.items()) ^ set(baseline.resident.items())),
        "device_id_mismatches": device_mismatches,
        "observed_attempts": attempts, "observed_uploads": uploads,
        "observed_evictions": evictions, "observed_final_resident": len(observed_resident),
        "decode_tokens": len(decode_tokens),
    }
    return states, validation


def warm_rows(states, warm_sets, counts, capacity, fractions, expert_limit=None):
    baseline = states["baseline"]
    total_history = sum(counts.values())
    rows = []
    for name, state in states.items():
        policy, fraction = ("baseline", 0) if name == "baseline" else name
        placed = state.warm_initial
        for window in ("full", "decode"):
            metrics = state.windows[window]
            base = baseline.windows[window]
            row = {
                "policy": policy, "window": window, "warm_fraction": f"{fraction:.2f}",
                "warm_limit": expert_limit if expert_limit is not None and name != "baseline" else warm_limit(capacity, fraction),
                "warm_candidates": len(warm_sets[name]),
                "warm_experts": len(placed), "warm_placed": len(placed),
                "warm_bytes": len(placed) * EXPERT_BYTES,
                "historical_count_coverage": f"{sum(counts.get(key, 0) for key in placed) / total_history:.6f}" if total_history else "0.000000",
                "selections": metrics["selections"], "resident_hits": metrics["resident_hits"],
                "misses": metrics["misses"], "fallback_rows": metrics["fallback_rows"],
                "fallback_rows_avoided": base["fallback_rows"] - metrics["fallback_rows"],
                "promotion_attempts": metrics["promotion_attempts"],
                "runtime_uploads": metrics["runtime_uploads"],
                "uploads_avoided": base["runtime_uploads"] - metrics["runtime_uploads"],
                "evictions": metrics["evictions"], "evictions_changed": metrics["evictions"] - base["evictions"],
                "warm_hits": metrics["warm_hits"], "cold_dynamic_hits": metrics["cold_dynamic_hits"],
                "warm_ever_hit": len(state.warm_hit_experts if window == "full" else state.warm_hit_experts_decode),
                "warm_never_hit": len(placed - (state.warm_hit_experts if window == "full" else state.warm_hit_experts_decode)),
                "warm_evicted_before_hit": len(state.warm_evicted_before_hit),
                "warm_survive_decode_start": state.decode_start_warm,
                "warm_survive_decode_end": len(state.warm_live),
                "resident_at_decode_start": state.decode_start_resident,
                "final_resident": len(state.resident),
                "first_hit_tokens": ";".join(f"{token}:{n}" for token, n in sorted(state.first_hit_tokens.items())),
            }
            for i in range(len(state.used)):
                placed_gpu = sum(owner == i for owner in state.warm_initial_owners.values())
                row[f"warm_used_gpu{i}"] = placed_gpu * EXPERT_BYTES
                row[f"warm_placed_gpu{i}"] = placed_gpu
            rows.append(row)
    for row in rows:
        if row["policy"] != "historical":
            continue
        oracle = next(r for r in rows if r["policy"] == "oracle_not_deployable" and
                      r["window"] == row["window"] and r["warm_fraction"] == row["warm_fraction"])
        for metric in ("uploads_avoided", "fallback_rows_avoided"):
            denominator = oracle[metric]
            row["historical_oracle_" + metric + "_ratio"] = (
                f"{row[metric] / denominator:.6f}" if denominator > 0 else "NA")
    return rows


def run_warm(args):
    report = sys.stderr if args.csv == "-" else sys.stdout
    dimensions, counts = read_usage(args.warm_usage, args.n_layers, args.n_experts,
                                    args.first_dense)
    trace = list(events(args.trace))
    frequencies = Counter()
    for kind, value in trace:
        if kind == "S" and value[1] > 0:
            frequencies[value[2], value[3]] += value[4]
    historical = ranked_usage(counts)
    oracle = ranked_usage(frequencies)
    fractions = ([args.warm_experts / args.capacity] if args.warm_experts is not None else
                 [float(value) for value in args.warm_fractions.split(",")])
    if not fractions or any(not 0 <= value <= 1 for value in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError("warm fractions must be distinct values between zero and one")
    if args.warm_experts is not None:
        if not 0 <= args.warm_experts <= args.capacity:
            raise ValueError("warm expert limit is outside shared capacity")
    if bool(args.device_capacities) == bool(args.device_log):
        raise ValueError("supply exactly one of --device-capacities or --device-log")
    devices = None
    if args.device_log:
        capacities, devices = capacities_from_log(args.device_log)
        print("device capacities: " + ",".join(f"{device}:{slots}" for device, slots in zip(devices, capacities)), file=report)
    else:
        capacities = tuple(int(value) for value in args.device_capacities.split(","))
    warm_sets = {"baseline": []}
    for fraction in fractions:
        limit = args.warm_experts if args.warm_experts is not None else warm_limit(args.capacity, fraction)
        warm_sets["historical", fraction] = historical[:limit]
        warm_sets["oracle_not_deployable", fraction] = oracle[:limit]
    states, validation = replay_warm(trace, capacities, args.capacity,
                                     args.heat_min, args.heat_margin, warm_sets, devices)
    if args.decode_tokens is not None and validation["decode_tokens"] != args.decode_tokens:
        raise ValueError(f"expected {args.decode_tokens} decode tokens, found {validation['decode_tokens']}")
    print("baseline validation: " + " ".join(f"{key}={value}" for key, value in validation.items()), file=report)
    if any(validation[key] for key in ("selection_mismatches", "attempt_mismatches",
                                      "upload_owner_mismatches", "eviction_mismatches",
                                      "final_resident_mismatches", "device_id_mismatches")):
        raise ValueError("baseline replay differs from observed trace; warm results withheld")
    rows = warm_rows(states, warm_sets, counts, args.capacity, fractions, args.warm_experts)
    columns = list(rows[0])
    for row in rows[1:]:
        columns.extend(key for key in row if key not in columns)
    print(f"usage_dimensions={dimensions[0]}x{dimensions[1]} historical_candidates={len(historical)}", file=report)
    print("policy window fraction placed uploads avoided fallback_rows avoided evictions warm_never_hit hist/oracle_upload hist/oracle_fallback", file=report)
    for row in rows:
        print(f"{row['policy']:21} {row['window']:6} {row['warm_fraction']:>8} "
              f"{row['warm_experts']:6} {row['runtime_uploads']:7} {row['uploads_avoided']:7} "
              f"{row['fallback_rows']:13} {row['fallback_rows_avoided']:7} "
              f"{row['evictions']:9} {row['warm_never_hit']:14} "
              f"{row.get('historical_oracle_uploads_avoided_ratio', 'NA'):>18} "
              f"{row.get('historical_oracle_fallback_rows_avoided_ratio', 'NA'):>20}", file=report)
    if args.csv:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        if args.csv == "-":
            print(output.getvalue(), end="")
        else:
            Path(args.csv).write_text(output.getvalue(), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--capacity", type=int, required=True,
                        help="total whole-expert slots (847 on the 2x3070 run)")
    parser.add_argument("--device-capacities", type=str,
                        help="ordered per-device slots, e.g. 424,423")
    parser.add_argument("--device-log", help="read ordered capacities from final [glm53-cuda-device] lines")
    parser.add_argument("--heat-min", type=int, default=2)
    parser.add_argument("--heat-margin", type=int, default=1)
    parser.add_argument("--warm-usage", help="immutable historical .coli_usage; enables warm replay")
    parser.add_argument("--warm-fractions", default="0,0.25,0.50,0.75,1.00")
    parser.add_argument("--warm-experts", type=int, help="additional explicit warm expert limit")
    parser.add_argument("--n-layers", type=int, help="expected usage layer count")
    parser.add_argument("--n-experts", type=int, help="expected usage expert count")
    parser.add_argument("--first-dense", type=int, default=0)
    parser.add_argument("--csv", help="write deterministic warm metrics CSV (- for stdout)")
    parser.add_argument("--decode-tokens", type=int, help="require this many contiguous decode tokens")
    args = parser.parse_args()
    if args.warm_usage:
        try:
            run_warm(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return
    old = compare(args.trace, args.capacity)
    print(f"decode_tokens={old['tokens']} selections={old['events']} "
          f"capacity={old['capacity']} routed_rows={old['total_rows']}")
    for label in ("current", "lru", "static_cold", "static_preloaded"):
        hits = old[label + "_rows"]
        display = "ORACLE_NOT_DEPLOYABLE_static_preloaded" if label == "static_preloaded" else label
        print(f"{display}_rows={hits} rows_per_token={hits / old['tokens']:.3f} "
              f"hit_fraction={hits / old['total_rows']:.4f}")
    print_replay("global", replay(events(args.trace), (args.capacity,), args.heat_min, args.heat_margin))
    if args.device_capacities:
        try:
            capacities = tuple(int(piece) for piece in args.device_capacities.split(","))
        except ValueError as exc:
            parser.error(f"invalid --device-capacities: {exc}")
        if sum(capacities) != args.capacity:
            parser.error("sum of --device-capacities must equal --capacity")
        print_replay("device-aware " + args.device_capacities,
                     replay(events(args.trace), capacities, args.heat_min, args.heat_margin))
    print("# static_preloaded is ORACLE / NOT DEPLOYABLE; "
          "replay assumes equal-size experts and successful uploads.")


if __name__ == "__main__":
    main()
