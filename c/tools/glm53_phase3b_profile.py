#!/usr/bin/env python3
"""Validate one complete Phase 3B run and emit decode-only metric deltas."""
import argparse
import csv
import math
import sys

TIMES = ('ffn_dense_s', 'ffn_sparse_s', 'sparse_router_s', 'sparse_topk_s',
         'sparse_shared_s', 'sparse_union_s', 'sparse_join_wait_s',
         'sparse_routed_tier_s')
COUNTS = ('host_cache_hits', 'host_cache_misses', 'host_expert_miss_bytes',
          'gpu_group_attempts', 'gpu_group_successes', 'cpu_routed_fallback_calls',
          'kda_cuda_calls', 'kda_h2d_bytes', 'kda_d2h_bytes', 'kda_state_pushes',
          'kda_state_push_bytes', 'kda_state_pulls', 'kda_state_pull_bytes')
METRICS = TIMES + COUNTS
PREFIX = '[glm53-phase3b-profile] '


def decode_rows(lines):
    previous = None
    final = False
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.startswith(PREFIX):
            continue
        try:
            pairs = [word.split('=', 1) for word in line.split()[1:]]
            fields = dict(pairs)
            if len(fields) != len(pairs):
                raise ValueError('duplicate fields')
            phase = fields['phase']
            current = {k: float(fields[k]) if k in TIMES else int(fields[k])
                       for k in METRICS + ('decode_tokens',)}
            if any(not math.isfinite(v) or v < 0 for v in current.values()):
                raise ValueError('nonfinite or negative value')
            if final:
                raise ValueError('records after final; interleaved runs')
            if previous is None:
                if phase != 'start' or current['decode_tokens'] != 0:
                    raise ValueError('missing phase=start baseline')
            else:
                if phase not in ('decode', 'prefill', 'final'):
                    raise ValueError('unexpected phase/repeated start; interleaved runs')
                if any(current[k] < previous[k] for k in current):
                    raise ValueError('decreasing counters/timers; interleaved or corrupt log')
                increment = current['decode_tokens'] - previous['decode_tokens']
                if increment != (1 if phase == 'decode' else 0):
                    raise ValueError('decode token increment must be exactly 1 for decode, 0 otherwise')
                if phase == 'decode':
                    rows.append({'decode_tokens': current['decode_tokens'],
                                 **{k + '_delta': current[k] - previous[k]
                                    for k in METRICS}})
            previous = current
            final = phase == 'final'
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Phase 3B line {number}: {exc}') from exc
    if previous is None:
        raise ValueError('no Phase 3B phase=start baseline found')
    if not final:
        raise ValueError('missing phase=final; incomplete log')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    args = parser.parse_args()
    try:
        with open(args.log, encoding='utf-8', errors='strict') as stream:
            rows = decode_rows(stream)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    writer = csv.DictWriter(sys.stdout, fieldnames=['decode_tokens'] +
                           [k + '_delta' for k in METRICS])
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
