# Phase 3B: behavior-neutral FFN and ownership attribution

Base: `0845b558280670dcde8869435b2935ef68ebdce3`. Phase 2F and Phase 3A
remain closed. This change adds attribution, not an optimization.

Enable both `GLM53_CUDA_PROFILE=1` and `GLM53_PHASE3B_PROFILE=1`.
The latter is read once in full-model initialization after normal CUDA profile
initialization. It cannot enable the normal profiler. Unset, `0`, or any value
other than exactly `1` disables it. Disabled profiling adds no clock reads or
Phase 3B records and leaves the existing CUDA record formats unchanged.

## Records and metric definitions

`[glm53-phase3b-profile]` emits cumulative start, prefill, decode and final
snapshots. `decode_tokens` is the existing normal CUDA decode counter.
Prefill snapshots are necessary because reused cache/fallback/KDA counters
include prefill, while new FFN timers/group counters collect decode only.
Subtract consecutive snapshots; do not subtract start directly from decode
for raw counters. Final is emitted before KDA teardown. No report adds a join.

| Metric | Exact definition |
|---|---|
| `ffn_dense_s`, `ffn_sparse_s` | Existing outer FFN `phase_s`, classified by layer `< first_dense` or otherwise, for explicit single-token decode. Covers the whole `ffn_layer` call, including allocations, routing, loading, compute, promotion, joins, scatter, cleanup and in-call diagnostic reporting. Excludes the site's preceding hyperconnection pre/norm and following hyperconnection post. |
| `sparse_router_s` | Router dot products and sigmoid scores. |
| `sparse_topk_s` | Top-k selection and normalized routing weights. |
| `sparse_shared_s` | Shared expert `mlp3` call. |
| `sparse_union_s` | Union allocation, OOM check and distinct-expert construction; zero on non-streaming paths. Does not mean all routed planning. |
| `sparse_routed_tier_s` | From union completion through sparse FFN return (including late join, grouped/serial tier decisions, loading, fallback compute, scatter, promotions, diagnostics and frees). Non-streaming: from before resident CPU routed loop through cleanup. |
| `sparse_join_wait_s` | Delta of existing `promotion_join_s` across the entire sparse FFN. Includes synchronous batch joins and deferred promotion joins exactly once. Excludes CUDA stream synchronization, which remains in existing expert API envelopes. |
| `host_cache_hits`, `host_cache_misses`, `host_expert_miss_bytes` | Existing `GModel.hits`, `miss`, `ebytes`, without duplicate accounting. Miss bytes describe expert bytes associated with misses, not physical disk bytes. |
| `gpu_group_attempts` | Actual calls to `g53_cuda_ffn_grouped` during decode, including calls that refuse and return zero. Counts layer-level calls, not per-device issues. |
| `gpu_group_successes` | Those calls that return handled (`1`) with no CUDA-tier failure. A handled call can also compute CPU misses; it is not a count of GPU-only layers or GPU rows. |
| `cpu_routed_fallback_calls` | Existing `fallback_compute_rows`: actual streamed routed host `mlp3` calls, including recovery recomputation. Excludes shared/dense MLPs, non-streaming resident CPU path and diagnostic replay. |
| `kda_cuda_calls`, `kda_h2d_bytes`, `kda_d2h_bytes` | Existing successful KDA tier call count and logical token transfer bytes. These counters retain their existing successful-call semantics; failed-attempt transfers are not added. |
| `kda_state_pushes`, `kda_state_push_bytes`, `kda_state_pulls`, `kda_state_pull_bytes` | Existing successful state push/pull counts and bytes. No new KDA implementation or timing. |

Dense plus sparse totals equal decode-only existing `ffn_s` modulo log
rounding. Router, top-k, shared, union and routed-body intervals are mutually
non-overlapping but do **not** cover every sparse-site instruction: entry
joins, initial allocations, route recording and scratch setup remain in the
outer residual. The join diagnostic overlaps the outer sparse total and can
overlap routed-body time; never add it to the above subphase sum.

Existing `disk_s`, fallback, promotion, upload, issue/take and worker timings
are nested/overlapping diagnostics within the routed body, not additional
partition terms. `disk_s` remains unchanged; no physical-I/O timer is invented.
These are wall timings, not CPU consumption or exact blocked time.

## Clock budget

Only explicit S=1 sparse decode calls enable the new timing regions:

- Router: before scoring and after scoring (2 reads).
- Top-k: before selection and after normalization (2 reads).
- Shared: before and after shared MLP (2 reads).
- Union: before allocation and after construction (2 reads).
- Routed end: after cleanup on the grouped or serial return (1 read).

Streaming total: 9 reads. On existing late-join opportunities the first 8
reuse the original clock locations; only the routed-end read is additional.
Without such an opportunity all 9 are additional. Non-streaming skips union
and substitutes one routed-start read: 8 reads total. Dense split, join
diagnostic, counters and reports add zero clock reads. Disabled Phase 3B
retains exactly the original late-join clock predicates.

## Ownership boundaries and transfer limitations

Grouped execution copies the CPU activation into per-device host input rows.
`backend_cuda.cu::expert_group_issue_impl` packs staging, enqueues activation
H2D, device expert computation and output D2H. `coli_cuda_expert_group_take`
synchronizes the existing stream and returns pinned host output. GLM53 copies,
validates and scatters results on the host. CPU misses and shared MLP also
produce host values. These boundaries interrupt a device-resident chain.

Expert activation/result byte counters are deliberately absent: GLM53 success
counts cannot measure which backend transfer calls actually succeeded before
a partial failure. Exact accounting would require a backend counter API and
instrumentation at individual transfer call sites. This bounded change does
not add that interface and does not estimate bytes. Serial expert transfers
and weight-upload bytes likewise remain unmeasured here. KDA bytes are exposed
separately, as defined above, never mixed with weight uploads.

## Rig smoke

Use the unchanged Phase 3A production environment, including four OMP threads,
eight GPUs, the frozen usage snapshot and warm residency off. H1_FILE must
point to the exact held-out H1 prompt. After building this commit on the rig:

```bash
set -euo pipefail
unset GLM53_CUDA_KDA
export GLM53_CUDA_PROFILE=1 GLM53_PHASE3B_PROFILE=1
: "${H1_FILE:?Set exact held-out H1 prompt path}"
printf '%s  %s\n' \
  7a246a49fc1dc3fb6360c131153c22012ca08949da390455654652d76e18a953 \
  "$COLI_USAGE" | sha256sum -c -
mkdir -p artifacts/phase3/3b-attribution
run=$(mktemp -d "$PWD/artifacts/phase3/3b-attribution/smoke.XXXXXX")
git rev-parse HEAD > "$run/revision.txt"
sha256sum c/glm53 "$COLI_USAGE" "$H1_FILE" > "$run/frozen.sha256"
env | sort > "$run/environment.txt"
/usr/bin/timeout --signal=TERM --kill-after=30s 1800s \
  c/glm53 --model /srv/models-fast/colibri/glm53-flash-i4 \
  --prompt "$(cat "$H1_FILE")" --greedy 128 \
  > "$run/stdout.log" 2> "$run/stderr.log"
python3 c/tools/glm53_phase3b_profile.py "$run/stderr.log" > "$run/phase3b.csv"
python3 c/tools/glm53_cuda_profile.py "$run/stderr.log" > "$run/cuda.csv"
sha256sum -c "$run/frozen.sha256"
python3 - "$run" <<'PY'
import csv, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / 'phase3b.csv').open()))
assert len(rows) == 128 and int(rows[-1]['decode_tokens']) == 128
out = (root / 'stdout.log').read_text()
assert re.search(r'^decode 128 token in ', out, re.M)
log = (root / 'stderr.log').read_text()
errors = re.findall(r'\berrors=(\d+)', log)
assert errors and all(int(v) == 0 for v in errors)
assert not re.search(r'step failed|state push failed|committed state pull failed|glm53-kda-cuda-error', log)
assert re.search(r'kda_cuda_calls=[1-9]\d*', log)
print('PASS Phase3B smoke: 128 snapshots, active KDA, no reported failures')
PY
```

Repeat with original H2/H3 files after smoke acceptance. Compare semantic
output with frozen Phase 3A artifacts. Windows/WSL fake-backend checks are
correctness checks for instrumentation, not real-GPU performance validation.
