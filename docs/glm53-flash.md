# GLM-5.3-Flash engine (`c/glm53.c`)

A sibling engine for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(321B parameters, 45 layers + MTP, with a vision tower), following the
one-engine-per-family pattern (`colibri.c` = GLM-5.2, `kimi_k3.c`, `inkling.c`).
It shares `st.h`, `json.h`, `tok.h`, `quant.h` and `hyper_connections.h`, and
touches nothing in the other engines.

```
make glm53
./glm53 --model <dir> --prompt "Ciao, come stai?" --greedy 32
```

Or through the launcher, which is where images and chat templates live:

```
coli chat  --model <dir> --no-think
coli serve --model <dir>
coli web   --model <dir>
```

## Getting the model

The engine reads a converted container, not the HF snapshot. One pass does both
the download and the conversion, one shard at a time:

```
python3 tools/convert_glm53.py --outdir /path/glm53_i4 --min-free-gb 30
```

Peak disk is the output plus a single 5 GB source shard, never the repository's
328 GB. On the reference machine: 62 shards, **194.7 GB out, 25 hours**.

Routed experts become int4 group-scaled at 64 (`name` U8 nibbles + `name.qs`
F32 scales, the same container GLM-5.2 uses). Everything else stays BF16 — 9.7B
parameters of 321, about 18 GB — so the precision of the dense set is a load-time
choice and retuning it never means downloading the repository again. Every one of
the 92 tensor kinds is classified explicitly and an unrecognised name stops the
conversion, because a converter that skips what it does not know produces a
checkpoint that loads and is quietly missing a tensor.

## Architecture notes

Four pieces differ from everything else in this repository:

**KDA (Kimi Delta Attention)** on 34 of the 45 layers: `q,k,v = SiLU(ShortConv4(Wx))`,
q and k L2-normalised with the epsilon **inside** the square root (FLA's
convention, deliberately unlike `F.normalize`), then the gated delta rule. The
output gate is low-rank here where Kimi K3's is a single projection.

**DSA with k-pooling** on the other 11. Keys are grouped into pools of
`index_kpool`; the pooled key is a per-channel softmax mixture rather than an
average, so a pool is not forced to describe itself by its mean. A pool is
selectable only if it is complete and causally visible, the incomplete tail is
appended when asked for (hence a row `topk + pool - 1` wide, not `topk`), and
ties go to the lower index so selection is deterministic.

**mHC hyper-connections**, shared with DeepSeek V4 through
`hyper_connections.h` — measured bit-compatible, not assumed.

**Clamped SwiGLU in the text MLP**, not only in the vision tower: `gate` has a
ceiling, `up` is clamped both ways. Plain SiLU here gives a model that speaks
well and is wrong.

### MLA is absorbed

Caching expanded keys costs 1.39 MB per token across the DSA layers — 11.9 GB at
8192 positions, on an engine that exists to fit in small memory. `kv_b_proj` is
folded into the two ends instead, so the cache holds the 512-wide latent:

```
score_j = q · (W_k c_j) = (W_kᵀ q) · c_j
out     = Σ_j a_j (W_v c_j) = W_v (Σ_j a_j c_j)
```

An identity, not an approximation: only the order of the products changes.
**33 KB per token**, which is 1.1 GB at 32k positions and 4.4 GB at the full
128k. The weights do not grow either, because transposing W_k keeps its element
count.

## Memory and speed

Measured on the real checkpoint, 6 physical cores, 25 GB RAM, model on an
ordinary disk:

| | |
|---|---|
| resident weights at `GLM53_BITS=4` | ~12 GB |
| KV state | 33 KB per token (1.1 GB at 32k) |
| prefill workspace | flat, ~60 MB at any context |
| decode | ~44 s/token cold, ~20 s/token with a warm expert cache |

**The disk is the wall, and it is worth knowing where it sits.** One token
touches 42 sparse layers × 8 experts × 14.2 MB = 4.8 GB. Measured with
`O_DIRECT`, the reference disk gives 72 MB/s at queue depth 1, 185 at QD4 and
207 at QD16 — saturating near 200 MB/s. That puts a **floor of 24 seconds per
token** on this hardware: what the disk takes to deliver the bytes, with any CPU
and any GPU. Faster silicon does not move it; fewer bytes would.

GPU acceleration does not remove that disk floor. The Metal/Vulkan paths and
the optional CUDA expert tier below accelerate computation; CUDA resident
expert hits additionally avoid rereading those experts from disk.

## Optional CUDA expert tier

On Linux (including a configured WSL CUDA toolkit), build for an RTX 3070:

```bash
cd c
make glm53 CUDA=1 CUDA_ARCH=sm_86
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=auto ./glm53 --model /path/glm53_i4 --prompt "Hello" --greedy 32
```

The default build/run is unchanged. Explicit CUDA requests require a CUDA
build, working devices, and the streaming int4-gs64 container. `COLI_GPU=N`
retains the single-device path (default device 0). A nonempty `COLI_GPUS`
overrides it with an ordered comma-separated list, for example
`COLI_GPUS=0,1,2,3,4,5,6,7`. IDs must be unique and valid; the shared backend
currently supports up to 16 selected devices. An empty plural form is ignored.
The existing Windows DLL build convention remains `CUDA_DLL=1`, not `CUDA=1`.

Only the normal full-model CLI and SERVE loader enables this tier. Segment,
Edge, and range-only model instances never initialize it, even when a Segment
covers every layer or `COLI_CUDA=1` is inherited. Their inactive tier cannot
shut down the process-global CUDA backend owned by another model, and their
CPU capabilities and `numeric_class` remain unchanged. The CPU-only adapter
build also ignores this CUDA opt-in instead of rejecting it.

The RAM cache (`GLM53_EXPERT_GB`) stays intact. On-demand promotion begins
after an expert has been selected for at least two rows by default; a full
device tier replaces its least frequently selected resident only for a hotter
candidate. `GLM53_CUDA_HEAT_MIN` (positive integer, default `2`) and
`GLM53_CUDA_HEAT_MARGIN` (nonnegative integer, default `1`) control admission
and replacement. At full capacity, replacement requires incoming heat greater
than victim heat plus the margin. Setting the margin to `0` restores the prior
replacement policy. Invalid values fail initialization. The effective values
appear once on the CUDA startup line. Heat remains cumulative for this
model lifetime. Uploads follow the normal disk-to-RAM read and
own their device memory independently of
RAM-slot eviction. Resident hits bypass disk and RAM reads. Cold misses use
the existing host path and become eligible for subsequent CUDA execution.

`GLM53_CUDA_PARALLEL_PROMOTE=1` enables an experimental, bounded cache-fill
prototype on multi-GPU runs (default off). It plans only consecutive promotions
that need no eviction, reserving budget and least-allocated owners in logical
order. It copies each expert's host pieces before the RAM slot can be reused,
uploads at most one expert per device at a time, waits for upload completion,
then publishes in logical order. An eviction boundary drains the batch and
uses the unchanged serial victim path. A worker failure drains the batch and
enables the existing model-wide host fallback. Single-device runs remain serial.

`GLM53_CUDA_PROMOTE_OVERLAP=1` is a separate opt-in (default off) and only
applies when parallel promotion is enabled. An eviction-free, non-full batch
at the end of a grouped one-token **decode** FFN starts its upload workers but keeps the
tensors unpublished while the model computes CPU mHC and KDA/MLA work. The
model joins before the next FFN reads CUDA residency or failure state. Final
owned layers, prefill, full batches, eviction/owner boundaries, and serial
replacements still join immediately. Teardown also joins before freeing CUDA
state. Even a one-token prefill chunk stays synchronous. No CUDA attention or
dense operation is overlapped by this path.
`GLM53_CUDA_PROMOTE_LATE_JOIN=1` is a further opt-in (default off) requiring
both `GLM53_CUDA_PARALLEL_PROMOTE=1` and `GLM53_CUDA_PROMOTE_OVERLAP=1`.
Only for an already deferred one-token sparse decode batch, it moves the
next FFN's join from function entry to immediately before the first CUDA-tier
branch. Router scoring/top-k, the CPU shared-expert MLP, and union construction
may run while uploads finish. The model still joins and publishes in logical
order before reading tier residency or failure state; prefill and every other
join/flush path retain their previous location.

`GLM53_MLA_OUT_ROWS4=1` is a separate, default-off CPU attention experiment.
For a one-token MLA output projection it reuses KDA KO's AVX2 rows4 int4
kernel only when the output matrix is fmt4/gs64, has a row count divisible by
four, and its input dimension is divisible by 64. Other formats/geometries,
non-AVX2 builds, and Metal/Vulkan builds keep the generic `mv` path. It does
not change KDA dispatch or the CUDA expert tier. When CUDA profiling is on,
the cumulative `[glm53-attn-split]` line includes
`mla_out_rows4_calls` and `mla_out_generic_calls`; `mla_out_s` retains its
existing inclusive timing interval.

`CUDA_EXPERT_GB` is a total decimal-GB cap across all selected devices, or
`auto` (default): the sum of free VRAM minus 2 GB of runtime headroom on each
device, floored at zero per device. A numeric cap is clamped to that total;
zero forces host execution. Experts are placed whole on the least-allocated
device with enough allowance, breaking ties by list order. The cap is shared
dynamically, not multiplied by the device count or rigidly divided; every
device must also stay within its own allowance. The startup line shows the
total cap and each device's usable bytes and whole-expert capacity (these
individual maxima are subject to the shared cap). Single-GPU execution retains
the original serial gate/up/host-clamp/down path. With multiple devices,
decode-scale resident experts issue one clamped gs64 group per participating
device. CPU misses run while groups are pending; every issued device is drained
before any routed contribution is scattered. Results accumulate in stable
logical order, and a failed issue, take, or nonfinite result recomputes the
operation on the CPU without publishing a partial GPU result. Promotions wait
until all groups have drained. Larger prefill blocks retain the serial path.
This tier does not upload attention, dense, or shared-expert weights. The
model/container format is unchanged.

`[glm53-cuda]` diagnostics report resident expert count, allocated expert
VRAM bytes (excluding runtime scratch), successfully executed expert rows,
host fallback rows, uploads, and errors. They appear at startup, after each
forward through the final sparse layer, and at teardown. A CUDA upload or
execution failure disables further CUDA work for this model lifetime; the
existing host MLP recomputes the affected expert before any result is scattered.
Initialization/configuration failures are explicit startup errors.

Validation (a skip is not a pass):

```bash
make glm53-cuda-tier-check                         # fake backend, no GPU
make glm53-cuda-check CUDA=1 CUDA_ARCH=sm_86        # real CUDA tensor numerics
CUDA_VISIBLE_DEVICES=0,1 make glm53-cuda-multidev-check CUDA=1 CUDA_ARCH=sm_86
python3 tests/glm53_cuda_harness.py --binary ./glm53 --fixture ~/glm53_stream-i4
```

The portable target also destroys Segment/Edge instances while a full-model
owner has resident experts on one or two devices, verifies execution, and checks
that shutdown happens exactly once when the owner is released. Source tests
pin initialization to the full-model loader and preserve adapter CPU metadata.
The multi-device fake test covers ordered parsing/initialization, deterministic
placement, shared-budget accounting, uneven headroom, partial upload rollback,
CPU fallback signaling, and cleanup on every device and initialization errors.
Phase 2 can group experts by owner for parallel issue/take execution once the
GLM53 clamped SwiGLU contract is supported. Dense/KDA/MLA placement is later work;
this phase adds no tensor parallelism, matrix sharding, or NCCL.

The numerical test uses asymmetric matrices, distinct per-group scales and
clamp-saturating inputs, and overwrites host weights after upload. It compares
gate, up, and clamped-down projection outputs with CPU dequantization, reports
measured maximum absolute, relative and tolerance-scaled errors, and rejects
nonfinite results. The fixed envelope is `abs(error) <= 0.002 + 0.002*abs(cpu)`;
this is a provisional acceptance bound, not a measured NVIDIA accuracy claim.
Run the real-device target and retain its reported errors before validating
the target GPU. Fake-backend agreement is not CUDA numerical validation. The model
harness uses the streaming fixture described below, requires nonzero resident
bytes, executions and fallbacks, compares CPU/CUDA teacher-forcing and greedy
tokens, and checks zero-budget and injected-error fallback. Token equality is
a fixture regression check; near ties in real models may differ with floating
point accumulation order. GPU detection or `nvidia-smi` alone proves no tensor
execution. CUDA kernel and full-model results must be measured on the target
hardware before claiming performance.

### Opt-in CUDA profiling

Set `GLM53_CUDA_PROFILE=1` with the CUDA tier. The profiler is off
by default. It uses GLM53's monotonic host clock, adds no CUDA events or
synchronization beyond the upload completion required by the parallel-promotion
prototype, and emits cumulative `[glm53-cuda-profile]` records at
startup (`phase=start`), after each prefill chunk or decode forward, and before
shutdown (`phase=final`). `phase=decode` labels an actual one-token decode
forward, even when prefill chunks also contain one token. The timing checks
avoid clock calls when profiling is off. The normal `[glm53-cuda]` diagnostics
remain independent of this opt-in. Startup model loading and CUDA initialization
finish before `phase=start`; their time is outside this profile.

With parallel promotion enabled, `[glm53-cuda-promotion]` reports planned batch
size/device totals, planning time, summed per-worker dispatch-to-start wait, caller join wait,
batch wall time, and concurrent versus serial promotions. Per-device worker
active time appears in `[glm53-cuda-promotion-device]`. `upload_s` remains the
sum of per-expert upload work durations; these durations can overlap and must
not be read as elapsed wall time. The batch wall field is elapsed time. The
optional selection trace adds `U` records with each expert's upload duration.
Heat-ineligible attempts pass through a pending batch without reserving an
owner or staging expert bytes. `[glm53-cuda-promotion-flush]` counts only drains
of nonempty batches by end boundary, repeated owner, capacity/possible eviction,
full one-task-per-device batch, failure/disabled state, staging failure, or
other serial-only cases. `heat_noop_passthrough` counts below-minimum attempts
encountered while a batch is pending. `[glm53-cuda-promotion-batch-size]`
reports completed batches by size from 1 through the selected device count.
The existing `concurrent` field counts experts in batches larger than one; it
does not prove simultaneous DMA. `serial` includes singleton batches and
serial replacement uploads. `batch_devices` sums occupied device slots over
all batches, rather than counting distinct devices across the run.
`eviction_boundaries` counts pending batches drained by the capacity check,
not total evictions; `flush_capacity_or_eviction` gives the same drain count.
The inactive/failed fast path does not drain a batch in the current call graph,
so `flush_failure_or_disabled` normally stays zero.
`[glm53-cuda-promotion-overlap]` reports deferred end batches/experts,
caller dispatch time, dispatch-to-join lifetime, mandatory join wait, and
hidden time measured as the union of worker-active intervals overlapping
CPU-only work after dispatch and before the mandatory join. It excludes idle
gaps and does not sum concurrent workers.
`deferred_already_complete_at_join` samples whether every worker had finished
before the barrier; `join_next_ffn` and the `forced_*` fields identify barrier
reasons. Per-device worker active time remains in the existing device lines.
`[glm53-cuda-late-join-opportunity]` measures only FFNs entered with a pending
deferred batch. With late join disabled, `late_join_current_wait_s` is the caller
time blocked at entry; `late_join_prelude_s` is the subsequent CPU
router, top-k, shared-expert MLP, union, and miscellaneous setup through the
first CUDA-tier branch. The component fields partition that measured prelude.
`late_join_hideable_upper_s` sums `min(join wait, prelude)` per opportunity.
It is a counterfactual upper estimate, **not** elapsed time saved: concurrent
uploads could slow the prelude. With late join enabled, these two old-estimate
fields remain zero; the measured prelude fields reflect CPU work with the
outstanding batch. `[glm53-cuda-late-join-actual]` reports the effective switch,
actual late joins, caller-blocked wait at their barrier, and whether all workers
were already complete. `late_join_hidden_s` / `prelude_inflight_s` are the same
union of worker-active intervals clipped to the CPU prelude; router, top-k,
shared, and union `*_inflight_s` clip that union to each substage. They do not
sum worker durations or imply an equal end-to-end wall-time saving.
`[glm53-cuda-decode-profile]` reports decode-only deltas for the named phase,
promotion, and KDA/MLA substage timers. It snapshots each explicit decode
forward separately, so prefill before or between decode calls is excluded.
Its `mla_proj_s` parent includes four sequential child intervals:
`mla_query_path_s` (q_a, RMS, q_b), `mla_latent_path_s` (kv_a, RMS),
`mla_absorbed_q_s` (the per-head kv_b key-transpose matvecs), and
`mla_index_path_s` (indexer projections, key norm, and head-weight scaling).
`mla_proj_residual_s` is the parent minus these children; it includes timer
and loop overhead, not another kernel. `mla_absorbed_q_calls` and
`mla_absorbed_q_rows` are decode-only counts of per-head matvec calls and
their output rows. The child clocks and counts run only with
`GLM53_CUDA_PROFILE=1`; no kernel or OpenMP dispatch changes.
The original cumulative lines and their field meanings are unchanged. Worker
`upload_s` is summed worker time; `promotion_s` and join waits measure caller
blocking, and the phase/substage fields are nested rather than exclusive.
With overlap enabled, `promotion_s` counts caller-blocked planning, dispatch,
serial work, and join time; `batch_wall_s` includes deferred dispatch-to-join
lifetimes and can overlap attention time. Neither should be summed with other
phase timings as an exclusive elapsed interval.

At shutdown, profiling also prints `[glm53-cuda-cache]` totals and a
`[glm53-cuda-cache-layer]` row for each selected sparse layer: resident
selection hits/misses and rows, promotions, evictions, re-promotions, and
promotions evicted without a subsequent resident selection. Victim last-use
distance is measured in routed expert selections, not tokens. Individual
evictions are recorded in an internal spool without per-event log lines.
Set `GLM53_CUDA_TRACE=/path/cache.csv` alongside profiling to retain a
selection/eviction trace for offline comparison. For example, after a run
whose tier fits 847 whole experts:

```bash
python3 tools/glm53_cuda_cache_sim.py /path/cache.csv --capacity 847
python3 tools/glm53_cuda_cache_sim.py /path/cache.csv --capacity 847 --device-capacities 424,423
```

The script compares observed resident rows with global-capacity LRU, a
future-aware static top-frequency set, and cumulative-heat admission/replacement
counterfactuals. It reports both global and device-aware policy replays when
ordered per-device capacities are supplied. The static set is an upper bound.
Older selection/eviction traces infer promotion timing from each decode-layer
selection group. New traces also contain promotion-attempt (`A`) and successful
upload (`P`) events; the latter include owner index and physical device ID so
the replay can check its device placement. Compare the replayed current-policy
rows and event mismatch counts with the observed values before interpreting
counterfactuals. The upload-time estimate is rough accounting, not a speed
prediction; the replay does not model transient allocations or failed uploads.

The fields are cumulative within one model run:

| Field | What it measures |
|---|---|
| `tokens`, `decode_tokens`, `forwards` | Input rows processed in all forwards, rows in explicit decode forwards, and forward calls. |
| `cuda_rows` | Successful CUDA routed-expert row executions (`executed` in the normal diagnostic). |
| `fallback_rows` | Routed rows sent to the existing fallback path. |
| `fallback_compute_rows` | Actual host `mlp3` calls in that path. |
| `uploads`, `evictions`, `errors` | Successfully published whole experts, replacements of resident experts, and CUDA upload/execution errors. Teardown frees are not evictions. |
| `resident`, `vram_bytes`, `budget_bytes`, `tier_full` | Current device expert count, allocated expert VRAM, budget, and whether another expert would exceed it. `tier_full` says nothing about steady state. |
| `upload_s` | Wall time around gate/up/down tensor upload calls, including failed attempts. |
| `gate_s`, `up_s`, `down_s` | Synchronous `coli_cuda_matmul` call-boundary wall time, including whatever H2D transfer, kernel execution, D2H transfer and completion wait the existing API performs. These are not kernel-only times. |
| `clamp_s` | Host clamped-SwiGLU call time. |
| `fallback_compute_s` | Host `mlp3` call time in fallback; excludes scatter/add and expert loading. |
| `promotion_s`, `eviction_s` | Entire synchronous promotion decision/upload attempt, and the resident free within a replacement. `upload_s` and `eviction_s` are subsets of `promotion_s`. |
| `group_issue_s`, `group_take_s` | Host call-boundary time for grouped decode work. Device execution can overlap CPU misses between these calls. |
| `cuda_expert_s` | Sum of serial gate/up/clamp/down and grouped issue/take call times, including attempted calls before a failure; excludes promotion, routing and scatter. |
| `disk_s` | Existing GLM53 `t_disk`: expert load-batch wall time (plus single-slot reads). It does not isolate physical NVMe I/O from RAM/page-cache or allocation work. |
| `attn_s`, `ffn_s`, `head_s` | Existing GLM53 cumulative phase timers. They overlap the expert sub-times and must not be added to them. |
| `forward_s`, `decode_forward_s` | Whole forward-call wall time, and its explicit decode subset. Decode excludes sampling, text output and the profile record itself. |

Successful CUDA rows and fallback rows are **not always a partition**: a late
CUDA failure can lead to host recomputation of rows already attempted. Do not
derive a CUDA share from those counters for a window containing errors. The
backend API does not expose separate H2D, kernel-only and D2H timings for these
individual matmuls, so the profiler cannot attribute activation-transfer cost
more narrowly without backend instrumentation.

On the first non-finite grouped expert result, the CUDA tier logs the layer,
expert, device, group position, output coordinate and input range. It replays
that expert with the serial CUDA path and CPU `mlp3` before enabling the normal
model-wide fallback. Set `GLM53_CUDA_DIAG_GROUP=1` to capture and compare every
member of the failing device group in private scratch; this adds work only on
the failure path. For the first bad expert it also reports gate/up/down scale
geometry and finite ranges, then CPU gate, up, SwiGLU and down stage ranges
with separate NaN and infinity counts. The synthetic production-geometry
check is included in
`make glm53-cuda-multidev-check CUDA=1 CUDA_ARCH=sm_86` and skips if fewer than
two CUDA devices are visible.

To inspect the layer-17 expert-166 scale corruption without inference, run
`./glm53 --model DIR --expert-scale-audit`. This uses ERef file offsets for
direct `pread` comparison with the expert slot and `Mat` scale pointers, and
reports direct-file controls for experts 165 and 167. Run
`./glm53 --model DIR --expert-scale-scan` for a scale-piece-only scan of all
layer-17 experts; it prints only experts with non-finite scales or finite
scales above `1e6` in magnitude. Both flags can be combined.

For a 64-token RTX 3070 run with the validated 4 GB tier, from `c/`:

```bash
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=4 GLM53_CUDA_PROFILE=1 \
GLM53_VERBOSE=1 GLM53_MAXT=1024 DRAFT=0 \
./glm53 --model /srv/models-fast/colibri/glm53-flash-i4 \
  --prompt "Answer only with the word OK." --greedy 64 \
  > glm53-cuda-64.log 2>&1
python3 tools/glm53_cuda_profile.py glm53-cuda-64.log --warm-window 16 \
  > glm53-cuda-64.csv
```

The parser prints per-decode-token deltas, including `uploads_delta`,
`evictions_delta`, `errors_delta` and `decode_forward_s_delta`. It identifies a
trailing 16-token **steady working-set candidate** when the continuous decode
window has zero errors, uploads and evictions, even if `tier_full=0`.
`tier_full` remains a separate occupancy gauge. The CSV never computes a
CUDA/fallback ratio.

### Relationship to Qwen3.8 streaming placement

`qwen36_tier.c` already has a Qwen3.8 FP8 streaming mode (`qt_init_fp8`), in
addition to Qwen3.6's full-RAM mode. The `cap == n_experts` requirement applies
only outside that streaming mode. Qwen3.8 temporarily points at the current
RAM slot in `qt_note`, copies weights/scales into owned staging buffers in
`enqueue_locked`, and calls `stream_forget` before returning. The uploader
then creates owned device tensors. It therefore supports recycled RAM slots.

GLM53 deliberately duplicates these placement/residency concepts: per-expert
heat and resident state; a VRAM budget accounting for allocation footprint;
promotion while streamed bytes are in hand; replacement of a colder resident;
owned device copies independent of RAM eviction; resident dispatch with host
fallback; and rollback/freeing of device tensors. GLM53 completes upload
synchronously while the slot is live, so it needs no staging queue or retained
host pointers. It is a separate minimal implementation, not a new residency idea.

The policies differ: Qwen uses per-device placement, a background upload queue,
in-flight protection and grouped issue/take; admission uses
`hot > cold + (cold >> 2) + 4`, with heat decay every 1024 issue ticks and an
optional persisted heat file. GLM53 uses one device, cumulative selected-row
counts, admission after two selections, and strictly `hot > cold` replacement.
Qwen's numeric budget is GiB and its automatic allowance reserves 1 GiB;
GLM53 uses decimal GB, clamps numeric caps to free memory, and reserves 2 GB.
GLM53 does not duplicate Qwen's trunk placement, warmstart planner or FP8 LUT
handling. Reusing the whole Qwen tier would need an int4 streaming entry point
and a clamped-activation execution path: its generic fused/grouped expert
execution uses plain SiLU. No such refactor or kernel change is included here.

## Vision

Reachable from every surface: a path pasted in `coli chat` (read by the client,
sent as a data URI), a file attached or dropped in `coli web`, or an OpenAI
`image_url` part with a base64 data URI. A local path in the request is read
only under `COLI_IMAGE_ROOT` (see `docs/ENVIRONMENT.md`): a file read by the
server happens with its own rights, and an inference client is not the
operator. Remote URLs are refused rather than fetched — a request should not
make the server open a network connection of the sender's choosing.

`tools/glm53_image.py` does the preprocessing and is pinned against the official
`Glm5NextImageProcessor`: identical geometry on every shape tried, bit-identical
pixels wherever no resampling happens, and 0.03 worst case where it does, which
is Pillow's bicubic against torchvision's. The image is scaled with its aspect
kept and padded, never stretched, and padded with zeros *before* normalisation.

Images travel in their own `IMAGE` frame, announced immediately before the
`SUBMIT` they belong to (see `docs/serve_protocol.md`). The engine holds one
pending image and drops an older one rather than answering about the previous
photo without saying so.

`GLM53_MAX_IMAGE_TOKENS` matters more than it looks. The checkpoint's own
ceiling is 8000 tokens per image, which is 2691 for an ordinary 1080p photo — on
an engine that streams experts from disk, a prefill nobody will sit through.
Each image token covers 28×28 pixels, so 256 keeps ordinary text legible and 64
keeps only shapes and colours. The image is shrunk, not cropped: what is lost is
detail rather than pieces.

## Reasoning and tools

The generation prompt opens `<think>` and the model closes it, which is what the
official template does; `--no-think` closes it immediately instead, and the
answer starts at the first word. That form is not in the template — it does not
contemplate switching reasoning off — but it is exactly what the template writes
in front of a past turn that had no reasoning, so the model has seen it.
`--effort` picks the level the template understands: low, high or max.

Both are time controls on this engine, not matters of taste.

Tool calling is complete. GLM-5.3 declares tools differently from GLM-5.2 (its
own preamble, its own JSON serialisation, its own spacing inside `<tools>`) but
emits calls identically, so the existing parser handles them unchanged. The
whole rendering is pinned byte for byte against `chat_template.jinja`
(`tests/glm53_chat_template_harness.py`).

## Environment

See `docs/ENVIRONMENT.md` for the table. The ones that change the shape of a run:
`GLM53_BITS` (dense precision, default 4), `GLM53_EXPERT_GB` (expert cache;
measured from available memory when unset), `GLM53_MAX_IMAGE_TOKENS`,
`GLM53_PREFILL_CHUNK`, `GLM53_MAXT`.

## Tests

```
python3 tools/make_glm53_multimodal_tiny.py --output ~/glm53_mm_tiny
python3 tools/make_glm53_streaming_pair.py --fixture ~/glm53_mm_tiny --output ~/glm53_stream
python3 tests/glm53_multimodal_tiny_harness.py --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/glm53_streaming_harness.py --binary ./glm53 \
        --quantized ~/glm53_stream-i4 --dequantized ~/glm53_stream-deq
python3 tests/glm53_serve_harness.py        --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/glm53_vision_serve_harness.py --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/glm53_chat_template_harness.py --template <model>/chat_template.jinja
make VK=1 glm53 && python3 tests/glm53_vulkan_harness.py --binary ./glm53 --fixture ~/glm53_mm_tiny
```

The generators want transformers 5.16.1, pinned because an oracle written by a
different version is a different oracle. Each harness skips with the command that
builds what it is missing rather than throwing, and exits 2 when it does: a skip
verified nothing and must not read as a pass.

The harnesses are named `glm53_*_harness.py` so `make test-python` does not
collect them as empty unittest modules. `tests/test_glm53_oracles.py` wraps the
two stdlib-only oracles for unittest; it runs them when `GLM53_TINY`
(`tools/make_glm53_tiny.py`) and `GLM53_MM_TINY` (the multimodal fixture above)
point at their fixtures, as the GLM-5.3 CI job does, and skips with that reason
otherwise.

Two of the generators refuse to write a fixture that cannot fail: one rejects a
degenerate model that answers the same token everywhere, the other a fixture
whose answer does not change when the image is inverted. The vision test does
not check that the model answers — it would answer anyway, ignoring the pixels —
but that **two different images give two different answers**.

`tools/check_glm53_container.c` compares a real converted expert against a numpy
dequantisation of the same bytes. It needs a converted checkpoint, so it does not
run in CI, and it is the first thing to reach for when the real model answers
strangely.
