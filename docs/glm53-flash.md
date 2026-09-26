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

## Optional single-GPU CUDA expert tier

On Linux (including a configured WSL CUDA toolkit), build for an RTX 3070:

```bash
cd c
make glm53 CUDA=1 CUDA_ARCH=sm_86
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=auto ./glm53 --model /path/glm53_i4 --prompt "Hello" --greedy 32
```

The default build/run is unchanged. Explicit CUDA requests require a CUDA
build, a working device, and the streaming int4-gs64 container. `COLI_GPUS`
is rejected: this implementation supports exactly one `COLI_GPU` ordinal.
The existing Windows DLL build convention remains `CUDA_DLL=1`, not `CUDA=1`.

Only the normal full-model CLI and SERVE loader enables this tier. Segment,
Edge, and range-only model instances never initialize it, even when a Segment
covers every layer or `COLI_CUDA=1` is inherited. Their inactive tier cannot
shut down the process-global CUDA backend owned by another model, and their
CPU capabilities and `numeric_class` remain unchanged. The CPU-only adapter
build also ignores this CUDA opt-in instead of rejecting it.

The RAM cache (`GLM53_EXPERT_GB`) stays intact. On-demand promotion begins
after an expert has been selected for at least two rows; a full device tier
replaces its least frequently selected resident only for a hotter candidate.
Heat is cumulative for this model lifetime. Uploads are synchronous, after
the normal disk-to-RAM read, and own their device memory independently of
RAM-slot eviction. Resident hits bypass disk and RAM reads. Cold misses use
the existing host path and become eligible for subsequent CUDA execution.

`CUDA_EXPERT_GB` is a decimal-GB cap, or `auto` (default): free VRAM minus
2 GB of runtime headroom. A numeric cap is also clamped to that allowance;
zero forces host execution. This tier does not upload attention, dense or
shared-expert weights. Gate/up/down run through existing CUDA resident
matmuls, with GLM53's exact clamped SwiGLU on the host between projections.
The generic fused CUDA expert API uses plain SiLU and is unsuitable here.
Extra activation transfers and synchronous promotion mean speedup is not
guaranteed. The model/container format is unchanged.

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
python3 tests/glm53_cuda_harness.py --binary ./glm53 --fixture ~/glm53_stream-i4
```

The portable target also destroys Segment/Edge instances while a full-model
owner has a resident expert, verifies that expert still executes, and checks
that shutdown happens exactly once when the owner is released. Source tests
pin initialization to the full-model loader and preserve adapter CPU metadata.

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

### Hardware validation

Validated on Linux with an NVIDIA GeForce RTX 3070 (Ampere `sm_86`,
8 GB-class VRAM) and CUDA Toolkit 12.4. Build used:

```bash
make -B glm53 CUDA=1 CUDA_ARCH=sm_86 CUDA_HOME=/usr \
  EXTRA_LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
```

**Real-device tensor validation.** `make glm53-cuda-check ...` executed real
CUDA int4-gs64 matmuls. Measured CPU-vs-CUDA projection differences were:

| Projection | max_abs | max_rel |
|---|---:|---:|
| gate | 1.52587891e-05 | 5.32183816e-06 |
| up | 1.52587891e-05 | 4.44259501e-06 |
| clamped-down | 1.13248825e-06 | 0.000106291905 |

The test used the provisional tolerance `atol = 0.002`, `rtol = 0.002`:

```text
PASS GLM53 CUDA tier: gs64 numerics, owned tensors, heat eviction, cleanup
```

**Tiny streaming-model validation.** `tests/glm53_cuda_harness.py` passed on
the same NVIDIA GPU:

```text
PASS GLM53 model CUDA: executed tensors, residency, zero-budget and injected-failure fallback
```

This verifies actual model CUDA execution, VRAM residency, host fallback with
CUDA disabled by budget, and injected GPU failure fallback. GPU detection
alone cannot satisfy this test.

**Real checkpoint validation.** The converted GLM-5.3-Flash int4-gs64 container
had 45 layers (42 sparse), hidden size 4096, and 288 routed experts with top-8
routing. Routed expert storage was reported as 175 GB int4-g64. Command:

```bash
COLI_CUDA=1 \
COLI_GPU=0 \
CUDA_EXPERT_GB=4 \
GLM53_VERBOSE=1 \
GLM53_MAXT=1024 \
DRAFT=0 \
./glm53 \
  --model /srv/models-fast/colibri/glm53-flash-i4 \
  --prompt "Answer only with the word OK." \
  --greedy 8
```

| Observed metric | Value |
|---|---:|
| CUDA expert budget | 4,000,000,000 bytes |
| Final resident experts | 282 |
| Final allocated expert VRAM | 3,991,928,832 bytes |
| Successful CUDA expert rows | 434 |
| Host fallback rows | 1,582 |
| Uploads | 323 |
| CUDA errors | 0 |
| RAM expert-cache hits | 241 |
| RAM expert-cache misses | 1,084 |
| Expert bytes read | 15,344,861,184 |
| Load time | 63.4 s |
| Prefill | 7 tokens in 11.9 s |
| Decode | 8 tokens in 10.8 s = 0.743 tok/s |

The full-model run demonstrates dynamic promotion from the existing RAM
streaming cache into device-owned VRAM, CUDA execution of resident routed
experts, continued host execution for misses, and eviction/replacement after
the 4 GB tier filled, completing with zero CUDA errors. GLM53 clamped SwiGLU
remains on the host; the generic fused plain-SiLU CUDA expert path is not used.
Multi-GPU remains unsupported and out of scope.

The 0.743 tok/s observation is neither a speedup claim nor representative
performance: this was a short cold/adaptive validation run with synchronous
promotion and host-side clamped SwiGLU activation transfers. It is not a
comparison with unrelated CPU or Vulkan runs. Generated text was repeated
`!`, so this prompt does not establish semantic correctness of the full 321B
checkpoint. The tiny CPU-vs-CUDA harness supplies correctness evidence for the
CUDA path; the full checkpoint run supplies execution/residency/fallback
evidence.

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
