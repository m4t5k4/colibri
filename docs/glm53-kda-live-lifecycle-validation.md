# CUDA KDA live lifecycle validation

This is the next correctness gate before a default-on experiment. Keep
`GLM53_CUDA_KDA` default OFF. Do not tune kernels, transfers, placement or cache
policy. The checkpoint-free lifecycle suite proves host bookkeeping with a
device double; it does not establish live CUDA SERVE behavior.

## Freeze the rig

Use one revision and binary for each OFF/ON pair, with the established eight-GPU
production environment. Record commit, binary hash, environment and snapshot
hash before and after. Run engines sequentially, not simultaneously.

```bash
cd ~/colibri-glm53-cuda
git rev-parse HEAD
git status --short
sha256sum c/glm53 artifacts/phase2e/usage-snapshot-before-phase2e
```

The usage hash must remain
`7a246a49fc1dc3fb6360c131153c22012ca08949da390455654652d76e18a953`.
Retain the Phase 2F3 production flags and set these additional overrides:

```bash
export SERVE=1 SERVE_BATCH=1 KV_SLOTS=2 COLI_PIN_SLOTS=4
export SNAP=/srv/models-fast/colibri/glm53-flash-i4
export COLI_USAGE="$PWD/artifacts/phase2e/usage-snapshot-before-phase2e"
export USAGE_SAVE=0 GLM53_CUDA_WARM_RESIDENCY=0
export GLM53_CUDA_PROFILE=1 GLM53_VERBOSE=1
```

Use `GLM53_CUDA_KDA=0` for the reference and `=1` for the live CUDA run.
Do not alter the binary between those runs. The protocol driver must retain
stdin until every request has finished, then close it and wait for normal
shutdown so final KDA/expert counters are captured.

## Stock SERVE protocol cases

The stock multi-slot/decode-prefill/reset comparison is runnable with:

```bash
python3 -B -m unittest discover -s c/tests -p test_glm53_kda_serve_lifecycle.py -v
bash c/tests/run_glm53_kda_serve_lifecycle.sh
```

If the child exits before the first response, first run only the OFF diagnostic:

```bash
bash c/tests/run_glm53_kda_serve_lifecycle.sh --diagnostic-one-request \
  --startup-timeout 900 --read-timeout 600 --request-timeout 1800
```

This never starts ON or requests 2-8. The driver uses raw pipes and complete
write handling like the production gateway, drains READY/STAT/EMAP before SUBMIT,
and bounds startup, idle reads and whole requests. On failure, `0.failure.json`
records the pre-cleanup poll/exit/signal, stdin status, startup frame identities,
last complete frame, request ID and stderr tail. It separately labels termination
of a still-live child; an already-exited child's status is preserved. Exact input
bytes are saved as `0.stdin.bin`. The runner records after-hashes even on failure.
Successful one-request completion is not a full lifecycle-validation pass.

The failure report also snapshots Linux `/proc/<pid>/status`, `stat`, `wchan`,
all fd symlink targets, fdinfo for 0/1/2 and each task's wchan before cleanup.
When the raw stdout read returns zero bytes, it takes a second snapshot directly
in the reader thread at that event. Parent pipe identities at launch and failure
allow child fd 1 to be compared to the expected pipe. Permission/disappearance
races are recorded as errors rather than suppressing the rest of the snapshot.
On Linux, FIONREAD on the parent's stdin pipe also reports currently queued bytes
without consuming them. The exact stdin capture size/hash, successful writer byte
count and escaped first request distinguish logged intent from completed writes.
These observations do not by themselves prove the child parsed a consumed frame.

`FileIO.read(65536)` on a blocking Linux pipe can return short nonempty data;
zero bytes indicate EOF. A nonblocking raw stream can instead return None when
it would block. The driver now diagnoses None explicitly, never as EOF, and
records raw stream type, pipe identity/blocking mode and terminal event kind.

### Optional fd-filtered syscall diagnostic

First use the untraced one-request run above. If additional syscall evidence is
needed and installed strace supports `--trace-fds`, run the following manually:

```bash
cd ~/colibri-glm53-cuda
command -v strace
strace --help | grep -- '--trace-fds'
# Stop here if either check fails; do not fall back to unfiltered tracing.
mkdir -p artifacts/phase2f-live
trace=$(mktemp "$PWD/artifacts/phase2f-live/stdio-strace.XXXXXX")
strace -f -tt -T -yy -s 160 --trace-fds=0,1,2 \
  -e trace=read,write,writev,close,dup,dup2,dup3,fcntl,poll,ppoll,select,pselect6 \
  -o "$trace" \
  bash c/tests/run_glm53_kda_serve_lifecycle.sh --diagnostic-one-request \
    --startup-timeout 900 --read-timeout 600 --request-timeout 1800
```

This traces the launcher/driver and descendants so the actual glm53 PID remains
the Popen PID in the report. Use that PID to identify engine calls. Descriptor
filtering excludes normal model-file I/O on other descriptors and prints only
160 payload bytes per call. It includes stdin reads and stdout/stderr operations
plus fd-related polls. It does not capture Python's read side if that pipe uses
a descriptor other than 0/1/2. Tracing still perturbs execution: do not use this
run for performance claims, and do not broaden it to all model-file I/O.

Interpretation: missing/replaced child fd 1 supports A; matching fd 1 with a
zero-byte raw read requires investigating descriptor-sharing/races or the reader
(B), not assuming a model failure. Queued complete request bytes or traced stdin
reads establish consumption evidence for C/D. A zero queued byte count alone
does not prove successful parsing or identify where request processing hangs.

The shell script establishes the production environment and runs the same binary
OFF then ON, without rebuilding. It writes logs, request/results JSON and hashes
under a fresh `artifacts/phase2f-live/serve.*` directory. The Python driver freezes
the OFF payloads for ON, checks token IDs and DATA bytes after each request, tests
reset against the same prompt in another fresh slot, and rejects missing REUSE
coverage. Healthy final calls must match the actual SERVE decode-forward count;
all KDA/expert errors and fallbacks must be zero. This stock script does not test
direct stale-state pin/save or injected failures.

`SUBMIT <id> <slot> <payload_bytes> <max_tokens> 0 1 logprobs=1 [pin=1]`
followed by the UTF-8 payload and a newline requests deterministic generation.
Wait for READY before submitting, and DONE before the next request. DATA and
ECHO payloads are byte-counted; they are not line-delimited text. Save raw stdout
and stderr separately. Use 16 generated tokens per turn initially.

Run the following identical request schedule OFF and ON:

| Case | Request sequence | Required evidence |
| --- | --- | --- |
| Session switching | Generate in slot 0, then slot 1, then continue slot 0, then slot 1; repeat twice | Same outputs as OFF; outgoing stale state materialized and incoming state pushed |
| Decode/prefill/decode | Append the emitted text and a multi-token suffix to the previous prompt in one slot, then generate again | `REUSE <id> <reused> <prompt_tokens>` has reused > 0 and a multi-token new suffix; CPU prefill then new CUDA decode calls |
| Pin/restore/continue | Pin prefix P, generate on it, extend to P+A, then request P+B and generate | `[PIN]` confirms save; REUSE confirms the pinned prefix restored; continuation matches OFF |
| Reset/fresh | After GPU decoding, submit an unrelated prompt in the same slot | REUSE is zero; result matches the same prompt in a fresh slot/reference session |

Use distinct prompts such as a virtual-memory explanation in slot 0 and a C
binary-search explanation in slot 1. For text-based continuation, re-tokenizing
decoded output can change token boundaries. Require observed REUSE rather than
assuming concatenation preserved the cached prefix; otherwise report the case
as unexercised and use the direct driver below.

Compare generated token IDs when supplied by the logprobs metadata, DATA payload
bytes and actual emitted counts. Do not compare rounded timing/PROF/HITS fields
or require bit-identical floating-point logprobs. Capture REUSE/PIN diagnostics
alongside the semantic comparison. Do not assume 34 * emitted tokens is the
CUDA call count: SERVE emits the last token without forwarding it, and may stop
early. The driver must count actual decode forwards.

## Boundaries stock SERVE cannot prove

There is no explicit RESET protocol command. Reset is triggered by incompatible
prefix reuse in `serve_one()`. Also, `pin=1` saves after `forward_prefill()` and
before generation, not immediately after GPU decode. Public protocol runs do
not alone prove a snapshot copied directly from stale GPU-owned host state.

The production `coli_cuda_kda_step()` wrapper always passes `KDA_OK`; no runtime
environment flag exposes failure injection. Do not simulate a transactional
failure by returning false after a successful committed step, resetting CUDA,
or causing real device loss.

## Required test-only live-CUDA driver

Pending implementation, the exact proposed file changes are:

- `c/tests/test_glm53_kda_live_lifecycle.c`: new driver including production
  lifecycle functions, loading real weights and calling the real CUDA backend.
- `c/tests/glm53_kda_live_backend.cu`: new test translation unit including
  `backend_cuda.cu`, privately renaming the normal step/push wrappers and
  supplying test-only wrappers that pass the existing transactional fault
  argument. Kernels, streams and device allocations remain the real backend.
- `c/tests/glm53_kda_live_test_api.h`: new private arm/observe API for that test
  executable only; no environment variables or production header additions.
- `c/Makefile`: a separate live test backend object/executable, never linked into
  `glm53` and never substituted for the normal production backend object.
- This document: exact build/run commands once that target exists.

No production source changes are required by this proposed arrangement. There
is intentionally no build/run command for the unimplemented driver yet.

Prepare a separate test executable, not a production feature flag. Reuse the
production functions and backend object, with the real checkpoint and all 34
initialized KDA objects. The driver must explicitly exercise:

1. Two actual GSession instances alternating owners after successful GPU tokens.
2. An exact token-ID continuation through CPU multi-token prefill and back to
   GPU decode, without text re-tokenization ambiguity.
3. `slot_pin_save()` immediately while host state is stale, then advance several
   GPU tokens, `slot_pin_restore()` and continue from that checkpoint.
4. `slot_reset()` while GPU state is newer, then create a fresh session and
   reproduce a fresh CPU-reference generation.
5. A targeted post-recurrence/pre-commit failure after at least two successful
   GPU tokens in one layer. Route it through the real `g53_kda_try_decode()`
   fallback, not an isolated backend retry. Also test a failed push after CPU
   prefill invalidates a previously successful device generation.

Expose the existing fault argument only in a separately compiled test backend
behind a test-only macro. Bind injection to a layer/object and successful-token
ordinal, consume it once, and record committed/next generation evidence. The
normal backend must remain unable to inject faults. This driver/hook is a next
implementation step; it is not an existing target or runnable stock flag.

After every transition, compare final output, recurrent state and convolution
history to CPU continuation. Use the established CUDA regression tolerances,
reject nonfinite values, and report observed errors. Push/pull and checkpoint
copies without arithmetic must compare exactly. Pull stale state at observation
boundaries only: pulling after every GPU token would mask the lifecycle being
tested.

## Acceptance and reporting

Healthy cases require zero KDA errors/fallbacks and zero expert errors, actual
CUDA decode calls, unchanged hashes and semantic equality. Report state pushes,
pulls, invalidations, per-device calls and final allocation/cache counters.
If aggregate reporting cannot identify a transition, add test-driver assertions
or explicit test-only observations; do not infer coverage from a model completing.

For each forced failure require: committed state/window unchanged, next generation
changed for the post-recurrence fault, host recovery from the committed generation,
the failed token applied once on CPU, only the targeted layer disabled, subsequent
CPU continuation correct, other CUDA KDA layers active, expert errors still zero
and promotion continuing. One intentional KDA error is expected for a single
injection; later CPU fallback counts may grow for the disabled layer. Never fold
injected failures into the healthy zero-error requirement.

Do not authorize a default-on experiment until both the stock SERVE comparison
and these direct real-CUDA lifecycle cases pass. This is correctness validation,
not Phase 2F4 performance work.
