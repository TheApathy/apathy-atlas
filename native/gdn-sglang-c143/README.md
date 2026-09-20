# SGLang c143 GDN SM121 shadow

This surface is absent from Atlas kernel manifests and production dispatch.
It exposes three caller-workspace C ABIs solely for the accompanying common-input
gate:

- v1 preserves the original compact tensors: BF16 Q `[M,2048]`, K
  `[M,2048]`, V `[M,6144]`, FP32 log-decay/beta `[M,48]`;
- v2 accepts the exact Qwen3.8 C1 production layout: a BF16 QKV row with
  offsets `0/2048/4096`, row strides `10240/10240/10240`, and FP32 interleaved
  `[alpha(48),beta(48)]` rows with stride 96. A same-stream conversion kernel
  writes compact `log(max(alpha,1e-30))` into the v2 workspace tail before the
  unchanged three-pass core.
- v3 retains v2's exact signature and layout admission but replaces its scalar
  one-program-per-head recurrence/output spines with pinned-c143 geometry:
  `(NT,H)` KKT and W/U programs, four `BV=32` state programs per head, and two
  `BV=64` output programs per chunk/head. The recurrence contractions and
  output `A@V` use SM121 BF16 tensor-core MMA while the master state stays FP32.

All remain default-unrouted. Build and, only with a separately reserved GPU,
run the four-way functional gate. It retains Atlas, production-v2, and SGLang
as the three oracles and adds production-v3 as the candidate. The v2 arm still
requires compact-v1 == production-v2 exact output/state bytes; v3 must pass all
canary/determinism/finite/numerical screens and median <2.5 ms at M2079 and
<10 ms at M8192. Timing admission additionally requires near-balanced position
counts for all four arms, a positive paired median win over current Atlas, and
v3 no slower than 1.20x pinned SGLang's same-process median:

```bash
bash native/gdn-sglang-c143/build.sh
python3 native/gdn-sglang-c143/gate.py \
  --m 2079 --m 8192 --warmup 2 --reps 11 \
  --output native/gdn-sglang-c143/result-v3-$(date -u +%Y%m%dT%H%M%SZ).json
```

Per-shape receipts are emitted only after every correctness and comparative
performance gate for that shape passes. The final result file is written only
after both required shapes pass. Output creation is exclusive: an existing or
symlink path is rejected so a failed rerun cannot leave a stale result under
the requested name.

The port consumes FP32 state in Atlas `[H,K,V]` order and emits compact BF16
`[M,48,128]`. Its workspace is caller-owned and must remain live through all
same-stream launches. The ABI performs no allocation, free, synchronization,
graph capture or production routing.

## Workspace ownership

`atlas_gdn_c143_workspace_size_v2(M)` and `_v3(M)` are exact and include the compact
log-decay tail. Current receipts are 130,565,952 bytes at M=2,079 and
506,462,208 bytes at M=8,192 for both versions. v3 aliases its solved-A phase
with its later entry-state/corrected-value phase, so the redesign adds no
workspace capacity. The exact query also admits 1,048,576 tokens at
64,827,162,624 bytes; this is ABI capacity evidence only, not a long-context
quality or runtime qualification. A future production caller should allocate one
workspace in its process/forward arena and reuse it sequentially across the 48
SSM layers on the same stream. It must not allocate one workspace per layer.
If forward calls can overlap across streams, each concurrently usable stream
requires a distinct retained workspace (or an externally proven exclusion).
The allocation may not be freed or reused until its stream has completed, and
graph capture must stay disabled until the library and address lifetime are
explicitly made replay-stable.

The build injects the candidate source SHA256 into
`atlas_gdn_c143_abi_identity()`. The gate compares that exact identity with the
current source before any workspace query or launch, so a stale or substituted
shared library fails closed. v3 also rejects a null stream instead of silently
using CUDA's default stream. The same build-time binding covers the Atlas WY32
reference bridge and both its wrapper and included production-source bytes, so
a stale reference binary cannot make the comparative screen pass.

## Production-feasibility audit

Atlas's existing `ssm_conv_out_f32` arena is `M * 16384 * 4` bytes. It is large
enough for v3 at both qualified shapes: 136,249,344 versus 130,565,952 bytes at
M=2,079 (5,683,392-byte margin), and 536,870,912 versus 506,462,208 bytes at
M=8,192 (30,408,704-byte margin). In the monolithic path that arena is dead
after the same-stream FlashInfer QKVZ projection and until the FlashInfer SSM-O
projection. In the single-sequence three-phase path, all phase-1 launches
precede the full GDN call and all phase-3 launches follow it on the same stream.
Those two boundaries therefore provide a sequential caller-workspace window;
no per-layer allocation is required.

The ABI is deliberately exact batch one. Atlas's multi-sequence phased path
instead calls `prefill_gdn_full_batched` with stacked rows and a device array of
state pointers. It cannot call this ABI directly. That path must remain on its
existing fallback unless a separately qualified batched ABI or explicit
per-sequence serialization is implemented. Capacity and lifetime feasibility
do not constitute production routing or runtime qualification.

The authoritative signatures and element-versus-byte units are in
`atlas_gdn_c143.h`. v2 rejects nulls, undersized workspaces, misalignment, and
any base-offset/row-stride tuple other than the exact Qwen3.8 C1 production
layout before launching a CUDA kernel.

The v2 runtime gate passed every numerical/canary gate but was rejected at
8.3364/32.5435 ms versus pinned SGLang 2.1197/8.4502 ms. The diagnosis is the
v2 CUDA port's scalar recurrence and output contractions, not the layout
adapter. v3 is a static-ready redesign only: no production-quality or
performance claim exists until the four-way runtime gate has run for both
lengths and all subsequent real-layer state/output and task-quality gates pass.
Before importing SGLang, the harness requires the exact c143 commit, a clean
tracked donor worktree, all four audited donor source hashes plus LICENSE, and
then verifies that every loaded FLA dependency resolves under that pinned tree.
