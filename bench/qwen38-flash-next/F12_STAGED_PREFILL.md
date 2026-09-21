# F12: staged Flash-Next prefill

2026-09-05, root implementation in `perf/qwen38-flash-next` at HEAD
`11e76f29a68ed5e22be8fd14cc72e318bdcc6111` plus preserved dirty source.
Experimental paths remain default-off. No 2,000 tok/s or fleet-wide completion
claim. GLM, DeepSeek and dense Qwen runtime branches are not modified by F12.

## Implemented

- Correct sigmoid multisequence normalization for sigmoid SSM models. The new
  wrapper calls the shipping scalar sigmoid body; generic SiLU is unchanged.
- Exact-family HC preparation in tiles up to32, with mixed-row staging and
  saved injection scales. Attention still executes causal token-ordered core.
- Exact-family tiled SSM projections, batched FP32 BA gates, sequence conv and
  ordered FP32 GDN without verification snapshots, then sigmoid normalization
  and a batched saved-HC injection. This is not a tensor-core chunked FLA port.
- Compact original-layout NVFP4 expert scheduling, reusing the existing private
  planner/GEMM arithmetic and persistent workspace. No transpose-weight copy.
- Strict geometry, arena, alias, kernel and selector admission before effects;
  checked device planner status before accepting routed results.
- Optional same-input HC, SSM and compact-MoE numerical replay diagnostics.

All F12 paths are currently bounded to canonical single-sequence eager
Flash-Next, BF16 residual/KV, FP32 SSM state and prompt end at most2048.
Singleton chunks retain decode. Speculation and multimodal are not qualified.

## Selectors

For staged candidate execution, explicitly set all four to1:

```text
ATLAS_QWEN4_PREFILL_MOE_BATCH
ATLAS_QWEN4_PREFILL_HC_EXACT
ATLAS_QWEN4_PREFILL_SSM_EXACT
ATLAS_QWEN4_PREFILL_MOE_COMPACT
```

For numerical qualification only, also set all three to1:

```text
ATLAS_QWEN4_PREFILL_HC_CHECK
ATLAS_QWEN4_PREFILL_SSM_CHECK
ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK
```

CHECK runs have **no timing eligibility**. They compare finite raw BF16/F32
values, not tolerance-promoted text. SSM replays from identical initial hidden,
H and conv state and compares hidden, H, conv and saved HC scales. Compact
replays shipping gate/up and down on identical inputs, routing metadata and
activation, with independent output clearing. Errors propagate before reduction.

## Completed gates

- Current-tree CPU integration:617 library +63 focused tests =680 passed.
  Original verbose tool output was truncated; complete quiet repeat summary:
  `/var/tmp/atlas-f12-microgate.QKtSjJRN/cpu-results.txt`.
- Synthetic norm GPU:66/66 cases, rows1..2048 including tile boundaries,
  repeated reset, zero/positive/negative gates. Byte-exact scalar sigmoid;
  old SiLU differs in every case; finite results, poisoned padding/redzones
  and immutable inputs verified.
- Synthetic HC GPU:36/36 cases, rows1..2048, repeated reset. Normalization,
  down/activation/up, mixing/packing, saved scales and injection byte-exact;
  finite results, redzones and immutable inputs verified.

These synthetic gates do not establish actual-model or full persistent-state
parity. See `f12_microgate/` for source and CPU fixtures. Both GPU executables
used `-O3 --use_fast_math --fmad=false -arch=sm_121f -Xptxas -O3`.

Evidence directory: `/var/tmp/atlas-f12-microgate.QKtSjJRN/`.
Norm ELF SHA256 `21234a581eba052ffff299d80aa247ded4b79a4f6a0dc4ee633f873bb8303558`.
Norm raw JSONL SHA256 `d3301c5f66a7b67d72c4c6faa08decd02bf5717edfc6a8601953e1461509bcec`.
HC ELF SHA256 `d8882080de84f87a76fabf40003181cd258aed6861289516f1d34cc6e621afa5`.
Raw files retain diagnostic FNV integers exactly; FNV is not provenance.

## Native and actual-model gates

Native build output: `/var/tmp/atlas-flashnext-f12-native.s8bgRjIL`.
Build exit0 in2m26s;160 selected kernels. Server ELF SHA256
`ad1c5168fc73daf6005b2fd80dcefab645441dfb3132fd1326f09074a5a0b602`.
Generated target table SHA256
`9ff69d03e81fbd5007da8fd844c71aba5c6db9f3be299fc4c725ada7fef6d88d`.
Prebuild source fingerprint:
`5b84760c3f1cc7b5bed8f922de0d3221dd56227ac988a0937afce05f778c6644`.
`f12_microgate/source_fingerprint.sh` hashes tracked/untracked nonignored build
inputs and expands directory aliases; bench/docs are not build inputs.
Build uses the exact F11 native command, substituting this target directory;
the target KERNEL.toml places `--fmad=false` after fast-math.

Source fingerprint unchanged after native build and checked model run.
Checked run: `/var/tmp/atlas-flashnext-f12-checked.0ACdzpge/summary.json`.
Complete server log SHA256
`acd402e80f0684703f1b05411940de083049a300417ea900aa1da5d7e5735fe5`.
PID2552020/start8358210 received binding-checked SIGINT and exited0; GPU and
port8898 were empty after drain. No checked-run timings are credited.

Eight short requests (canary, arithmetic, retrieval, Python addition; each
twice) passed expected output/count/cache0/stop checks. One exact256 and one
exact2048 token corpus request each completed32-token continuation. All ten
requests passed960HC,360SSM and480compact numerical checks, with20 complete
layer receipts and no reported error. At2048, continuation crosses the initial
QSA dense region, but QSA cache/ranking has not been independently captured.

The256 output matches historical scalar C3;2048 differs in wording. Both
describe86 numbered shelves and blue/red folder counts, but this is **not** a
full scalar-output parity pass. F8's grouped router/shared GEMMs and routed
rounding differ from scalar. Same-binary F8-only comparison completed at
`/var/tmp/atlas-flashnext-f12-f8baseline.TTxOxC4s/`: canary,2x256 and2x2048
all match checked F12 output. The wording difference predates F12's core and
compact-schedule changes. Baseline server drained with exit0.
Do not promote local F12 numerical checks to F8-versus-scalar equality.

## Uninstrumented diagnostic measurements

Candidate evidence: `/var/tmp/atlas-flashnext-f12-fast.WwHVUdv2/summary.json`.
One warmup plus five samples per bin, temperature0, max32 outputs, cache0.
All outputs/counts/finish match F8 baseline and checked F12;26 complete layer
receipts across13 requests, no CHECK or runtime-profile activation. Source
fingerprint unchanged. PID2573876/start8443406 drained with SIGINT/exit0;
GPU and port8898 empty afterward.

| Input tokens | Candidate median TTFT | Effective prefill tok/s | F8 baseline TTFT, two-sample median | Diagnostic ratio |
| --- | ---: | ---: | ---: | ---: |
|256|2.697375s|94.9071|4.450359s|1.6499x|
|2048|17.395284s|117.7331|33.627992s|1.9332x|

Effective prefill means input/serverTTFT, **not isolated GPU prefill**. This
is an A-then-B diagnostic, not ABBA or a production-qualified performance win.
Reported decode medians38.571/18.337 tok/s respectively are a32-token response
metric, not representative decode or DFlash2 qualification. No2000+ claim.

Complete candidate server log SHA256
`4f6371404bbfe094e2a95d19c45f2908a0ef976824515fdbfe88b8546dac960e`.
The checked and timing runs are distinct; never aggregate their timings.

## Provisional profile and next implementation targets

CUDA-only profile: `/var/tmp/atlas-flashnext-f12-profile.wlYRUqpI/summary.json`.
Uncollected startup/warmup, then one2048/max1 request, output `The`, cache0.
Collector stopped/exported before binding-checked server SIGINT; exit0,
GPU/8898 empty, and Nsight session automatically closed. No lane is retained.

Nsight reports **possible missing CUDA events** and unsupported Unified Memory
tracing. Consequently this is provisional attribution, not complete GPU timing
qualification. It does not invalidate the separate uninstrumented five-run
measurement. The trace contains452075 kernel events, one process and one
stream, with16.262s summed kernel duration inside a17.466s kernel span.
Host CUDA API waiting overlaps device execution and must not be added to it.

Largest observed kernel groups: exact M32 projections5.440s, compact expert
GEMM3.033s, attention Q projection1.772s, paged attention1.751s, attention SW
projection1.121s, dense GEMV0.895s, GDN recurrence0.659s. Within M32 projections:
QKVZ2.467s, HC up1.239s, SSM output0.842s, HC down0.451s, HC injection0.441s.
These are observed trace totals, not exhaustive independently qualified costs.

Next bounded candidates, each default-off and independently tested:

1. SSM output projection, then QKVZ: existing original-layout
   `ops::w4a16_gemm`/`self.w4a16_gemm_k`, same buffers/weights, no new weight
   copy. This changes reduction/rounding; preserve strict exact CHECK and give
   a separate numerical/quality gate. Do not silently select generic FP8
   prefill dispatch or missing Flash-Next `w4a16_gemm_pipe`.
2. Batch remaining causal attention projections/core and HC projections.
   Preserve actual QSA/KV state, dense-window boundaries and continuation.
3. Profile/optimize original-layout expert GEMM data access. GLM and DeepSeek
   already compact their active work and require their own EXL3 implementations;
   do not transplant this NVFP4 kernel by filename.

Exact scalar MoE batching can reuse existing token-count-parameterized batch3
kernels as an oracle, but must not reuse the small logits arena for M2048
shared-gate scratch. It is not a guaranteed speed improvement over MMA.

Still separate: F8 grouped FFN versus scalar router/shared precision, full
KV/QSA/PLE and next-token state capture, chunk/reset/interleaving boundaries,
long-QSA ranking, representative coding and vision. Do not promote short text
equality or path engagement to these qualifications.
