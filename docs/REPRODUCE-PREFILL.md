# Reproducing the DeepSeek-V4-Flash prefill numbers (historical frontier: 1,062 tok/s @ N=2,410)

> **HISTORICAL THROUGHPUT LADDER (verified 2026-08-10; not a current-tree
> rerun).** The original 792 tok/s headline was followed by `d8ebbd90` (884)
> and `e66088e9` (905, TTFT 2.66 s). Commit `887a697c` then ran a controlled
> same-configuration cuBLASLt A/B with 20 measured requests per arm:
> `ATLAS_V4_PREFILL_CUBLASLT=1` measured min/median/max TTFT
> 2.10/2.27/2.82 s (1,148/1,062/854 tok/s), versus 2.76/2.81/3.63 s
> (872/856/663 tok/s) with it off; tool-eval remained 90/100. Thus 1,062
> tok/s is the latest verified historical prefill frontier, while 905 remains
> the exact preceding campaign rung rather than being rewritten. The earlier
> rungs' opt-outs extend the bisect table in §2:
> `ATLAS_V4_PREFILL_TC2=0`, `ATLAS_MOE_SHARED_K64=0`,
> `ATLAS_MOE_PREFILL_EXACT_TILES=0`, and
> `ATLAS_EXTRA_NVCC_FLAGS="-DATLAS_MOE_NO_WIDE_DEQUANT_STORE"` /
> `"-DATLAS_MOE_NO_PACKED_EPILOGUE"` for the SASS-level toggles.

Everything below runs from a clean checkout of `combined-residency` on one
GB10 (DGX Spark, sm_121). Record the exact commit, checkpoint, binary, flags,
and probe output for each new run; the historical figures below do not retain
all of those artifacts.

## 0. Prerequisites

- One GB10, 128 GB unified memory, no other process holding the GPU
  (a resident server will OOM the microtests — kill it first).
- `DeepSeek-V4-Flash-162B` at `/home/flocka/models/DeepSeek-V4-Flash-162B`
  and the 0731 drafter at `.../DeepSeek-V4-Flash-0731-drafter`.
- `tool-eval-bench` (the quality gate) — any recent build.

## 1. Build — the flag that is NOT optional

```bash
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo build --release
```

**The default target is `qwen3-next-80b-a3b`.** A plain `cargo build --release`
compiles the wrong kernel set and the server dies at boot with
`No compiled kernel target matches model_type 'deepseek_v4'`. Microtests can
mask this because `kernels/gb10/common/` compiles into every target set.

## 2. Serve

```bash
scripts/dsflash-serve-bench.sh <run-name> 5 \
  ATLAS_V4_ATTN_NVFP4=1 ATLAS_V4_ATTN_RELEASE_BF16=1 \
  ATLAS_MTP_GATE_FORCE=1 ATLAS_DFLASH_LOW_GEAR=1
```

The launcher now resolves the current checkout by default (override `REPO`,
`BIN`, `MODEL`, or `DRAFTER` explicitly), validates the executable/checkpoint
identity, and writes `<log>.planned.receipt.json` before starting the server.
The planned receipt binds the full git commit plus dirty-tree hash, binary SHA-256,
model/drafter config/index/tokenizer hashes, every shard referenced by each
weight index, normalized argv, the raw explicit no-secret environment map, and
a SHA-256 of the kernel-visible inherited environment after explicit overrides.
The full map is never serialized, so inherited secrets are bound without
disclosure.
Missing shards,
checkpoint-escaping shard paths, secret-bearing argv, a missing binary gate
string, or a missing executable fail before launch. `PRINT_CONFIG_ONLY=1`
performs this preflight and prints the effective config without starting a child.
Receipt and decode-probe files are exclusive-create artifacts: reuse of a run
name fails instead of overwriting evidence, so choose a fresh name per run.

For a real launch the script then waits up to `ATLAS_BENCH_READY_TIMEOUT`
seconds (default 600), verifies the current Git/tree identity, live `/proc`
executable hash, exact argv and selected environment, PID start time,
listening-port ownership, `/v1/models` identity, unchanged checkpoint shards,
and the single GPU's UUID/name/driver plus activation-time clocks, power, and
thermal state. Only then does it write
`<log>.receipt.json` as `ACTIVE_VERIFIED`; a mismatch kills the child it just
started and fails closed. The active checks are CPU-testable and do not claim
which speculative engine served an individual request; that evidence is carried
separately in each response's `usage.atlas_engine` counters.
The HTTP identity probe rejects redirects and requires the serving PID to own
the exact requested literal IPv4 or IPv6 loopback listener, not merely the same
port.

Both attention flags are required TOGETHER (`RELEASE_BF16` alone aborts with
"requires successful NVFP4 transcodes"). The pair is quality-positive and
frees ~8 GiB. Boot takes ~6 minutes (weight load + NVFP4 transcode); the launcher
returns only after its active receipt is verified rather than using a fixed sleep.

Flags that select the optimizations under test (all DEFAULT ON — listed so a
bisect can turn them off one at a time):

| flag | `=0` restores | landed in |
|---|---|---|
| `ATLAS_V4_PREFILL_TC` | scalar prefill attention | `6d8216a3`, `889abbfe` |
| `ATLAS_V4_COMP_GEMM_TC` | scalar compressor GEMMs | `1b4bd7f2`, `0e368e6f` |
| `ATLAS_V4_KV_PIPELINED` | scalar kv_proj GEMM | `415734fa` |
| `ATLAS_V4_WOA_INPLACE` | wo_a gather/scatter | `af92018a` |
| `ATLAS_HC_TILED` | one-block-per-token `hc_pre` | `17481eaa` |
| `ATLAS_VERIFY_EXACT_GEMV` | drifting batched verify GEMVs | `090830de` |

Experimental opt-ins remain default-off. The exact max profile admits only
the native winners below and forces rejected selectors to zero:

| flag | effect | required promotion evidence |
|---|---|---|
| `ATLAS_V4_PREFILL_TC2_WARP0=1` | beneath TC2 only, have warp 0 compute and broadcast the exact-N=2,410 tile softmax instead of repeating it in four warps | exact CSA/HCA/dense output bytes and malformed no-write canaries, GB10 occupancy plus same-boot ABBA timing against TC2, output/cache hashes, and five-run zero-cache TTFT |
| `ATLAS_EXL3_PREFILL_N256=1` | widen the exact K64/K2 routed grouped-prefill strip from N64/N128 to N256 | N64/N128/N256 byte parity at both production shapes and M boundaries, wrong-block canaries, SASS resource gate, GB10 occupancy query, same-boot CUDA timing, five-run TTFT |
| `ATLAS_EXL3_PREFILL_W2A8=1` | use the model-specific five-launch routed W2A8 core for the exact DeepSeek K2/top-6/TP1/no-EP shape | native conversion/fragment dumps, both production-shape projection and final-tail parity, malformed-launch canaries, real checkpoint routing histogram, GB10 occupancy, same-boot timing, quality/output hashes, and at least 20 fresh zero-cache TTFT samples per arm |
| `ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN=1` | beneath the base W2A8 gate only, replace its separate gate, up, and post/SwiGLU/down-A8 launches with the exact fused N128 producer | base W2A8 qualification first; byte-exact A8 and FP32 scales for all 14,460 rows/256 experts, route/guard/tail/malformed coverage, GB10 occupancy and same-boot ABBA timing against the three-launch W2A8 incumbent with the checkpoint's real histogram, then active-receipt quality/output hashes and at least 20 fresh zero-cache TTFT samples per arm |
| `ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256=1` | beneath the fused N128 W2A8 gate only, widen its M32 gate/up producer from N128 to N256 | exact N128/N256 FP8 and scale bytes at the 14,460-row route and M32 boundaries, malformed no-write canaries, GB10 occupancy and same-boot ABBA timing, then composed output hashes and TTFT |
| `ATLAS_EXL3_PREFILL_W2A8_N256_DOWN=1` | beneath the base W2A8 gate only, replace its final N64 down projection with the exact 512-thread N256 down projection | base W2A8 qualification first; exact N64/N256 down bytes at the production shape and M boundaries, wrong-block no-write and guard coverage, GB10 occupancy and same-boot timing on the checkpoint routing histogram, then active-receipt quality/output hashes and at least 20 fresh zero-cache TTFT samples per arm |
| `ATLAS_EXL3_SHARED_PREFILL_FP8=1` | predequant only the EXL3 shared expert to persistent FP8 and keep BF16 input/one GEMM launch | +1.008-GiB residency, per-shape cosine/logit parity, full output hashes, tool/prose quality, five-run TTFT |
| `ATLAS_V4_PREFILL_QB_ROPE_FUSED=1` | replace exact-N=2,410 q_b normalization plus forward Q/K RoPE with the scale-independent fused kernel, then re-extract only K's rotated tail for the incumbent cache assembler | exact full-buffer Q/K bytes, malformed no-write canaries, retained calibration/cache/output hashes, GB10 occupancy and same-boot timing, and five-run zero-cache TTFT |
| `ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED=1` | jointly dispatch registered q_b norm/forward-RoPE and strided-K FP8-cache kernels only at the exact 2,410-token V4 shape | a scale-aware path or truly loaded static per-layer scales first (the current K2 checkpoint has neither); then native module parity for Q/K/both cache pools, 2,409/2,411 and every dtype/diagnostic/graph/profile/calibration fallback, output hashes, five-run zero-cache TTFT |
| `ATLAS_V4_PREFILL_KV_ALIAS=1` | skip the dead V4 `k_out`-to-`v_out` prefill scratch copy | exact-shape source contract, gate-off/on output and cache parity, five-run zero-cache TTFT; keep separate from the joint-fusion result |
| `ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED=1` | replace inverse RoPE extract/rotate/writeback with one exact-shape in-place BF16 kernel | inverse-only exact-byte runner, malformed no-write canaries, retained-BF16 cuBLASLt output/cache hashes, and same-boot TTFT |
| `ATLAS_V4_PREFILL_HC_RMS_FUSED=1` | fuse exact-N=2,410 `hc_pre_finish` with the following DeepSeek-vanilla RMSNorm at both attention and FFN sites while retaining materialized BF16 `hidden` | native byte-exact `hidden`/`normed`/`post`/`comb` parity, malformed no-write canaries, SM121a resource gate, same-boot ABBA timing, both engagement receipts, output/cache hashes, and five-run zero-cache TTFT |

The shared-expert FP8 arm changes the baseline BF16-MMA numerics. Its offline
allocation, dispatch, and SASS gates make it safe to test but do not qualify it
for release or allow a throughput claim.

The routed W2A8 candidate is a default-off, model-specific serving opt-in. It
requires exact bits-2 gate/up `(2048,4096)` and down `(4096,2048)`, top-6,
TP1 with no communicator or EP, direct persistent fixed-K2/fixed-shape prefill,
dual-pre and fused-post, no incompatible M128/N128/N256 selector, no graph
capture, all four handles, and checked arena capacity. Any W2A8 mismatch keeps
the incumbent path before mutation. Once the five-launch core starts, errors
propagate; its raw down BF16 result receives the incumbent H128/SVH post unless
the existing fused post-unpermute tail owns that transform.

The optional refinements are also strict default-off selectors. The first
fused selector replaces the base arm's three middle launches with one exact
N128 gate/up-to-down-A8 producer; its subordinate N256 selector widens that
same one-launch producer while falling back to N128 on any miss. The separate
N256-down selector replaces only the final N64 down launch. Only the exact
string `1` enables a selector, all remain subordinate to the admitted base
W2A8 arm, and a missing handle or eligibility failure falls back before the
corresponding mutation.

### Current native-gated EXL3-K2 plain-prefill result

The historical 1,062 tok/s result used the 144-expert FP8/NVFP4 checkpoint,
whereas the routed W2A8 chain requires the 256-expert EXL3 K2 checkpoint. The
following is therefore a separate model/profile frontier, not an extension of
the historical measurement and not a 2,000 tok/s claim:

```bash
unset GAMMA DSPARK_TOKENS DRAFTER
unset ATLAS_PROFILE ATLAS_DIAG_V4_ALL_LAYERS ATLAS_OP_DUMP
unset ATLAS_DUMP_EXPERT_IDS DFLASH_TRAIN_DUMP
unset ATLAS_EXL3_SHARED_PREFILL_FP8 ATLAS_V4_PROJ_FP8MMA
unset ATLAS_FP8_KV_EMA_RECAL ATLAS_FP8_KV_HEADROOM
unset ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED
unset ATLAS_DEBUG_SYNC_KERNELS ATLAS_PREFILL_HOST_TIMING ATLAS_V4_STAGE_SYNCS
unset ATLAS_MOE_PREFILL_ZERO
unset ATLAS_DUMP_EMBED CUDA_LAUNCH_BLOCKING
unset ATLAS_EXL3_FIXED_K2 ATLAS_EXL3_SPLIT ATLAS_EXL3_VERIFY_WORKLIST
unset ATLAS_EXL3_FUSED ATLAS_MAX_BATCH_TOKENS ATLAS_KV_EXTERNAL_RESERVE_GB
unset ATLAS_PEAK_MEM_MULT
unset ATLAS_PREFILL_MAX_REQUIRE_ARMS
unset FP8_KV_CALIBRATION_TOKENS

export MODEL=/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1
export MAX_SEQ_LEN=4096 MAX_PREFILL_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.96 OOM_GUARD=2048
export ATLAS_KV_OVERCOMMIT=0
export ATLAS_PREFILL_MAX_REQUIRE_ARMS=1
export ATLAS_V4_PREFILL_CUBLASLT=1
export ATLAS_V4_ATTN_RELEASE_BF16=0 ATLAS_V4_ATTN_NVFP4=0
export ATLAS_V4_PREFILL_TC=1 ATLAS_V4_PREFILL_TC2=1
export ATLAS_V4_PREFILL_TC2_WARP0=0
export ATLAS_V4_PREFILL_QB_ROPE_FUSED=0
export ATLAS_V4_COMP_GEMM_TC=1 ATLAS_V4_KV_PIPELINED=1
export ATLAS_V4_WOA_INPLACE=1 ATLAS_HC_TILED=1
export ATLAS_V4_PREFILL_KV_ALIAS=1
export ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED=1
export ATLAS_V4_PREFILL_HC_RMS_FUSED=1

export ATLAS_EXL3_PREFILL_DIRECT=1
export ATLAS_EXL3_PREFILL_PERSISTENT=1
export ATLAS_EXL3_PREFILL_FIXED_K2=1
export ATLAS_EXL3_PREFILL_FIXED_SHAPE=1
export ATLAS_EXL3_PREFILL_FUSED_POST=1
export ATLAS_EXL3_PREFILL_DUAL_PRE=1
export ATLAS_EXL3_PREFILL_W2A8=1
export ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN=1
export ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256=0
export ATLAS_EXL3_PREFILL_W2A8_N256_DOWN=1
export ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE=1
export ATLAS_EXL3_HROW_FIXED_SHAPE=1
export ATLAS_EXL3_PREFILL_FUSED_BLEND=0
export ATLAS_MOE_SHARED_K64=4

export ATLAS_EXL3_PREFILL_M128=0 ATLAS_EXL3_PREFILL_K64=0
export ATLAS_EXL3_PREFILL_N128=0 ATLAS_EXL3_PREFILL_N256=0

scripts/build-exl3-prefill-max.sh
scripts/exl3-prefill-max.sh
```

For the release-grade exact-shape measurement, use the one-command GB10
qualifier rather than the generic multi-length probe:

```bash
python3 scripts/qualify-exl3-prefill-max.py \
  --output-dir "/tmp/atlas-exl3-prefill-max-$(date -u +%Y%m%dT%H%M%SZ)"
```

The preceding 2026-08-27 GB10 record enables packed half2 E4M3 production in the
fused-GU N128 and N256-down K2 W2A8 kernels. Before endpoint promotion, a
frozen same-source ABBA runner required exact intermediate FP8/scales, raw and
final BF16, production aliases, immutable inputs, and clean redzones. It
measured incumbent composed times **23.1101923/23.1488485 ms** and candidate
times **22.9780083/22.0857124 ms**, a **1.026525x** direct-chain speedup. SASS
fell from 2,760 to 2,664 instructions in fused-GU N128 and from 904 to 856 in
N256 down, with unchanged register/shared-memory counts and no stack, local
memory, or spills. The exhaustive CPU oracle checks all 65,536 legal K2
windows byte-exactly, including signed zero, subnormal, overflow, and 1,131 RNE
tie cases.
The frozen pair identity, source hashes, compile commands, binary/SASS hashes,
resource census, and exact 1.01 admission threshold are retained in the
[`native build receipt`](probes/exl3-prefill-packed-e4m3-20260827-native-build-receipt.txt).

The promoted exact-shape qualifier then completed one warmup plus 20 measured
requests at exactly N=2,410 with zero cached tokens. It classified
`ACTIVE_VERIFIED`, with median TTFT **2.912403 s** and median prefill
**827.495410 tok/s** (range **635.804702--838.434061 tok/s**). This is
**+1.4398%** over the preceding 815.750465 tok/s EXL3-K2 median. All six
required optional-arm receipts engaged, the source-bound binary and receipts
were unchanged after measurement, and the receipted server stopped cleanly.
`target_2000_tok_s_met` remains false.

The promoted raw samples and output hashes are retained in
[`exl3-prefill-packed-e4m3-20260827-qualification.json`](probes/exl3-prefill-packed-e4m3-20260827-qualification.json),
with the matching
[`active receipt`](probes/exl3-prefill-packed-e4m3-20260827-active-receipt.json)
and
[`source-bound build receipt`](probes/exl3-prefill-packed-e4m3-20260827-build-receipt.txt).
The preceding 815.750465 tok/s qualification and its receipts remain preserved
under the original `exl3-prefill-max-20260827-*` names. These results are a
separate 256-expert EXL3-K2 frontier; neither may be called a regression from
the historical 1,062 tok/s result, which used a different 144-expert
FP8/NVFP4 checkpoint and predates the current receipt standard.

A subsequent default-off experiment replaced N256-down's 16 redundant
per-warp route scans with one warp plus a block barrier. Native parity passed,
but selector-only ABBA timing measured incumbent means **22.71709635 ms** and
candidate means **22.72614005 ms**, or **0.999602x**, below the required 1.01
promotion floor. It remains disabled and receives no endpoint run. The
[`rejection result`](probes/exl3-n256-route-guard-20260827-native-result.json)
and
[`native build receipt`](probes/exl3-n256-route-guard-20260827-native-build-receipt.txt)
are retained to prevent repeating the neutral design.

The preceding 2026-08-28 GB10 record additionally promotes K64 `cp.async`
double buffering in the fused-GU N128 kernel. Its frozen one-factor native
ABBA gate retained exact intermediate FP8/scales, raw/final BF16, production
aliases and hashes, clean redzones, immutable inputs, and 16 malformed-route
cases. Incumbent composed times were **23.1274643/23.3668232 ms** versus
candidate **20.0922079/19.8693762 ms**, a **1.163475x** speedup. Registers
remain 128, shared memory rises from 40,960 to 48,128 bytes, and the candidate
has zero stack, local memory, spills, or atomics. The production wrapper
unconditionally normalizes both the packed-E4M3 and double-buffer selectors to
one; a hostile compile proves command-line zero definitions cannot demote the
released kernel. The experimental component still defaults both selectors to
zero, and the post-promotion AB builder uses a generated experiment-only
wrapper so it can continue to construct a real incumbent.

The fresh exact-shape endpoint qualifier classified `ACTIVE_VERIFIED` after
one warmup and 20 measured requests at exactly N=2,410 with zero cached tokens.
Median prefill is **848.973992 tok/s**, median TTFT is **2.838731 s**, and the
range is **724.603987--877.236679 tok/s**. This is **+21.478582 tok/s** or
**+2.5956%** over the preceding 827.495410 median, while median TTFT improves
by 0.073672 s. All six required arms engaged exactly once, source/binary and
receipts remained unchanged, and the server stopped cleanly.
`target_2000_tok_s_met` remains false. Retained evidence is the
[`qualification`](probes/exl3-prefill-double-buffer-20260828-qualification.json),
[`active receipt`](probes/exl3-prefill-double-buffer-20260828-active-receipt.json),
[`source-bound build receipt`](probes/exl3-prefill-double-buffer-20260828-build-receipt.txt),
[`native result`](probes/exl3-fused-gu-n128-double-buffer-20260827-native-result.txt),
and [`native build receipt`](probes/exl3-fused-gu-n128-double-buffer-20260827-native-build-receipt.txt).

The current 2026-08-28 GB10 record also promotes K64 `cp.async` double
buffering in the exact N256-down projection. Its frozen one-factor native ABBA
gate retained exact intermediate FP8/scales, raw/final BF16, production
aliases and hashes, clean redzones, immutable inputs, every route offset, and
16 malformed-route no-write cases. Incumbent times were
**20.1511278/19.6476240 ms** versus candidate
**18.4733438/18.3480082 ms**, a **1.080861x** speedup. Only the N256-down
cubin changed: registers rose from 97 to 102, shared memory from 10,240 to
19,456 bytes, and the candidate retained zero stack, local memory, spills, and
atomics. The production wrapper forces the selector to one even under a
hostile command-line zero definition; the reusable component defaults it to
zero, and the rejected single-warp route guard remains zero.

The fresh exact-shape qualifier classified `ACTIVE_VERIFIED` after one
warmup and 20 measured requests at exactly N=2,410 with zero cached tokens.
Median prefill is **884.612726 tok/s**, median TTFT is **2.724358 s**, and the
range is **759.579927--894.451958 tok/s**. This is
**+35.638734 tok/s** or **+4.1979%** over the preceding 848.973992 median,
while median TTFT improves by **0.114373 s** or **4.0290%**. All six required
arms engaged exactly once, source/binary and receipts remained unchanged, and
the server stopped cleanly. `target_2000_tok_s_met` remains false; the
remaining exact-model gap is **1,115.387274 tok/s**. Retained evidence is the
[`qualification`](probes/exl3-prefill-n256-double-buffer-20260828-qualification.json),
[`active receipt`](probes/exl3-prefill-n256-double-buffer-20260828-active-receipt.json),
[`planned receipt`](probes/exl3-prefill-n256-double-buffer-20260828-planned-receipt.json),
[`source-bound build receipt`](probes/exl3-prefill-n256-double-buffer-20260828-build-receipt.txt),
[`native result`](probes/exl3-n256-down-double-buffer-20260828-native-result.txt),
and [`native build receipt`](probes/exl3-n256-down-double-buffer-20260828-native-build-receipt.txt).

The qualifier derives the `ATLAS_*` process environment from
`exl3-prefill-max.sh`, rejects any additional inherited Atlas experiment,
launches that wrapper, and binds the exact source tree, binary, checkpoint,
argv, full process environment, boot, and GPU to an active benchmark receipt
before its first inference request. The max-build receipt is hashed before
launch and must remain byte-identical after all samples. It uses the live
`/tokenize` endpoint to
build a unique nonce-bearing token-ID vector and submits that vector through
`/v1/completions`, so every warmup and measured request is exactly N=2,410
without relying on the different chat-template count. The final usage must
report exactly 2,410 prompt tokens and zero cached tokens. One warmup and 20
measured requests are retained, and the command then requires the exact-shape
W2A8 core/tail plus V4 HC-finish/RMS at both attention and FFN sites,
K/V-alias, and inverse-RoPE engagement log
records. It writes bounded `qualification.json` plus the server log and
planned/active receipts in a new, non-overwritten output directory, then stops
the receipted PID. Each raw sample includes prompt and streamed-output hashes,
one validated terminal `finish_reason`, and its completion-token count;
repeated usage, ambiguous terminal ordering, or content after terminal is
rejected. `target_2000_tok_s_met=true` means both a median of at least
2,000 tok/s and median TTFT at or below 1.205 s; it does not replace the quality
gate or establish a result unless the command succeeds on GB10 with unchanged
receipts and required engagement records.

Build the binary first with the exact DeepSeek target from §1. The launcher's
wrapper first runs `check-exl3-prefill-max-model.py`, verifies the source-bound
build receipt, then passes and logs this exact profile; any trailing
`NAME=VALUE` argument is an explicit, visibly last A/B override.
`ATLAS_PREFILL_MAX_REQUIRE_ARMS=1` turns silent exact-shape fallback into a
qualification failure for W2A8, fused-GU N128, N256-down, fused unpermute,
HC-finish/RMS, K/V alias, and inverse RoPE. The native-losing fused-GU N256,
fused blend, TC2 warp-0, and q_b/forward-RoPE selectors must be literal zero. A valid
run must also contain the emitted
`ATLAS_PREFILL_MAX_ARMS_RECEIPT` and `V4_PREFILL_MAX_ARM_ENGAGED` records; the
flag remains default-off outside this qualification wrapper. The explicit
4,096-token cap keeps
the N=2,410 qualification prompt in one initial prefill pass; do not infer a
fixed chunk count from `exl3-serve.sh`'s 1,024-token default because scheduler
buffer state and the MLA correctness gate also affect continuation chunks.
Keeping `ATLAS_V4_ATTN_RELEASE_BF16=0` preserves the cuBLASLt projection arm
and its roughly 8.06-GiB BF16 mirrors. Keep GAMMA/DSpark/DFlash unset: the
historical full-drafter plus those mirrors did not fit one GB10. The W2A8 and
tail selectors above still require native parity, same-boot timing, output
hashes, and the quality gate before any result may replace the historical
frontier. Keep `FP8_KV_CALIBRATION_TOKENS` unset: this K2 checkpoint has zero
`*.k_scale` and zero `*.v_scale` tensors, so its MODEL.toml 256-token online
calibration is the only admitted FP8 scale source. The server now rejects FP8
KV when both online calibration and checkpoint scales are absent. The joint
q_b/RoPE/cache arm remains excluded from this checkpoint's max profile because
it requires frozen scales; its offline probe is evidence for a future
scale-aware/static-scale path, not permission to force calibration to zero.
The separate `ATLAS_V4_PREFILL_QB_ROPE_FUSED=1` arm has no cache-scale,
cache-dtype, or calibration dependency. It retains the incumbent cache writer
and re-extracts only K's rotated 64-wide tail; diagnostics, graph capture,
profiling, non-V4 models, and every non-exact shape fall back before mutation.
The q_b/forward-RoPE, K/V-alias, and inverse-RoPE arms remain exact-shape and
strict-default-off. The 2026-08-27 native gate rejected q_b/forward-RoPE even
at a diagnostic 1.00000012 speedup floor, so the max profile leaves it off.
The fixed immutable inverse-only runner passed exact parity and measured
0.471931 ms incumbent versus 0.179099 ms candidate, or 2.635032x, at its
precommitted 1.01 floor.
K/V alias remains enabled and is checked again by the end-to-end output/cache
gate.

The HC-finish/RMS arm is likewise strict default-off and exact to DeepSeek-V4,
N=2,410, H=4,096, HC=4, 43 layers, 20 Sinkhorn iterations, FP8 KV serving,
vanilla checkpoint norm semantics, TP1/EP1, and the unprofiled, non-diagnostic,
non-graph, single-sequence path. It checks every pointer, range, arena extent,
and alias before `hc_expand`; strict qualification fails before any layer
mutation on a mismatch. Successful candidate launches emit separate
attention/FFN engagement receipts. The fallback retains the incumbent
`hc_pre_mix_tiled` → `hc_pre_finish` → `rms_norm_vanilla` ordering.

The TC2 warp-0 arm is also now production-reachable only at the exact DeepSeek
V4 `(N=2410,nq=64,nkv=1,hd=512)` shape, with TC2 and then TC as unchanged
fallbacks. Its model eliminates 95.579 GB of gross replicated `sSp` reads
before charging the new warp-0 broadcast traffic, and avoids 7.467 billion
redundant exponential operations across the prefill. Those are source-topology
counts—not net shared-memory traffic, measured DRAM traffic, elapsed time, or
tok/s. Its 2026-08-27 immutable native probe missed the pre-registered 1.01
admission threshold, so the max profile retains TC2 and disables warp-0.

V4 compressed-pool persistence is now scale-consistent with that online path:
the raw KV write performs the first observation and freezes the scale before
the pool is quantized, and both prefill and decode use the same scale-aware FP8
conversion. The pool count is published only after that conversion is queued.
This repairs the prior provisional-scale mismatch; it is not a throughput
measurement and still requires native output/cache parity.

The build helper compiles the exact DeepSeek target and writes a sidecar binding
the release-binary SHA-256 to the current commit plus all dirty/untracked Cargo,
crate, and kernel build inputs. The max launcher refuses a missing, stale,
replaced, symlinked, or wrong-semantics binary instead of silently launching a
different kernel set.

Offline CPU fragment/numeric/dispatch contracts and SM121a resource
compilation are reproducible with:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_w2a8_numeric_model --test exl3_w2a8_component_model \
  --test exl3_w2a8_dispatch_model --test exl3_w2a8_emitter_probe_model \
  --test exl3_w2a8_fused_gu_down_integration_model \
  --test exl3_w2a8_fused_gu_down_emit_n256_model \
  --test exl3_w2a8_fused_gu_down_n256_integration_model \
  --test exl3_w2a8_n256_down_dispatch_model \
  --test exl3_w2a8_composed_probe_model
bash scripts/check-exl3-prefill-w2a8-sass.sh
bash scripts/check-exl3-prefill-w2a8-n128-sass.sh
bash scripts/check-exl3-prefill-w2a8-n256-sass.sh
bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-sass.sh
bash scripts/check-exl3-prefill-w2a8-fused-gu-n256-sass.sh
bash scripts/check-exl3-prefill-w2a8-probe-build.sh
bash scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh
bash scripts/check-exl3-prefill-w2a8-composed-probe-build.sh 1.01
```

Passing those commands does not qualify output numerics or throughput. Native
conversion/fragment dumps, projection parity, quality, occupancy, and
same-boot timing are still required before promotion or a default change.
The same SASS command also compiles the isolated H128-to-A8 producer emitters.
They preserve the incumbent BF16 boundaries and standalone K128 reduction
topology while avoiding a global BF16 sidecar; the current static resource
ceilings are 36/40 registers, 5,120 B shared, and zero spills for dual-H4096
and down-H2048 respectively. DeepSeek-only wrappers expose them to the exact
serving opt-in; this static evidence does not qualify the runtime path.
The W2A8 N256 experiment doubles N128 to a 512-thread, 16-warp M64xN256 CTA.
Both exact GU/down instantiations compile for SM121a at 97 registers/thread,
53,248 allocation-granularity-rounded registers/block, 10,240 B cuobjdump
shared memory, 16 static E4M3 QMMA sites, and zero stack/local/spills. A
DeepSeek-only production wrapper and registry handle now make the down shape
reachable only under `ATLAS_EXL3_PREFILL_W2A8=1` plus the strict subordinate
`ATLAS_EXL3_PREFILL_W2A8_N256_DOWN=1`. Relative to the base arm's N64 down
launch, its N256 strip removes 12,288 logical CTAs and 1,421,475,840 logical A8
activation-read bytes per layer: 528,384 CTAs and 61,123,461,120 bytes
(56.92 GiB) across 43 layers. This is source-shape arithmetic, not measured
traffic or time; the N256-down path subsequently passed as part of the exact
composed native runner described below.
Every warp now scans the complete 257-entry routing vector before any trellis
resolution or output store, so a malformed late offset cannot leave partial
output from earlier expert CTAs.

The fused-N128 experiment follows the gate/up-to-down-A8
mega-fusion concept from Entrpi/ds4 commit `da027a1`, adapted to Atlas's EXL3
trellis and K128-scaled E4M3 contracts. One 256-thread CTA computes an exact
N128 gate tile and up tile sequentially, retains their BF16 boundaries, and
emits the post-H128/SwiGLU/down-pre result as A8 plus FP32 scale; it does not
run the down GEMM. The SM121a static gate reports 3,376 instructions, 128
registers/thread, 40,960 B shared memory, and zero stack, local memory, spills,
or atomics. Its opcode
census is 32 E4M3 QMMA, 52 BF16 conversions, 36 saturating E4M3 conversions,
4 `MUFU.EX2`, 166 FFMA, 193 shuffles, and 7 `BAR.SYNC` sites. The CPU/source
model removes two launches and 236,912,640 logical BF16 bytes per layer at
2,410 tokens/top-6: 86 launches and 10,187,243,520 bytes (9.488 GiB) across
43 layers. Those counts are structural, not measured traffic or time. The
contract and SASS/resource gates are:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_w2a8_fused_gu_down_emit_n128_model
bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-sass.sh
bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh 1.01
```

The last command builds one receipted SM121a executable containing the fused
candidate and its exact three-launch N128 gate/up/emitter incumbent. It rejects
non-finite, out-of-range, or implicit speed thresholds before CUDA use and extracts
the candidate, grouped-N128, and emitter cubins. To retain an immutable runner
for GB10, choose a new output path and pre-register the one timing threshold:

```bash
W2A8_FUSED_GU_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-fused-gu-probe \
  bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh 1.01
/tmp/atlas-w2a8-fused-gu-probe/run-fused-gu-n128-probe.sh
```

The generated runner verifies its receipt, executable, embedded build ID, and
all extracted cubins before it tests exact FP8 and FP32-scale bytes across all
14,460 production rows and 256 experts, validates the complete 257-entry
expert-offset route before any write, covers routing boundaries and an empty
expert, swaps opposite output poisons and guards, and exercises 20 malformed
geometries. Its ABBA timing uses a
balanced synthetic 256-expert histogram, not a measured serving histogram. The
example speedup threshold is an admission input, not an observed result. A
DeepSeek-only wrapper, registry handle, and prevalidated serving dispatch now
make this candidate reachable only under `ATLAS_EXL3_PREFILL_W2A8=1` plus the
strict subordinate `ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN=1`; the three-launch
middle-stage incumbent remains its fallback. The standalone runner remains
unexecuted, while the composed N128-plus-N256-down runner below passed native
parity and timing. Neither establishes end-to-end throughput by itself.

The composed promotion runner measures the complete production-shape core,
not either candidate kernel in isolation. Its incumbent is the base W2A8
five-launch chain; its candidate is the shared pre-emitter, fused N128
gate/up-to-down-A8 stage, and N256 down projection (three launches). To retain
the exact current kernel/serving sources, tools, binary, seven extracted
cubins, canonical binary64 threshold, repository commit, and status shape for
a GB10 run:

```bash
W2A8_COMPOSED_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-composed-probe \
  bash scripts/check-exl3-prefill-w2a8-composed-probe-build.sh 1.01
/tmp/atlas-w2a8-composed-probe/run-composed-probe.sh

W2A8_FUSED_GU_N256_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-fused-gu-n256 \
  bash scripts/check-exl3-prefill-w2a8-fused-gu-n256-probe-build.sh 1.01
/tmp/atlas-w2a8-fused-gu-n256/run-fused-gu-n256-probe.sh
```

Record the builder's `runner_sha256` outside the output directory and verify it
with an independent `sha256sum` before execution. The runner necessarily cannot
make its own mutable file trustworthy; the builder's printed hash is an external
handoff value, while the runner's embedded checks bind the receipt, binary,
build ID, threshold, and cubins. Every direct CUDA input plus the W2A8 state,
dispatch, caller, phase, and post-tail Rust sources is content-hashed. The separately labeled
`git_status_shape_sha256` detects added/removed status entries but is not a
dirty-content hash.

The generated runner takes no arguments. It validates exact intermediate FP8
data/scales, raw down BF16, and final H128-transformed BF16 under balanced,
empty-expert, skewed, and explicit 63/64/65/127/128/129 M64-boundary routes at
the production 14,460-row total; protects immutable inputs and redzones under
swapped poisons; requires first- and late-offset malformed launches to leave
output untouched; and byte-compares both chains through the production's three
destructively aliased arena buffers. Whole-chain ABBA CUDA-event timing uses
that same alias schedule. The native result is recorded with the inventory
below.

Under the source access model, the fused stage removes 236,912,640 logical
BF16 bytes per layer, its N128 gate/up strip removes 1,895,301,120 logical A8
re-read bytes relative to N64, and N256 down removes another 1,421,475,840.
The N128-composed total is 3,553,689,600 bytes per layer and 152,808,652,800
bytes (142.314 GiB) plus 86 launches across 43 layers. Its 2026-08-27 immutable
GB10 runner passed exact intermediate/final parity and measured 49.275759 ms
for the five-launch incumbent versus 47.531616 ms for fused N128 plus N256
down, or 1.036694x, at the precommitted 1.01 floor. The separate fused-GU
N128-to-N256 probe missed that floor, so the max profile does not add its
otherwise-modeled 977,264,640 bytes per layer. Logical source accesses are not
measured DRAM traffic or saved end-to-end time.

Future GB10 promotion of either subordinate gate requires all of the following
without borrowing evidence between the two candidates:

1. Qualify the base W2A8 arm first, including native emitter and projection
   parity, final-tail parity, malformed-launch no-write canaries, checked arena
   bounds, native cubin resource checks, and a GB10 occupancy query.
2. For fused GU/down-A8, use the receipted harness to byte-compare FP8 data and
   FP32 scales against the exact three-launch incumbent over all 14,460 rows,
   all 256 experts, the complete 257-entry route, empty/boundary experts,
   opposite poisons, guards, tails, and every malformed geometry. Repeat its
   same-process ABBA timing with the checkpoint's real routing histogram and a
   precommitted speedup floor of at least 1.01.
3. For N256 down, byte-compare the production N64 and N256 down BF16 outputs
   with immutable inputs at the exact `(N=4096,K=2048)` shape, the production
   14,460 expanded rows, all covered M boundaries, guards, tails, and the wrong
   512-thread/grid cases that must leave poison untouched. Record actual GB10
   occupancy and same-process timing with the checkpoint's real histogram.
4. Run separate same-boot end-to-end A/B cohorts with only the candidate's
   subordinate flag changed and the base W2A8 flag on in both arms. Preserve an
   `ACTIVE_VERIFIED` receipt and every raw `RESULT_JSON`, require at least 20
   post-warmup requests per arm at the same API-reported prompt count with
   `cached_tokens == 0`, and rerun the 90/100 zero-failure quality gate plus
   deterministic output hashes. If both selectors are proposed together, test
   all four off/fused/N256/both combinations; neither individual result proves
   the composition.

The V4 prefill RoPE experiments apply the existing interleaved YaRN formulas
directly to the trailing dimensions 448..511 of resident `[N,64,512]`
Q/attention-output and `[N,1,512]` K tensors. The forward arm remains isolated
and unregistered; the inverse-only arm is production-reachable behind the
strict exact-shape `ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED=1` flag. At N=2,410,
replacing both the forward
extract-Q/extract-K/RoPE/writeback-Q/writeback-K chain and the inverse
extract/RoPE/writeback chain with two direct kernels removes six launches and
159,175,680 logical BF16 bytes per layer: 258 launches and 6,844,554,240 bytes
across 43 layers. Those are source-shape counts, not measured traffic or time.
The CUDA 13 SM121a gate rejects injected nvcc flags and pins both kernels to 24
registers, zero shared/stack/local/spills/atomics, and exact SASS topology:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test v4_prefill_rope_fused_model
bash scripts/check-v4-prefill-rope-fused-sass.sh
```

The combined forward-plus-inverse component still needs native byte-parity and
isolated timing, followed by an end-to-end paired GB10 run, before any serving
or throughput claim. The max profile uses only the production inverse arm,
whose exact standalone budget is 78,970,880 bytes and two launches per layer,
or 3,395,747,840 bytes (3.163 GiB) and 86 launches across 43 layers. A receipted
native harness is available; choose the minimum total forward-plus-inverse
speedup before building it:

```bash
V4_ROPE_PROBE_OUTPUT_DIR=/tmp/atlas-v4-rope-probe \
  bash scripts/check-v4-prefill-rope-fused-probe-build.sh 1.01
/tmp/atlas-v4-rope-probe/run-v4-prefill-rope-fused-probe.sh
```

The generated runner revalidates its receipt, executable, embedded build ID,
and extracted cubins, then requires exact full-buffer Q/K/output bytes across
three position/frequency/mscale cases, 31 pointer/shape/block/grid no-write
canaries under full-buffer poison, and ABBA CUDA-event timing at N=2,410. The
example threshold is an admission input, not
an observed result. Only the offline build has run here; native parity and
speed remain unqualified.

### Settled offline V4 prefill fusion set

The fusion components are now source- and probe-locked: pure q_b RMSNorm plus
direct forward RoPE; vanilla q_a RMSNorm
plus W8A8 activation quantization, with a second probe through the real
downstream `w8a8_gemm_pipelined` path; contiguous latent+RoPE cache assembly
directly into FP8 K/V pages; its strided-K-full composition sibling; direct
inverse RoPE; and inverse RoPE plus W8A8 activation quantization. The q_b and
strided-K modules share one strict default-off production gate, while direct
inverse RoPE has its own exact-shape gate. The remaining fusion kernels stay
unregistered. The FP8 cache candidates apply only to the FP8-cache path. The
W8A8 candidates apply only to the released-BF16,
FP8-native projection arm (`ATLAS_V4_ATTN_RELEASE_BF16=1` and
`ATLAS_V4_PROJ_FP8MMA=1`) when its existing eligibility accepts the exact shape
and buffers. Forward q_b and inverse-RoPE fusion must bypass to the incumbent
materialized-BF16 chain when diagnostics are active, so `diag_this` observes
the same boundary.

The separate host-only `ATLAS_V4_PREFILL_KV_ALIAS=1` experiment is also
default-off. It skips only the dead `k_out`-to-`v_out` device copy at the exact
single-KV-head 512-dimension V4 shape; diagnostics, graph capture, and every
other shape retain the incumbent copy. Its source/traffic contract is CPU-only:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test v4_prefill_kv_alias_model
```

That gate does not prove that enabling the alias improves GB10 TTFT.

Run the complete CPU/source and SM121a static gate from the repository root:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test v4_prefill_attn_tc2_warp0_model \
  --test v4_prefill_attn_tc2_warp0_probe_model \
  --test v4_prefill_attn_tc2_warp0_integration_model \
  --test v4_prefill_qb_norm_rope_fused_model \
  --test v4_prefill_qb_norm_rope_fused_probe_model \
  --test v4_prefill_qb_rope_integration_model \
  --test v4_prefill_qa_norm_w8a8_quant_fused_model \
  --test v4_prefill_qa_norm_w8a8_quant_fused_probe_model \
  --test v4_prefill_qa_norm_w8a8_pipeline_probe_model \
  --test v4_prefill_cache_assemble_fp8_fused_model \
  --test v4_prefill_cache_assemble_fp8_fused_probe_model \
  --test v4_prefill_cache_kfull_fp8_fused_model \
  --test v4_prefill_cache_kfull_fp8_fused_probe_model \
  --test v4_prefill_inverse_rope_w8a8_quant_fused_model \
  --test v4_prefill_inverse_rope_w8a8_quant_fused_probe_model \
  --test v4_prefill_inverse_rope_integration_model \
  --test v4_prefill_rope_fused_inverse_probe_model \
  --test v4_prefill_qb_rope_cache_joint_integration_model \
  --test v4_prefill_qb_rope_cache_joint_probe_model \
  --test v4_prefill_fusion_budget_model
bash scripts/check-v4-prefill-attn-tc2-warp0-sass.sh
bash scripts/check-v4-prefill-qb-norm-rope-fused-sass.sh
bash scripts/check-v4-prefill-qa-norm-w8a8-quant-fused-sass.sh
bash scripts/check-v4-prefill-cache-assemble-fp8-fused-sass.sh
bash scripts/check-v4-prefill-cache-kfull-fp8-fused-sass.sh
bash scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-sass.sh
```

Build immutable promotion runners into fresh paths with the runbook's stricter
precommitted minimum speedup of `1.01`, then execute exactly those generated
runners on GB10:

```bash
V4_TC2_WARP0_PROBE_OUTPUT_DIR=/tmp/atlas-v4-tc2-warp0-probe \
  bash scripts/check-v4-prefill-attn-tc2-warp0-probe-build.sh 1.01
/tmp/atlas-v4-tc2-warp0-probe/run-v4-tc2-warp0-probe.sh

V4_QB_NORM_ROPE_PROBE_OUTPUT_DIR=/tmp/atlas-v4-qb-norm-rope-probe \
  bash scripts/check-v4-prefill-qb-norm-rope-fused-probe-build.sh 1.01
/tmp/atlas-v4-qb-norm-rope-probe/run-v4-prefill-qb-norm-rope-fused-probe.sh

V4_QB_ROPE_CACHE_JOINT_PROBE_OUTPUT_DIR=/tmp/atlas-v4-qb-rope-cache-joint \
  bash scripts/check-v4-prefill-qb-rope-cache-joint-probe-build.sh 1.01
/tmp/atlas-v4-qb-rope-cache-joint/run-v4-prefill-qb-rope-cache-joint-probe.sh

V4_QA_PROBE_OUTPUT_DIR=/tmp/atlas-v4-qa-norm-w8a8-probe \
  bash scripts/check-v4-prefill-qa-norm-w8a8-quant-fused-probe-build.sh 1.01
/tmp/atlas-v4-qa-norm-w8a8-probe/run-v4-prefill-qa-norm-w8a8-quant-fused-probe.sh

V4_QA_PIPELINE_PROBE_OUTPUT_DIR=/tmp/atlas-v4-qa-w8a8-pipeline-probe \
  bash scripts/check-v4-prefill-qa-norm-w8a8-pipeline-probe-build.sh 1.01
/tmp/atlas-v4-qa-w8a8-pipeline-probe/run-v4-prefill-qa-norm-w8a8-pipeline-probe.sh

V4_CACHE_FP8_PROBE_OUTPUT_DIR=/tmp/atlas-v4-cache-fp8-probe \
  bash scripts/check-v4-prefill-cache-assemble-fp8-fused-probe-build.sh 1.01
/tmp/atlas-v4-cache-fp8-probe/run-v4-prefill-cache-assemble-fp8-fused-probe.sh

V4_CACHE_KFULL_PROBE_OUTPUT_DIR=/tmp/atlas-v4-cache-kfull-probe \
  bash scripts/check-v4-prefill-cache-kfull-fp8-fused-probe-build.sh 1.01
/tmp/atlas-v4-cache-kfull-probe/run-v4-prefill-cache-kfull-fp8-fused-probe.sh

V4_INVERSE_ROPE_PROBE_OUTPUT_DIR=/tmp/atlas-v4-inverse-rope-probe \
  bash scripts/check-v4-prefill-rope-fused-inverse-probe-build.sh 1.01
/tmp/atlas-v4-inverse-rope-probe/run-v4-prefill-rope-fused-inverse-probe.sh

V4_INV_QUANT_PROBE_OUTPUT_DIR=/tmp/atlas-v4-inverse-quant-probe \
  bash scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-probe-build.sh 1.01
/tmp/atlas-v4-inverse-quant-probe/run-v4-prefill-inverse-rope-w8a8-quant-fused-probe.sh
```

Every output directory must be absent before its build. The joint runner times
the complete materialized incumbent q_b norm/extract/YaRN/writeback/cache chain
against the two production candidates together, while requiring exact full Q,
K, valid FP8 K/V cache bytes, holes, tails, guards, and immutable inputs under
swapped poisons. Promotion requires exact incumbent/candidate parity: BF16
output bytes where retained, FP8 bytes, and FP32 scale bits, including the full
downstream q_a pipeline output. There is no tolerance-based substitute. Most
optional runners in this matrix remain unexecuted; only the selected inverse
runner and the explicitly recorded rejection probes have native evidence.

At N=2,410, the isolated source-modeled inventories include 159,175,680 B for
combined direct forward+inverse RoPE, 197,427,200 B for q_b fusion, 14,807,040 B
for q_a fusion, 11,105,280 B for cache fusion, 59,228,160 B for adding W8A8
quantization to an already-direct inverse arm, and 4,935,680 B for K-to-V
aliasing per layer. They are not additive: several arms replace overlapping
materialization chains, and the strided-K-full cache variant is a composition
unlock whose avoided K-tail recreation is already absent from the direct-RoPE
budget. The current max profile enables inverse-RoPE and K-to-V aliasing from
this inventory, for 83,906,560 gross logical bytes per layer and 3,607,982,080
bytes (3.360 GiB) across 43 layers. Inverse RoPE removes 86 kernel launches per
pass; the alias additionally removes 43 D2D copy enqueues, not kernel launches.
The q_b arm's 197,427,200-byte-per-layer model and 172 launches are excluded
after its native timing rejection. The larger 446,679,040-byte-per-layer /
19,207,198,720-byte
portfolio total applies only to a conditional released-BF16, FP8-projection,
frozen/static-FP8-cache configuration. It is not the current max profile.
The scale-independent V4 set is disjoint from the routed W2A8 model above:
together with the HC-finish/RMS arm they account for 158,114,508,800 logical
bytes (147.256 GiB), 258 kernel-launch reductions, and 43 D2D-copy enqueue
reductions per pass. Those
are source-topology counts only; TC2 warp-0's shared-memory/exp reductions are
not added to this byte inventory.

The `hc_pre_finish` + vanilla RMSNorm production arm uses one exact
1,024-thread H=4,096/HC=4 CTA per token. Each thread owns the same two packed
BF16 words as DeepSeek-V4's vanilla RMS kernel (`x*rms*weight`); the HC collapse
is explicitly rounded to BF16, stored to the observable `hidden` output, and
widened again before the unchanged 32-warp XOR/warp-sum reduction.
At N=2,410, replacing finish plus RMS at both HC sites removes two launches,
39,485,440 logical BF16 reread bytes, and 77,120 CTAs per layer: 86 launches,
1,697,873,920 bytes (1.581 GiB), and 3,316,160 CTAs across 43 layers. The offline gate
reports 48 registers, 144 B ptxas static shared memory (1,168 B from
`cuobjdump`), and zero stack/local/spills/atomics:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test v4_hc_pre_finish_rms_fused_model
bash scripts/check-v4-hc-pre-finish-rms-fused-sass.sh
```

The arm is registered only in the DeepSeek-V4 target and selected only by
`ATLAS_V4_PREFILL_HC_RMS_FUSED=1`; every guard failure retains the incumbent
before mutation unless max qualification makes it an error. Build its immutable
SM121a runner with one pre-registered binary64 speedup threshold on the GB10
host, then run the receipt-bound probe with no arguments:

```bash
V4_HC_PROBE_OUTPUT_DIR=/tmp/atlas-v4-hc-fused-probe \
  bash scripts/check-v4-hc-pre-finish-rms-fused-probe-build.sh 1.01
/tmp/atlas-v4-hc-fused-probe/run-v4-hc-fused-probe.sh
```

The runner binds the exact incumbent, candidate, tools, sources, binary, and
cubins, plus the threshold's canonical decimal text and exact binary64 value.
It requires byte-exact full-buffer `hidden`, `normed`, `post`, and `comb` parity for four
deterministic Sinkhorn/epsilon cases under opposite poisons, checks ten
malformed geometries for no writes, and uses six-sample ABBA CUDA-event timing
at N=2,410. The threshold above is a build-time admission input, not a measured
result; the generated runner rejects arguments. The 2026-08-27 immutable GB10
run passed byte-exact parity and measured 1.105437 ms incumbent versus
0.910739 ms candidate, or 1.213781x. End-to-end qualification remains the
throughput gate.

The dedicated emitter build produces a receipted SM121a runner that compares
both fused emitters byte-for-byte against the incumbent BF16 H128 transforms
followed by `per_token_group_quant_fp8`. It covers gathered and identity
routing, row boundaries through 129, two output poisons, inactive tails, and
every geometry guard. Persist it for a GB10 run without overwriting evidence:

```bash
W2A8_EMITTER_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-emitter-probe \
  bash scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh
/tmp/atlas-w2a8-emitter-probe/run-emitter-probe.sh
```

The build alone is not native parity: only a successful generated-runner
execution on GB10 supplies the exact FP8-byte and FP32-scale comparison.
The build script accepts an explicit `W2A8_PROBE_OUTPUT_DIR` when the GU/down
N64/N128/N256 GB10 parity binaries and `build-receipt.txt` need to be retained. The output
directory must be empty, so reruns cannot overwrite evidence. The receipt binds
the source files, CUDA tools, compile flags, commit/status, exact host-binary
identity, build-script hash, exact per-shape commands, and stable extracted
cubins. Use the generated `run-{gu,down}-probe.sh` wrappers: they verify and
print the receipt/binary hashes, while each binary prints the matching embedded
build ID. Unreceipted `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS` are rejected.
The build revalidates source/tool/repository identity after all six
compilations before it writes the receipt or runners. Every binary requires
`<min_cosine> <max_abs_error> <min_end_to_end_speedup>` arguments; choose and
record them before inspecting the result rather than tuning them after the run.
The fail-closed admission ranges are cosine `0.99..1`, absolute error `(0..1]`,
and speedup `(1..100]`. The speedup denominator includes per-token/per-K128 A8
quantization plus W2A8, rather than timing the candidate kernel alone.

The existing N64/N128 pair runners remain available as a compatibility
admission path. The generated `run-{gu,down}-three-width-probe.sh` runners
execute receipted N64, N128, and N256 binaries for each GU/down shape with the
same explicit thresholds and immutable inputs, then require all nine hashes,
input/trellis identities, and raw BF16 dumps to match exactly across all three
widths. Building this runner is still only offline evidence: until it succeeds
on GB10, N256 has no native parity, timing, or throughput result.

## 3. Measure prefill

```bash
ATLAS_BENCH_RECEIPT=serve-<run-name>.log.receipt.json \
  python3 scripts/prefill_probe.py 8977      # TTFT at ~1k/2k/3k tokens
```

For each length the probe runs one warmup followed by five measured requests.
It uses a fresh nonce at the start of every request, requires the final
streaming usage chunk, and aborts unless
`prompt_tokens_details.cached_tokens == 0`. It reports both total and fresh
tokens and divides only verified fresh tokens by TTFT. The human summary is the
five-run median plus min/max range; the final `RESULT_JSON=` line contains the
warmup and every raw sample for each length. Missing cache accounting or first
reasoning/content timing is a failed benchmark, not zero. The active-receipt
digest and GPU snapshot are copied into that JSON and the run is classified
`ACTIVE_VERIFIED`; omitting the
receipt leaves the probe usable for diagnostics but visibly classifies it
`UNGRADED`. Passing the pre-launch `.planned.receipt.json` is visibly `PLANNED`
and is not a release-grade runtime result. Both prefill and decode probes
revalidate the same receipt, repository, process, executable, checkpoint, and
GPU identity after all warmups and measurements and before publishing their
final report; receipt v2 also binds and rechecks the canonical Linux boot ID.
Drift still present at that final boundary aborts instead of retaining
`ACTIVE_VERIFIED`. The boot ID proves only a same-boot cohort, not a shared
model-load or autotune instance; use repeated paired trials for promotion.
The historical 792/884/905 runs used
`scripts/dsflash-serve-bench.sh` while prefix caching defaulted off, so they are
not invalidated; their zero-cache provenance is inferred from that default
rather than backed by a retained usage receipt from this hardened probe. The
later `887a697c` cuBLASLt result used 20 measured requests per arm because the
observed within-serve TTFT spread was 32--34%; its 2.27 s / 1,062 tok/s median,
not either favorable five-run subset, is the historical comparison baseline.
Future prefill promotion cohorts must likewise retain at least 20 post-warmup
samples per arm (for example, four complete five-sample probe reports) and must
not select the fastest short cohort.

### 2,000 tok/s promotion line

At the historical N=2,410 geometry, 2,000 tok/s means median TTFT at or below
**1.205 s**. From the verified historical 2.27 s / 1,062 tok/s n=20 median that
requires removing **1.065 s (46.9%)**. The earlier 2.66 s / 905 tok/s point
remains a separate documented campaign rung, not the denominator for the
current target gap. Kernel launch counts, SASS resources, and logical traffic
savings are prioritization evidence only; promotion requires active receipts,
at least 20 fresh zero-cache samples per arm at the same API-reported prompt
count, median TTFT at or below 1.205 s, and the existing quality gates.

or, for the historical roughly 2,410-token geometry quoted in the campaign
doc, use the API-reported prompt count and take a five-run median (warm up with
one short request first — the first request after boot pays lazy-init):

```
prompt: "run=<fresh-random-nonce>\nSummarize in one sentence: " + 120 x
        "Fact {i}: division {i} reported revenue of {1000+i*7} units at
         margin {10+i%20} percent."
max_tokens=4, temperature=0, stream=true, include_usage=true
require cached_tokens=0
prefill tok/s = (prompt_tokens - cached_tokens) / TTFT
```

Expect **TTFT 3.03–3.10 s → 780–795 tok/s**. Run-to-run spread is ±1%;
report the median of 5, not a single run.

## 4. The quality gate — mandatory before believing any number

```bash
tool-eval-bench --base-url http://127.0.0.1:8977/v1 --short --no-live \
  --json-file /tmp/teb.json
```

The bar is **90/100 with 0 failures** (12 pass / 3 partial). TC-14 is the
known-borderline scenario that flips on numerics; the other 14 must not move.
A speed change that drops this is a regression regardless of tok/s.

## 5. Profiling (how the campaign found its targets)

```bash
scripts/dsflash-serve-bench.sh prof 5 <flags...> ATLAS_PROFILE=1
# run one prefill, note the wall-clock start time, then:
python3 scripts/prefill_waterfall.py serve-prof.log <seconds-since-midnight>
```

Prints the per-bucket waterfall. Sanity check that validates the method: the
`xw_ffn_block` wrapper must equal the sum of its interior buckets to <1 ms.

```bash
ATLAS_GEMM_SHAPE_LOG=1   # every unique (kernel, M, N, K) logged once
```

This is the tool that found most of the campaign's wins — it shows which
kernel each projection ACTUALLY dispatches, which repeatedly was not the
fastest available one. Cross-reference each logged shape against
`dense_gemm_microtest`.

**`nsys` cannot trace prefill on this model** — the ~6-minute weight load
overruns its buffers and the trace is silently dropped, `--delay` included.
`ncu` is not installed. The in-tree probes above are the instrument.

## 6. Kernel-level oracles (run with the GPU free)

```bash
# GEMM kernel comparison at any shape (cosine vs CPU + CUDA-event timing)
cargo run --release -p spark-model --example dense_gemm_microtest \
  --features cuda,gpu-examples -- <kernel> <M> <N> <K>
#   kernels: dense_gemm_bf16 | dense_gemm_tc | dense_gemm_bf16_pipelined | ...

# The campaign's headline comparison:
#   dense_gemm_tc            2410 1024 4096 -> 8.07 ms
#   dense_gemm_bf16_pipelined 2410 1024 4096 -> 0.44 ms   (cosine 1.000000)

cargo run --release -p spark-model --example prefill_attn_tc_microtest \
  --features cuda,gpu-examples -- 0x7C21 <S> <window> <ratio>
bash scripts/check-v4-prefill-attn-tc2-sass.sh
cargo run --release -p spark-model --example hc_pre_microtest --features cuda,gpu-examples
cargo run --release -p spark-model --example w8a8_gemm_microtest --features cuda,gpu-examples
cargo run --release -p spark-model --example w4a16_gemv_grouped_microtest --features cuda,gpu-examples
```

## 7. Artifacts

Run output (`serve-*.log`, `probe-*.json`, `*.bin`) is **gitignored** — it
differs every run and must never gate a reproduction. Curated results that
back a claim in the docs are committed under `docs/probes/`.

## 8. What the numbers are NOT

- Not a single-run best: everything quoted is a median with the spread stated.
- Not comparable to the reference stack's `1055 tok/s` — that figure is at
  252,047 tokens; ours is at 2,410. The prefill curve here is nearly flat with
  length (590 @ 1525 vs 610 @ 3065 measured mid-campaign) because the HCA
  layers attend full-causally, so length is not an escape hatch in either
  direction.
- Not decode: decode numbers live in `docs/SESSION-STATE.md` and are measured
  with `scripts/decode_ab_probe.py`. Its default contract is one warmup plus
  five measured runs; set `ATLAS_BENCH_RECEIPT` to the matching launcher
  active receipt to bind live provenance. The probe validates request-scoped
  speculative-step, serial-token, and low-gear counters, rejects inconsistent
  classifications, and reports medians separately when engine modes differ.
  An older server without those counters remains `UNVERIFIED` and gets no
  aggregate median.
